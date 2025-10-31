
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Point, Hourly, Stations
from tqdm import tqdm
import sys
import numpy as np
from typing import Optional, List
# 5 sec: depends a lot on the internet connection
# 30 sec – the south west corner of the black diamond
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
# 30 sec

def get_hourly_example(location: Point, start: datetime, end: datetime ) -> pd.DataFrame: 
    example_df = Hourly(location, start, end) 
    return example_df.fetch() 

def isClean(df: pd.DataFrame) -> bool:
    """
    Return True if the DataFrame contains no NaN values.
    If NaNs exist, print each column with its NaN count and return False.
    """
    _is_clean = False
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    
    if nan_columns.empty:
        _is_clean = True
    else:
        print("NaN values detected in the following columns:")
        for col, count in nan_columns.items():
            print(f"  → Column: '{col}', NaN count: {count}")

    return _is_clean


class CyclicConversion:
    """
    Convert time- and angle-based features into cyclic (sin/cos) representations
    to preserve their periodic nature.

    Example:
        23:00 and 01:00 are numerically far apart but cyclically close.
        Using sin/cos encodings helps the model learn continuity across cycles.

    Parameters
    ----------
    add_time : bool, optional
        If True, adds cyclic encodings for hour, weekday, and day-of-year.
    add_angle : bool, optional
        If True, replaces 'wdir' (wind direction, degrees) with sin/cos encoding.
    """

    def __init__(self, add_time=True, add_angle=True):
        self.add_time = add_time
        self.add_angle = add_angle

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cyclic feature transformations on the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a DatetimeIndex (for temporal encoding)
            and optionally a 'wdir' column (for angular encoding).

        Returns
        -------
        pd.DataFrame
            DataFrame with new sin/cos features and removed raw cyclic columns.
        """
        df = df.copy()

        # --- Temporal encoding ---
        if self.add_time:
            # Extract temporal components from index
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['dayofyear'] = df.index.dayofyear

            # Encode each as sine/cosine pairs to preserve cyclic continuity
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['sin_week'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['cos_week'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['sin_year'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
            df['cos_year'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

            # Drop raw integer-based time features (not cyclically meaningful)
            df = df.drop(columns=['hour', 'dayofweek', 'dayofyear'], errors='ignore')

        # --- Angular encoding (wind direction) ---
        if self.add_angle and 'wdir' in df.columns:
            # Convert wind direction (°) to unit circle representation
            df['sin_wdir'] = np.sin(2 * np.pi * df['wdir'] / 360)
            df['cos_wdir'] = np.cos(2 * np.pi * df['wdir'] / 360)
            df = df.drop(columns=['wdir'], errors='ignore')

        return df

class PseudoNormaliser:
    """
    Lightweight fixed-constant normaliser.
    Scales meteorological features to approximately [-1, 1]
    using predefined constants.

    Features are grouped by normalisation type:
      - scale–shift: linear transform  (x - shift) / scale
      - log: nonlinear transform  log(x + log_offset) with replacement for zeros

    This avoids dataset-specific fitting and ensures consistent scaling
    across datasets.
    """

    def __init__(self):
        # --- Linear (scale–shift) features ---
        # Each tuple: (shift, scale)
        self.scale_shift = {
            # temperature-related (°C)
            'temp': (15.0, 20.0),   # roughly [-5, +35]
            'dwpt': (0.0, 20.0),
            # relative humidity (%)
            'rhum': (50.0, 50.0),   # [0, 100]
            # snow (mm)
            'snow': (0.0, 50.0),    # mostly 0; rare spikes
            # wind (km/h)
            'wspd': (20.0, 20.0),   # [0, 40]
            'wpgt': (30.0, 30.0),   # [0, 60]
            # pressure (hPa)
            'pres': (1013.0, 30.0), # [980, 1040]
            # sunshine (minutes)
            'tsun': (30.0, 30.0),   # [0, 60]
        }

        # --- Log-transformed features ---
        # Each tuple: (log_offset, zero_replacement)
        #   log_offset → value added before log
        #   zero_replacement → used when raw value == 0
        self.log_transform = {
            'prcp': (1.0, -1.0),  # log10(x + 1), replace 0 with -1
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Exclude NaNs from processing but keep their positions ---
        nan_mask = df.isna()

        # --- Apply linear transforms ---
        for col, (shift, scale) in self.scale_shift.items():
            if col in df.columns:
                df[col] = (df[col] - shift) / scale

        # --- Apply log transforms ---
        for col, (log_offset, zero_value) in self.log_transform.items():
            if col in df.columns:
                coldata = df[col]
                transformed = pd.Series(np.nan, index=coldata.index, dtype=float)
                transformed[coldata > 0] = np.log10(coldata[coldata > 0] + log_offset)
                transformed[coldata == 0] = zero_value
                df[col] = transformed

        # --- Restore NaNs explicitly ---
        df[nan_mask] = np.nan

        # Do NOT fill NaNs — they’ll be handled later through masking
        return df
    
class CategoricalEncoder:
    """
    Encode categorical weather condition feature ('coco') using either:
      - one-hot encoding (dense numerical expansion), or
      - embedding indices (integer codes suitable for learnable embeddings).

    Parameters
    ----------
    column : str, optional
        Name of the categorical column to encode (default: 'coco').
    method : {'onehot', 'embedding'}, optional
        Encoding strategy.
    num_classes : int, optional
        Number of unique classes (default: 27 for Meteostat weather codes).

    Notes
    -----
    - One-hot encoding expands the column into multiple binary features.
    - Embedding mode keeps a single integer feature in range [0, num_classes-1].
      The integer can be later mapped to a learnable embedding matrix in the model.
    """

    def __init__(self, column='coco', method='embedding', num_classes=27):
        self.column = column
        self.method = method
        self.num_classes = num_classes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply categorical encoding transformation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the categorical feature.

        Returns
        -------
        pd.DataFrame
            DataFrame with the categorical column transformed according to the selected method.
        """
        df = df.copy()

        # Skip if the specified column is not found
        if self.column not in df.columns:
            return df

        # --- One-hot encoding ---
        if self.method == 'onehot':
            one_hot = pd.get_dummies(
                df[self.column].astype(int),
                prefix=self.column,
                columns=[self.column],
                drop_first=False
            )
            df = df.drop(columns=[self.column])
            df = pd.concat([df, one_hot], axis=1)

        # --- Embedding encoding ---
        elif self.method == 'embedding':
            # Fill missing values and convert to integer IDs
            df[self.column] = df[self.column].astype('Int64')  # nullable integer type
            # df[self.column] = df[self.column].fillna(0).astype(int)
            # Clip to valid index range for embedding layer (0–num_classes−1)
            # df[self.column] = df[self.column].clip(0, self.num_classes - 1)
            df[self.column] = df[self.column].mask(
            df[self.column].notna(),
            df[self.column].clip(0, self.num_classes - 1)
        )


        # --- Invalid method ---
        else:
            raise ValueError("method must be either 'onehot' or 'embedding'")

        return df


class MeteoPreprocessor:
    """
    Composes cyclic, categorical, and pseudo-normalisation transformations.
    Accepts either pandas.DataFrame (preferred) or numpy.ndarray input.
    """

    def __init__(self,
                 use_cyclic: bool = True,
                 categorical_mode: str = 'embedding'):
        self.use_cyclic = use_cyclic
        self.cyclic = CyclicConversion() if use_cyclic else None
        self.encoder = CategoricalEncoder(method=categorical_mode)
        self.scaler = PseudoNormaliser()

    def transform(self, data):
        """Apply transformations, supporting both pandas and numpy input."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if self.cyclic:
                df = self.cyclic.transform(df)
            df = self.encoder.transform(df)
            df = self.scaler.transform(df)
            return df

        elif isinstance(data, np.ndarray):
            # Only apply numeric scaling on numpy arrays (fast path)
            return self._transform_numpy(data)

        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

    def _transform_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Fast-path numeric-only transformation for numpy arrays."""
        # Example: apply fixed scaling assuming same column order as training
        # For simplicity, use global constants from PseudoNormaliser
        arr = arr.copy()
        # You can implement constant-based scaling here
        return arr

    def __call__(self, data):
        return self.transform(data)


class MeteoDataset(Dataset):
    """
    Dataset for meteorological sequence-to-sequence forecasting.
    Optionally applies preprocessing (cyclic, categorical, normalisation).
    """
    ALL_FEATURES = [
        "station", 
        "time", 
        "temp", 
        "dwpt", 
        "rhum", 
        "prcp", 
        "wspd", 
        "wpgt",
        "pres", 
        "tsun", 
        "snow",
        # "coco", 
        "sin_hour", "cos_hour",
        "sin_week", "cos_week", 
        "sin_year", "cos_year",
        "sin_wdir", "cos_wdir", 
    ]

    def __init__(self,
                 source_data: pd.DataFrame,
                 window_size: int = 24,
                 horizon: int = 12,
                 overlap: bool = True,
                 strict: bool = False):
        self.window_size = window_size
        self.horizon = horizon
        self.overlap = overlap
        self.strict = strict
        self.preprocessor = MeteoPreprocessor()
        
        # --- Defensive copy ---
        self.source_data = source_data.copy(deep=True)
        
        # Ensure all columns exist, even if entirely NaN
        for col in self.ALL_FEATURES:
            if col not in source_data.columns:
                source_data[col] = np.nan


        # --- Ensure datetime index BEFORE preprocessing ---
        if not isinstance(self.source_data.index, pd.DatetimeIndex):
            if "time" in self.source_data.columns:
                self.source_data["time"] = pd.to_datetime(self.source_data["time"])
                self.source_data = self.source_data.set_index("time")
            else:
                raise ValueError("No DatetimeIndex or 'time' column found for cyclic encoding.")

        # --- Apply preprocessing ---
        print("🧭 Applying in-dataset preprocessing...")
        self.source_data = self.preprocessor(self.source_data)

        # --- Now drop metadata (AFTER preprocessing) ---
        for col in ['station', 'time']:
            if col in self.source_data.columns:
                self.source_data = self.source_data.drop(columns=[col])

        # --- Convert nullable dtypes to float ---
        self.source_data = self.source_data.convert_dtypes()
        for col in self.source_data.columns:
            if pd.api.types.is_extension_array_dtype(self.source_data[col].dtype):
                self.source_data[col] = self.source_data[col].astype(float)

        # --- Identify usable columns ---
        self.available_features = self._validate_features()

        # --- Subset to usable columns ---
        self.source_data = self.source_data[self.available_features]

        # --- Convert to NumPy float32 array ---
        self.values = self.source_data.values.astype(np.float32)

        # --- Build sliding windows ---
        self.windows = self._make_windows()

        # === 🧾 Summary ===
        if len(self.windows) > 0:
            x0, y0 = self.windows[0]
            print(f"📊 MeteoDataset built: {len(self.windows)} samples | "
                  f"Input shape: {x0.shape} | Target shape: {y0.shape} | "
                  f"Features: {len(self.available_features)} ({self.available_features})")
        else:
            print("⚠️ MeteoDataset built with 0 samples!")

    # ==============================================================
    def _get_available_features(self) -> List[str]:
        return self.available_features
    
    def _validate_features(self) -> List[str]:
        """Keep all declared columns; warn if some are entirely NaN."""
        cols = list(self.source_data.columns)
        all_nan = [c for c in cols if self.source_data[c].isna().all()]

        if all_nan:
            print(f"⚠️ Columns all NaN but retained for consistency: {all_nan}")

        return cols

    # ==============================================================
    def _make_windows(self):
        """Construct (X, Y) window pairs."""
        X, Y = [], []
        step = 1 if self.overlap else self.window_size
        n = len(self.values)
        for i in range(0, n - self.window_size - self.horizon + 1, step):
            x = self.values[i : i + self.window_size]
            y = self.values[i + self.window_size : i + self.window_size + self.horizon]
            X.append(x)
            Y.append(y)
        return list(zip(X, Y))

    # ==============================================================
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Create masks: True where data is valid
        x_mask = ~torch.isnan(x_tensor)
        y_mask = ~torch.isnan(y_tensor)

        # Optionally replace NaNs with zeros for numerical stability
        x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0)

        return x_tensor, y_tensor, x_mask, y_mask

class MeteoDatasetModule(LightningDataModule):
    """
    Time-aware DataModule for Meteostat forecasting.
    Prevents leakage by inserting temporal gaps between train, val, and test.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 window_size: int = 24,
                 horizon: int = 12,
                 gap: int = 60,
                 batch_size: int = 128,
                 num_workers: int = 2,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 shuffle_train: bool = True):
        super().__init__()

        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.gap = gap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle_train = shuffle_train

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ===========================================================
    def setup(self, stage: Optional[str] = None):
        """
        Chronologically split with optional 'gap' zones between splits.
        Gaps ensure no window from training overlaps with validation/test data.
        """
        n = len(self.data)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        # compute gap indices (avoid overlap)
        train_end = n_train - self.gap
        val_start = n_train + self.gap
        val_end = n_train + n_val - self.gap
        test_start = n_train + n_val + self.gap

        # enforce boundaries
        train_end = max(train_end, 0)
        val_start = min(val_start, n)
        val_end = min(val_end, n)
        test_start = min(test_start, n)

        # slice subsets
        df_train = self.data.iloc[:train_end].copy()
        df_val   = self.data.iloc[val_start:val_end].copy()
        df_test  = self.data.iloc[test_start:].copy()

        # sanity check (no overlap)
        assert df_train.index.max() < df_val.index.min(), "Train–Val overlap detected!"
        assert df_val.index.max() < df_test.index.min(), "Val–Test overlap detected!"

        # instantiate datasets
        self.train_dataset = MeteoDataset(df_train, self.window_size, self.horizon)
        self.val_dataset = MeteoDataset(df_val, self.window_size, self.horizon)
        self.test_dataset = MeteoDataset(df_test, self.window_size, self.horizon)

    # ===========================================================
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_available_features(self):
        return self.train_dataset._get_available_features()


