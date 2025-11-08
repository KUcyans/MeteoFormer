
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Point, Hourly, Stations
from tqdm import tqdm
import sys
import numpy as np
from typing import Optional, List

from dataclasses import dataclass
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

# ==============================================================
@dataclass(frozen=True)
class PreprocessingContext:
    use_cyclic: bool = True
    categorical_mode: str = "embedding"

    def build(self):
        return MeteoPreprocessor(self.use_cyclic, self.categorical_mode)

@dataclass(frozen=True)
class ForecastContext:
    window: int
    horizon: int
    causal: bool = False
    overlap: bool = True
    strict: bool = False
    gap: int = 60
    val_ratio: float = 0.2
    test_ratio: float = 0.1

@dataclass(frozen=True)
class ModelContext:
    d_model: int
    n_heads: int
    d_ff: int
    num_layers: int
    dropout: float
    activation: str = "gelu"

@dataclass(frozen=True)
class ExperimentContext:
    preprocessing: PreprocessingContext
    forecast: ForecastContext
    model: ModelContext


# ==============================================================
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
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # angle reconstruction
        if self.add_time:
            # hours
            df['hour'] = (np.degrees(np.arctan2(df['sin_hour'],df['cos_hour'])) / 360 * 24) % 24
            df['dayofweek'] = (np.degrees(np.arctan2(df['sin_week'],df['cos_week'])) / 360 * 7) % 7
            df['dayofyear'] = (np.degrees(np.arctan2(df['sin_year'],df['cos_year'])) / 360 * 365) % 365

            df = df.drop(columns=[
                'sin_hour','cos_hour',
                'sin_week','cos_week',
                'sin_year','cos_year'
            ], errors='ignore')

        if self.add_angle:
            df['wdir'] = (np.degrees(np.arctan2(df['sin_wdir'],df['cos_wdir'])) % 360)
            df = df.drop(columns=['sin_wdir','cos_wdir'], errors='ignore')

        return df

# ==============================================================
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
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col,(shift,scale) in self.scale_shift.items():
            if col in df.columns:
                df[col] = df[col] * scale + shift

        for col,(offset,zero_val) in self.log_transform.items():
            if col in df.columns:
                x = df[col]
                out = np.where(x==zero_val, 0, (10**x) - offset )
                df[col] = out

        return df

# ==============================================================    
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
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.method == 'onehot' and self.onehot_columns is not None:
            # inverse onehot -> coco
            oh = df[self.onehot_columns].values
            coco = oh.argmax(axis=1).astype("Int64")
            df[self.column] = coco
            df = df.drop(columns=self.onehot_columns, errors='ignore')

        # embedding mode: already integer; nothing to do
        return df

# ==============================================================
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

    def transform(self, df):
        df = self.cyclic.transform(df)
        df = self.encoder.transform(df)
        df = self.scaler.transform(df)
        return df
    
    def inverse_transform(self, df):
        df = self.scaler.inverse_transform(df)
        df = self.encoder.inverse_transform(df)
        df = self.cyclic.inverse_transform(df)
        return df

    # def __call__(self, data):
    #     return self.transform(data)

# ==============================================================
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

    def __init__(
        self,
        source_data: pd.DataFrame,
        forecast: ForecastContext,
        preprocessing: PreprocessingContext
    ):


        # store contexts
        self.forecast_ctx = forecast
        self.preprocessor = MeteoPreprocessor(
            use_cyclic=preprocessing.use_cyclic,
            categorical_mode=preprocessing.categorical_mode
        )

        self.window_size = forecast.window
        self.horizon = forecast.horizon
        self.overlap = forecast.overlap
        self.strict = forecast.strict

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

        # --- Apply preprocessing from context ---
        print("🧭 Applying in-dataset preprocessing...")
        self.source_data = self.preprocessor.transform(self.source_data)

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

def temporal_split(df: pd.DataFrame, fc) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pure chronological split with gap zones.

    Parameters
    ----------
    df : pd.DataFrame
        full chronological dataframe (DatetimeIndex)
    fc : ForecastContext
        must contain: val_ratio, test_ratio, gap

    Returns
    -------
    (df_train, df_val, df_test)
    """
    n = len(df)
    n_test = int(n * fc.test_ratio)
    n_val  = int(n * fc.val_ratio)
    n_train = n - n_val - n_test

    # compute gap boundaries
    train_end = n_train - fc.gap
    val_start = n_train + fc.gap
    val_end   = n_train + n_val - fc.gap
    test_start= n_train + n_val + fc.gap

    # clamp boundaries defensively
    train_end = max(train_end, 0)
    val_start = min(val_start, n)
    val_end   = min(val_end, n)
    test_start= min(test_start, n)

    # slice
    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[val_start:val_end].copy()
    df_test  = df.iloc[test_start:].copy()

    # Print shapes for REPL feedback
    print(f"Split shapes → Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")

    # assert strict chronological non-overlap
    if len(df_train) > 0 and len(df_val) > 0:
        assert df_train.index.max() < df_val.index.min(), "Train–Val overlap"
    if len(df_val) > 0 and len(df_test) > 0:
        assert df_val.index.max() < df_test.index.min(), "Val–Test overlap"

    return df_train, df_val, df_test


def make_datasets(df: pd.DataFrame, ctx: ExperimentContext):
    df_train, df_val, df_test = temporal_split(df, ctx.forecast)   # a pure function, pure output
    train_ds = MeteoDataset(df_train, ctx.forecast, ctx.preprocessing)
    val_ds   = MeteoDataset(df_val,   ctx.forecast, ctx.preprocessing)
    test_ds  = MeteoDataset(df_test,  ctx.forecast, ctx.preprocessing)
    return train_ds, val_ds, test_ds


def make_dataloaders(df: pd.DataFrame, ctx: ExperimentContext,batch_size: int=128, num_workers: int=2):
    train_ds, val_ds, test_ds = make_datasets(df, ctx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl


def make_predict_loader(df_future: pd.DataFrame, ctx: ExperimentContext,
                        batch_size: int = 128, num_workers: int = 2):
    pred_ds = MeteoDataset(df_future, ctx.forecast, ctx.preprocessing)
    pred_dl = DataLoader(pred_ds, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    return pred_dl