"""
VanillaTransformer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from DataPipelineWorkShop import (ForecastContext, 
                                  ModelContext, 
                                  TrainingContext,
                                  MeteoPreprocessor,
                                  inverse_predictions_to_df,
                                  build_optimizer_and_scheduler)
from InputPositionType import build_position_module
from AttentionCore import MultiHeadSelfAttention
from typing import List
import logging
import abc
from enum import Enum
# 30 sec

class FFN(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used inside Transformer encoder blocks.
    Expands and projects the feature dimension with non-linearity and dropout.

    Formula:
        FFN(x) = Dropout(W2 * Activation(W1 * x)) + residual

    Args:
        d_model (int): Input and output feature dimension.
        d_ff (int): Hidden expansion dimension (usually 2–4× d_model).
        dropout (float): Dropout probability.
        activation (str): Activation function: 'gelu', 'relu', or 'silu'.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, S, D_model)

        Returns:
            Tensor of shape (B, S, D_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
    LayerNorm → MultiHeadSelfAttention (+ residual)
    LayerNorm → Feed-Forward Network (+ residual)

    Supports both NaN masking and causal masking.

    Args:
        model_ctx (ModelContext): the model context
    """

    def __init__(self, model_ctx: ModelContext):
        super().__init__()
        d_model = model_ctx.d_model
        n_heads = model_ctx.n_heads
        d_ff = model_ctx.d_ff
        dropout = model_ctx.dropout
        activation = model_ctx.starter_activation
        attention_type = model_ctx.attention_type

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=attention_type,
            )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout=dropout, activation=activation)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, D)
            mask: Optional mask of shape (B, S) where True = valid
            causal: Whether to apply causal masking (for autoregressive forecasting)

        Returns:
            Tensor of shape (B, S, D)
        """

        # --- Multi-head attention sublayer ---
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x, mask=mask, causal=causal)
        x = residual + self.dropout1(attn_out)

        # --- Feed-forward sublayer ---
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout2(ffn_out)

        return x

## ----- starter & closer style -----
'''
TAE
Transformer AutoEncoder ( it isn't an autoencoder bc it doesn't reconstruct input, but forecast future)

OH
Output head
'''

class StarterMeteoVanillaTransformerEncoder(nn.Module):
    def __init__(
            self,
            model_ctx: ModelContext,
            forecast_ctx: ForecastContext,
            input_features: List[str],
        ):
        super().__init__()
        # store features
        self.input_features = input_features
        self.feature_dim = len(input_features)

        # short aliases (for convenience & readability inside code)
        d_model = model_ctx.d_model
        num_layers = model_ctx.starter_num_layers
        input_position_type = model_ctx.input_position_type

        window = forecast_ctx.window
        self.causal = forecast_ctx.causal

        self.d_model = d_model
        self.window = window

        # --- Input projection ---
        self.input_proj = nn.Linear(self.feature_dim, d_model)

        # --- Positional encoding: absolute, sinusoidal, or no positional encoding ---
        self.position = build_position_module(
            input_position_type=input_position_type,
            d_model=d_model,
            max_len=window,
        )

        # --- Stack of encoder blocks ---
        self.layers = nn.ModuleList([
            EncoderBlock(model_ctx=model_ctx)
            for _ in range(num_layers)
        ])

        # --- Normalisation and output projection ---
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # B, S, _ = x.shape
        x = self.input_proj(x)
        x = self.position(x)

        for layer in self.layers:
            x = layer(x, mask=mask, causal=self.causal)

        return self.final_norm(x)

class MeteoTaskCloser(nn.Module, abc.ABC):
    # subclasses override these
    TARGETS: List[str] = []          # group of variables this closer predicts
    HEAD_DEPTH: int = None           # if None → use model_ctx.closer_num_layers
    HIDDEN_MULTIPLIER: int = 4       # 4×d_model by default

    def __init__(
        self,
        model_ctx: ModelContext,
        input_features: List[str],
    ):
        super().__init__()

        # ===== 1) Subclass declares its own target variables =====
        raw_targets = self.TARGETS

        # ===== 2) Resolve and index targets =====
        self.target_features = self._resolve_target_features(
            input_features, raw_targets
        )
        self.target_indices = [input_features.index(f) for f in self.target_features]
        logging.info(f"Closer targets resolved to: {self.target_features}")
        logging.info(f"Closer target indices: {self.target_indices}")
        self.out_dim = len(self.target_features)

        # ===== 3) Build FFN head with subclass overrides =====
        d_model = model_ctx.d_model
        depth = model_ctx.closer_num_layers or self.HEAD_DEPTH
        dropout = model_ctx.dropout
        hidden_dim = self.HIDDEN_MULTIPLIER * d_model

        act = model_ctx.closer_activation
        if act == "gelu":
            activation = nn.GELU()
        elif act == "relu":
            activation = nn.ReLU()
        elif act == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

        layers = []
        for _ in range(depth - 1):
            layers += [
                nn.Linear(d_model, hidden_dim),
                activation,
                nn.Dropout(dropout),
            ]
            d_model = hidden_dim

        layers.append(nn.Linear(d_model, self.out_dim))
        self.net = nn.Sequential(*layers)

    @abc.abstractmethod
    def forward(self, H):
        pass
    
    def get_target_features(self):
        return self.target_features
    
    @staticmethod
    def _resolve_target_features(input_features: list, target_features: list) -> list:
        """
        Resolve target features against available input features.

        Automatically expands or replaces derived variables (e.g. wdir → sin_wdir, cos_wdir)
        and validates that all targets exist in input_features.

        Returns:
            List of resolved target feature names present in input_features.
        Raises:
            ValueError if a target or its derived components are missing.
        """
        DERIVED_FEATURES = {
            "wdir": ["sin_wdir", "cos_wdir"],
            "hour": ["sin_hour", "cos_hour"],
            "week": ["sin_week", "cos_week"],
            "year": ["sin_year", "cos_year"],
        }

        resolved_targets = []
        for t in target_features:
            if t in input_features:
                resolved_targets.append(t)
            elif t in DERIVED_FEATURES:
                derived = [f for f in DERIVED_FEATURES[t] if f in input_features]
                if len(derived) == len(DERIVED_FEATURES[t]):
                    logging.info(f"⚠️ Target '{t}' expanded to derived features {derived}")
                    resolved_targets.extend(derived)
                else:
                    missing = set(DERIVED_FEATURES[t]) - set(derived)
                    raise ValueError(f"Derived target '{t}' missing expected components: {missing}")
            else:
                raise ValueError(f"Target '{t}' not found in available features: {t}")

        return resolved_targets

class ThermoCloser(MeteoTaskCloser):
    TARGETS = ["temp",]
    HEAD_DEPTH = 4
    HIDDEN_MULTIPLIER = 4

    def forward(self, H):
        return self.net(H)

class PrecipitationCloser(MeteoTaskCloser):
    TARGETS = ["prcp"]
    HEAD_DEPTH = 6
    HIDDEN_MULTIPLIER = 8

    def forward(self, H):
        return self.net(H)
    
class WindSpeedCloser(MeteoTaskCloser):
    TARGETS = ["wspd"]
    HEAD_DEPTH = 8                 
    HIDDEN_MULTIPLIER = 8             

    def forward(self, H):
        return self.net(H)
    
class WindDirectionCloser(MeteoTaskCloser):
    TARGETS = ["sin_wdir", "cos_wdir"]
    HEAD_DEPTH = 8                 
    HIDDEN_MULTIPLIER = 8             

    def forward(self, H):
        return self.net(H)

class WindCloser(MeteoTaskCloser):
    TARGETS = ["wspd", "sin_wdir", "cos_wdir"]
    HEAD_DEPTH = 8                 
    HIDDEN_MULTIPLIER = 8             

    def forward(self, H):
        return self.net(H)
    
class ThermodynamicCloser(MeteoTaskCloser):
    TARGETS = ["temp", "rhum", "pres", "dwpt"]
    HEAD_DEPTH = 4
    HIDDEN_MULTIPLIER = 4

    def forward(self, H):
        return self.net(H)
    
class CloserType(Enum):
    THERMO        = (0, ThermoCloser, "thermo")
    PRECIPITATION = (1, PrecipitationCloser, "precipitation")
    WINDSPEED     = (2, WindSpeedCloser, "wind_speed")
    WINDDIRECTION = (3, WindDirectionCloser, "wind_direction")
    THERMODYNAMIC = (4, ThermodynamicCloser, "thermodynamic")
    WIND          = (5, WindCloser, "wind")
    
    def __init__(self, 
                 val:int,
                 type:MeteoTaskCloser,
                 string:str):
        self._val = val
        self._type = type
        self._string = string
    @property
    def val(self):
        return self._val
    
    @property 
    def string(self):
        return self._string
    
    @property 
    def type(self) -> MeteoTaskCloser:
        return self._type   
    @staticmethod
    def from_string(name: str):
        name = name.lower().strip()
        for closer_type in CloserType:
            if closer_type.string == name:
                return closer_type
        raise ValueError(f"No closer type named '{name}'")
    
    def get_raw_target_features(self):
        return self.type.TARGETS


class MeteoVanillaTransformerEncoder(LightningModule):
    """
    Vanilla Transformer Encoder for meteorological forecasting.

    Features:
      - Learnable absolute positional embeddings
      - NaN-aware attention masking
      - Optional causal masking (for autoregressive forecasts)
      - Fully self-contained (no external Encoder class)
    """

    def __init__(
        self,
        model_ctx: ModelContext,
        forecast_ctx: ForecastContext,
        input_features: List[str],
        closer_type: CloserType,
        training_ctx: TrainingContext,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_ctx", "forecast_ctx", "training_ctx"])

        # store contexts
        self.model_ctx = model_ctx
        self.forecast_ctx = forecast_ctx
        self.training_ctx = training_ctx
        self.horizon = forecast_ctx.horizon
        
        # store features
        self.input_features = input_features

        # short aliases (for convenience & readability inside code)
        # --- Input projection ---
        self.starter = StarterMeteoVanillaTransformerEncoder(
            model_ctx=model_ctx,
            forecast_ctx=forecast_ctx,
            input_features=input_features,
        )
        self.closer = closer_type.type(
            model_ctx=model_ctx,
            input_features=input_features,
        )
        
        # --- preprocessor ---
        self.preprocessor = MeteoPreprocessor()

        # --- Loss ---
        self.loss_fn = nn.MSELoss()
        self._test_outputs = []


    # ==============================================================
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, S, F)
            mask: Optional (B, S) boolean mask where True = valid (non-NaN)
            max seq length >= self.window
        Returns:
            Predicted sequence (B, S, F)
        """
        B, S, _ = x.shape

        H = self.starter(x, mask=mask)
        out = self.closer(H)

        return out

    # ==============================================================
    def _compute_loss(self, preds, targets, y_mask):
        """
        preds:   (B, S, out_dim)
        targets: (B, S, F_total)
        y_mask:  (B, S, F_total)
        """
        idx = self.closer.target_indices

        preds_future   = preds[:,   -self.horizon:, :]
        targets_future = targets[:, -self.horizon:, idx]
        mask_future    = y_mask[:, -self.horizon:, idx]

        valid = mask_future.all(-1)                          # (B, H)

        sq_error = (preds_future - targets_future) ** 2       # (B, H, out_dim)
        sq_error = sq_error.mean(dim=-1)                      # (B, H)

        horizon_weights = torch.linspace(
            0.75,
            1.25,
            steps=self.horizon,
            device=preds.device,
            dtype=preds.dtype,
        )                                                     # (H,)

        weighted_error = sq_error * horizon_weights[None, :]  # (B, H)
        valid_weights = horizon_weights[None, :].expand_as(sq_error)

        # normalised weighted loss
        loss = weighted_error[valid].sum() / valid_weights[valid].sum()

        return loss
    
    def _log_prediction_stats_to_wandb(
        self,
        preds,
        targets,
        y_mask,
        prefix: str,
    ):
        """
        Log simple statistical summaries of predictions, targets, and residuals
        over the forecast horizon.

        preds:   (B, S, out_dim)
        targets: (B, S, F_total)
        y_mask:  (B, S, F_total)
        """

        idx = self.closer.target_indices

        preds_f = preds[:, -self.horizon:, :]          # (B, H, out_dim)
        targets_f = targets[:, -self.horizon:, idx]    # (B, H, out_dim)
        mask_f = y_mask[:, -self.horizon:, idx]        # (B, H, out_dim)

        valid = mask_f.bool()

        pred_valid = preds_f[valid]
        target_valid = targets_f[valid]
        residual_valid = pred_valid - target_valid

        # Avoid logging empty tensors
        if pred_valid.numel() == 0:
            return

        stats = {
            f"{prefix}/pred_mean": pred_valid.mean(),
            f"{prefix}/pred_median": pred_valid.median(),
            f"{prefix}/pred_std": pred_valid.std(unbiased=False),
            f"{prefix}/pred_min": pred_valid.min(),
            f"{prefix}/pred_max": pred_valid.max(),

            f"{prefix}/target_mean": target_valid.mean(),
            f"{prefix}/target_median": target_valid.median(),
            f"{prefix}/target_std": target_valid.std(unbiased=False),
            f"{prefix}/target_min": target_valid.min(),
            f"{prefix}/target_max": target_valid.max(),

            f"{prefix}/residual_mean": residual_valid.mean(),
            f"{prefix}/residual_median": residual_valid.median(),
            f"{prefix}/residual_std": residual_valid.std(unbiased=False),
            f"{prefix}/residual_min": residual_valid.min(),
            f"{prefix}/residual_max": residual_valid.max(),
        }

        self.log_dict(
            stats,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
    # ==============================================================
    def training_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch

        preds = self.forward(x, mask=x_mask.any(-1))
        loss = self._compute_loss(preds, y, y_mask)

        self.log("train_loss", loss, prog_bar=True)

        # --- Periodic printing/logging every ~1/3 of an epoch ---
        if self.trainer is not None and self.trainer.train_dataloader is not None:
            period = max(100, len(self.trainer.train_dataloader) // 3)

            if batch_idx % period == 0:
                df_pred_inv = inverse_predictions_to_df(
                    preds,
                    self.get_target_features(),
                    self.preprocessor,
                    self.horizon,
                )

                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

                logging.info(f"\n[Epoch {self.current_epoch} | Batch {batch_idx}]")
                logging.info(f"Train Loss: {loss.item():.6f} | LR: {current_lr:.2e}")
                logging.info(
                    "Preds inverse-transformed:\n"
                    + df_pred_inv.describe()
                    .loc[["mean", "std", "min", "max"]]
                    .to_string()
                )

                self.log("lr", current_lr, prog_bar=True)

                # --- W&B statistical monitoring ---
                self._log_prediction_stats_to_wandb(
                    preds=preds,
                    targets=y,
                    y_mask=y_mask,
                    prefix="train_stats",
                )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch

        preds = self.forward(x, mask=x_mask.any(-1))
        loss = self._compute_loss(preds, y, y_mask)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # --- Periodic validation printing/logging ---
        if self.trainer is not None and self.trainer.val_dataloaders is not None:
            try:
                period = max(200, len(self.trainer.val_dataloaders) // 3)
            except TypeError:
                period = 200

            if batch_idx % period == 0:
                df_pred_inv = inverse_predictions_to_df(
                    preds,
                    self.get_target_features(),
                    self.preprocessor,
                    self.horizon,
                )

                logging.info(f"\nValidation: Epoch {self.current_epoch}, Batch {batch_idx}")
                logging.info(f"Val Loss: {loss.item():.6f}")
                logging.info(
                    "Preds inverse-transformed:\n"
                    + df_pred_inv.describe()
                    .loc[["mean", "std", "min", "max"]]
                    .to_string()
                )

                self._log_prediction_stats_to_wandb(
                    preds=preds,
                    targets=y,
                    y_mask=y_mask,
                    prefix="val_stats",
                )

        return loss

    def test_step(self, batch, batch_idx):
        x, y_true, x_mask, y_mask = batch

        y_pred = self.forward(x, mask=x_mask.any(-1))

        idx = self.closer.target_indices

        # Slice future windows
        y_pred_f = y_pred[:, -self.horizon:, :]
        y_true_f = y_true[:, -self.horizon:, idx]
        y_mask_f = y_mask[:, -self.horizon:, idx]

        diff = (y_pred_f - y_true_f) * y_mask_f

        mse = (diff ** 2).sum(dim=[0, 1])
        count = y_mask_f.sum(dim=[0, 1])

        out = {"mse_sum": mse, "count": count}

        # === NEW: store results ourselves ===
        self._test_outputs.append(out)

        return out

    def on_test_epoch_end(self):
        outputs = self._test_outputs

        mse_total = torch.stack([o["mse_sum"] for o in outputs]).sum(dim=0)
        count_total = torch.stack([o["count"] for o in outputs]).sum(dim=0)

        rmse_per_feature = torch.sqrt(mse_total / count_total)
        rmse_avg = rmse_per_feature.mean()

        self.log("rmse_avg", rmse_avg, prog_bar=True)

        target_feats = self.get_target_features()
        for i, feat in enumerate(target_feats):
            self.log(f"rmse_{feat}", rmse_per_feature[i])

        # Clear for next evaluation run
        self._test_outputs = []


    def predict_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        # return only the last horizon (forecast window)
        return preds[:, -self.horizon:, :]

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            model=self,
            trainer=self.trainer,
            training_ctx=self.training_ctx,
        )
    def _total_steps(self):
        try:
            return len(self.trainer.train_dataloader) * self.trainer.max_epochs
        except Exception:
            # fallback estimate (useful for debugging or unit tests)
            return 1000
        
    def get_onecycle_config(self):
        return {
            "scheduler": self.training_ctx.scheduler,
            "max_lr": self.training_ctx.max_lr,
            "div_factor": self.training_ctx.div_factor,
            "final_div_factor": self.training_ctx.final_div_factor,
            "pct_start": self.training_ctx.pct_start,
            "anneal_strategy": self.training_ctx.anneal_strategy,
            "three_phase": self.training_ctx.three_phase,
            "total_steps": self._total_steps(),
        }

    def get_optimizer_config(self):
        return {
            "optimizer": self.training_ctx.optimizer,
            "max_lr": self.training_ctx.max_lr,
            "weight_decay": self.training_ctx.weight_decay,
            "beta1": self.training_ctx.beta1,
            "beta2": self.training_ctx.beta2,
            "eps": self.training_ctx.eps,
        }
    
    def get_target_features(self):
        return self.closer.get_target_features()
    
    