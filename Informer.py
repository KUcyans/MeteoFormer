"""
Informer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from DataPipelineWorkShop import (ForecastContext, 
                                  ModelContext, 
                                  MeteoPreprocessor,
                                  TrainingContext, 
                                  inverse_predictions_to_df,
                                  build_optimizer_and_scheduler)
from InputPositionType import build_position_module
from AttentionCore import MultiHeadSelfAttention
from typing import List
import logging
import abc
from enum import Enum

# (B, S, F) → (B, S, D)
import math
import abc
import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# If you keep ModelContext/ForecastContext in DataPipelineWorkShop, import them as you did:
# from DataPipelineWorkShop import ForecastContext, ModelContext
# And reuse your closer classes by importing them:
from VanillaTransformer import MeteoTaskCloser, CloserType

# =============================================================================
# FFN (reuse your existing one)
# =============================================================================

class FFN(nn.Module):
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
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# InformerEncoderLayer (LN -> ProbSparseAttn -> res, LN -> FFN -> res)
# =============================================================================

class InformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_ctx: ModelContext,
        factor: int
    ):
        super().__init__()
        d_model = model_ctx.d_model
        n_heads = model_ctx.n_heads
        d_ff = model_ctx.d_ff
        activation = model_ctx.starter_activation
        attention_type = model_ctx.attention_type
        dropout = model_ctx.dropout
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=attention_type,
            factor=factor,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout=dropout, activation=activation)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual + self.drop1(self.attn(x, mask=mask, causal=causal))

        residual = x
        x = self.norm2(x)
        x = residual + self.drop2(self.ffn(x))
        return x


# =============================================================================
# ConvDistillLayer + mask downsampler
# =============================================================================

class ConvDistillLayer(nn.Module):
    """
    Distilling layer (downsample in time) with LayerNorm.

    Input:  (B,S,D)
    Output: (B,S',D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.LayerNorm(d_model)  # LN over feature dim D
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,D)
        x = x.transpose(1, 2)              # (B,D,S)
        x = self.down_conv(x)              # (B,D,S)
        x = x.transpose(1, 2)              # (B,S,D)

        x = self.norm(x)                   # (B,S,D)
        x = self.activation(x)             # (B,S,D)

        x = x.transpose(1, 2)              # (B,D,S)
        x = self.pool(x)                   # (B,D,S')
        x = x.transpose(1, 2)              # (B,S',D)
        return x

    @staticmethod
    def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(1)  # (B,1,S)
        m = F.max_pool1d(m, kernel_size=3, stride=2, padding=1)
        return (m.squeeze(1) > 0.5)

# =============================================================================
# StarterMeteoInformerHourglassEncoder
# =============================================================================

class StarterMeteoInformerHourglassEncoder(nn.Module):
    """
    Starter encoder with hourglass structure (narrowing only):
        input_len S_in
          -> encoder layers
          -> optional distil between layers (S decreases)
    Returns:
        H: (B, S_out, d_model)
        mask_out: (B, S_out) bool
    """

    def __init__(
        self,
        model_ctx: ModelContext,
        forecast_ctx: ForecastContext,
        input_features: List[str],
        distil: bool,
        factor: int,
    ):
        super().__init__()

        self.input_features = input_features
        self.feature_dim = len(input_features)

        d_model = model_ctx.d_model
        num_layers = model_ctx.starter_num_layers
        input_position_type = model_ctx.input_position_type

        self.causal = forecast_ctx.causal
        self.window = forecast_ctx.window
        self.distil = distil

        self.input_proj = nn.Linear(self.feature_dim, d_model)

        # --- Positional encoding: absolute, sinusoidal, or no positional encoding ---
        self.position = build_position_module(
                input_position_type=input_position_type,
                d_model=d_model,
                max_len=self.window,
            )

        self.layers = nn.ModuleList([
            InformerEncoderLayer(
                model_ctx=model_ctx,
                factor=factor,
            )
            for _ in range(num_layers)
        ])

        # distil between layers (not after last layer)
        if self.distil and num_layers > 1:
            self.distil_layers = nn.ModuleList([
                ConvDistillLayer(d_model) for _ in range(num_layers - 1)
            ])
        else:
            self.distil_layers = None

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B,S_in,F), where S_in is forecast_ctx.window
        mask: (B,S_in) bool, True=valid
        """
        B, S, _ = x.shape
        if S > self.window:
            x = x[:, -self.window:, :]
            if mask is not None:
                mask = mask[:, -self.window:]

        x = self.input_proj(x)
        x = self.position(x)

        cur_mask = mask

        for i, layer in enumerate(self.layers):
            x = layer(x, mask=cur_mask, causal=self.causal)

            # distil between layers
            if self.distil_layers is not None and i < len(self.distil_layers):
                x = self.distil_layers[i](x)
                if cur_mask is not None:
                    cur_mask = ConvDistillLayer.downsample_mask(cur_mask)

        x = self.final_norm(x)
        return x, cur_mask

# =============================================================================

# =============================================================================

class ResamplingCloser(nn.Module):
    def __init__(
        self,
        model_ctx: ModelContext,
        input_features: List[str],
        closer_type: CloserType,
        prediction_len: int
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.closer = closer_type.type(
            model_ctx=model_ctx,
            input_features=input_features,
        )

        self.target_indices = self.closer.target_indices

    def forward(self, H: torch.Tensor, H_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if H_mask is not None:
            H = H * H_mask.unsqueeze(-1).to(H.dtype)

        B, S, D = H.shape
        if S != self.prediction_len:
            H = F.interpolate(
                H.transpose(1, 2),
                size=self.prediction_len,
                mode="nearest"
            ).transpose(1, 2)

        return self.closer(H)

    def get_target_features(self):
        return self.closer.get_target_features()

# =============================================================================
# LightningModule wrapper (reuse your closer + loss logic)
# =============================================================================

class MeteoInformerHourglassTransformer(LightningModule):
    """
    Hourglass Informer-style model that reuses your closer classes unchanged.

    Output shape matches vanilla closer output:
        preds: (B, prediction_len, out_dim)
    Loss uses the last horizon positions:
        preds[:, -horizon:, :] vs y[:, -horizon:, idx]
    """

    def __init__(
        self,
        model_ctx: ModelContext,
        forecast_ctx: ForecastContext,
        training_ctx: TrainingContext,
        input_features: List[str],
        closer_type: CloserType,
        factor: int, # related to ProbSparse Attention query selection
        distil: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_ctx", "forecast_ctx", "training_ctx"])

        self.model_ctx = model_ctx
        self.forecast_ctx = forecast_ctx
        self.training_ctx = training_ctx
        self.horizon = forecast_ctx.horizon
        self.input_features = input_features
        self.prediction_len = forecast_ctx.horizon

        self.starter = StarterMeteoInformerHourglassEncoder(
            model_ctx=model_ctx,
            forecast_ctx=forecast_ctx,
            input_features=input_features,
            distil=distil,
            factor=factor,
        )

        self.closer = ResamplingCloser(
            model_ctx=model_ctx,
            input_features=input_features,
            closer_type=closer_type,
            prediction_len=self.prediction_len,
        )
        
        # --- preprocessor ---
        self.preprocessor = MeteoPreprocessor()

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H, H_mask = self.starter(x, mask=mask)
        out = self.closer(H, H_mask=H_mask)
        return out

    def _compute_loss(self, preds, targets, y_mask):
        idx = self.closer.target_indices

        preds_future   = preds[:,   -self.horizon:, :]
        targets_future = targets[:, -self.horizon:, idx]
        mask_future    = y_mask[:,  -self.horizon:, idx]

        valid = mask_future.all(-1)  # (B,H)
        return self.loss_fn(preds_future[valid], targets_future[valid])

    def training_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        loss = self._compute_loss(preds, y, y_mask)
        self.log("train_loss", loss)
        
        # --- Periodic printing (every ~1/3 of an epoch) ---
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
                    + df_pred_inv.describe().loc[["mean", "std", "min", "max"]].to_string()
                )
                self.log("lr", current_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        loss = self._compute_loss(preds, y, y_mask)
        self.log("val_loss", loss, prog_bar=True)
        
        # --- Periodic validation printing ---
        if self.trainer is not None and self.trainer.val_dataloaders is not None:
            period = max(200, len(self.trainer.val_dataloaders) // 3)
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
                    + df_pred_inv.describe().loc[["mean", "std", "min", "max"]].to_string()
                )

        return loss

    def predict_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
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