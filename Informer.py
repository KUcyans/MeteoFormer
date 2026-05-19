import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from DataPipelineWorkShop import ForecastContext, ModelContext, MeteoPreprocessor, inverse_predictions_to_df
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
# ProbSparseAttentionCore
# =============================================================================

class ProbSparseAttentionCore(nn.Module):
    """
    Core ProbSparse attention (Informer-style) operating on already-projected Q, K, V.

    Inputs:
        q, k, v: (B, H, S, Dh)
        key_valid:  (B, S) bool, True = valid key timestep
        query_valid:(B, S) bool, True = valid query timestep
        causal: bool

    Output:
        out: (B, H, S, Dh)
    """

    def __init__(self, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _mask_invalid_keys(scores: torch.Tensor, key_valid: Optional[torch.Tensor]) -> torch.Tensor:
        # scores: (B,H,Q,K)
        if key_valid is None:
            return scores
        km = key_valid.unsqueeze(1).unsqueeze(1)  # (B,1,1,K)
        return scores.masked_fill(~km, float("-inf"))

    @staticmethod
    def _mask_future_keys_for_selected(scores_top: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """
        scores_top: (B,H,u,S)
        top_idx:    (B,H,u) query positions (absolute indices)
        """
        B, H, u, S = scores_top.shape
        qpos = top_idx.unsqueeze(-1)  # (B,H,u,1)
        kpos = torch.arange(S, device=scores_top.device).view(1, 1, 1, S)
        future = kpos > qpos
        return scores_top.masked_fill(future, float("-inf"))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, S, Dh = q.shape
        scale = Dh ** -0.5

        # u ~ factor * ln(S)
        u = min(S, max(1, int(self.factor * math.log(S + 1))))
        k_sample = min(S, max(1, int(self.factor * math.log(S + 1))))

        # sample some keys for cheap importance estimation
        sample_idx = torch.randint(0, S, (k_sample,), device=q.device)
        k_sampled = k[:, :, sample_idx, :]  # (B,H,k',Dh)

        approx = torch.einsum("bhsd,bhkd->bhsk", q, k_sampled) * scale  # (B,H,S,k')

        # mask sampled keys
        if key_valid is not None:
            key_valid_sample = key_valid[:, sample_idx]  # (B,k')
            approx = self._mask_invalid_keys(approx, key_valid_sample)

        approx_max = approx.max(dim=-1).values
        approx_mean = approx.mean(dim=-1)
        importance = approx_max - approx_mean  # (B,H,S)

        # prevent invalid queries from being selected
        if query_valid is not None:
            importance = importance.masked_fill(~query_valid.unsqueeze(1), float("-inf"))

        top_idx = importance.topk(k=u, dim=-1).indices  # (B,H,u)

        # exact attention only for top queries
        q_top = torch.gather(
            q, dim=2, index=top_idx.unsqueeze(-1).expand(B, H, u, Dh)
        )  # (B,H,u,Dh)

        scores_top = torch.einsum("bhud,bhkd->bhuk", q_top, k) * scale  # (B,H,u,S)
        scores_top = self._mask_invalid_keys(scores_top, key_valid)

        if causal:
            scores_top = self._mask_future_keys_for_selected(scores_top, top_idx)

        attn_top = torch.softmax(scores_top, dim=-1)
        attn_top = self.dropout(attn_top)

        out_top = torch.einsum("bhuk,bhkd->bhud", attn_top, v)  # (B,H,u,Dh)

        # default context for non-selected queries
        if not causal:
            if key_valid is None:
                context = v.mean(dim=2, keepdim=True)
            else:
                w = key_valid.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                denom = w.sum(dim=2, keepdim=True).clamp_min(1.0)
                context = (v * w).sum(dim=2, keepdim=True) / denom
            out = context.expand(B, H, S, Dh).contiguous()
        else:
            out = torch.zeros((B, H, S, Dh), device=v.device, dtype=v.dtype)

        # scatter top outputs into out
        index = top_idx.unsqueeze(-1).expand(B, H, u, Dh)

        if out_top.dtype != out.dtype:
            out_top = out_top.to(out.dtype)

        out.scatter_(dim=2, index=index, src=out_top)

        # zero out invalid query positions (matches your vanilla semantics)
        if query_valid is not None:
            out = out * query_valid.unsqueeze(1).unsqueeze(-1).to(out.dtype)

        return out


# =============================================================================
# InformerMultiHeadSelfAttention (drop-in replacement for MultiHeadAttention)
# =============================================================================

class InformerMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention using ProbSparseAttentionCore.
    Matches your vanilla signature: forward(x, mask=None, causal=False) -> (B,S,D)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.core = ProbSparseAttentionCore(factor=factor, dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        B, S, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B,H,S,Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        key_valid = mask  # (B,S) True=valid
        query_valid = mask

        out = self.core(q, k, v, key_valid=key_valid, query_valid=query_valid, causal=causal)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)
        return self.dropout(out)


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
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        factor: int = 5
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = InformerMultiHeadSelfAttention(d_model, n_heads, dropout=dropout, factor=factor)
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
        model_ctx,
        forecast_ctx,
        input_features: List[str],
        distil: bool = True,
        factor: int = 5,
    ):
        super().__init__()

        self.input_features = input_features
        self.feature_dim = len(input_features)

        d_model = model_ctx.d_model
        n_heads = model_ctx.n_heads
        d_ff = model_ctx.d_ff
        num_layers = model_ctx.starter_num_layers
        dropout = model_ctx.dropout
        activation = model_ctx.starter_activation

        self.causal = forecast_ctx.causal
        self.window = forecast_ctx.window
        self.distil = distil

        self.input_proj = nn.Linear(self.feature_dim, d_model)

        # positional embedding defined on the input window grid (S_in)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.window, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.layers = nn.ModuleList([
            InformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
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
        x = x + self.pos_embedding[:, :x.shape[1], :]

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
        preset_len: int,
    ):
        super().__init__()
        self.preset_len = preset_len
        self.closer = closer_type.type(
            model_ctx=model_ctx,
            input_features=input_features,
        )

        self.target_indices = self.closer.target_indices

    def forward(self, H: torch.Tensor, H_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if H_mask is not None:
            H = H * H_mask.unsqueeze(-1).to(H.dtype)

        B, S, D = H.shape
        if S != self.preset_len:
            H = F.interpolate(
                H.transpose(1, 2),
                size=self.preset_len,
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
        preds: (B, preset_len, out_dim)
    Loss uses the last horizon positions:
        preds[:, -horizon:, :] vs y[:, -horizon:, idx]
    """

    def __init__(
        self,
        model_ctx,
        forecast_ctx,
        input_features: List[str],
        closer_type,                 # your CloserType enum
        preset_len: Optional[int] = None,
        distil: bool = True,
        factor: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_ctx", "forecast_ctx"])

        self.model_ctx = model_ctx
        self.forecast_ctx = forecast_ctx
        self.horizon = forecast_ctx.horizon
        self.input_features = input_features
        self.preset_len = preset_len if preset_len is not None else forecast_ctx.window


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
            preset_len=self.preset_len,
        )
        
        # --- preprocessor ---
        self.preprocessor = MeteoPreprocessor()

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H, H_mask = self.starter(x, mask=mask)
        out = self.closer(H, H_mask=H_mask)
        return out, H_mask

    def _compute_loss(self, preds, targets, y_mask):
        idx = self.closer.target_indices

        preds_future   = preds[:,   -self.horizon:, :]
        targets_future = targets[:, -self.horizon:, idx]
        mask_future    = y_mask[:,  -self.horizon:, idx]

        valid = mask_future.all(-1)  # (B,H)
        return self.loss_fn(preds_future[valid], targets_future[valid])

    def training_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds, _ = self.forward(x, mask=x_mask.any(-1))
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
        preds, _ = self.forward(x, mask=x_mask.any(-1))
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
        preds, _ = self.forward(x, mask=x_mask.any(-1))
        return preds[:, -self.horizon:, :]

    def configure_optimizers(self):
        # copy your existing OneCycle config, unchanged
        self._onecycle_cfg = {
            "max_lr": 3e-4,
            "div_factor": 25.0,
            "final_div_factor": 1e2,
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "three_phase": False,
        }
        self._onecycle_cfg["min_lr"] = (
            self._onecycle_cfg["max_lr"]
            / (self._onecycle_cfg["div_factor"] * self._onecycle_cfg["final_div_factor"])
        )
        self._onecycle_cfg["initial_lr"] = (
            self._onecycle_cfg["max_lr"] / self._onecycle_cfg["div_factor"]
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._onecycle_cfg["max_lr"],
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self._onecycle_cfg["max_lr"],
            total_steps=total_steps,
            pct_start=self._onecycle_cfg["pct_start"],
            anneal_strategy=self._onecycle_cfg["anneal_strategy"],
            div_factor=self._onecycle_cfg["div_factor"],
            final_div_factor=self._onecycle_cfg["final_div_factor"],
            three_phase=self._onecycle_cfg["three_phase"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def get_target_features(self):
        return self.closer.get_target_features()

    def get_onecycle_config(self):
        # small helper as in your vanilla
        def _total_steps():
            try:
                return len(self.trainer.train_dataloader) * self.trainer.max_epochs
            except Exception:
                return 1000
        return {**self._onecycle_cfg, "total_steps": _total_steps()}
