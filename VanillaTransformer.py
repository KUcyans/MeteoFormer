import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from DataPipelineWorkShop import ForecastContext, ModelContext
from typing import List
import logging
# 30 sec
# 

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with built-in NaN and causal masking support.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # ==============================================================
    def _make_attention_mask(self, x_mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Build combined attention mask from:
          - x_mask: per-timestep validity mask, shape (B, S)
          - causal: if True, disallow attending to future tokens

        Returns:
          attn_mask: (B, 1, S, S), dtype=bool
        """
        B, S = x_mask.shape

        # Base mask from data validity (True = keep, False = mask out)
        attn_mask = x_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        if causal:
            # Lower-triangular mask: allow attending to self and past only
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x_mask.device))
            attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        return attn_mask

    # ==============================================================
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
            mask: optional per-timestep validity mask (B, S) where True = valid
            causal: bool, apply causal (lookahead) masking if True

        Returns:
            Tensor of shape (B, S, D)
        """
        B, S, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        # Project Q, K, V
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Build attention mask
        attn_mask = None
        if mask is not None:
            attn_mask = self._make_attention_mask(mask, causal=causal)
            # Convert bool mask → float mask for scaled_dot_product_attention
            attn_mask = attn_mask.logical_not()  # True where to mask
            attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))

        # Core attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # handled manually above
        )

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output


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
    LayerNorm → MultiHeadAttention (+ residual)
    LayerNorm → Feed-Forward Network (+ residual)

    Supports both NaN masking and causal masking.

    Args:
        d_model (int): Input/hidden feature dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Feed-forward expansion dimension.
        dropout (float): Dropout probability.
        activation (str): Activation for FFN ('gelu', 'relu', or 'silu').
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
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
        target_features: List[str],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_ctx", "forecast_ctx"])  # we store contexts separately

        # store contexts
        self.model_ctx = model_ctx
        self.forecast_ctx = forecast_ctx

        # store features
        self.input_features = input_features
        self.feature_dim = len(input_features)
        self.target_features = self._resolve_target_features(target_features, input_features)
        self.target_indices = [input_features.index(t) for t in self.target_features]
        self.target_dim = len(self.target_features)

        # short aliases (for convenience & readability inside code)
        d_model = model_ctx.d_model
        n_heads = model_ctx.n_heads
        d_ff = model_ctx.d_ff
        num_layers = model_ctx.num_layers
        dropout = model_ctx.dropout
        activation = model_ctx.activation

        window = forecast_ctx.window
        horizon = forecast_ctx.horizon
        causal = forecast_ctx.causal

        self.d_model = d_model
        self.window = window
        self.horizon = horizon
        self.causal = causal

        # --- Input projection ---
        self.input_proj = nn.Linear(self.feature_dim, d_model)

        # --- Learnable absolute positional embedding ---
        self.pos_embedding = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # --- Stack of encoder blocks ---
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        # --- Normalisation and output projection ---
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, self.target_dim)

        # --- Loss ---
        self.loss_fn = nn.MSELoss()

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

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding (truncate if seq shorter than max_seq_len)
        x = x + self.pos_embedding[:, :S, :]

        # Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, mask=mask, causal=self.causal)

        # Final normalisation and output projection
        x = self.final_norm(x)
        out = self.output_head(x)

        return out

    # ==============================================================
    def _compute_loss(self, preds, targets, mask):
        """
        Compute MSE loss with masking applied.
        predictions for all S input tokens, 
        but only the last horizon is used for loss:
        pred (B, S, F)
        targets (B, S, F)
        mask (B, S
        """
        valid = mask.any(-1)
        preds_future = preds[:, -self.horizon:, :]
        targets_future = targets[:, -self.horizon:, :]
        if self.target_indices is not None:
            targets_future = targets_future[..., self.target_indices]
            mask = mask[..., self.target_indices]
        valid = mask.any(-1)
        return self.loss_fn(preds_future[valid], targets_future[valid])
    
    # ==============================================================
    def _resolve_target_features(self, target_features: list, input_features: list) -> list:
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

    # ==============================================================
    def training_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        # x: (B, horizon, feature_dim)
        # pred: (B, horizon, target_dim)
        loss = self._compute_loss(preds, y, y_mask)
        self.log("train_loss", loss)

        # --- Periodic printing (every ~1/3 of an epoch) ---
        if self.trainer is not None and self.trainer.train_dataloader is not None:
            period = max(100, len(self.trainer.train_dataloader) // 3)
            if batch_idx % period == 0:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                logging.info(f"\n[Epoch {self.current_epoch} | Batch {batch_idx}]")
                logging.info(f"Train Loss: {loss.item():.6f} | LR: {current_lr:.2e}")
                logging.info(f"Preds: mean={preds.mean().item():.4f}, std={preds.std().item():.4f}")
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
                logging.info(f"\nValidation: Epoch {self.current_epoch}, Batch {batch_idx}")
                logging.info(f"Val Loss: {loss.item():.6f}")
                logging.info(f"Preds: mean={preds.mean().item():.4f}, std={preds.std().item():.4f}")

        return loss

    def test_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        loss = self._compute_loss(preds, y, y_mask)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, x_mask, y_mask = batch
        preds = self.forward(x, mask=x_mask.any(-1))
        # return only the last horizon (forecast window)
        return preds[:, -self.horizon:, :]


    def configure_optimizers(self):
        max_lr = 3e-4
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=max_lr,
                                    weight_decay=1e-4,
                                    betas=(0.9, 0.999))

        # Compute total steps for OneCycleLR
        total_steps = self.trainer.estimated_stepping_batches
        # OneCycleLR configuration
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.1,          # fraction of cycle to reach max LR (default 0.3)
            anneal_strategy='cos',  # cosine annealing works well for transformers
            div_factor=25.0,        # initial LR = max_lr / div_factor
            final_div_factor=1e2,    # min LR = max_lr / (div_factor * final_div_factor)
            three_phase=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # OneCycle updates every step, not epoch
                "frequency": 1,
            },
        }
    def _total_steps(self):
        try:
            return len(self.trainer.train_dataloader) * self.trainer.max_epochs
        except Exception:
            # fallback estimate (useful for debugging or unit tests)
            return 1000


