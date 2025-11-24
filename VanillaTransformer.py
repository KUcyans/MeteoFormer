import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from DataPipelineWorkShop import ForecastContext, ModelContext
from typing import List
import logging
import abc
from enum import Enum
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
    def _make_attention_mask(self, x_mask: torch.Tensor, causal: bool) -> torch.Tensor:
        """
        x_mask: (B, S) where True = valid
        Returns:
            attn_mask: (B, H, S, S) boolean, True = ALLOW, False = BLOCK
        """
        B, S = x_mask.shape
        H = self.n_heads

        # Step 1: timestep validity mask -> (B,1,S)
        base = x_mask.unsqueeze(1)  # (B,1,S)

        # Expand to attention layout (keys dimension)
        base = base.unsqueeze(2).expand(B, 1, S, S)   # (B,1,S,S)
        base = base.expand(B, H, S, S)                # (B,H,S,S)

        if causal:
            causal_mask = torch.tril(
                torch.ones(S, S, dtype=torch.bool, device=x_mask.device)
            ).unsqueeze(0).unsqueeze(0)               # (1,1,S,S)

            causal_mask = causal_mask.expand(B, H, S, S)

            attn_mask = base & causal_mask            # True = allow
        else:
            attn_mask = base

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

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # (B,S) --> (B,H,S,S)
            attn_mask = self._make_attention_mask(mask, causal)

            # Convert to "True = mask out" for SDPA
            attn_mask = ~attn_mask

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


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
        n_heads = model_ctx.n_heads
        d_ff = model_ctx.d_ff
        num_layers = model_ctx.starter_num_layers
        dropout = model_ctx.dropout
        activation = model_ctx.starter_activation

        window = forecast_ctx.window
        self.causal = forecast_ctx.causal

        self.d_model = d_model
        self.window = window

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

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :S, :]

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
        depth = self.HEAD_DEPTH or model_ctx.closer_num_layers
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


class ThermodynamicCloser(MeteoTaskCloser):
    TARGETS = ["temp", "rhum", "pres", "dwpt"]
    HEAD_DEPTH = 4
    HIDDEN_MULTIPLIER = 4

    def forward(self, H):
        return self.net(H)

class ThermoCloser(MeteoTaskCloser):
    TARGETS = ["temp",]
    HEAD_DEPTH = 4
    HIDDEN_MULTIPLIER = 4

    def forward(self, H):
        return self.net(H)
    
class WindCloser(MeteoTaskCloser):
    TARGETS = ["wspd", "sin_wdir", "cos_wdir"]
    HEAD_DEPTH = 8                 
    HIDDEN_MULTIPLIER = 8             

    def forward(self, H):
        return self.net(H)
    
class PrecipitationCloser(MeteoTaskCloser):
    TARGETS = ["prcp"]
    HEAD_DEPTH = 6
    HIDDEN_MULTIPLIER = 8

    def forward(self, H):
        return self.net(H)

class CloserType(Enum):
    Thermo        = (0, ThermoCloser, "thermo")
    Thermodynamic = (1, ThermodynamicCloser, "thermodynamic")
    Wind          = (2, WindCloser, "wind")
    Precipitation = (3, PrecipitationCloser, "precipitation")
    
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_ctx", "forecast_ctx"])

        # store contexts
        self.model_ctx = model_ctx
        self.forecast_ctx = forecast_ctx
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

        idx = self.closer.target_indices   # <-- use closer’s indices

        preds_future   = preds[:,   -self.horizon:, :]       # (B, H, out_dim)
        targets_future = targets[:, -self.horizon:, idx]     # (B, H, out_dim)
        mask_future    = y_mask[:, -self.horizon:, idx]      # (B, H, out_dim)

        valid = mask_future.all(-1)                          # (B, H)

        return self.loss_fn(
            preds_future[valid],
            targets_future[valid]
        )

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
        self._onecycle_cfg = {
            "max_lr": 1e-4,
            "div_factor": 25.0,
            "final_div_factor": 1e3,
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
        
        self._optimizer_cfg = {
            "optimizer": "AdamW",
            "max_lr": self._onecycle_cfg["max_lr"],
            "weight_decay": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._optimizer_cfg["max_lr"],
            weight_decay=self._optimizer_cfg["weight_decay"],
            betas=(self._optimizer_cfg["beta1"], self._optimizer_cfg["beta2"]),
            eps=self._optimizer_cfg["eps"],
        )

        # Compute total steps for OneCycleLR
        total_steps = self.trainer.estimated_stepping_batches
        # OneCycleLR configuration
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
        
    def get_onecycle_config(self):
        return {
            **self._onecycle_cfg,
            "total_steps": self._total_steps()
        }
        
    def get_optimizer_config(self):
        return dict(self._optimizer_cfg)
    
    def get_target_features(self):
        return self.closer.get_target_features()
    
    