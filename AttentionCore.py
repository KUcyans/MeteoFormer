"""
AttentionCore.py
"""
import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionType(Enum):
    BASIC = "basic"
    T5 = "t5"
    ALIBI = "alibi"
    ROPE = "rope"
    PROBSPARSE = "probsparse"

    @classmethod
    def from_string(cls, name: str):
        name = name.lower().strip()
        for item in cls:
            if item.value == name:
                return item
        raise ValueError(
            f"Invalid attention type: {name}. "
            f"Choose from {[x.value for x in cls]}"
        )

# ============================================================
# Multi-head Self Attetion (wrapper) layer
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        factor: int = 5,
        attention_type: str = "basic",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.core = build_attention_core(
            attention_type=attention_type,
            n_heads=n_heads,
            head_dim=self.head_dim,
            dropout=dropout,
            factor=factor
        )

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
            # True: accounted in attention
            # False: excluded from attention

        attn_output = self.core(
            q, k, v,
            attn_mask=attn_mask,
            causal=causal,
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


# ============================================================
# Basic full attention
# ============================================================

class BasicAttentionCore(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        q, k, v: (B, H, S, Dh)
        attn_mask: optional bool mask, True = allow, False = block
        """
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=causal if attn_mask is None else False,
            # causal is already encoded in attn_mask
        )


# ============================================================
# ALiBi full attention
# ============================================================

class ALiBiAttentionCore(nn.Module):
    """
    Full attention with ALiBi bias.

    For causal=True:
        Bias penalises attention to distant past keys.

    For causal=False:
        Uses symmetric distance penalty |i - j|, which is more natural
        for encoder-style attention.
    """

    def __init__(
        self,
        n_heads: int,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        slopes = self._get_alibi_slopes(n_heads).view(1, n_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
        """
        Standard ALiBi slope construction.
        Works for both power-of-two and non-power-of-two head counts.
        """

        def get_slopes_power_of_2(n):
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)

            extra = ALiBiAttentionCore._get_alibi_slopes(2 * closest_power_of_2)
            extra = extra[0::2][: n_heads - closest_power_of_2].tolist()

            slopes = slopes + extra

        return torch.tensor(slopes, dtype=torch.float32)

    def _make_alibi_bias(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        causal: bool,
    ) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device).view(q_len, 1)
        k_pos = torch.arange(k_len, device=device).view(1, k_len)

        if causal:
            # distance to previous keys
            dist = (q_pos - k_pos).clamp(min=0)
        else:
            # encoder-style symmetric distance
            dist = (q_pos - k_pos).abs()

        dist = dist.to(dtype=dtype).view(1, 1, q_len, k_len)

        # negative bias: farther positions become less attractive
        bias = -self.slopes.to(device=device, dtype=dtype) * dist
        return bias  # (1, H, Q, K)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, Q, Dh = q.shape
        K = k.size(2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        bias = self._make_alibi_bias(
            q_len=Q,
            k_len=K,
            device=q.device,
            dtype=logits.dtype,
            causal=causal,
        )

        logits = logits + bias

        if attn_mask is not None:
            # attn_mask: True = allow, False = block
            logits = logits.masked_fill(
                ~attn_mask,
                torch.finfo(logits.dtype).min,
            )

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)


# ============================================================
# T5-style relative position bias
# ============================================================

class T5RelativePositionBias(nn.Module):
    """
    T5-style learned relative position bias.

    Produces a bias tensor of shape:
        (1, H, Q, K)

    This is added to attention logits before softmax.
    """

    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """
        relative_position: (Q, K)
            Usually key_position - query_position.
        """

        num_buckets = self.num_buckets
        max_distance = self.max_distance

        if causal:
            # Only attend to current/past positions.
            # Future positions get mapped as distance 0 here;
            # actual blocking is handled by attn_mask.
            relative_position = -torch.min(
                relative_position,
                torch.zeros_like(relative_position),
            )
        else:
            # Bidirectional case:
            # half buckets for negative positions, half for positive positions.
            num_buckets = num_buckets // 2

            sign_bucket = (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)

        # Now relative_position is non-negative distance.
        max_exact = num_buckets // 2

        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        buckets = torch.where(
            is_small,
            relative_position,
            relative_position_if_large,
        )

        if not causal:
            buckets = buckets + sign_bucket

        return buckets

    def forward(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
        causal: bool = False,
    ) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device).view(q_len, 1)
        k_pos = torch.arange(k_len, device=device).view(1, k_len)

        # key_position - query_position
        relative_position = k_pos - q_pos  # (Q, K)

        buckets = self._relative_position_bucket(
            relative_position=relative_position,
            causal=causal,
        )

        # (Q, K, H)
        values = self.relative_attention_bias(buckets)

        # (1, H, Q, K)
        values = values.permute(2, 0, 1).unsqueeze(0)

        return values


class T5AttentionCore(nn.Module):
    """
    Full attention with T5-style learned relative position bias.
    """

    def __init__(
        self,
        n_heads: int,
        dropout: float = 0.1,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = T5RelativePositionBias(
            n_heads=n_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, Q, Dh = q.shape
        K = k.size(2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        bias = self.relative_bias(
            q_len=Q,
            k_len=K,
            device=q.device,
            causal=causal,
        )

        logits = logits + bias.to(dtype=logits.dtype)

        if attn_mask is not None:
            # attn_mask: True = allow, False = block
            logits = logits.masked_fill(
                ~attn_mask,
                torch.finfo(logits.dtype).min,
            )

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)

# ============================================================
# RoPE: The Rotary Position Embedding
# ============================================================
class RoPEAttentionCore(nn.Module):
    """
    Full attention with Rotary Position Embedding.

    RoPE is applied to q and k before attention.
    q, k, v: (B, H, S, Dh)
    """

    def __init__(
        self,
        head_dim: int,
        dropout: float = 0.1,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, but got head_dim={head_dim}"
            )

        self.head_dim = head_dim
        self.dropout_p = dropout
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # (S, Dh/2)
        freqs = torch.einsum("s,d->sd", positions, self.inv_freq.to(device))

        sin = freqs.sin().to(dtype=dtype)
        cos = freqs.cos().to(dtype=dtype)

        # (1, 1, S, Dh/2), broadcast over B and H
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        return sin, cos

    @staticmethod
    def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """
        x:   (B, H, S, Dh)
        sin: (1, 1, S, Dh/2)
        cos: (1, 1, S, Dh/2)
        """
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        return x_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        _, _, q_len, _ = q.shape
        _, _, k_len, _ = k.shape

        if q_len != k_len:
            raise NotImplementedError(
                "This RoPE core currently assumes q_len == k_len for self-attention."
            )

        sin, cos = self._get_sin_cos(
            seq_len=q_len,
            device=q.device,
            dtype=q.dtype,
        )

        q = self._apply_rope(q, sin, cos)
        k = self._apply_rope(k, sin, cos)

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=causal if attn_mask is None else False,
            # causal is already encoded in attn_mask
        )
# ============================================================
# ProbSparse
# ============================================================
        
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
        attn_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, S, Dh = q.shape
        scale = Dh ** -0.5

        key_valid = None
        query_valid = None

        if attn_mask is not None:
            # attn_mask: (B, H, Q, K), True = allow

            # valid key if at least one query can attend to it
            key_valid = attn_mask.any(dim=2).any(dim=1)      # (B, K)

            # valid query if it can attend to at least one key
            query_valid = attn_mask.any(dim=-1).any(dim=1)   # (B, Q)

        # u ~ factor * ln(S)
        u = min(S, max(1, int(self.factor * math.log(S + 1))))
        k_sample = min(S, max(1, int(self.factor * math.log(S + 1))))

        sample_idx = torch.randint(0, S, (k_sample,), device=q.device)
        k_sampled = k[:, :, sample_idx, :]

        approx = torch.einsum("bhsd,bhkd->bhsk", q, k_sampled) * scale

        if key_valid is not None:
            key_valid_sample = key_valid[:, sample_idx]
            approx = self._mask_invalid_keys(approx, key_valid_sample)

        approx_max = approx.max(dim=-1).values
        approx_mean = approx.mean(dim=-1)
        importance = approx_max - approx_mean

        if query_valid is not None:
            importance = importance.masked_fill(
                ~query_valid.unsqueeze(1),
                torch.finfo(importance.dtype).min,
            )

        top_idx = importance.topk(k=u, dim=-1).indices

        q_top = torch.gather(
            q,
            dim=2,
            index=top_idx.unsqueeze(-1).expand(B, H, u, Dh),
        )

        scores_top = torch.einsum("bhud,bhkd->bhuk", q_top, k) * scale
        scores_top = self._mask_invalid_keys(scores_top, key_valid)

        if causal:
            scores_top = self._mask_future_keys_for_selected(scores_top, top_idx)

        attn_top = torch.softmax(scores_top, dim=-1)
        attn_top = self.dropout(attn_top)

        out_top = torch.einsum("bhuk,bhkd->bhud", attn_top, v)

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

        index = top_idx.unsqueeze(-1).expand(B, H, u, Dh)
        out.scatter_(dim=2, index=index, src=out_top.to(out.dtype))

        if query_valid is not None:
            out = out * query_valid.unsqueeze(1).unsqueeze(-1).to(out.dtype)

        return out

# ============================================================
# Builder
# ============================================================

def build_attention_core(
    attention_type: str,
    n_heads: int,
    head_dim: int,
    dropout: float,
    factor: int = 5,
):
    attention_type = AttentionType.from_string(attention_type)

    if attention_type == AttentionType.BASIC:
        return BasicAttentionCore(dropout=dropout)

    if attention_type == AttentionType.ALIBI:
        return ALiBiAttentionCore(
            n_heads=n_heads,
            dropout=dropout,
        )

    if attention_type == AttentionType.T5:
        return T5AttentionCore(
            n_heads=n_heads,
            dropout=dropout,
        )

    if attention_type == AttentionType.ROPE:
        return RoPEAttentionCore(
            head_dim=head_dim,
            dropout=dropout,
        )
    
    if attention_type == AttentionType.PROBSPARSE:
        return ProbSparseAttentionCore(
            factor=factor,
            dropout=dropout,
        )

    raise NotImplementedError(f"{attention_type.value} attention is not implemented yet")