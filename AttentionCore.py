import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionType(Enum):
    BASIC = "basic"
    T5 = "t5"
    ALIBI = "alibi"
    ROPE = "rope"

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
            is_causal=causal,  # causal is already encoded in attn_mask
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
# Builder
# ============================================================

def build_attention_core(
    attention_type: str,
    n_heads: int,
    head_dim: int,
    dropout: float,
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
        raise NotImplementedError(
            "RoPE is not implemented yet because it modifies q/k before logits."
        )

    raise NotImplementedError(f"{attention_type.value} attention is not implemented yet")