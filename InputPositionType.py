import torch
import torch.nn as nn
from enum import Enum
import math

class InputPositionType(Enum):
    NONE = "none"
    ABSOLUTE = "absolute"
    SINUSOIDAL = "sinusoidal"

    @classmethod
    def from_string(cls, name: str):
        name = name.lower().strip()
        for item in cls:
            if item.value == name:
                return item
        raise ValueError(
            f"Invalid position type: {name}. "
            f"Choose from {[x.value for x in cls]}"
        )

class NoPositionalEncoding(nn.Module):
    def forward(self, x):
        return x

class LearnableAbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        S = x.size(1)
        return x + self.pos_embedding[:, :S, :]

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        return x + self.pe[:, :S, :].to(dtype=x.dtype, device=x.device)


def build_position_module(input_position_type: str, d_model: int, max_len: int):
    input_position_type = InputPositionType.from_string(input_position_type)

    if input_position_type == InputPositionType.NONE:
        return NoPositionalEncoding()

    if input_position_type == InputPositionType.ABSOLUTE:
        return LearnableAbsolutePositionalEncoding(d_model, max_len)

    if input_position_type == InputPositionType.SINUSOIDAL:
        return SinusoidalPositionalEncoding(d_model, max_len=max(4096, max_len))

    raise NotImplementedError(f"{input_position_type.value} is not implemented yet")