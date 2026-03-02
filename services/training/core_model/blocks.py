"""
Transformer building blocks:
  - RMSNorm
  - MLP with GELU activation
  - DecoderBlock (pre-norm, attention → MLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from services.training.core_model.attention import CausalSelfAttention
from services.training.core_model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class MLP(nn.Module):
    """Feed-forward network: Linear → GELU → Linear."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))


class DecoderBlock(nn.Module):
    """
    Pre-norm Transformer decoder block:
        x = x + Attn(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
