"""
Multi-head causal self-attention with Rotary Position Embedding (RoPE).

Supports:
  - Standard MHA
  - Grouped Query Attention (GQA) via n_kv_heads
  - Optional Flash Attention 2
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from services.training.core_model.config import ModelConfig


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin cache tensors for RoPE.

    Returns:
        cos, sin: both shape (seq_len, head_dim)
    """
    half_dim = head_dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))  # (half_dim,)

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (seq_len,)
    angles = torch.outer(positions, inv_freq)  # (seq_len, half_dim)
    angles = torch.cat([angles, angles], dim=-1)  # (seq_len, head_dim)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by 90° (used in RoPE application)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    # Broadcast over batch and heads
    cos = cos[None, None, :, :]  # (1, 1, seq, head_dim)
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional GQA and RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.effective_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repeat factor

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # Try to import Flash Attention
        self._use_flash = False
        if config.use_flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore[import]
                self._flash_attn_func = flash_attn_func
                self._use_flash = True
            except ImportError:
                pass

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            cos, sin: (seq_len, head_dim) – RoPE cache
            attention_mask: (batch, 1, seq_len, seq_len) additive mask or None
        Returns:
            (batch, seq_len, hidden_size)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos[:T], sin[:T])

        # Expand KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self._use_flash:
            # Flash Attention expects (B, T, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            attn_out = self._flash_attn_func(q_fa, k_fa, v_fa, causal=True)
            attn_out = attn_out.reshape(B, T, -1)
        else:
            attn_out = self._sdpa(q, k, v, attention_mask, B, T)

        return self.o_proj(attn_out)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention (or PyTorch SDPA fused kernel)."""
        # Use PyTorch 2.0 SDPA when no external mask – fused kernel, memory-efficient
        if attention_mask is None and hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)
            if attention_mask is not None:
                scores = scores + attention_mask
            else:
                causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal, float("-inf"))
            scores = F.softmax(scores, dim=-1)
            scores = self.attn_dropout(scores)
            attn_out = torch.matmul(scores, v)  # (B, H, T, D)

        return attn_out.transpose(1, 2).contiguous().view(B, T, -1)
