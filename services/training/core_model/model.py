"""
LLMModel: decoder-only Transformer (~300M parameters).

Architecture:
  - Token embedding + optional weight-tied output projection
  - 24 × DecoderBlock (pre-norm, causal attention + MLP)
  - Final RMSNorm + linear head
  - RoPE positional encoding (no learned position embeddings)
  - Gradient checkpointing support

Parameter estimate (defaults):
  vocab 32000×1024 embed       = 32.8M
  24 layers × (4×1024² + 2×1024×4096 + 2×1024 norms)
    = 24 × (4.19M + 8.39M + 0.002M) ≈ 302.6M
  Final norm 1024              ≈ negligible
  Output head (tied)           = 0
  Total                        ≈ 302M
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from services.training.core_model.attention import build_rope_cache
from services.training.core_model.blocks import DecoderBlock, RMSNorm
from services.training.core_model.config import ModelConfig


@dataclass
class LLMOutput:
    """Output dataclass returned by LLMModel.forward()."""

    logits: torch.Tensor
    """Shape: (batch, seq_len, vocab_size)."""

    loss: Optional[torch.Tensor] = None
    """Cross-entropy loss when labels are supplied."""


class LLMModel(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head (optionally tied to embedding)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Pre-build RoPE cache (will be extended on demand)
        self._rope_seq_len: int = 0
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # RoPE cache management
    # ------------------------------------------------------------------

    def _get_rope_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._rope_seq_len or (
            self._cos_cache is not None and self._cos_cache.device != device
        ):
            cos, sin = build_rope_cache(
                max(seq_len, self.config.max_seq_len),
                self.config.head_dim,
                theta=self.config.rope_theta,
                device=device,
                dtype=dtype,
            )
            self._cos_cache = cos
            self._sin_cache = sin
            self._rope_seq_len = max(seq_len, self.config.max_seq_len)
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LLMOutput:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
            labels:         (batch, seq_len) target token IDs; -100 positions ignored
        Returns:
            LLMOutput with logits and optional loss
        """
        if inputs_embeds is not None:
            # PEFT may pass pre-computed embeddings; use them directly
            x = inputs_embeds
            B, T = x.shape[:2]
            device, dtype_ref = x.device, self.embed_tokens.weight.dtype
        else:
            assert input_ids is not None, "Either input_ids or inputs_embeds must be provided"
            B, T = input_ids.shape
            device, dtype_ref = input_ids.device, self.embed_tokens.weight.dtype
            x = self.embed_dropout(self.embed_tokens(input_ids))  # (B, T, H)
        cos, sin = self._get_rope_cache(T, device, dtype_ref)

        # Build additive causal attention bias from padding mask
        attn_bias: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # (B, 1, 1, T) → broadcast over heads and query positions
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            attn_bias = pad_mask * torch.finfo(dtype_ref).min

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(layer, x, cos, sin, attn_bias, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attn_bias)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift so that tokens < n predict token n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return LLMOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable) parameter count."""
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        """Minimal HuggingFace-compatible stub required by PEFT PeftModelForCausalLM."""
        return {"input_ids": input_ids}

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        # Rebuild config with checkpointing flag set — we mutate via object.__setattr__
        object.__setattr__(self.config, "use_gradient_checkpointing", True)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "LLMModel":
        """Load a model from a saved checkpoint directory."""
        import json
        import os

        cfg_path = os.path.join(checkpoint_path, "config.json")
        ckpt_path = os.path.join(checkpoint_path, "model.pt")

        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        config = ModelConfig(**cfg_dict)
        model = cls(config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and config to a directory."""
        import json
        import os

        os.makedirs(save_dir, exist_ok=True)
        cfg = self.config
        # ModelConfig.to_dict() returns model_dump(); handle other possible config objects
        if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
            cfg_dict = cfg.to_dict()
        elif hasattr(cfg, "model_dump") and callable(cfg.model_dump):
            cfg_dict = cfg.model_dump()
        else:
            cfg_dict = {}
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
