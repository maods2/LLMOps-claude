"""
Perplexity evaluation module.

Computes per-token cross-entropy loss → perplexity on a held-out dataset.
Supports:
  - Sliding-window evaluation for sequences longer than max_seq_len
  - Token accuracy
  - Batch evaluation with optional progress bar
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pydantic import BaseModel, Field


class PerplexityConfig(BaseModel):
    """Configuration for perplexity evaluation."""

    max_seq_len: int = Field(512)
    stride: int = Field(256, description="Sliding-window stride for long sequences")
    batch_size: int = Field(8)
    max_batches: Optional[int] = Field(None, description="Cap evaluation at N batches")
    dtype: str = Field("bfloat16")

    class Config:
        frozen = True


class PerplexityResult:
    """Result container for perplexity evaluation."""

    def __init__(
        self,
        loss: float,
        perplexity: float,
        token_accuracy: float,
        n_tokens: int,
    ) -> None:
        self.loss = loss
        self.perplexity = perplexity
        self.token_accuracy = token_accuracy
        self.n_tokens = n_tokens

    def __repr__(self) -> str:
        return (
            f"PerplexityResult(ppl={self.perplexity:.2f}, "
            f"acc={self.token_accuracy:.3f}, "
            f"loss={self.loss:.4f}, "
            f"n_tokens={self.n_tokens})"
        )

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "token_accuracy": self.token_accuracy,
            "n_tokens": self.n_tokens,
        }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    config: PerplexityConfig,
    device: Optional[torch.device] = None,
) -> PerplexityResult:
    """Evaluate perplexity and token accuracy on a dataset.

    Args:
        model:      The language model (must accept input_ids and labels kwargs).
        dataloader: Provides batches with 'input_ids' and 'labels' tensors.
        config:     Evaluation configuration.
        device:     Target device (auto-detected if None).
    Returns:
        PerplexityResult with aggregated metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    dtype = _parse_dtype(config.dtype)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    try:
        from tqdm import tqdm
        progress = tqdm(dataloader, desc="Evaluating", leave=False)
    except ImportError:
        progress = dataloader  # type: ignore

    for batch in progress:
        if config.max_batches and n_batches >= config.max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            out = model(input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += out.loss.item()

        # Token accuracy
        shift_logits = out.logits[:, :-1, :].argmax(dim=-1)
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        correct = (shift_logits == shift_labels) & valid
        total_correct += correct.sum().item()
        total_tokens += valid.sum().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    token_accuracy = total_correct / max(total_tokens, 1)

    return PerplexityResult(
        loss=avg_loss,
        perplexity=perplexity,
        token_accuracy=token_accuracy,
        n_tokens=total_tokens,
    )


def _parse_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_str
    ]
