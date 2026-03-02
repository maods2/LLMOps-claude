"""
Generation quality evaluation.

Generates text from a set of seed prompts and optionally computes:
  - n-gram diversity metrics
  - Length distribution
  - Coherence heuristics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizerFast


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    max_new_tokens: int = Field(128)
    temperature: float = Field(1.0)
    top_p: float = Field(0.9)
    top_k: int = Field(50)
    repetition_penalty: float = Field(1.1)
    do_sample: bool = Field(True)
    num_return_sequences: int = Field(1)

    model_config = ConfigDict(frozen=True)


@dataclass
class GenerationResult:
    """Result for a single generation sample."""

    prompt: str
    generated_text: str
    n_tokens: int
    unique_bigrams_ratio: float = 0.0


def generate_samples(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    prompts: list[str],
    config: GenerationConfig,
    device: torch.device | None = None,
) -> list[GenerationResult]:
    """Generate text for a list of prompts and compute diversity metrics.

    Args:
        model:     Language model.
        tokenizer: Tokenizer matching the model.
        prompts:   List of prompt strings.
        config:    Generation hyperparameters.
        device:    Target device.
    Returns:
        List of GenerationResult, one per prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = _greedy_or_sample(model, input_ids, config, tokenizer)

        generated = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        n_tokens = output_ids.shape[1] - input_ids.shape[1]
        diversity = _bigram_diversity(output_ids[0].tolist())
        results.append(
            GenerationResult(
                prompt=prompt,
                generated_text=generated,
                n_tokens=n_tokens,
                unique_bigrams_ratio=diversity,
            )
        )

    return results


def _greedy_or_sample(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> torch.Tensor:
    """Simple autoregressive generation loop (greedy or sampling)."""
    generated = input_ids.clone()
    eos_id = tokenizer.eos_token_id or -1
    past_tokens: list[int] = []

    for _ in range(config.max_new_tokens):
        out = model(generated)
        next_logits = out.logits[:, -1, :]  # (1, vocab)

        # Repetition penalty
        if config.repetition_penalty != 1.0 and past_tokens:
            for tok in set(past_tokens):
                if next_logits[0, tok] > 0:
                    next_logits[0, tok] /= config.repetition_penalty
                else:
                    next_logits[0, tok] *= config.repetition_penalty

        if config.do_sample:
            next_logits = next_logits / max(config.temperature, 1e-8)
            next_logits = _top_k_top_p_filter(next_logits, config.top_k, config.top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        tok_id = next_token.item()
        past_tokens.append(tok_id)  # type: ignore[arg-type]
        if tok_id == eos_id:
            break

    return generated


def _top_k_top_p_filter(
    logits: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        remove = cum_probs - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

    return logits


def _bigram_diversity(token_ids: list[int]) -> float:
    """Compute unique-bigrams / total-bigrams ratio."""
    if len(token_ids) < 2:
        return 0.0
    bigrams = [(token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)]
    return len(set(bigrams)) / len(bigrams)
