"""
Tokenizer wrapper around HuggingFace tokenizers.

Provides a unified interface for:
  - Loading pre-trained tokenizers (BPE / SentencePiece)
  - Training a tokenizer from scratch on local data
  - Encoding / decoding with padding and truncation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence
from transformers import PreTrainedTokenizerFast


_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]


def build_tokenizer_from_scratch(
    files: list[str],
    vocab_size: int = 32000,
    save_dir: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """Train a BPE tokenizer from raw text files.

    Args:
        files: Paths to plain-text training files.
        vocab_size: Target vocabulary size.
        save_dir: If provided, saves the tokenizer here.
    Returns:
        A HuggingFace-compatible fast tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormSequence([NFD(), Lowercase(), StripAccents()])  # type: ignore[call-arg]
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train(files=files, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
    )

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fast_tokenizer.save_pretrained(save_dir)

    return fast_tokenizer


def load_tokenizer(path: Union[str, Path]) -> PreTrainedTokenizerFast:
    """Load a saved tokenizer from disk or HuggingFace Hub.

    Args:
        path: Local directory path or HF Hub model id.
    Returns:
        Loaded tokenizer.
    """
    tok = PreTrainedTokenizerFast.from_pretrained(str(path))
    # Ensure padding token is set
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_default_tokenizer(cache_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
    """Return a suitable default tokenizer (GPT-2 BPE) for quick experiments.

    Downloads from HuggingFace if not cached.
    """
    from transformers import AutoTokenizer

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        "gpt2", cache_dir=cache_dir, use_fast=True
    )
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok
