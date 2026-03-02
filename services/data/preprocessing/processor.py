"""
Data preprocessing pipeline.

Responsibilities:
  - Tokenize raw text
  - Pack sequences into fixed-length chunks (for pretraining)
  - Format instruction-response pairs (for SFT)
  - Format preference pairs (for DPO)
  - Cache processed datasets to disk
"""

from __future__ import annotations

import torch
from datasets import Dataset, IterableDataset
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import PreTrainedTokenizerFast


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing pipeline."""

    max_seq_len: int = Field(512, description="Sequence length for packing")
    batch_size: int = Field(1000, description="Tokenisation batch size")
    num_workers: int = Field(4, description="DataLoader worker count")
    cache_dir: str | None = Field(None, description="Directory to cache processed datasets")
    seed: int = Field(42, description="Random seed for reproducibility")
    val_ratio: float = Field(0.05, description="Validation split ratio")
    test_ratio: float = Field(0.01, description="Test split ratio")
    text_column: str = Field("text", description="Source text column name")

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Tokenization + packing helpers
# ---------------------------------------------------------------------------

def _tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerFast,
    text_column: str = "text",
) -> dict[str, list[list[int]]]:
    """Tokenize a batch of texts without truncation."""
    encoded = tokenizer(
        batch[text_column],
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    return {"input_ids": encoded["input_ids"]}


def _pack_sequences(
    token_seqs: list[list[int]],
    seq_len: int,
    eos_id: int,
) -> list[list[int]]:
    """Concatenate token sequences with EOS, then chunk into fixed-length blocks.

    Returns a list of lists, each of length `seq_len`.
    """
    flat: list[int] = []
    for seq in token_seqs:
        flat.extend(seq)
        flat.append(eos_id)

    chunks = []
    for start in range(0, len(flat) - seq_len + 1, seq_len):
        chunks.append(flat[start : start + seq_len])
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PretrainDataset(TorchIterableDataset):
    """
    PyTorch IterableDataset that streams tokenised, packed sequences
    suitable for next-token prediction pretraining.
    """

    def __init__(
        self,
        hf_dataset: Dataset | IterableDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: PreprocessingConfig,
    ) -> None:
        super().__init__()
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.config = config
        self._eos_id = tokenizer.eos_token_id or 2

    def __iter__(self):  # type: ignore[override]
        buffer: list[int] = []
        seq_len = self.config.max_seq_len

        for example in self.dataset:
            text = example.get(self.config.text_column, "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)
            buffer.append(self._eos_id)

            while len(buffer) >= seq_len + 1:  # +1 for label shift
                chunk = buffer[: seq_len + 1]
                buffer = buffer[seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    """
    Dataset for Supervised Fine-Tuning on instruction/response pairs.

    Expected input: list of dicts with 'instruction' and 'response' keys.
    Loss is computed only on response tokens (instruction tokens masked with -100).
    """

    _TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        template: str | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.template = template or self._TEMPLATE

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        instruction = rec.get("instruction", rec.get("prompt", ""))
        response = rec.get("response", rec.get("output", ""))

        prompt_text = self.template.format(instruction=instruction, response="")
        full_text = self.template.format(instruction=instruction, response=response)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = self.tokenizer.encode(
            full_text + self.tokenizer.eos_token,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
        )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: self.max_seq_len]

        # Pad if shorter
        pad_id = self.tokenizer.pad_token_id or 0
        pad_len = self.max_seq_len - len(full_ids)
        input_ids = full_ids + [pad_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        labels = labels + [-100] * (self.max_seq_len - len(labels))

        return {
            "input_ids": torch.tensor(input_ids[: self.max_seq_len], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[: self.max_seq_len], dtype=torch.long),
            "labels": torch.tensor(labels[: self.max_seq_len], dtype=torch.long),
        }


class DPODataset(torch.utils.data.Dataset):
    """
    Dataset for Direct Preference Optimisation.

    Each record must have: 'prompt', 'chosen', 'rejected'.
    Returns tokenised chosen/rejected pairs with their prompt prefix masked.
    """

    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 512,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def _encode_pair(
        self, prompt: str, response: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(
            prompt + response + self.tokenizer.eos_token,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: self.max_seq_len]

        pad_id = self.tokenizer.pad_token_id or 0
        pad_len = self.max_seq_len - len(full_ids)
        input_ids = full_ids + [pad_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        labels = labels + [-100] * (self.max_seq_len - len(labels))

        return (
            torch.tensor(input_ids[: self.max_seq_len], dtype=torch.long),
            torch.tensor(attention_mask[: self.max_seq_len], dtype=torch.long),
            torch.tensor(labels[: self.max_seq_len], dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prompt, chosen, rejected = rec["prompt"], rec["chosen"], rec["rejected"]

        c_ids, c_mask, c_labels = self._encode_pair(prompt, chosen)
        r_ids, r_mask, r_labels = self._encode_pair(prompt, rejected)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels": r_labels,
        }


def build_pretrain_dataloader(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerFast,
    config: PreprocessingConfig,
) -> DataLoader:
    """Build a DataLoader for pretraining."""
    pt_dataset = PretrainDataset(dataset, tokenizer, config)
    return DataLoader(
        pt_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # IterableDataset: num_workers must be handled carefully
        pin_memory=torch.cuda.is_available(),
    )
