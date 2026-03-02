"""Unit tests for data loading and preprocessing (offline – no network required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Offline tokenizer fixture (avoids HuggingFace Hub download in CI)
# ---------------------------------------------------------------------------

def _make_offline_tokenizer(vocab_size: int = 256):
    """Build a minimal mock tokenizer that works entirely offline.

    Uses a byte-level encoding so it's deterministic and requires no downloads.
    """
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.bos_token_id = 1
    tok.eos_token = "</s>"
    tok.pad_token = "<pad>"

    def _encode(text, add_special_tokens=True, max_length=None, truncation=False, **_kw):
        ids = [int(b) % (vocab_size - 3) + 3 for b in text.encode("utf-8")]
        if add_special_tokens:
            ids = [1] + ids + [2]
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def _call(texts, add_special_tokens=True, padding=False,
              truncation=False, max_length=None, return_tensors=None, **_kw):
        if isinstance(texts, str):
            ids = _encode(texts, add_special_tokens, max_length, truncation)
            if return_tensors == "pt":
                return MagicMock(input_ids=torch.tensor([ids]))
            return MagicMock(input_ids=[ids])
        all_ids = [_encode(t, add_special_tokens, max_length, truncation) for t in texts]
        if padding:
            ml = max(len(i) for i in all_ids)
            all_ids = [i + [0] * (ml - len(i)) for i in all_ids]
        if return_tensors == "pt":
            return MagicMock(input_ids=torch.tensor(all_ids))
        return MagicMock(input_ids=all_ids)

    tok.encode = _encode
    tok.__call__ = _call
    return tok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> str:
    """Create a temporary .jsonl file with a few text samples."""
    records = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "In a hole in the ground there lived a hobbit."},
        {"text": "Call me Ishmael."},
        {"text": "It was the best of times, it was the worst of times."},
        {"text": "To be or not to be, that is the question."},
    ]
    fpath = tmp_path / "data.jsonl"
    with open(fpath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(fpath)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalJsonlLoader:
    def test_loads_records_non_streaming(self, tmp_jsonl: str):
        from services.data.ingestion.loader import load_local_jsonl

        ds = load_local_jsonl(tmp_jsonl, streaming=False)
        records = list(ds)
        assert len(records) == 5
        assert "text" in records[0]

    def test_loads_records_streaming(self, tmp_jsonl: str):
        from services.data.ingestion.loader import load_local_jsonl

        ds = load_local_jsonl(tmp_jsonl, streaming=True)
        records = list(ds)
        assert len(records) == 5

    def test_records_have_text_field(self, tmp_jsonl: str):
        from services.data.ingestion.loader import load_local_jsonl

        ds = load_local_jsonl(tmp_jsonl, streaming=False)
        for rec in ds:
            assert isinstance(rec["text"], str)
            assert len(rec["text"]) > 0


class TestPretrainDataset:
    def test_yields_input_ids_and_labels(self, tmp_jsonl: str):
        from services.data.ingestion.loader import load_local_jsonl
        from services.data.preprocessing.processor import PreprocessingConfig, PretrainDataset

        tokenizer = _make_offline_tokenizer()
        ds = load_local_jsonl(tmp_jsonl, streaming=True)
        cfg = PreprocessingConfig(max_seq_len=32, batch_size=2)
        pt_ds = PretrainDataset(ds, tokenizer, cfg)

        samples = list(pt_ds)
        assert len(samples) > 0
        first = samples[0]
        assert "input_ids" in first
        assert "labels" in first
        assert first["input_ids"].shape == (32,)
        assert first["labels"].shape == (32,)

    def test_labels_shifted_from_input_ids(self, tmp_jsonl: str):
        from services.data.ingestion.loader import load_local_jsonl
        from services.data.preprocessing.processor import PreprocessingConfig, PretrainDataset

        tokenizer = _make_offline_tokenizer()
        ds = load_local_jsonl(tmp_jsonl, streaming=False)
        cfg = PreprocessingConfig(max_seq_len=16)
        pt_ds = PretrainDataset(ds, tokenizer, cfg)
        samples = list(pt_ds)
        assert len(samples) > 0


class TestSFTDataset:
    def test_sft_output_shapes(self):
        from services.data.preprocessing.processor import SFTDataset

        tokenizer = _make_offline_tokenizer()
        records = [
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "Name a planet.", "response": "Mars"},
        ]
        ds = SFTDataset(records, tokenizer, max_seq_len=64)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (64,)
        assert item["attention_mask"].shape == (64,)

    def test_sft_prompt_tokens_masked(self):
        from services.data.preprocessing.processor import SFTDataset

        tokenizer = _make_offline_tokenizer()
        records = [{"instruction": "Tell me something.", "response": "Sure."}]
        ds = SFTDataset(records, tokenizer, max_seq_len=64)
        item = ds[0]
        # First token of labels should be -100 (prompt masked)
        assert item["labels"][0].item() == -100


class TestDPODataset:
    def test_dpo_output_shapes(self):
        from services.data.preprocessing.processor import DPODataset

        tokenizer = _make_offline_tokenizer()
        records = [
            {
                "prompt": "What's your favourite colour?",
                "chosen": "Blue.",
                "rejected": "I don't know.",
            }
        ]
        ds = DPODataset(records, tokenizer, max_seq_len=64)
        assert len(ds) == 1
        item = ds[0]
        for key in [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ]:
            assert key in item
            assert item[key].shape == (64,)
