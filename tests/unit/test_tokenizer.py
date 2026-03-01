"""Unit tests for the tokenizer wrapper (offline – no network required)."""

from __future__ import annotations

import pytest


class TestTokenizerOffline:
    """Tests that work without network access (BPE from scratch)."""

    def test_build_bpe_tokenizer_from_scratch(self, tmp_path):
        """Train a BPE tokenizer from local text files."""
        from services.data.tokenization.tokenizer import build_tokenizer_from_scratch

        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models predict the next token.",
        ] * 20

        text_file = tmp_path / "corpus.txt"
        text_file.write_text("\n".join(texts))

        tokenizer = build_tokenizer_from_scratch(
            files=[str(text_file)],
            vocab_size=100,
            save_dir=str(tmp_path / "tokenizer"),
        )

        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token is not None

    def test_encode_with_scratch_tokenizer(self, tmp_path):
        """Tokenizer trained from scratch should encode text to integer ids."""
        from services.data.tokenization.tokenizer import build_tokenizer_from_scratch

        texts = ["hello world test sentence."] * 30
        text_file = tmp_path / "corpus.txt"
        text_file.write_text("\n".join(texts))

        tok = build_tokenizer_from_scratch(files=[str(text_file)], vocab_size=80)
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_save_and_load_tokenizer(self, tmp_path):
        """Saved tokenizer should be loadable and produce same vocab size."""
        from services.data.tokenization.tokenizer import (
            build_tokenizer_from_scratch,
            load_tokenizer,
        )

        texts = ["the quick brown fox."] * 30
        text_file = tmp_path / "corpus.txt"
        text_file.write_text("\n".join(texts))

        save_dir = str(tmp_path / "tok")
        tok1 = build_tokenizer_from_scratch(
            files=[str(text_file)], vocab_size=80, save_dir=save_dir
        )
        tok2 = load_tokenizer(save_dir)
        assert tok2.vocab_size == tok1.vocab_size


@pytest.mark.network
class TestTokenizerHubDependent:
    """Requires HuggingFace Hub access – skipped in environments without internet."""

    def test_get_default_tokenizer(self):
        from services.data.tokenization.tokenizer import get_default_tokenizer
        tok = get_default_tokenizer()
        assert tok.vocab_size > 0

    def test_load_tokenizer_from_hub(self):
        from services.data.tokenization.tokenizer import load_tokenizer
        tok = load_tokenizer("gpt2")
        assert tok.vocab_size > 0
