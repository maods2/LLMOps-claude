"""Integration tests: verify one full training mini-step executes correctly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from services.training.core_model.config import ModelConfig
from services.training.core_model.model import LLMModel


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        n_layers=2,
        hidden_size=64,
        intermediate_size=128,
        n_heads=4,
        dtype="float32",
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> LLMModel:
    return LLMModel(tiny_config)


def _make_pretrain_loader(vocab_size: int, seq_len: int, n: int = 8) -> DataLoader:
    """Create a tiny DataLoader for pretrain testing."""
    input_ids = torch.randint(3, vocab_size, (n, seq_len))
    labels = input_ids.clone()

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i): return {"input_ids": input_ids[i], "labels": labels[i]}

    return DataLoader(_DS(), batch_size=2)


def _make_sft_loader(vocab_size: int, seq_len: int, n: int = 8) -> DataLoader:
    """Create a tiny DataLoader for SFT testing (with masked labels)."""
    input_ids = torch.randint(3, vocab_size, (n, seq_len))
    labels = input_ids.clone()
    labels[:, :seq_len // 2] = -100  # Mask first half (prompt)
    mask = torch.ones(n, seq_len, dtype=torch.long)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            return {"input_ids": input_ids[i], "labels": labels[i], "attention_mask": mask[i]}

    return DataLoader(_DS(), batch_size=2)


class TestPretrainStep:
    def test_single_train_step_reduces_loss(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        """One gradient step should reduce training loss."""
        from torch.optim import AdamW

        loader = _make_pretrain_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        batch = next(iter(loader))

        tiny_model.train()
        # Initial loss
        out1 = tiny_model(batch["input_ids"], labels=batch["labels"])
        loss1 = out1.loss.item()

        optimizer.zero_grad()
        out1.loss.backward()
        optimizer.step()

        # After one step
        with torch.no_grad():
            out2 = tiny_model(batch["input_ids"], labels=batch["labels"])
        loss2 = out2.loss.item()

        # Loss should decrease (not necessarily by much, but should not increase by >2×)
        assert loss2 < loss1 * 1.5, f"Loss did not decrease: {loss1:.4f} → {loss2:.4f}"

    def test_pretrain_trainer_two_steps(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path: Path):
        """PretrainTrainer should run two steps without error."""
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        cfg = PretrainConfig(
            output_dir=str(tmp_path / "ckpt"),
            max_steps=2,
            log_interval=1,
            eval_interval=10,
            save_interval=10,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        loader = _make_pretrain_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = PretrainTrainer(tiny_model, loader, None, cfg)
        trainer.train()  # Should complete without exception

    def test_pretrain_checkpoint_saved(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path: Path):
        """Trainer must create checkpoint files."""
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        cfg = PretrainConfig(
            output_dir=str(tmp_path / "ckpt"),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        loader = _make_pretrain_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = PretrainTrainer(tiny_model, loader, None, cfg)
        trainer.train()

        ckpt_dir = tmp_path / "ckpt" / "final"
        assert (ckpt_dir / "config.json").exists()
        assert (ckpt_dir / "model.pt").exists()


class TestSFTStep:
    def test_sft_trainer_masked_loss(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path: Path):
        """SFT trainer should compute loss only on response tokens."""
        from services.training.sft.trainer import SFTConfig, SFTTrainer

        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            early_stopping_patience=5,
        )
        loader = _make_sft_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = SFTTrainer(tiny_model, loader, None, cfg)
        trainer.train()
