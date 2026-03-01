"""Integration tests for the DPO trainer."""

from __future__ import annotations

import copy

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


def _make_dpo_loader(vocab_size: int, seq_len: int, n: int = 4) -> DataLoader:
    """Build a minimal DPO DataLoader."""
    chosen_ids = torch.randint(3, vocab_size, (n, seq_len))
    rejected_ids = torch.randint(3, vocab_size, (n, seq_len))
    chosen_mask = torch.ones(n, seq_len, dtype=torch.long)
    rejected_mask = torch.ones(n, seq_len, dtype=torch.long)
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, : seq_len // 2] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, : seq_len // 2] = -100

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            return {
                "chosen_input_ids": chosen_ids[i],
                "chosen_attention_mask": chosen_mask[i],
                "chosen_labels": chosen_labels[i],
                "rejected_input_ids": rejected_ids[i],
                "rejected_attention_mask": rejected_mask[i],
                "rejected_labels": rejected_labels[i],
            }

    return DataLoader(_DS(), batch_size=2)


class TestDPOTrainer:
    def test_dpo_loss_is_positive(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        """DPO loss must be a positive scalar."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(tiny_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir="/tmp/dpo_test",
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            offload_ref_to_cpu=False,
        )
        loader = _make_dpo_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = DPOTrainer(tiny_model, ref_model, loader, None, cfg)

        batch = next(iter(loader))
        loss, metrics = trainer._dpo_loss(batch)
        assert loss.item() > 0
        assert "reward_margin" in metrics

    def test_dpo_one_step(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path):
        """DPO trainer should complete one step without error."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(tiny_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir=str(tmp_path / "dpo"),
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            offload_ref_to_cpu=False,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
        )
        loader = _make_dpo_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = DPOTrainer(tiny_model, ref_model, loader, None, cfg)
        trainer.train()

    def test_dpo_reference_free_mode(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path):
        """Reference-free DPO should work without a ref model."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        cfg = DPOConfig(
            beta=0.1,
            reference_free=True,
            output_dir=str(tmp_path / "dpo_free"),
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
        )
        loader = _make_dpo_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = DPOTrainer(tiny_model, None, loader, None, cfg)
        batch = next(iter(loader))
        loss, metrics = trainer._dpo_loss(batch)
        assert loss.item() > 0

    def test_dpo_reward_margin_tracked(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        """Reward margin metric should be a finite float."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(tiny_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir="/tmp/dpo_margin",
            max_steps=1,
            offload_ref_to_cpu=False,
            dtype="float32",
            gradient_checkpointing=False,
        )
        loader = _make_dpo_loader(tiny_config.vocab_size, tiny_config.max_seq_len)
        trainer = DPOTrainer(tiny_model, ref_model, loader, None, cfg)
        batch = next(iter(loader))
        _, metrics = trainer._dpo_loss(batch)
        margin = metrics["reward_margin"]
        assert isinstance(margin, float)
        import math
        assert math.isfinite(margin)
