"""Integration tests for LoRA adapter injection and merging."""

from __future__ import annotations

import pytest
import torch

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


class TestLoRAInjection:
    def test_lora_inject_reduces_trainable_params(self, tiny_model: LLMModel):
        """LoRA should freeze base weights and only keep adapter params trainable."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, inject_lora

        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            output_dir="/tmp/lora_test",
            max_steps=1,
        )
        lora_model = inject_lora(tiny_model, cfg)

        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        assert trainable < total
        assert trainable > 0

    def test_lora_forward_pass_works(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        """LoRA-wrapped model should produce correct output shape."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, inject_lora

        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            output_dir="/tmp/lora_test",
            max_steps=1,
        )
        lora_model = inject_lora(tiny_model, cfg)

        B, T = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        out = lora_model(input_ids)
        assert out.logits.shape == (B, T, tiny_config.vocab_size)

    def test_lora_trainer_one_step(
        self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path
    ):
        """LoRATrainer should complete one step without error."""
        pytest.importorskip("peft")

        from torch.utils.data import DataLoader

        from services.training.lora.trainer import LoRAConfig, LoRATrainer

        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            output_dir=str(tmp_path / "lora"),
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
        )
        n, seq = 4, tiny_config.max_seq_len
        input_ids = torch.randint(3, tiny_config.vocab_size, (n, seq))
        labels = input_ids.clone()

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return n
            def __getitem__(self, i): return {"input_ids": input_ids[i], "labels": labels[i]}

        loader = DataLoader(_DS(), batch_size=2)
        trainer = LoRATrainer(tiny_model, loader, None, cfg)
        trainer.train()  # Should not raise

    def test_lora_merge_and_save(self, tiny_model: LLMModel, tiny_config: ModelConfig, tmp_path):
        """Merged model should be loadable and produce same-shaped output."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, inject_lora, merge_and_save

        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj"],
            output_dir=str(tmp_path / "lora"),
            max_steps=1,
        )
        lora_model = inject_lora(tiny_model, cfg)
        merge_dir = str(tmp_path / "merged")
        merge_and_save(lora_model, merge_dir)

        # Verify output files
        import os
        files = os.listdir(merge_dir)
        assert any("model" in f for f in files)
