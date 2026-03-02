"""Unit tests for the core LLM model."""

from __future__ import annotations

import pytest
import torch

from services.training.core_model.attention import (
    apply_rope,
    build_rope_cache,
    rotate_half,
)
from services.training.core_model.blocks import MLP, DecoderBlock, RMSNorm
from services.training.core_model.config import ModelConfig
from services.training.core_model.model import LLMModel, LLMOutput

# ---------------------------------------------------------------------------
# Tiny model config for fast CPU tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        n_layers=2,
        hidden_size=64,
        intermediate_size=128,
        n_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        use_flash_attention=False,
        tie_word_embeddings=True,
        dtype="float32",
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> LLMModel:
    return LLMModel(tiny_config)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.dtype == x.dtype

    def test_non_zero_output(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert not torch.all(out == 0)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_cache_shapes(self):
        seq_len, head_dim = 32, 16
        cos, sin = build_rope_cache(seq_len, head_dim)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8, 16)  # (B, H, T, D)
        rot = rotate_half(x)
        assert rot.shape == x.shape

    def test_apply_rope_shapes(self):
        B, H, T, D = 2, 4, 16, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        cos, sin = build_rope_cache(T, D)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_changes_values(self):
        B, H, T, D = 1, 2, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        cos, sin = build_rope_cache(T, D)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert not torch.allclose(q, q_rot)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_mlp_forward(self, tiny_config: ModelConfig):
        mlp = MLP(tiny_config)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_no_nan(self, tiny_config: ModelConfig):
        mlp = MLP(tiny_config)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        out = mlp(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# DecoderBlock
# ---------------------------------------------------------------------------

class TestDecoderBlock:
    def test_forward_shape(self, tiny_config: ModelConfig):
        block = DecoderBlock(tiny_config)
        B, T, H = 2, 16, tiny_config.hidden_size
        x = torch.randn(B, T, H)
        cos, sin = build_rope_cache(T, tiny_config.head_dim)
        out = block(x, cos, sin)
        assert out.shape == x.shape

    def test_forward_no_nan(self, tiny_config: ModelConfig):
        block = DecoderBlock(tiny_config)
        B, T = 2, 16
        x = torch.randn(B, T, tiny_config.hidden_size)
        cos, sin = build_rope_cache(T, tiny_config.head_dim)
        out = block(x, cos, sin)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# LLMModel
# ---------------------------------------------------------------------------

class TestLLMModel:
    def test_forward_logits_shape(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        out = tiny_model(input_ids)
        assert isinstance(out, LLMOutput)
        assert out.logits.shape == (B, T, tiny_config.vocab_size)
        assert out.loss is None

    def test_forward_with_labels_returns_loss(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        labels = input_ids.clone()
        out = tiny_model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()
        assert out.loss.item() > 0

    def test_label_masking_with_minus100(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 1, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        labels = input_ids.clone()
        labels[:, :8] = -100  # Mask first half
        out = tiny_model(input_ids, labels=labels)
        assert out.loss is not None and out.loss.item() > 0

    def test_weight_tying(self, tiny_model: LLMModel):
        assert tiny_model.lm_head.weight is tiny_model.embed_tokens.weight

    def test_num_parameters(self, tiny_model: LLMModel):
        n = tiny_model.num_parameters()
        assert n > 0

    def test_parameter_estimate_method(self, tiny_config: ModelConfig):
        est = tiny_config.estimate_params()
        model = LLMModel(tiny_config)
        actual = model.num_parameters()
        # Estimate should be within 5% of actual
        assert abs(est - actual) / actual < 0.05

    def test_no_nan_in_output(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 2, 8
        input_ids = torch.randint(1, tiny_config.vocab_size, (B, T))
        out = tiny_model(input_ids)
        assert not torch.isnan(out.logits).any()

    def test_attention_mask_respected(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        mask = torch.ones(B, T)
        mask[0, 8:] = 0  # Mask second half of first sequence
        out_masked = tiny_model(input_ids, attention_mask=mask)
        out_full = tiny_model(input_ids)
        # Logits should differ when mask is applied
        assert not torch.allclose(out_masked.logits, out_full.logits)

    def test_gradient_flows(self, tiny_model: LLMModel, tiny_config: ModelConfig):
        B, T = 1, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        labels = input_ids.clone()
        out = tiny_model(input_ids, labels=labels)
        out.loss.backward()
        # Check at least one parameter has non-None gradient
        has_grad = any(p.grad is not None for p in tiny_model.parameters())
        assert has_grad

    def test_save_and_load(self, tiny_model: LLMModel, tmp_path):
        tiny_model.save_pretrained(str(tmp_path))
        loaded = LLMModel.from_pretrained(str(tmp_path))
        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            tiny_model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch at {n1}"
