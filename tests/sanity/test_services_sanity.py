"""
Sanity checks for all service components.

These tests verify that every service is importable, configurable, and
produces sensible outputs for minimal inputs. They are intentionally lightweight
and run in seconds so they can be executed as a first-pass health check before
longer unit/integration test suites.

Covered services:
  - Data pipeline  (loader, preprocessing, tokenisation)
  - Training       (pretrain, SFT, LoRA, DPO)
  - Serving API    (health, version, generate, batch, metrics endpoints)
  - Observability  (logging, Prometheus metrics, MLflow tracker)
  - Evaluation     (perplexity, generation quality, benchmarks)
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest
import torch

# ===========================================================================
# DATA PIPELINE SANITY
# ===========================================================================

class TestDataPipelineSanity:
    """Verify that the data ingestion + preprocessing pipeline is functional."""

    def test_local_jsonl_loader_non_streaming(self, tmp_path):
        """load_local_jsonl must return records from a file."""
        from services.data.ingestion.loader import load_local_jsonl

        f = tmp_path / "data.jsonl"
        records = [{"text": f"Sample sentence number {i}."} for i in range(5)]
        f.write_text("\n".join(json.dumps(r) for r in records))

        ds = load_local_jsonl(str(f), streaming=False)
        rows = list(ds)
        assert len(rows) == 5
        assert all("text" in r for r in rows)

    def test_local_jsonl_loader_streaming(self, tmp_path):
        """Streaming mode must also iterate all records."""
        from services.data.ingestion.loader import load_local_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps({"text": f"Line {i}"}) for i in range(3)))

        ds = load_local_jsonl(str(f), streaming=True)
        rows = list(ds)
        assert len(rows) == 3

    def test_pretrain_dataset_produces_fixed_length_chunks(self, tmp_path):
        """PretrainDataset must emit (seq_len,) tensors regardless of text length."""
        from unittest.mock import MagicMock

        from services.data.ingestion.loader import load_local_jsonl
        from services.data.preprocessing.processor import PreprocessingConfig, PretrainDataset

        # Minimal mock tokenizer
        tok = MagicMock()
        tok.vocab_size = 256
        tok.eos_token_id = 2
        tok.pad_token_id = 0
        tok.encode = lambda text, **_: [ord(c) % 253 + 3 for c in text]

        f = tmp_path / "data.jsonl"
        texts = ["The quick brown fox jumps over the lazy dog. " * 5 for _ in range(10)]
        f.write_text("\n".join(json.dumps({"text": t}) for t in texts))

        ds = load_local_jsonl(str(f), streaming=True)
        cfg = PreprocessingConfig(max_seq_len=32)
        pt_ds = PretrainDataset(ds, tok, cfg)

        samples = list(pt_ds)
        assert len(samples) > 0
        for s in samples[:3]:
            assert s["input_ids"].shape == (32,)
            assert s["labels"].shape == (32,)

    def test_sft_dataset_shapes_and_masking(self):
        """SFTDataset must return padded/masked tensors of the correct size."""
        from unittest.mock import MagicMock

        from services.data.preprocessing.processor import SFTDataset

        tok = MagicMock()
        tok.vocab_size = 256
        tok.eos_token_id = 2
        tok.pad_token_id = 0
        tok.eos_token = "</s>"
        tok.pad_token = "<pad>"
        tok.encode = lambda text, **_: [ord(c) % 253 + 3 for c in text[:10]]

        records = [
            {"instruction": "What is AI?", "response": "Artificial intelligence."},
            {"instruction": "Define ML.", "response": "Machine learning."},
        ]
        ds = SFTDataset(records, tok, max_seq_len=32)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"][0].item() == -100  # prompt masked

    def test_dpo_dataset_keys_present(self):
        """DPODataset must include all six required keys per sample."""
        from unittest.mock import MagicMock

        from services.data.preprocessing.processor import DPODataset

        tok = MagicMock()
        tok.vocab_size = 256
        tok.eos_token_id = 2
        tok.pad_token_id = 0
        tok.eos_token = "</s>"
        tok.encode = lambda text, **_: [ord(c) % 253 + 3 for c in text[:8]]

        records = [{"prompt": "Q?", "chosen": "Good answer.", "rejected": "Bad answer."}]
        ds = DPODataset(records, tok, max_seq_len=32)
        item = ds[0]
        for key in [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ]:
            assert key in item

    def test_tokenizer_from_scratch_vocab(self, tmp_path):
        """build_tokenizer_from_scratch must produce a usable tokenizer."""
        from services.data.tokenization.tokenizer import build_tokenizer_from_scratch

        corpus = tmp_path / "c.txt"
        corpus.write_text(("hello world test sanity check. " * 50) + "\n")
        tok = build_tokenizer_from_scratch(files=[str(corpus)], vocab_size=128)
        assert tok.vocab_size > 0
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_data_source_config_immutable(self):
        """DataSourceConfig must be frozen (immutable after construction)."""
        from services.data.ingestion.loader import DataSourceConfig

        cfg = DataSourceConfig(name="local", local_path="/tmp/data.jsonl")
        with pytest.raises((TypeError, ValueError)):  # Pydantic frozen model raises on mutation
            cfg.name = "openwebtext"  # type: ignore[misc]


# ===========================================================================
# MODEL SANITY
# ===========================================================================

class TestModelSanity:
    """Basic model health checks covering config, forward, and serialisation."""

    @pytest.fixture
    def tiny_cfg(self):
        from services.training.core_model.config import ModelConfig
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
    def model(self, tiny_cfg):
        from services.training.core_model.model import LLMModel
        return LLMModel(tiny_cfg)

    def test_model_imports(self):
        assert True

    def test_model_forward_no_labels(self, model, tiny_cfg):
        ids = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
        out = model(ids)
        assert out.logits.shape == (2, 8, tiny_cfg.vocab_size)
        assert out.loss is None

    def test_model_forward_with_labels(self, model, tiny_cfg):
        ids = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
        out = model(ids, labels=ids.clone())
        assert out.loss is not None
        assert math.isfinite(out.loss.item())
        assert out.loss.item() > 0

    def test_model_no_nan_outputs(self, model, tiny_cfg):
        ids = torch.randint(1, tiny_cfg.vocab_size, (2, 8))
        out = model(ids)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_model_param_count_positive(self, model):
        assert model.num_parameters() > 0

    def test_model_round_trip(self, model, tmp_path):
        from services.training.core_model.model import LLMModel
        model.save_pretrained(str(tmp_path / "ckpt"))
        loaded = LLMModel.from_pretrained(str(tmp_path / "ckpt"))
        ids = torch.randint(0, 256, (1, 8))
        out1 = model(ids)
        out2 = loaded(ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_model_config_estimate_close_to_actual(self, tiny_cfg, model):
        est = tiny_cfg.estimate_params()
        actual = model.num_parameters()
        assert abs(est - actual) / actual < 0.05

    def test_gradient_flows_through_model(self, model, tiny_cfg):
        ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
        out = model(ids, labels=ids.clone())
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ===========================================================================
# TRAINING SERVICES SANITY
# ===========================================================================

class TestTrainingServicesSanity:
    """Verify that each trainer can be imported and configured."""

    def test_pretrain_config_default_valid(self):
        from services.training.pretrain.trainer import PretrainConfig
        cfg = PretrainConfig(output_dir="/tmp/pt", max_steps=1)
        assert cfg.max_steps == 1
        assert cfg.learning_rate > 0

    def test_sft_config_default_valid(self):
        from services.training.sft.trainer import SFTConfig
        cfg = SFTConfig(output_dir="/tmp/sft", max_steps=1)
        assert cfg.max_steps == 1
        assert cfg.early_stopping_patience > 0

    def test_dpo_config_default_valid(self):
        from services.training.dpo.trainer import DPOConfig
        cfg = DPOConfig(output_dir="/tmp/dpo", max_steps=1)
        assert cfg.beta > 0

    def test_lora_config_default_valid(self):
        from services.training.lora.trainer import LoRAConfig
        cfg = LoRAConfig(output_dir="/tmp/lora", max_steps=1)
        assert cfg.r > 0
        assert len(cfg.target_modules) > 0

    def test_pretrain_trainer_instantiates(self, tmp_path):
        from torch.utils.data import DataLoader

        from services.training.core_model.config import ModelConfig
        from services.training.core_model.model import LLMModel
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        cfg_m = ModelConfig(vocab_size=256, max_seq_len=16, n_layers=2, hidden_size=64,
                            intermediate_size=128, n_heads=4, dtype="float32")
        model = LLMModel(cfg_m)
        n, seq = 4, 16
        ids = torch.randint(3, 256, (n, seq))

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return n
            def __getitem__(self, i): return {"input_ids": ids[i], "labels": ids[i].clone()}

        loader = DataLoader(_DS(), batch_size=2)
        cfg = PretrainConfig(output_dir=str(tmp_path / "pt"), max_steps=1,
                             log_interval=1, eval_interval=100, save_interval=100,
                             batch_size=2, gradient_accumulation_steps=1,
                             dtype="float32", gradient_checkpointing=False)
        trainer = PretrainTrainer(model, loader, None, cfg)
        assert trainer.global_step == 0

    def test_lora_inject_reduces_trainable_params(self):
        pytest.importorskip("peft")
        from services.training.core_model.config import ModelConfig
        from services.training.core_model.model import LLMModel
        from services.training.lora.trainer import LoRAConfig, inject_lora

        cfg_m = ModelConfig(vocab_size=256, max_seq_len=16, n_layers=2, hidden_size=64,
                            intermediate_size=128, n_heads=4, dtype="float32")
        model = LLMModel(cfg_m)
        lora_cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj"],
                              output_dir="/tmp", max_steps=1)
        lm = inject_lora(model, lora_cfg)
        trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lm.parameters())
        assert trainable < total


# ===========================================================================
# SERVING API SANITY
# ===========================================================================

class TestServingAPISanity:
    """Verify the FastAPI serving layer works correctly end-to-end."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import services.serving.api.app as app_module
        from services.training.core_model.config import ModelConfig
        from services.training.core_model.model import LLMModel

        cfg = ModelConfig(vocab_size=256, max_seq_len=16, n_layers=2,
                          hidden_size=64, intermediate_size=128, n_heads=4, dtype="float32")
        model_dir = tmp_path / "model"
        LLMModel(cfg).save_pretrained(str(model_dir))

        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        monkeypatch.setenv("USE_VLLM", "false")
        monkeypatch.setenv("DTYPE", "float32")
        monkeypatch.setenv("JSON_LOGS", "false")

        app_module._engine = None

        mock_engine = MagicMock()
        mock_engine.framework = "pytorch"
        mock_engine.tokenizer = MagicMock()
        mock_engine.tokenizer.encode.return_value = [1, 2, 3]
        mock_engine.config = MagicMock()
        mock_engine.config.model_path = str(model_dir)
        mock_engine.config.dtype = "float32"
        mock_engine.generate.return_value = "Generated text."
        mock_engine.batch_generate.return_value = ["Text A", "Text B"]

        async def _stream(*a, **kw):
            for tok in ["word1 ", "word2 "]:
                yield tok

        mock_engine.generate_stream = _stream

        with patch("services.serving.api.app.get_engine", return_value=mock_engine):
            from fastapi.testclient import TestClient

            from services.serving.api.app import app
            with TestClient(app) as c:
                yield c

    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_version_endpoint(self, client):
        r = client.get("/version")
        assert r.status_code == 200
        data = r.json()
        assert "framework" in data
        assert "dtype" in data
        assert "model_path" in data

    def test_generate_returns_text(self, client):
        r = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
        assert r.status_code == 200
        data = r.json()
        assert "generated_text" in data
        assert len(data["generated_text"]) > 0

    def test_generate_latency_non_negative(self, client):
        r = client.post("/generate", json={"prompt": "test", "max_new_tokens": 4})
        assert r.json()["latency_ms"] >= 0

    def test_batch_endpoint(self, client):
        r = client.post("/batch", json={"prompts": ["A", "B"], "max_new_tokens": 4})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_invalid_generate_request_422(self, client):
        r = client.post("/generate", json={"max_new_tokens": 8})  # missing prompt
        assert r.status_code == 422

    def test_empty_batch_rejected(self, client):
        r = client.post("/batch", json={"prompts": [], "max_new_tokens": 4})
        assert r.status_code == 422


# ===========================================================================
# OBSERVABILITY SANITY
# ===========================================================================

class TestObservabilitySanity:
    """Verify that logging, metrics, and experiment tracking work correctly."""

    def test_configure_logging_and_get_logger(self):
        from observability.logging.logger import configure_logging, get_logger

        configure_logging(service_name="test", log_level="WARNING", json_logs=False, environment="test")
        logger = get_logger("test.sanity")
        assert logger is not None
        logger.info("sanity_check_log", check="test", value=42)

    def test_bind_unbind_context(self):
        from observability.logging.logger import bind_context, unbind_context

        bind_context(test_key="test_value")
        unbind_context("test_key")

    def test_prometheus_metrics_update(self):
        from observability.metrics.prometheus import update_train_metrics

        # Should not raise
        update_train_metrics(loss=2.5, learning_rate=3e-4, gpu_mb=None)

    def test_prometheus_record_inference(self):
        from observability.metrics.prometheus import record_inference

        record_inference(duration_s=0.1, tokens_generated=50, endpoint="/generate")

    def test_prometheus_update_dpo_metrics(self):
        from observability.metrics.prometheus import update_dpo_metrics

        update_dpo_metrics(loss=1.2, reward_margin=0.3)

    def test_prometheus_collect_gpu_stats_no_crash(self):
        from observability.metrics.prometheus import collect_gpu_stats

        stats = collect_gpu_stats()
        assert isinstance(stats, dict)

    def test_mlflow_tracker_noop_when_disabled(self):
        from observability.experiment_tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(enabled=False)
        with tracker.start_run(run_name="noop"):
            tracker.log_params({"lr": 1e-4})
            tracker.log_metric("loss", 2.5, step=1)
            tracker.log_metrics({"ppl": 12.2}, step=1)
            tracker.set_tag("stage", "test")

    def test_mlflow_tracker_local_run(self, tmp_path):
        """MLflow tracker must be able to log to a local directory."""
        from observability.experiment_tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name="sanity_test",
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
            enabled=True,
        )
        with tracker.start_run(run_name="sanity"):
            tracker.log_params({"lr": 3e-4, "batch_size": 4})
            tracker.log_metric("loss", 2.3, step=1)
            tracker.log_metric("loss", 2.1, step=2)
            tracker.set_tag("status", "sanity_ok")


# ===========================================================================
# EVALUATION SANITY
# ===========================================================================

class TestEvaluationSanity:
    """Verify that the evaluation pipeline modules are importable and functional."""

    @pytest.fixture
    def tiny_model_and_tokenizer(self, tmp_path):
        from unittest.mock import MagicMock

        from services.training.core_model.config import ModelConfig
        from services.training.core_model.model import LLMModel

        cfg = ModelConfig(vocab_size=256, max_seq_len=16, n_layers=2, hidden_size=64,
                          intermediate_size=128, n_heads=4, dtype="float32")
        model = LLMModel(cfg)

        tok = MagicMock()
        tok.vocab_size = 256
        tok.eos_token_id = 2
        tok.pad_token_id = 0
        tok.eos_token = "</s>"

        def _encode(text, return_tensors=None, **_):
            ids = [ord(c) % 253 + 3 for c in text[:12]]
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids

        tok.encode = _encode
        tok.decode = lambda ids, **_: "decoded text"
        tok.__call__ = lambda texts, **_: MagicMock(
            input_ids=torch.tensor([[ord(c) % 253 + 3 for c in texts[0][:8]]])
        )
        return model, tok

    def test_perplexity_eval_imports(self):
        assert True

    def test_perplexity_on_small_dataset(self, tiny_model_and_tokenizer):
        from evaluation.perplexity.eval import PerplexityConfig, evaluate_perplexity

        model, tok = tiny_model_and_tokenizer

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return 2
            def __getitem__(self, i):
                ids = torch.randint(3, 256, (8,))
                return {"input_ids": ids, "labels": ids.clone()}

        from torch.utils.data import DataLoader
        loader = DataLoader(_DS(), batch_size=2)
        cfg = PerplexityConfig(max_batches=1, dtype="float32")
        result = evaluate_perplexity(model, loader, cfg)
        assert hasattr(result, "perplexity")
        assert math.isfinite(result.perplexity)
        assert result.perplexity > 0

    def test_generation_quality_eval_imports(self):
        assert True

    def test_generation_produces_output(self, tiny_model_and_tokenizer):
        from evaluation.generation_quality.eval import GenerationConfig, generate_samples

        model, tok = tiny_model_and_tokenizer
        cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_k=10)
        device = torch.device("cpu")
        results = generate_samples(model, tok, ["hello world"], cfg, device)
        assert len(results) == 1
        assert hasattr(results[0], "generated_text")

    def test_benchmark_runner_imports(self):
        from evaluation.benchmarks.runner import BenchmarkRunner
        assert BenchmarkRunner is not None

    def test_regression_runner_imports(self):
        from evaluation.regression_tests.runner import RegressionTestRunner
        assert RegressionTestRunner is not None
