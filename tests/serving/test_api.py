"""Serving API tests using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.training.core_model.config import ModelConfig
from services.training.core_model.model import LLMModel


@pytest.fixture(autouse=True)
def _set_env(monkeypatch, tmp_path):
    """Set environment variables and create a dummy model checkpoint."""
    # Create a tiny model and save it
    cfg = ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        n_layers=2,
        hidden_size=64,
        intermediate_size=128,
        n_heads=4,
        dtype="float32",
    )
    model_dir = tmp_path / "model"
    model = LLMModel(cfg)
    model.save_pretrained(str(model_dir))

    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    monkeypatch.setenv("USE_VLLM", "false")
    monkeypatch.setenv("DTYPE", "float32")
    monkeypatch.setenv("JSON_LOGS", "false")


@pytest.fixture
def client():
    """Create a TestClient with mocked inference engine."""
    # Reset the lazy engine singleton
    import services.serving.api.app as app_module
    app_module._engine = None

    # Patch InferenceEngine to use a mock
    mock_engine = MagicMock()
    mock_engine.framework = "pytorch"
    mock_engine.tokenizer = MagicMock()
    mock_engine.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_engine.config = MagicMock()
    mock_engine.config.model_path = "/fake/path"
    mock_engine.config.dtype = "float32"
    mock_engine.generate.return_value = "This is generated text."

    async def _fake_stream(*args, **kwargs):
        for token in ["This ", "is ", "a ", "stream."]:
            yield token

    mock_engine.generate_stream = _fake_stream
    mock_engine.batch_generate.return_value = ["Result A", "Result B"]

    with patch("services.serving.api.app.get_engine", return_value=mock_engine):
        from services.serving.api.app import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_schema(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert "status" in data


class TestVersionEndpoint:
    def test_version_endpoint_exists(self, client: TestClient):
        response = client.get("/version")
        assert response.status_code == 200

    def test_version_returns_expected_fields(self, client: TestClient):
        response = client.get("/version")
        data = response.json()
        assert "model_path" in data
        assert "framework" in data
        assert "dtype" in data


class TestGenerateEndpoint:
    def test_generate_returns_200(self, client: TestClient):
        payload = {"prompt": "Once upon a time", "max_new_tokens": 32}
        response = client.post("/generate", json=payload)
        assert response.status_code == 200

    def test_generate_response_has_text(self, client: TestClient):
        payload = {"prompt": "Hello world", "max_new_tokens": 16}
        response = client.post("/generate", json=payload)
        data = response.json()
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        assert len(data["generated_text"]) > 0

    def test_generate_response_has_latency(self, client: TestClient):
        payload = {"prompt": "Test", "max_new_tokens": 8}
        response = client.post("/generate", json=payload)
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_generate_invalid_request_returns_422(self, client: TestClient):
        # Missing required 'prompt' field
        response = client.post("/generate", json={"max_new_tokens": 32})
        assert response.status_code == 422

    def test_generate_with_params(self, client: TestClient):
        payload = {
            "prompt": "The meaning of life is",
            "max_new_tokens": 64,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 200


class TestBatchEndpoint:
    def test_batch_returns_results(self, client: TestClient):
        payload = {"prompts": ["Hello", "World"], "max_new_tokens": 16}
        response = client.post("/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_empty_prompts_rejected(self, client: TestClient):
        payload = {"prompts": [], "max_new_tokens": 16}
        response = client.post("/batch", json=payload)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_endpoint_exists(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200
