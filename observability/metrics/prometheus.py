"""
Prometheus metrics for training and serving.

Exposes an HTTP endpoint on a configurable port for scraping.
Provides training and inference metric gauges/histograms.
"""

from __future__ import annotations

import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

TRAIN_LOSS = Gauge("llm_train_loss", "Current training loss")
TRAIN_PPL = Gauge("llm_train_perplexity", "Current training perplexity")
TRAIN_LR = Gauge("llm_train_learning_rate", "Current learning rate")
TRAIN_STEP = Counter("llm_train_steps_total", "Total training steps completed")
GPU_MEMORY_MB = Gauge("llm_gpu_memory_mb", "GPU allocated memory in MB")
GPU_UTILISATION = Gauge("llm_gpu_utilisation_pct", "GPU utilisation percentage")

# ---------------------------------------------------------------------------
# DPO-specific
# ---------------------------------------------------------------------------

DPO_REWARD_MARGIN = Gauge("llm_dpo_reward_margin", "DPO reward margin (chosen - rejected)")
DPO_LOSS = Gauge("llm_dpo_loss", "DPO training loss")

# ---------------------------------------------------------------------------
# Serving metrics
# ---------------------------------------------------------------------------

INFERENCE_REQUESTS = Counter(
    "llm_inference_requests_total", "Total inference requests", ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "Inference request latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
INFERENCE_TOKENS = Histogram(
    "llm_inference_tokens_generated",
    "Number of tokens generated per request",
    buckets=[10, 50, 100, 200, 500, 1000],
)
ACTIVE_REQUESTS = Gauge("llm_active_requests", "Current number of in-flight requests")

# ---------------------------------------------------------------------------
# Metric update helpers
# ---------------------------------------------------------------------------


def update_train_metrics(
    loss: float,
    learning_rate: float,
    gpu_mb: int | None = None,
    gpu_util_pct: float | None = None,
) -> None:
    """Update training metrics gauges."""
    import math
    TRAIN_LOSS.set(loss)
    TRAIN_PPL.set(math.exp(min(loss, 20)))
    TRAIN_LR.set(learning_rate)
    TRAIN_STEP.inc()
    if gpu_mb is not None:
        GPU_MEMORY_MB.set(gpu_mb)
    if gpu_util_pct is not None:
        GPU_UTILISATION.set(gpu_util_pct)


def update_dpo_metrics(loss: float, reward_margin: float) -> None:
    """Update DPO-specific metrics."""
    DPO_LOSS.set(loss)
    DPO_REWARD_MARGIN.set(reward_margin)


def record_inference(
    duration_s: float,
    tokens_generated: int,
    endpoint: str = "/generate",
    status: str = "200",
) -> None:
    """Record a completed inference request."""
    INFERENCE_REQUESTS.labels(endpoint=endpoint, status=status).inc()
    INFERENCE_LATENCY.observe(duration_s)
    INFERENCE_TOKENS.observe(tokens_generated)


def collect_gpu_stats() -> dict:
    """Collect GPU stats using PyTorch CUDA APIs."""
    try:
        import torch
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            GPU_MEMORY_MB.set(mem_mb)
            return {"gpu_memory_mb": mem_mb}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = 8090) -> None:
    """Start the Prometheus HTTP metrics server (idempotent).

    Args:
        port: Port for the /metrics endpoint.
    """
    global _server_started
    with _server_lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True
            print(f"Prometheus metrics server started on :{port}")
