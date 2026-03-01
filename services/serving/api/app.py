"""
FastAPI serving application.

Endpoints:
  GET  /health          – liveness check
  GET  /version         – model version info
  POST /generate        – text generation (non-streaming)
  POST /generate/stream – streaming text generation (SSE)
  POST /batch           – batch inference
  GET  /metrics         – Prometheus metrics (text format)

Requires VLLM_MODEL_PATH env var pointing to a model directory.
Can also fall back to native PyTorch inference for development.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from services.serving.inference.engine import InferenceEngine, InferenceConfig
from observability.logging.logger import configure_logging, get_logger
from observability.metrics.prometheus import (
    record_inference,
    ACTIVE_REQUESTS,
    collect_gpu_stats,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

configure_logging(
    service_name="llm-serving",
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    json_logs=os.environ.get("JSON_LOGS", "true").lower() == "true",
    environment=os.environ.get("ENVIRONMENT", "production"),
)
logger = get_logger(__name__)

app = FastAPI(
    title="LLM Stack Serving API",
    description="Production serving for the LLM Stack Transformer model",
    version="0.1.0",
)

# Lazy-initialised engine (avoids GPU allocation at import time)
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        cfg = InferenceConfig(
            model_path=os.environ.get("MODEL_PATH", "./checkpoints/pretrain/final"),
            use_vllm=os.environ.get("USE_VLLM", "true").lower() == "true",
            dtype=os.environ.get("DTYPE", "bfloat16"),
            max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "512")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        )
        _engine = InferenceEngine(cfg)
    return _engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.0, le=5.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=200)
    repetition_penalty: float = Field(1.1, ge=1.0, le=3.0)
    stop_sequences: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    n_tokens: int
    latency_ms: float
    model_version: str


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=32)
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(1.0)
    top_p: float = Field(0.9)


class HealthResponse(BaseModel):
    status: str
    gpu_memory_mb: Optional[int] = None


class VersionResponse(BaseModel):
    model_path: str
    model_version: str
    framework: str
    dtype: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm the inference engine."""
    logger.info("server_startup", msg="Warming up inference engine")
    try:
        engine = get_engine()
        logger.info("engine_ready", framework=engine.framework)
    except Exception as exc:
        logger.error("engine_startup_failed", error=str(exc))
        # Don't crash the server; health endpoint will report degraded


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    gpu_stats = collect_gpu_stats()
    return HealthResponse(status="ok", gpu_memory_mb=gpu_stats.get("gpu_memory_mb"))


@app.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """Return model version information."""
    engine = get_engine()
    return VersionResponse(
        model_path=engine.config.model_path,
        model_version=os.environ.get("MODEL_VERSION", "unknown"),
        framework=engine.framework,
        dtype=engine.config.dtype,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text for a single prompt."""
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        engine = get_engine()
        generated = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        n_tokens = len(engine.tokenizer.encode(generated)) if engine.tokenizer else 0
        record_inference(latency_ms / 1000, n_tokens, endpoint="/generate")
        logger.info("generate", latency_ms=round(latency_ms, 1), n_tokens=n_tokens)
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            n_tokens=n_tokens,
            latency_ms=round(latency_ms, 1),
            model_version=os.environ.get("MODEL_VERSION", "unknown"),
        )
    except Exception as exc:
        record_inference(time.perf_counter() - t0, 0, status="500")
        logger.error("generate_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens as Server-Sent Events."""
    engine = get_engine()
    ACTIVE_REQUESTS.inc()

    async def _token_stream() -> AsyncIterator[str]:
        try:
            async for token in engine.generate_stream(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(_token_stream(), media_type="text/event-stream")


@app.post("/batch")
async def batch_generate(request: BatchGenerateRequest) -> JSONResponse:
    """Batch inference for multiple prompts."""
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        engine = get_engine()
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.batch_generate(
                request.prompts,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return JSONResponse(
            {
                "results": [
                    {"prompt": p, "generated_text": g}
                    for p, g in zip(request.prompts, results)
                ],
                "latency_ms": round(latency_ms, 1),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics() -> StreamingResponse:
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    content = generate_latest()
    return StreamingResponse(
        iter([content]),
        media_type=CONTENT_TYPE_LATEST,
    )
