# LLM Stack

A fully modular, production-grade monorepo for training, fine-tuning, evaluating, and serving a ~300M-parameter decoder-only Transformer language model within an **8 GB GPU** constraint.

---

## Architecture Overview

```
llm-stack/
│
├── services/
│   ├── training/
│   │   ├── core_model/     # Transformer model (attention, RoPE, RMSNorm, MLP)
│   │   ├── pretrain/       # Pretraining loop + CLI
│   │   ├── sft/            # Supervised Fine-Tuning loop + CLI
│   │   ├── lora/           # LoRA adapter training + merge + CLI
│   │   └── dpo/            # Direct Preference Optimisation + CLI
│   │
│   ├── serving/
│   │   ├── api/            # FastAPI app (generate, stream, batch, health)
│   │   └── inference/      # InferenceEngine (vLLM or PyTorch backend)
│   │
│   └── data/
│       ├── ingestion/      # Dataset loaders (wikitext2, openwebtext, tinystories, jsonl)
│       ├── preprocessing/  # Tokenisation, packing, SFT/DPO dataset classes
│       └── tokenization/   # Tokenizer wrappers
│
├── evaluation/
│   ├── benchmarks/         # Config-driven benchmark runner
│   ├── perplexity/         # Perplexity + token-accuracy evaluation
│   ├── generation_quality/ # Text generation + diversity metrics
│   └── regression_tests/   # Prompt → expected-output regression suite
│
├── observability/
│   ├── logging/            # Structured logging (structlog)
│   ├── metrics/            # Prometheus metrics (training + serving)
│   └── experiment_tracking/# MLflow integration
│
├── infra/
│   ├── docker/             # Dockerfile.training, Dockerfile.serving
│   ├── compose/            # docker-compose.yml + prometheus.yml
│   └── k8s/               # Kubernetes Job + Deployment + PVC manifests
│
├── tests/
│   ├── unit/               # Model, tokenizer, dataloader unit tests
│   ├── integration/        # Training step, LoRA, DPO integration tests
│   └── serving/            # FastAPI endpoint tests
│
├── configs/                # YAML configs for every stage
├── .github/workflows/ci.yml# GitHub Actions CI (lint → type → test → validate → build)
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Memory Estimation (8 GB GPU)

| Component | Memory |
|---|---|
| Model weights ~302M (BF16) | ~604 MB |
| Gradient buffer (BF16) | ~604 MB |
| AdamW optimizer states (FP32) | ~2.4 GB |
| Activations (batch=8, seq=512, grad-ckpt) | ~1.5 GB |
| CUDA overhead + workspace | ~0.9 GB |
| **Total (estimate)** | **~6.1 GB** |

With gradient checkpointing enabled and batch size ≤ 8, the stack fits comfortably within 8 GB.

For LoRA / DPO with a frozen reference model, use `offload_ref_to_cpu: true` (default) to keep the reference on CPU, reducing VRAM by ~604 MB.

---

## Quick Start

### 1. Install

```bash
# Create environment
python3.10 -m venv .venv && source .venv/bin/activate

# Install all dependencies
make install

# Create sample datasets
make setup-data
```

### 2. Run Tests (no GPU required)

```bash
make test
```

### 3. Train from Scratch

```bash
# Optionally edit configs/pretrain.yaml first
make pretrain
```

Checkpoints are written to `./checkpoints/pretrain/`.

### 4. Supervised Fine-Tuning (SFT)

Prepare your data as a JSONL file with `instruction` and `response` keys:
```jsonl
{"instruction": "What is the speed of light?", "response": "Approximately 3×10⁸ m/s."}
```

```bash
# Point configs/sft.yaml → data.train_path to your data
make sft
```

### 5. LoRA Fine-Tuning

```bash
# Edit configs/lora.yaml → lora.r, target_modules, etc.
make lora
```

To merge the LoRA adapter into the base model:
```bash
make merge
# or during training:
python -m services.training.lora.train --config configs/lora.yaml --merge
```

### 6. DPO (Direct Preference Optimisation)

Prepare preference data:
```jsonl
{"prompt": "Tell me about AI.", "chosen": "Detailed accurate answer.", "rejected": "IDK"}
```

```bash
make dpo
```

The trainer uses `β=0.1` by default. The reference model is offloaded to CPU to save VRAM.

### 7. Evaluate

```bash
make evaluate
# Or point to a specific checkpoint:
MODEL_PATH=./checkpoints/sft/best make evaluate
```

Results are written to `./eval_results/benchmark_results.json`.

### 8. Serve with vLLM

```bash
# Production (GPU + vLLM):
make serve

# Development (CPU, no GPU required):
make serve-dev
```

API endpoints:
- `GET  /health`             – liveness check
- `GET  /version`            – model version info
- `POST /generate`           – single generation
- `POST /generate/stream`    – SSE streaming generation
- `POST /batch`              – batch generation
- `GET  /metrics`            – Prometheus metrics

Example:
```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time,", "max_new_tokens": 64}'
```

---

## Docker

### Build Images

```bash
make docker-build
```

### Run Full Stack (Dev Mode, no GPU required)

```bash
make docker-up
# Services: serving-dev (:8000), prometheus (:9090), mlflow (:5000)
```

### Run Full Stack (Production, GPU required)

```bash
docker compose -f infra/compose/docker-compose.yml --profile production up
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./checkpoints/pretrain/final` | Path to model checkpoint |
| `USE_VLLM` | `true` | Enable vLLM inference backend |
| `DTYPE` | `bfloat16` | Model dtype |
| `MAX_SEQ_LEN` | `512` | Maximum context length |
| `GPU_MEM_UTIL` | `0.85` | vLLM GPU memory utilisation |
| `LOG_LEVEL` | `INFO` | Python log level |
| `JSON_LOGS` | `true` | Emit JSON-structured logs |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking server URI |

---

## Cloud Deployment (AWS/GCP)

### AWS (EKS)

```bash
# 1. Push images to ECR
REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com make docker-build
docker tag llm-stack-serving:latest $REGISTRY/llm-stack-serving:latest
docker push $REGISTRY/llm-stack-serving:latest

# 2. Apply K8s manifests
kubectl create namespace llm-stack
kubectl apply -f infra/k8s/serving-deployment.yaml
kubectl apply -f infra/k8s/training-job.yaml

# 3. S3 checkpoint sync (optional)
# Set env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_CHECKPOINT_BUCKET
```

### GCP (GKE)

Substitute `REGISTRY=gcr.io/<project-id>` and use GKE node pools with `accelerator: nvidia-t4-gpu` labels.

### S3-Compatible Storage

Set `S3_CHECKPOINT_BUCKET` to sync checkpoints automatically. The training container uses `boto3` to upload checkpoints after each save.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every PR to `main`:

| Step | Blocks merge? | Description |
|---|---|---|
| **Lint** (`ruff`) | Yes | Style + import checks |
| **Type check** (`mypy`) | Yes | Static type analysis |
| **Unit tests** | Yes | Model, tokenizer, dataloader |
| **Integration tests** | Yes | Training steps, LoRA, DPO |
| **Serving tests** | Yes | API endpoint validation |
| **Validate training step** | Yes | Forward + backward pass |
| **Validate serving** | Yes | Health + version endpoints |
| **Build Docker** | No (on push) | Builds both images |
| **Push Docker** | No (main only) | Pushes to GHCR |

The `ci-gate` job (required by branch protection) fails if **any** of the above fail.

---

## Model Specification

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Layers | 24 |
| Hidden size | 1024 |
| Intermediate size | 4096 |
| Attention heads | 16 |
| KV heads | 16 (full MHA; set `n_kv_heads=8` for GQA) |
| Context length | 512 |
| Positional encoding | RoPE |
| Normalisation | RMSNorm (pre-norm) |
| Activation | GELU |
| Weight tying | Yes (embed ↔ output) |
| Flash Attention | Optional |
| Total parameters | ~302M |

---

## Configuration System

All stages are configured via YAML files in `configs/`. Every config is validated at runtime with **Pydantic v2**. Override individual fields via environment variables or CLI flags.

```bash
# CLI override example:
python -m services.training.pretrain.train \
  --config configs/pretrain.yaml \
  --max_steps 5000 \
  --learning_rate 1e-4
```

---

## Makefile Reference

| Command | Description |
|---|---|
| `make install` | Install Python dependencies |
| `make setup-data` | Create sample JSONL datasets |
| `make lint` | Ruff lint |
| `make format` | Auto-format code |
| `make typecheck` | Mypy type check |
| `make test` | All tests (CPU) |
| `make preprocess` | Cache training datasets |
| `make pretrain` | Pretrain from scratch |
| `make sft` | Supervised fine-tuning |
| `make lora` | LoRA fine-tuning |
| `make dpo` | DPO preference optimisation |
| `make merge` | Merge LoRA into base model |
| `make evaluate` | Run benchmarks |
| `make serve` | Start API server (GPU) |
| `make serve-dev` | Start API server (CPU) |
| `make docker-build` | Build Docker images |
| `make docker-up` | Start dev stack |
| `make docker-down` | Stop stack |
| `make clean` | Remove build artefacts |

---

## Observability

- **Structured logging**: JSON logs via `structlog`. Set `JSON_LOGS=false` for human-readable output.
- **Prometheus metrics**: Exposed at `:8090/metrics` (serving) or via `observability/metrics/prometheus.py`.
- **MLflow**: Set `MLFLOW_TRACKING_URI` to your MLflow server. Metrics and artifacts auto-logged.
- **GPU tracking**: `GPU_MEMORY_MB` and `GPU_UTILISATION_PCT` gauges updated every log interval.

---

## License

MIT
