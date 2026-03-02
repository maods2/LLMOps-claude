"""
Pretraining loop for the decoder-only Transformer.

Features:
  - Mixed precision (BF16/FP16)
  - Gradient accumulation
  - Gradient checkpointing
  - Cosine LR schedule with warm-up
  - Checkpoint save/resume
  - Structured logging + Prometheus metrics
  - GPU memory tracking
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from services.training.core_model.model import LLMModel


class PretrainConfig(BaseModel):
    """Pretraining hyperparameters."""

    # I/O
    output_dir: str = Field("./checkpoints/pretrain", description="Output directory")
    resume_from: str | None = Field(None, description="Path to checkpoint to resume from")

    # Training loop
    max_steps: int = Field(10_000, description="Maximum training steps")
    eval_interval: int = Field(500, description="Evaluate every N steps")
    save_interval: int = Field(1_000, description="Save checkpoint every N steps")
    log_interval: int = Field(10, description="Log metrics every N steps")

    # Optimiser
    learning_rate: float = Field(3e-4, description="Peak learning rate")
    weight_decay: float = Field(0.1, description="AdamW weight decay")
    beta1: float = Field(0.9, description="AdamW beta1")
    beta2: float = Field(0.95, description="AdamW beta2")
    grad_clip: float = Field(1.0, description="Gradient clip norm (0 = disabled)")
    warmup_steps: int = Field(500, description="LR warm-up steps")

    # Batch
    batch_size: int = Field(8, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        4, description="Accumulation steps (effective batch = batch×accum)"
    )

    # Precision
    dtype: str = Field("bfloat16", description="Training dtype: float32 | float16 | bfloat16")

    # Memory
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")

    model_config = ConfigDict(frozen=True)


def _cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _get_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_schedule_with_warmup(step, warmup_steps, max_steps),
    )


class PretrainTrainer:
    """Manages the pretraining loop."""

    def __init__(
        self,
        model: LLMModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: PretrainConfig,
        logger=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or _default_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _parse_dtype(config.dtype)

        self.model.to(self.device)
        if config.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.optimizer = AdamW(
            _get_param_groups(model, config.weight_decay),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available(),
        )
        self.scheduler = _get_scheduler(
            self.optimizer, config.warmup_steps, config.max_steps
        )
        self.scaler = GradScaler("cuda", enabled=(self.dtype == torch.float16))

        self.global_step: int = 0
        self.best_val_loss: float = float("inf")

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full pretraining loop."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self.model.train()
        data_iter = iter(self.train_loader)
        accum_loss = 0.0
        t0 = time.perf_counter()

        while self.global_step < self.config.max_steps:
            # ----------------------------------------------------------
            # Gradient accumulation
            # ----------------------------------------------------------
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.dtype != torch.float32),
                ):
                    out = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # ----------------------------------------------------------
            # Optimiser step
            # ----------------------------------------------------------
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            accum_loss += step_loss

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = accum_loss / self.config.log_interval
                perplexity = math.exp(min(avg_loss, 20))
                lr = self.scheduler.get_last_lr()[0]
                gpu_mb = _gpu_memory_mb()
                self.logger.info(
                    "train_step",
                    step=self.global_step,
                    loss=round(avg_loss, 4),
                    perplexity=round(perplexity, 2),
                    lr=f"{lr:.2e}",
                    gpu_mb=gpu_mb,
                    steps_per_sec=round(self.config.log_interval / elapsed, 2),
                )
                accum_loss = 0.0
                t0 = time.perf_counter()

            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            if self.global_step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self._evaluate()
                self.model.train()
                self.logger.info(
                    "val_step",
                    step=self.global_step,
                    val_loss=round(val_loss, 4),
                    val_ppl=round(math.exp(min(val_loss, 20)), 2),
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}")

        self._save_checkpoint("final")
        self.logger.info("training_complete", steps=self.global_step)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in self.val_loader:  # type: ignore[union-attr]
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=(self.dtype != torch.float32),
            ):
                out = self.model(input_ids, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = Path(self.config.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ckpt_dir))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            ckpt_dir / "training_state.pt",
        )
        self.logger.info("checkpoint_saved", path=str(ckpt_dir))

    def _load_checkpoint(self, path: str) -> None:

        state_path = Path(path) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.global_step = state["global_step"]
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.logger.info("checkpoint_loaded", path=path, step=self.global_step)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def _gpu_memory_mb() -> int:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() // (1024 * 1024)
    return 0


def _get_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    """Separate parameters into decay / no-decay groups."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _default_logger():
    import structlog
    return structlog.get_logger()
