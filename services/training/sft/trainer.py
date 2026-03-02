"""
Supervised Fine-Tuning (SFT) trainer.

Extends PretrainTrainer with:
  - Masked loss (only response tokens contribute)
  - Instruction-response template formatting
  - Early stopping support
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from services.training.core_model.model import LLMModel
from services.training.pretrain.trainer import (
    _get_param_groups,
    _get_scheduler,
    _gpu_memory_mb,
    _parse_dtype,
)


class SFTConfig(BaseModel):
    """SFT-specific configuration (extends pretrain settings)."""

    output_dir: str = Field("./checkpoints/sft")
    resume_from: str | None = Field(None)

    max_steps: int = Field(3_000)
    eval_interval: int = Field(200)
    save_interval: int = Field(500)
    log_interval: int = Field(10)

    learning_rate: float = Field(2e-5)
    weight_decay: float = Field(0.01)
    beta1: float = Field(0.9)
    beta2: float = Field(0.999)
    grad_clip: float = Field(1.0)
    warmup_steps: int = Field(100)

    batch_size: int = Field(4)
    gradient_accumulation_steps: int = Field(8)

    dtype: str = Field("bfloat16")
    gradient_checkpointing: bool = Field(True)

    # SFT-specific
    early_stopping_patience: int = Field(5, description="Stop if val loss doesn't improve for N evals")

    model_config = ConfigDict(frozen=True)


class SFTTrainer:
    """Supervised fine-tuning trainer with masked loss on responses."""

    def __init__(
        self,
        model: LLMModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: SFTConfig,
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

        from torch.amp import GradScaler
        from torch.optim import AdamW

        self.optimizer = AdamW(
            _get_param_groups(model, config.weight_decay),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.scheduler = _get_scheduler(
            self.optimizer, config.warmup_steps, config.max_steps
        )
        self.scaler = GradScaler("cuda", enabled=(self.dtype == torch.float16))

        self.global_step = 0
        self.best_val_loss = float("inf")
        self._no_improve_count = 0

    def train(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.model.train()
        data_iter = iter(self.train_loader)
        accum_loss = 0.0
        t0 = time.perf_counter()

        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)  # -100 on prompt tokens
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

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            accum_loss += step_loss

            if self.global_step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = accum_loss / self.config.log_interval
                self.logger.info(
                    "sft_step",
                    step=self.global_step,
                    loss=round(avg_loss, 4),
                    ppl=round(math.exp(min(avg_loss, 20)), 2),
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    gpu_mb=_gpu_memory_mb(),
                    steps_per_sec=round(self.config.log_interval / elapsed, 2),
                )
                accum_loss = 0.0
                t0 = time.perf_counter()

            if self.global_step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self._evaluate()
                self.model.train()
                self.logger.info("sft_val", step=self.global_step, val_loss=round(val_loss, 4))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._no_improve_count = 0
                    self._save_checkpoint("best")
                else:
                    self._no_improve_count += 1
                    if self._no_improve_count >= self.config.early_stopping_patience:
                        self.logger.info("early_stopping", step=self.global_step)
                        break

            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}")

        self._save_checkpoint("final")

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:  # type: ignore[union-attr]
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
            total += out.loss.item()
            n += 1
        return total / max(n, 1)

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = Path(self.config.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ckpt_dir))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            ckpt_dir / "training_state.pt",
        )


def _default_logger():
    import structlog
    return structlog.get_logger()
