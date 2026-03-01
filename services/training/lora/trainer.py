"""
LoRA (Low-Rank Adaptation) fine-tuning using HuggingFace PEFT.

Features:
  - Injects LoRA into Q, K, V, O attention projections
  - Configurable rank (r) and alpha
  - Optional 4-bit QLoRA via bitsandbytes
  - Adapter-only saving / loading
  - Merge adapter weights back into base model
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    # LoRA hyperparameters
    r: int = Field(16, description="LoRA rank (low-rank decomposition dim)")
    lora_alpha: float = Field(32.0, description="LoRA scaling factor alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Modules to apply LoRA to",
    )
    bias: str = Field("none", description="Bias mode: none | all | lora_only")

    # QLoRA
    use_4bit: bool = Field(False, description="Use 4-bit quantisation (QLoRA)")
    bnb_4bit_compute_dtype: str = Field("bfloat16", description="Compute dtype for 4-bit ops")

    # Training
    output_dir: str = Field("./checkpoints/lora")
    max_steps: int = Field(2_000)
    eval_interval: int = Field(200)
    save_interval: int = Field(500)
    log_interval: int = Field(10)

    learning_rate: float = Field(3e-4)
    weight_decay: float = Field(0.01)
    warmup_steps: int = Field(100)
    grad_clip: float = Field(1.0)

    batch_size: int = Field(4)
    gradient_accumulation_steps: int = Field(8)

    dtype: str = Field("bfloat16")
    gradient_checkpointing: bool = Field(True)

    class Config:
        frozen = True


class _HFCompatConfig:
    """HuggingFace-style config shim so PEFT can introspect our Pydantic ModelConfig.

    PEFT calls either:
      - model_config.to_dict()  (HF PretrainedConfig)
      - dataclasses.asdict()    (Python dataclass)
      - model_config.get(key)   (dict-like)
    We support all three interfaces.
    """

    model_type: str = "custom"

    def __init__(self, model_config) -> None:
        # Store the underlying dict so all access is fast
        if hasattr(model_config, "model_dump"):
            self._d = model_config.model_dump()
        elif hasattr(model_config, "__dict__"):
            self._d = dict(model_config.__dict__)
        else:
            self._d = {}
        self._d.setdefault("model_type", "custom")

    def to_dict(self) -> dict:
        return self._d

    def get(self, key: str, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key: str):
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        return self._d.get(key)


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Wrap the model with PEFT LoRA adapters.

    Args:
        model:  Base LLMModel or any nn.Module.
        config: LoRA configuration.
    Returns:
        PEFT-wrapped model (only adapter params trainable).
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        inference_mode=False,
    )
    return get_peft_model(model, peft_config)


def load_qlora_model(base_model_path: str, config: LoRAConfig) -> nn.Module:
    """Load a 4-bit quantised model and inject LoRA (QLoRA setup).

    Requires bitsandbytes.
    """
    import bitsandbytes as bnb
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )

    from services.training.core_model.model import LLMModel

    model = LLMModel.from_pretrained(base_model_path)
    model = prepare_model_for_kbit_training(model)
    return inject_lora(model, config)


def merge_and_save(
    peft_model: nn.Module,
    save_dir: str,
) -> None:
    """Merge LoRA weights into the base model and save.

    Args:
        peft_model: A PEFT model with LoRA adapters.
        save_dir:   Directory to save the merged model.
    """
    merged = peft_model.merge_and_unload()  # type: ignore[attr-defined]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # If the merged model is our LLMModel it has save_pretrained.
    # The config may be the _HFCompatConfig shim; LLMModel.save_pretrained handles that.
    if hasattr(merged, "save_pretrained"):
        merged.save_pretrained(save_dir)
    else:
        torch.save(merged.state_dict(), Path(save_dir) / "model.pt")


class LoRATrainer:
    """LoRA fine-tuning trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: LoRAConfig,
        logger=None,
    ) -> None:
        self.config = config
        self.logger = logger or _default_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _parse_dtype(config.dtype)

        # Inject LoRA
        self.model = inject_lora(model, config).to(self.device)

        if config.gradient_checkpointing:
            try:
                self.model.enable_input_require_grads()  # type: ignore[attr-defined]
            except AttributeError:
                pass

        self.train_loader = train_loader
        self.val_loader = val_loader

        from torch.optim import AdamW
        from torch.cuda.amp import GradScaler

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(
            "lora_setup",
            trainable_params=sum(p.numel() for p in trainable_params),
            total_params=sum(p.numel() for p in self.model.parameters()),
        )

        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        from services.training.pretrain.trainer import _get_scheduler
        self.scheduler = _get_scheduler(self.optimizer, config.warmup_steps, config.max_steps)
        self.scaler = GradScaler(enabled=(self.dtype == torch.float16))

        self.global_step = 0
        self.best_val_loss = float("inf")

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
                    "lora_step",
                    step=self.global_step,
                    loss=round(avg_loss, 4),
                    ppl=round(math.exp(min(avg_loss, 20)), 2),
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    steps_per_sec=round(self.config.log_interval / elapsed, 2),
                )
                accum_loss = 0.0
                t0 = time.perf_counter()

            if self.global_step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self._evaluate()
                self.model.train()
                self.logger.info("lora_val", step=self.global_step, val_loss=round(val_loss, 4))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_adapter("best")

            if self.global_step % self.config.save_interval == 0:
                self._save_adapter(f"step_{self.global_step:07d}")

        self._save_adapter("final")

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

    def _save_adapter(self, tag: str) -> None:
        """Save only the LoRA adapter weights (not the full model)."""
        adapter_dir = Path(self.config.output_dir) / tag
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_dir))  # type: ignore[attr-defined]
        self.logger.info("lora_adapter_saved", path=str(adapter_dir))

    def merge_and_save(self, save_dir: str) -> None:
        """Merge LoRA adapters into base and save full model."""
        merge_and_save(self.model, save_dir)
        self.logger.info("lora_merged", path=save_dir)


def _parse_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_str
    ]


def _default_logger():
    import structlog
    return structlog.get_logger()
