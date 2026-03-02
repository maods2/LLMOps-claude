"""
Direct Preference Optimisation (DPO) trainer.

Implements the DPO objective from Rafailov et al. (2023):

    L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[
        log σ(β * log π_θ(y_w|x)/π_ref(y_w|x) - β * log π_θ(y_l|x)/π_ref(y_l|x))
    ]

Key properties:
  - No reward model required
  - No PPO / RL outer loop
  - Low memory: reference model run in no-grad, can be offloaded to CPU
  - Tracks reward margin: E[r_w - r_l] as a training signal indicator
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pydantic import BaseModel, ConfigDict, Field


class DPOConfig(BaseModel):
    """DPO training configuration."""

    output_dir: str = Field("./checkpoints/dpo")
    resume_from: Optional[str] = Field(None)

    max_steps: int = Field(1_000)
    eval_interval: int = Field(100)
    save_interval: int = Field(200)
    log_interval: int = Field(10)

    learning_rate: float = Field(5e-6)
    weight_decay: float = Field(0.01)
    warmup_steps: int = Field(50)
    grad_clip: float = Field(1.0)

    batch_size: int = Field(2)
    gradient_accumulation_steps: int = Field(8)

    # DPO-specific
    beta: float = Field(0.1, description="KL penalty coefficient (temperature)")
    reference_free: bool = Field(
        False, description="If True, skip reference model (implicit uniform ref)"
    )
    offload_ref_to_cpu: bool = Field(
        True, description="Keep reference model on CPU to save VRAM"
    )

    dtype: str = Field("bfloat16")
    gradient_checkpointing: bool = Field(True)

    model_config = ConfigDict(frozen=True)


def _compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute per-token log probabilities and average over response tokens.

    Labels are -100 for prompt tokens; we average only over response positions.

    Returns:
        (batch,) tensor of mean log-probs per sequence.
    """
    device = input_ids.device
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(dtype != torch.float32),
    ):
        out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, 1:]       # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    # Gather log-prob of target tokens; -100 positions zeroed
    response_mask = (shift_labels != -100).float()
    clamped = shift_labels.clamp(min=0)
    token_log_probs = log_probs.gather(-1, clamped.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    token_log_probs = token_log_probs * response_mask

    # Sum / normalise over response tokens (avoid div-by-zero)
    n_tokens = response_mask.sum(dim=-1).clamp(min=1.0)
    return token_log_probs.sum(dim=-1) / n_tokens  # (B,)


class DPOTrainer:
    """
    Direct Preference Optimisation trainer.

    Args:
        policy_model:    The model being trained (π_θ).
        ref_model:       The frozen reference model (π_ref). Pass None if reference_free=True.
        train_loader:    DataLoader yielding DPO-formatted batches.
        val_loader:      Optional validation DataLoader.
        config:          DPOConfig.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: DPOConfig,
        logger=None,
    ) -> None:
        self.config = config
        self.logger = logger or _default_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _parse_dtype(config.dtype)

        self.policy = policy_model.to(self.device)
        if config.gradient_checkpointing:
            self.policy.enable_gradient_checkpointing()

        # Reference model: frozen, optionally on CPU
        self.ref: Optional[nn.Module] = None
        if not config.reference_free and ref_model is not None:
            self.ref = ref_model
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
            if config.offload_ref_to_cpu:
                self.ref.to("cpu")
            else:
                self.ref.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        from torch.optim import AdamW
        from torch.amp import GradScaler
        from services.training.pretrain.trainer import _get_scheduler

        self.optimizer = AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = _get_scheduler(self.optimizer, config.warmup_steps, config.max_steps)
        self.scaler = GradScaler("cuda", enabled=(self.dtype == torch.float16))

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------

    def _dpo_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Compute DPO loss for one batch.

        Returns:
            loss scalar and a dict of metrics.
        """
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # Policy log-probs
        pi_lp_chosen = _compute_log_probs(
            self.policy, chosen_ids, chosen_mask, chosen_labels, self.dtype
        )
        pi_lp_rejected = _compute_log_probs(
            self.policy, rejected_ids, rejected_mask, rejected_labels, self.dtype
        )

        # Reference log-probs
        if self.ref is not None:
            ref_device = next(self.ref.parameters()).device
            with torch.no_grad():
                ref_lp_chosen = _compute_log_probs(
                    self.ref,
                    chosen_ids.to(ref_device),
                    chosen_mask.to(ref_device),
                    chosen_labels.to(ref_device),
                    self.dtype,
                ).to(self.device)
                ref_lp_rejected = _compute_log_probs(
                    self.ref,
                    rejected_ids.to(ref_device),
                    rejected_mask.to(ref_device),
                    rejected_labels.to(ref_device),
                    self.dtype,
                ).to(self.device)
        else:
            # Reference-free: uniform reference → log-probs cancel
            ref_lp_chosen = torch.zeros_like(pi_lp_chosen)
            ref_lp_rejected = torch.zeros_like(pi_lp_rejected)

        # DPO objective
        log_ratio_chosen = pi_lp_chosen - ref_lp_chosen
        log_ratio_rejected = pi_lp_rejected - ref_lp_rejected
        logits_dpo = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits_dpo).mean()

        # Reward margin (positive = model prefers chosen)
        reward_margin = (log_ratio_chosen - log_ratio_rejected).mean().item()
        chosen_reward = log_ratio_chosen.mean().item()
        rejected_reward = log_ratio_rejected.mean().item()

        return loss, {
            "reward_margin": reward_margin,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.policy.train()
        data_iter = iter(self.train_loader)
        accum_loss = 0.0
        accum_margin = 0.0
        t0 = time.perf_counter()

        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_margin = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss, metrics = self._dpo_loss(batch)
                scaled_loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                step_loss += scaled_loss.item()
                step_margin += metrics["reward_margin"] / self.config.gradient_accumulation_steps

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            accum_loss += step_loss
            accum_margin += step_margin

            if self.global_step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = accum_loss / self.config.log_interval
                avg_margin = accum_margin / self.config.log_interval
                self.logger.info(
                    "dpo_step",
                    step=self.global_step,
                    loss=round(avg_loss, 4),
                    reward_margin=round(avg_margin, 4),
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    steps_per_sec=round(self.config.log_interval / elapsed, 2),
                )
                accum_loss = 0.0
                accum_margin = 0.0
                t0 = time.perf_counter()

            if self.global_step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self._evaluate()
                self.policy.train()
                self.logger.info("dpo_val", step=self.global_step, val_loss=round(val_loss, 4))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")

            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}")

        self._save_checkpoint("final")

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.policy.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:  # type: ignore[union-attr]
            loss, _ = self._dpo_loss(batch)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = Path(self.config.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy, "save_pretrained"):
            self.policy.save_pretrained(str(ckpt_dir))  # type: ignore[attr-defined]
        else:
            torch.save(self.policy.state_dict(), ckpt_dir / "model.pt")
        self.logger.info("dpo_checkpoint_saved", path=str(ckpt_dir))


def _parse_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_str
    ]


def _default_logger():
    import structlog
    return structlog.get_logger()
