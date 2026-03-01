"""
MLflow experiment tracking integration.

Provides a lightweight wrapper around the MLflow client so training code
does not depend directly on MLflow internals.

Features:
  - Auto-creates experiments if they don't exist
  - Logs hyperparameters, metrics, and artefacts
  - Context-manager interface for clean run lifecycle management
  - Gracefully no-ops if MLflow is not installed or tracking URI is unset
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


class MLflowTracker:
    """Thin wrapper around MLflow for experiment tracking.

    Usage:
        tracker = MLflowTracker(experiment_name="pretrain")
        with tracker.start_run(run_name="run_001", tags={"stage": "pretrain"}):
            tracker.log_params({"lr": 3e-4, "batch_size": 8})
            tracker.log_metric("loss", 2.3, step=100)
            tracker.log_artifact("./checkpoints/best/model.pt")
    """

    def __init__(
        self,
        experiment_name: str = "llm-stack",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and _MLFLOW_AVAILABLE
        self._active_run = None

        if self.enabled:
            uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Iterator[None]:
        """Context manager that starts an MLflow run and ends it on exit."""
        if not self.enabled:
            yield
            return

        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            self._active_run = run
            try:
                yield
            finally:
                self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters (only at run start, not per step)."""
        if self.enabled and mlflow.active_run():
            # MLflow param values must be strings ≤ 500 chars
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar metric."""
        if self.enabled and mlflow.active_run():
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once."""
        if self.enabled and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file or directory as an MLflow artefact."""
        if self.enabled and mlflow.active_run():
            p = Path(path)
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path=artifact_path)
            elif p.exists():
                mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag."""
        if self.enabled and mlflow.active_run():
            mlflow.set_tag(key, value)

    def log_model_config(self, config_dict: dict) -> None:
        """Convenience: log a model config dict as parameters."""
        flat = {f"model.{k}": v for k, v in config_dict.items()}
        self.log_params(flat)
