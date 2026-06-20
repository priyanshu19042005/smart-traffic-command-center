"""
registry.py
===========
**Model Registry** — a thin abstraction that logs params/metrics and persists
artifacts, backed by **MLflow** when available and a **local JSON + joblib**
store otherwise. The rest of the code never imports MLflow directly, so the
platform runs identically with or without it.

Layout (local backend)::

    models/
      <task>/
        model.joblib            # fitted sklearn Pipeline
        metadata.json           # params, metrics, feature names, timestamp
        feature_importance.csv
      registry.json             # index of all runs (the "registry")
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


def _mlflow_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("mlflow") is not None


class ModelRegistry:
    """Backend-agnostic experiment + artifact store."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()
        backend = self.cfg.models.registry_backend
        self.use_mlflow = (backend == "mlflow") or (backend == "auto" and _mlflow_available())
        self.root = get_path("models_dir", cfg=self.cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if self.use_mlflow:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri((self.root / "mlruns").as_uri())
            mlflow.set_experiment(self.cfg.models.mlflow_experiment)
            log.info("ModelRegistry backend = MLflow (%s).", self.root / "mlruns")
        else:
            log.info("ModelRegistry backend = local (%s).", self.root)

    # -- logging a run ---------------------------------------------------
    def log_run(self, task: str, model, params: dict[str, Any],
                metrics: dict[str, float], feature_names: list[str],
                importances: Optional[dict[str, float]] = None,
                extra: Optional[dict[str, Any]] = None) -> Path:
        task_dir = self.root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        model_path = task_dir / "model.joblib"
        joblib.dump(model, model_path)

        meta = {
            "task": task,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": metrics,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            **(extra or {}),
        }
        (task_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")

        if importances:
            import csv
            with open(task_dir / "feature_importance.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["feature", "importance"])
                for k, v in sorted(importances.items(), key=lambda x: -x[1]):
                    w.writerow([k, v])

        self._update_index(task, meta)

        if self.use_mlflow:
            with self._mlflow.start_run(run_name=f"{task}"):
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                          if v == v})  # skip NaN
                try:
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(model, name="model")
                except Exception as exc:  # pragma: no cover
                    log.debug("mlflow log_model skipped: %s", exc)

        log.info("Registered '%s' | metrics=%s", task,
                 {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})
        return model_path

    def _update_index(self, task: str, meta: dict) -> None:
        index = {}
        if self.index_path.exists():
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        index[task] = {
            "created_at_utc": meta["created_at_utc"],
            "metrics": meta["metrics"],
            "best_model": meta.get("best_model"),
            "path": str((self.root / task / "model.joblib")),
        }
        self.index_path.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")

    # -- loading ---------------------------------------------------------
    def load(self, task: str):
        model_path = self.root / task / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No registered model for task '{task}'. Train first.")
        return joblib.load(model_path)

    def metadata(self, task: str) -> dict:
        p = self.root / task / "metadata.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def list_runs(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {}
