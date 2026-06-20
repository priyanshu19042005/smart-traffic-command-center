"""
predict.py
==========
**Prediction / inference pipeline.** Loads a registered model and scores new
(or held-out) events. Guarantees train/serve parity by reusing the exact fitted
``Pipeline`` (preprocessor + selector + estimator) saved by ``train.py``.

Usage
-----
>>> from src.models.predict import Predictor
>>> p = Predictor()                       # loads all registered tasks
>>> scored = p.predict_frame(events_df)   # adds pred_* columns
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.registry import ModelRegistry
from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


class Predictor:
    """Serve predictions for one or all registered tasks."""

    def __init__(self, cfg: Optional[Config] = None, tasks: Optional[list[str]] = None) -> None:
        self.cfg = cfg or load_config()
        self.registry = ModelRegistry(self.cfg)
        self.tasks = tasks or list(self.cfg.models.tasks)
        self.models: dict[str, Any] = {}
        self.meta: dict[str, dict] = {}
        for t in self.tasks:
            try:
                self.models[t] = self.registry.load(t)
                self.meta[t] = self.registry.metadata(t)
            except FileNotFoundError:
                log.warning("Task '%s' not trained yet; skipping.", t)

    def _features_for(self, task: str, df: pd.DataFrame) -> pd.DataFrame:
        names = self.meta[task].get("feature_names", [])
        cols = [c for c in names if c in df.columns]
        missing = set(names) - set(cols)
        if missing:
            # Models tolerate missing OHE categories; create empty cols to keep schema.
            for c in missing:
                df = df.assign(**{c: np.nan})
            cols = names
        return df[cols]

    def predict_task(self, task: str, df: pd.DataFrame) -> pd.DataFrame:
        if task not in self.models:
            raise FileNotFoundError(f"Model for '{task}' is not available.")
        model = self.models[task]
        X = self._features_for(task, df)
        task_cfg = getattr(self.cfg.models.tasks, task)
        out = pd.DataFrame(index=df.index)
        if task_cfg.type == "classification":
            proba = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba")
                     else model.predict(X))
            out[f"{task}_proba"] = np.round(proba, 4)
            out[f"{task}_pred"] = (proba >= 0.5).astype(int)
            out[f"{task}_label"] = np.where(
                out[f"{task}_pred"] == 1, str(task_cfg.positive_label), "other")
        else:
            out[f"{task}_pred"] = np.round(model.predict(X), 2)
        return out

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append predictions for every available task to a copy of *df*."""
        result = df.copy()
        for task in self.models:
            preds = self.predict_task(task, df)
            result = pd.concat([result, preds], axis=1)
        log.info("Scored %s rows across tasks %s.", len(df), list(self.models))
        return result


def predict_task(task: str, df: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    """Convenience wrapper for a single-task prediction."""
    return Predictor(cfg, tasks=[task]).predict_task(task, df)


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg)).head(10)
    scored = Predictor(cfg).predict_frame(feats)
    cols = [c for c in scored.columns if any(t in c for t in cfg.models.tasks)]
    print(scored[["id", "event_cause"] + cols].to_string(index=False))
