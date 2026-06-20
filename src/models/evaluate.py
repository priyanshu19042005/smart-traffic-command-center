"""
evaluate.py
===========
**Evaluation pipeline** — recomputes rich diagnostics for a trained task on a
fresh hold-out split and produces dashboard-ready artifacts:

* confusion matrix + classification report (classification)
* ROC & precision-recall curve points (classification)
* residual stats + predicted-vs-actual points (regression)
* the headline metrics table

Artifacts are written to ``outputs/eval_<task>.json`` and consumed by the
**ML Predictions** dashboard page.

Run::

    python -m src.models.evaluate            # all tasks
    python -m src.models.evaluate closure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.registry import ModelRegistry
from src.models.train import _prepare_xy, _clf_metrics, _reg_metrics
from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


def evaluate_task(task: str, df: pd.DataFrame, cfg: Optional[Config] = None,
                  persist: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    registry = ModelRegistry(cfg)
    model = registry.load(task)
    task_cfg = getattr(cfg.models.tasks, task)
    seed = cfg.project.random_seed

    X, y, target = _prepare_xy(df, task_cfg)
    stratify = y if task_cfg.type == "classification" else None
    _, Xte, _, yte = train_test_split(
        X, y, test_size=cfg.models.test_size, random_state=seed, stratify=stratify)

    report: dict[str, Any] = {"task": task, "type": task_cfg.type, "target": target,
                              "n_eval": int(len(Xte))}

    if task_cfg.type == "classification":
        from sklearn.metrics import (confusion_matrix, classification_report,
                                     roc_curve, precision_recall_curve)
        ypred = model.predict(Xte)
        yproba = (model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else ypred)
        report["metrics"] = _clf_metrics(yte, ypred, yproba)
        report["confusion_matrix"] = confusion_matrix(yte, ypred).tolist()
        report["classification_report"] = classification_report(
            yte, ypred, output_dict=True, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(yte, yproba)
            prec, rec, _ = precision_recall_curve(yte, yproba)
            # Downsample curves for compact JSON.
            step = max(1, len(fpr) // 100)
            report["roc_curve"] = {"fpr": fpr[::step].round(4).tolist(),
                                   "tpr": tpr[::step].round(4).tolist()}
            report["pr_curve"] = {"precision": prec[::step].round(4).tolist(),
                                  "recall": rec[::step].round(4).tolist()}
        except Exception as exc:
            log.debug("Curve computation skipped: %s", exc)
    else:
        ypred = model.predict(Xte)
        report["metrics"] = _reg_metrics(yte, ypred)
        resid = yte - ypred
        report["residuals"] = {"mean": float(resid.mean()), "std": float(resid.std())}
        idx = np.random.RandomState(seed).choice(
            len(yte), size=min(500, len(yte)), replace=False)
        report["scatter"] = {"actual": np.round(yte[idx], 2).tolist(),
                             "predicted": np.round(ypred[idx], 2).tolist()}

    log.info("Eval[%s]: %s", task, {k: round(v, 4) for k, v in report["metrics"].items()})
    if persist:
        out = get_path("outputs_dir", cfg=cfg) / f"eval_{task}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Eval report -> %s", out)
    return report


def evaluate_all(df: Optional[pd.DataFrame] = None, cfg: Optional[Config] = None) -> dict:
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_parquet(get_path("features", cfg=cfg))
    reports = {}
    for task in cfg.models.tasks:
        try:
            reports[task] = evaluate_task(task, df, cfg)
        except FileNotFoundError:
            log.warning("Skipping '%s' (not trained).", task)
    return reports


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    if len(sys.argv) > 1:
        print(json.dumps(evaluate_task(sys.argv[1], feats, cfg)["metrics"], indent=2))
    else:
        res = evaluate_all(feats, cfg)
        for t, r in res.items():
            print(t, {k: round(v, 4) for k, v in r["metrics"].items()})
