"""
train.py
========
**Training pipeline** for the three supervised tasks defined in config:

    M1 priority  (classification)  target = priority           (High vs Low)
    M2 closure   (classification)  target = requires_road_closure (imbalanced)
    M3 risk      (regression)      target = risk_score          (0-100 engineered)

For every task it runs:

1. **Data split**          — stratified train/test (classification).
2. **Model candidates**    — Random Forest, XGBoost, LightGBM
                             (XGB/LGBM fall back to sklearn Gradient/HistGB
                             when the libraries are absent).
3. **Cross-validation**    — k-fold CV picks the best model *family* on the
                             primary metric (F1 for clf, R² for reg).
4. **Feature selection**   — optional ``SelectFromModel`` (top-k importances).
5. **Hyperparameter tuning** — ``RandomizedSearchCV`` on the winning family.
6. **Evaluation**          — held-out metrics: Accuracy/Precision/Recall/F1/
                             ROC-AUC (clf) or R²/MAE/RMSE (reg).
7. **Registry**            — params, metrics, importances, artifact logged via
                             :class:`ModelRegistry` (MLflow or local).

Run::

    python -m src.models.train            # trains all tasks
    python -m src.models.train priority   # single task
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     KFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

from src.data_pipeline.feature_engineering import (
    build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
from src.models.registry import ModelRegistry
from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


HAS_XGB, HAS_LGBM = _has("xgboost"), _has("lightgbm")


# ---------------------------------------------------------------------------
# Model + hyperparameter factory
# ---------------------------------------------------------------------------
def _candidates(task_type: str, seed: int, class_weight: Optional[str]) -> dict[str, tuple]:
    """Return {name: (estimator, param_distributions)} for the task type."""
    cands: dict[str, tuple] = {}

    if task_type == "classification":
        from sklearn.ensemble import RandomForestClassifier
        cands["random_forest"] = (
            RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=-1),
            {"model__n_estimators": [200, 400, 600],
             "model__max_depth": [None, 8, 16, 24],
             "model__min_samples_leaf": [1, 2, 4],
             "model__max_features": ["sqrt", "log2", 0.5]},
        )
        if HAS_XGB:
            from xgboost import XGBClassifier
            cands["xgboost"] = (
                XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss",
                              tree_method="hist"),
                {"model__n_estimators": [200, 400, 600],
                 "model__max_depth": [3, 5, 7],
                 "model__learning_rate": [0.03, 0.06, 0.1],
                 "model__subsample": [0.8, 1.0],
                 "model__colsample_bytree": [0.8, 1.0]},
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            cands["hist_gb"] = (
                HistGradientBoostingClassifier(random_state=seed),
                {"model__max_iter": [200, 400],
                 "model__learning_rate": [0.03, 0.06, 0.1],
                 "model__max_depth": [None, 6, 10]},
            )
        if HAS_LGBM:
            from lightgbm import LGBMClassifier
            cands["lightgbm"] = (
                LGBMClassifier(random_state=seed, n_jobs=-1, class_weight=class_weight,
                               verbose=-1),
                {"model__n_estimators": [200, 400, 600],
                 "model__num_leaves": [31, 63, 127],
                 "model__learning_rate": [0.03, 0.06, 0.1],
                 "model__subsample": [0.8, 1.0]},
            )
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor
        cands["random_forest"] = (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {"model__n_estimators": [200, 400, 600],
             "model__max_depth": [None, 10, 20],
             "model__min_samples_leaf": [1, 2, 4]},
        )
        if HAS_XGB:
            from xgboost import XGBRegressor
            cands["xgboost"] = (
                XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist"),
                {"model__n_estimators": [200, 400, 600],
                 "model__max_depth": [3, 5, 7],
                 "model__learning_rate": [0.03, 0.06, 0.1],
                 "model__subsample": [0.8, 1.0]},
            )
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            cands["hist_gb"] = (
                HistGradientBoostingRegressor(random_state=seed),
                {"model__max_iter": [200, 400],
                 "model__learning_rate": [0.03, 0.06, 0.1]},
            )
        if HAS_LGBM:
            from lightgbm import LGBMRegressor
            cands["lightgbm"] = (
                LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1),
                {"model__n_estimators": [200, 400, 600],
                 "model__num_leaves": [31, 63, 127],
                 "model__learning_rate": [0.03, 0.06, 0.1]},
            )
    return cands


def _fast_candidates(task_type: str, seed: int, class_weight: Optional[str]) -> dict[str, tuple]:
    """A single, compact Random Forest — for resource-constrained environments
    (e.g. Streamlit Cloud). No CV/tuning, ``n_jobs=1`` to cap memory."""
    if task_type == "classification":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=120, max_depth=16, n_jobs=1,
                                     class_weight=class_weight, random_state=seed)
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_estimators=120, max_depth=16, n_jobs=1,
                                    random_state=seed)
    return {"random_forest": (est, {})}


def _build_pipeline(estimator, preprocessor, task_type: str, cfg: Config, seed: int,
                    use_selection: Optional[bool] = None) -> Pipeline:
    steps = [("prep", preprocessor)]
    fs = cfg.models.feature_selection
    enabled = fs.enabled if use_selection is None else use_selection
    if enabled:
        from sklearn.feature_selection import SelectFromModel
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier as RF
        else:
            from sklearn.ensemble import RandomForestRegressor as RF
        steps.append(("select", SelectFromModel(
            RF(n_estimators=200, random_state=seed, n_jobs=-1),
            max_features=fs.top_k, threshold=-np.inf)))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clf_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        m["roc_auc"] = float("nan")
    return m


def _reg_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------
def _prepare_xy(df: pd.DataFrame, task_cfg: Config) -> tuple[pd.DataFrame, np.ndarray, str]:
    target = task_cfg.target
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in features; re-run the data pipeline.")
    df = df[df[target].notna()].copy()           # null targets dropped (e.g. unresolved)

    # Per-task leakage guard: exclude features that trivially encode the target.
    leak = list(getattr(task_cfg, "leakage_features", []) or [])
    feats = [c for c in (CATEGORICAL_FEATURES + NUMERIC_FEATURES)
             if c in df.columns and c not in leak]
    X = df[feats]

    if task_cfg.type == "classification":
        pos = str(task_cfg.positive_label).lower()
        y = (df[target].astype(str).str.lower() == pos).astype(int).to_numpy()
    else:
        y = df[target].astype(float).to_numpy()
        cap = getattr(task_cfg, "clip_max", None)
        lo = getattr(task_cfg, "min_target", 0)
        if cap is not None:
            y = np.clip(y, lo, cap)
    return X, y, target


# ---------------------------------------------------------------------------
# Train one task
# ---------------------------------------------------------------------------
def train_task(task_name: str, df: pd.DataFrame, cfg: Optional[Config] = None,
               registry: Optional[ModelRegistry] = None, fast: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    registry = registry or ModelRegistry(cfg)
    task_cfg = getattr(cfg.models.tasks, task_name)
    seed = cfg.project.random_seed
    log.info("===== Training task '%s' (%s) =====", task_name, task_cfg.type)

    X, y, target = _prepare_xy(df, task_cfg)
    log.info("Samples=%s | features=%s | target='%s'", len(X), X.shape[1], target)
    if task_cfg.type == "classification":
        log.info("Class balance: positive=%.1f%%", 100 * y.mean())

    stratify = y if task_cfg.type == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.models.test_size, random_state=seed, stratify=stratify)

    preprocessor, _, _ = build_preprocessor(X, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    class_weight = getattr(task_cfg, "class_weight", None) if task_cfg.type == "classification" else None
    scoring = (cfg.models.tuning.scoring_clf if task_cfg.type == "classification"
               else cfg.models.tuning.scoring_reg)
    cv_scores: dict[str, float] = {}

    if fast:
        # --- Fast path: one compact RF, no CV/tuning/selection (cloud-safe) ---
        best_name, (best_est, _) = next(iter(
            _fast_candidates(task_cfg.type, seed, class_weight).items()))
        model = _build_pipeline(best_est, preprocessor, task_cfg.type, cfg, seed,
                                use_selection=False)
        model.fit(Xtr, ytr)
        best_params = {"fast": True}
        log.info("Fast-trained %s (%s).", task_name, best_name)
    else:
        cands = _candidates(task_cfg.type, seed, class_weight)
        # --- 1) CV model selection -------------------------------------
        if task_cfg.type == "classification":
            cv = StratifiedKFold(cfg.models.cv_folds, shuffle=True, random_state=seed)
        else:
            cv = KFold(cfg.models.cv_folds, shuffle=True, random_state=seed)
        for name, (est, _) in cands.items():
            pipe = _build_pipeline(est, preprocessor, task_cfg.type, cfg, seed)
            try:
                score = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1).mean()
            except Exception as exc:
                log.warning("CV failed for %s: %s", name, exc)
                continue
            cv_scores[name] = score
            log.info("  CV %-14s %s=%.4f", name, scoring, score)
        if not cv_scores:
            raise RuntimeError(f"No model could be cross-validated for '{task_name}'.")
        best_name = max(cv_scores, key=cv_scores.get)
        log.info("Best family: %s (%s=%.4f)", best_name, scoring, cv_scores[best_name])

        # --- 2) Hyperparameter tuning on winner ------------------------
        best_est, param_dist = cands[best_name]
        best_pipe = _build_pipeline(best_est, preprocessor, task_cfg.type, cfg, seed)
        if cfg.models.tuning.enabled and param_dist:
            search = RandomizedSearchCV(
                best_pipe, param_dist, n_iter=cfg.models.tuning.n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, random_state=seed, refit=True)
            search.fit(Xtr, ytr)
            model, best_params = search.best_estimator_, search.best_params_
            log.info("Tuned %s: best CV %s=%.4f", best_name, scoring, search.best_score_)
        else:
            best_pipe.fit(Xtr, ytr)
            model, best_params = best_pipe, {}

    # --- 3) Held-out evaluation ----------------------------------------
    if task_cfg.type == "classification":
        ypred = model.predict(Xte)
        yproba = (model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else ypred)
        metrics = _clf_metrics(yte, ypred, yproba)
    else:
        ypred = model.predict(Xte)
        metrics = _reg_metrics(yte, ypred)
    log.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    # --- 4) Feature importances ----------------------------------------
    importances = _extract_importances(model)

    registry.log_run(
        task=task_name, model=model,
        params={"family": best_name, **best_params},
        metrics=metrics,
        feature_names=list(X.columns),
        importances=importances,
        extra={"best_model": best_name, "task_type": task_cfg.type,
               "target": target, "cv_scores": cv_scores,
               "n_train": len(Xtr), "n_test": len(Xte)},
    )
    return {"task": task_name, "best_model": best_name, "metrics": metrics}


def _extract_importances(model: Pipeline) -> dict[str, float]:
    """Map fitted-model importances back to (selected) feature names."""
    try:
        names = model.named_steps["prep"].get_feature_names_out()
        if "select" in model.named_steps:
            names = names[model.named_steps["select"].get_support()]
        est = model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
        elif hasattr(est, "coef_"):
            imp = np.abs(np.ravel(est.coef_))
        else:
            return {}
        return {str(n): float(v) for n, v in zip(names, imp)}
    except Exception as exc:  # pragma: no cover
        log.debug("Importance extraction failed: %s", exc)
        return {}


def train_all(df: Optional[pd.DataFrame] = None, cfg: Optional[Config] = None,
              fast: bool = False) -> dict[str, Any]:
    """Train every task defined in ``config.models.tasks``.

    Set ``fast=True`` for a compact, no-tuning fit (resource-constrained hosts).
    """
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_parquet(get_path("features", cfg=cfg))
    registry = ModelRegistry(cfg)
    results = {}
    for task_name in cfg.models.tasks:
        results[task_name] = train_task(task_name, df, cfg, registry, fast=fast)
    log.info("All tasks trained. Registry index: %s", registry.index_path)
    return results


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    if len(sys.argv) > 1:
        print(train_task(sys.argv[1], feats, cfg))
    else:
        print(train_all(feats, cfg))
