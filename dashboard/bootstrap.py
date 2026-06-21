"""
bootstrap.py
============
Makes the dashboard **self-sufficient on fresh deploys** (e.g. Streamlit
Community Cloud), where the git-ignored artifacts (`data/processed`, `outputs`,
`models`) don't exist — only the code and the raw CSV are present.

On first load it lazily builds whatever is missing, from the committed raw CSV:

* ``ensure_features``  — ingest → clean → feature engineering  (fast, ~5 s)
* ``ensure_analytics`` — road health, hotspots, resource allocation (~15 s)
* ``ensure_forecasts`` — incident forecasts (lazy: only on the Forecasting page)
* ``ensure_models``    — compact, no-tuning model fit (lazy: only on the ML page)

All steps are wrapped in ``@st.cache_resource`` so they run **once per
container** and are resilient: a failure in one (e.g. training OOM on a tiny
instance) is logged and never crashes the rest of the dashboard.
"""
from __future__ import annotations

import streamlit as st

from src.utils import get_path, load_config
from src.utils.logger import get_logger

log = get_logger("bootstrap")


def _cfg():
    return load_config()


@st.cache_resource(show_spinner="🛰️ Preparing data (first load only)…")
def ensure_features() -> bool:
    """Guarantee the engineered feature table exists; build it if not."""
    cfg = _cfg()
    if get_path("features", cfg=cfg).exists():
        return True
    log.info("Features missing — building from raw CSV.")
    from src.data_pipeline.ingest import ingest
    from src.data_pipeline.clean import clean
    from src.data_pipeline.feature_engineering import engineer_features
    raw, _ = ingest(cfg)
    engineer_features(clean(raw, cfg), cfg)        # persists parquet
    return True


@st.cache_resource(show_spinner="📊 Computing analytics (first load only)…")
def ensure_analytics() -> bool:
    """Build road-health, hotspots and resource-allocation artifacts if missing."""
    ensure_features()
    cfg = _cfg()
    out = get_path("outputs_dir", cfg=cfg)
    import pandas as pd
    feats = pd.read_parquet(get_path("features", cfg=cfg))

    try:
        from src.engines.road_health_engine import compute_road_health
        rhi = (compute_road_health(feats, cfg)
               if not (out / "road_health_zone.csv").exists()
               else {"zone": pd.read_csv(out / "road_health_zone.csv")})
    except Exception as exc:                       # pragma: no cover
        log.warning("Road-health bootstrap failed: %s", exc); rhi = None

    try:
        from src.engines.hotspot_engine import detect_hotspots
        hs = (detect_hotspots(feats, cfg)
              if not (out / "hotspots_all.csv").exists() else None)
        hs_all = hs.get("all") if hs else None
    except Exception as exc:                        # pragma: no cover
        log.warning("Hotspot bootstrap failed: %s", exc); hs_all = None

    try:
        if not (out / "resource_allocation.csv").exists():
            from src.engines.resource_allocator import allocate_resources
            rhi_zone = rhi["zone"] if rhi else None
            allocate_resources(feats, cfg, rhi_zone=rhi_zone, hotspots=hs_all)
    except Exception as exc:                        # pragma: no cover
        log.warning("Resource bootstrap failed: %s", exc)
    return True


@st.cache_resource(show_spinner="📈 Generating forecasts (first load only)…")
def ensure_forecasts() -> bool:
    cfg = _cfg()
    if (get_path("outputs_dir", cfg=cfg) / "forecasts.csv").exists():
        return True
    try:
        import pandas as pd
        from src.engines.forecast_engine import run_forecasts
        feats = pd.read_parquet(get_path("features", cfg=cfg))
        run_forecasts(feats, cfg, top_scopes=3)
    except Exception as exc:                        # pragma: no cover
        log.warning("Forecast bootstrap failed: %s", exc)
    return True


@st.cache_resource(show_spinner="🤖 Training models (first load only, compact)…")
def ensure_models() -> bool:
    """Fast-train compact models if none are registered (cloud-safe)."""
    cfg = _cfg()
    try:
        from src.models.registry import ModelRegistry
        if ModelRegistry(cfg).list_runs():
            return True
        import pandas as pd
        from src.models.train import train_all
        from src.models.evaluate import evaluate_all
        ensure_features()
        feats = pd.read_parquet(get_path("features", cfg=cfg))
        train_all(feats, cfg, fast=True)
        evaluate_all(feats, cfg)                    # eval_*.json for the perf tab
    except Exception as exc:                        # pragma: no cover
        log.warning("Model bootstrap skipped: %s", exc)
    return True
