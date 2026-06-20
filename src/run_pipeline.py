"""
run_pipeline.py
===============
**End-to-end orchestrator.** Runs the entire platform in dependency order and
writes every artifact the dashboard consumes.

Stages
------
1. ingest  -> validate -> clean -> feature engineering
2. road health index (corridor + zone)
3. hotspot detection (+ GeoJSON)
4. resource allocation
5. ML training (priority, closure, risk) + evaluation
6. forecasting (city + top corridors)

Each stage is independently toggleable via CLI flags, so you can refresh just
one part (e.g. ``--only forecast``) without recomputing everything.

Usage
-----
    python -m src.run_pipeline                 # full run
    python -m src.run_pipeline --skip forecast train
    python -m src.run_pipeline --only data rhi hotspots
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)

ALL_STAGES = ["data", "rhi", "hotspots", "resources", "train", "evaluate", "forecast"]


def _banner(title: str) -> None:
    log.info("=" * 70)
    log.info("STAGE: %s", title.upper())
    log.info("=" * 70)


def run(stages: list[str], cfg: Optional[Config] = None) -> dict:
    cfg = cfg or load_config()
    artifacts: dict = {}
    t0 = time.time()

    # ---- 1. DATA -------------------------------------------------------
    if "data" in stages:
        _banner("data pipeline")
        from src.data_pipeline.ingest import ingest
        from src.data_pipeline.validate import validate
        from src.data_pipeline.clean import clean
        from src.data_pipeline.feature_engineering import engineer_features
        raw, manifest = ingest(cfg)
        report = validate(raw, cfg)
        if not report.passed and cfg.validation.fail_on_error:
            raise SystemExit("Validation failed; aborting.")
        feats = engineer_features(clean(raw, cfg), cfg)
        artifacts["features"] = feats
    else:
        feats = pd.read_parquet(get_path("features", cfg=cfg))
        artifacts["features"] = feats

    # ---- 2. ROAD HEALTH ------------------------------------------------
    rhi = None
    if "rhi" in stages:
        _banner("road health index")
        from src.engines.road_health_engine import compute_road_health
        rhi = compute_road_health(feats, cfg)
        artifacts["road_health"] = rhi

    # ---- 3. HOTSPOTS ---------------------------------------------------
    hotspots = None
    if "hotspots" in stages:
        _banner("hotspot detection")
        from src.engines.hotspot_engine import detect_hotspots
        hotspots = detect_hotspots(feats, cfg)
        artifacts["hotspots"] = hotspots

    # ---- 4. RESOURCES --------------------------------------------------
    if "resources" in stages:
        _banner("resource allocation")
        from src.engines.resource_allocator import allocate_resources
        if rhi is None:
            from src.engines.road_health_engine import compute_road_health
            rhi = compute_road_health(feats, cfg, persist=False)
        hs_all = (hotspots or {}).get("all")
        artifacts["allocation"] = allocate_resources(
            feats, cfg, rhi_zone=rhi["zone"], hotspots=hs_all)

    # ---- 5. ML TRAIN + EVAL -------------------------------------------
    if "train" in stages:
        _banner("model training")
        from src.models.train import train_all
        artifacts["training"] = train_all(feats, cfg)
    if "evaluate" in stages:
        _banner("model evaluation")
        from src.models.evaluate import evaluate_all
        artifacts["evaluation"] = evaluate_all(feats, cfg)

    # ---- 6. FORECAST ---------------------------------------------------
    if "forecast" in stages:
        _banner("forecasting")
        from src.engines.forecast_engine import run_forecasts
        artifacts["forecasts"] = run_forecasts(feats, cfg)

    log.info("Pipeline complete in %.1fs. Stages: %s", time.time() - t0, stages)
    log.info("Artifacts in: %s", get_path("outputs_dir", cfg=cfg))
    return artifacts


def _parse() -> list[str]:
    p = argparse.ArgumentParser(description="Smart Traffic Command Center pipeline")
    p.add_argument("--only", nargs="+", choices=ALL_STAGES, help="run only these stages")
    p.add_argument("--skip", nargs="+", choices=ALL_STAGES, help="skip these stages")
    a = p.parse_args()
    if a.only:
        return a.only
    return [s for s in ALL_STAGES if not a.skip or s not in a.skip]


if __name__ == "__main__":  # pragma: no cover
    run(_parse())
