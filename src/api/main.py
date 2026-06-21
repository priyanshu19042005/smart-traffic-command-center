"""
main.py — Smart Traffic Command Center REST API (FastAPI)
=========================================================
Serves the platform's analytics artifacts and ML models over HTTP/JSON.

Endpoints (prefix ``/api/v1``)
------------------------------
* ``GET  /health``            service + model + artifact status
* ``GET  /stats``             headline KPIs
* ``GET  /road-health``       RHI table (``?level=zone|corridor``)
* ``GET  /hotspots``          hotspot table (``?cause=all|accident|...``)
* ``GET  /hotspots/geojson``  GIS-ready FeatureCollection
* ``GET  /forecast``          forecast series (``?scope=CITY``)
* ``GET  /resources``         resource-allocation plan
* ``GET  /quality``           data-quality validation report
* ``POST /predict``           score an incident (priority / closure / risk)

Interactive docs: ``/docs`` (Swagger UI) and ``/redoc``.

Run::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src import __version__
from src.api.schemas import (HealthResponse, IncidentRequest, PredictionResponse,
                             TaskPrediction)
from src.utils import get_path, load_config
from src.utils.logger import get_logger

log = get_logger("api")
CFG = load_config()
OUT = get_path("outputs_dir", cfg=CFG)
_RUSH = {8, 9, 10, 18, 19, 20}

# In-process state populated on startup.
STATE: dict = {"predictor": None, "events": None}


# --------------------------------------------------------------------------
# Lifespan: load models + a small events sample once.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.models.predict import Predictor
        STATE["predictor"] = Predictor(CFG)
        log.info("Loaded models: %s", list(STATE["predictor"].models))
    except Exception as exc:  # models not trained yet -> degraded mode
        log.warning("Models unavailable: %s", exc)
    feats = get_path("features", cfg=CFG)
    if feats.exists():
        STATE["events"] = pd.read_parquet(feats)
    yield
    STATE.clear()


app = FastAPI(
    title="Smart Traffic Command Center API",
    description="Incident intelligence, road health, hotspots, forecasting, "
                "resource allocation and ML predictions for the Astram dataset.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _csv(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        raise HTTPException(404, f"Artifact '{name}' not found. Run the pipeline first.")
    return pd.read_csv(p)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
def health():
    pred = STATE.get("predictor")
    artifacts = {
        "features": get_path("features", cfg=CFG).exists(),
        "road_health": (OUT / "road_health_zone.csv").exists(),
        "hotspots": (OUT / "hotspots_all.csv").exists(),
        "forecasts": (OUT / "forecasts.csv").exists(),
        "resources": (OUT / "resource_allocation.csv").exists(),
    }
    return HealthResponse(
        status="ok" if pred and pred.models else "degraded",
        version=__version__,
        models_loaded=list(pred.models) if pred else [],
        artifacts_available=artifacts,
    )


@app.get("/api/v1/stats", tags=["meta"])
def stats():
    df = STATE.get("events")
    if df is None:
        raise HTTPException(404, "Features not available. Run the pipeline.")
    return {
        "total_incidents": int(len(df)),
        "active": int((df["status"] == "active").sum()),
        "road_closures": int(df.get("requires_road_closure", pd.Series(dtype=bool)).sum()),
        "mean_risk_score": round(float(df["risk_score"].mean()), 2),
        "date_min": str(df["date"].min())[:10],
        "date_max": str(df["date"].max())[:10],
        "top_causes": df["event_cause"].value_counts().head(5).to_dict(),
    }


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
@app.get("/api/v1/road-health", tags=["analytics"])
def road_health(level: str = Query("zone", pattern="^(zone|corridor)$")):
    return _records(_csv(f"road_health_{level}.csv"))


@app.get("/api/v1/hotspots", tags=["analytics"])
def hotspots(cause: str = Query("all"), limit: int = Query(50, ge=1, le=500)):
    return _records(_csv(f"hotspots_{cause}.csv").head(limit))


@app.get("/api/v1/hotspots/geojson", tags=["analytics"])
def hotspots_geojson(cause: str = Query("all")):
    p = OUT / f"hotspots_{cause}.geojson"
    if not p.exists():
        raise HTTPException(404, f"GeoJSON for '{cause}' not found.")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/v1/forecast", tags=["analytics"])
def forecast(scope: str = Query("CITY")):
    fc = _csv("forecasts.csv")
    sub = fc[fc["scope"] == scope]
    if sub.empty:
        raise HTTPException(404, f"No forecast for scope '{scope}'. "
                                 f"Available: {sorted(fc['scope'].unique())}")
    summary = _csv("forecast_summary.csv")
    srow = summary[summary["scope"] == scope]
    return {
        "scope": scope,
        "horizons": (srow.iloc[0][["day", "week", "month"]].to_dict()
                     if not srow.empty else {}),
        "best_model": srow.iloc[0]["best_model"] if not srow.empty else None,
        "forecast": _records(sub[["date", "yhat", "yhat_lower", "yhat_upper"]]),
    }


@app.get("/api/v1/resources", tags=["analytics"])
def resources():
    return _records(_csv("resource_allocation.csv"))


@app.get("/api/v1/quality", tags=["analytics"])
def quality():
    p = OUT / "validation_report.json"
    if not p.exists():
        raise HTTPException(404, "Validation report not found.")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _enrich(req: IncidentRequest) -> pd.DataFrame:
    """Build a single-row feature frame from the request (train/serve parity)."""
    sev = CFG.features.cause_severity.to_dict()
    mass = CFG.features.veh_mass_weight.to_dict()
    row = {
        "event_cause": req.event_cause, "corridor": req.corridor, "zone": req.zone,
        "veh_type": req.veh_type, "event_type": req.event_type, "status": req.status,
        "gba_identifier": "unknown", "authenticated": "yes", "day_segment": "morning",
        "latitude": req.latitude, "longitude": req.longitude,
        "hour": req.hour, "dayofweek": 2, "month": 1, "day": 15, "week": 3,
        "is_weekend": 0, "is_rush_hour": int(req.hour in _RUSH),
        "hour_sin": np.sin(2 * np.pi * req.hour / 24),
        "hour_cos": np.cos(2 * np.pi * req.hour / 24),
        "dow_sin": np.sin(2 * np.pi * 2 / 7), "dow_cos": np.cos(2 * np.pi * 2 / 7),
        "dist_centre_km": 5.0,
        "cause_severity": sev.get(req.event_cause, 0.3),
        "veh_mass": mass.get(req.veh_type, 0.2),
        "corridor_load_norm": 0.3, "cell_repeat_count": 1, "is_segment_event": 0,
    }
    return pd.DataFrame([row])


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["ml"])
def predict(req: IncidentRequest):
    pred = STATE.get("predictor")
    if not pred or not pred.models:
        raise HTTPException(503, "Models not loaded. Train with `python -m src.models.train`.")
    scored = pred.predict_frame(_enrich(req)).iloc[0]
    out = PredictionResponse(model_versions={
        t: pred.meta.get(t, {}).get("best_model") for t in pred.models})
    if "priority_label" in scored:
        out.priority = TaskPrediction(label=scored["priority_label"],
                                      probability=float(scored["priority_proba"]))
    if "closure_label" in scored:
        out.closure = TaskPrediction(
            label="required" if scored["closure_pred"] == 1 else "not_required",
            probability=float(scored["closure_proba"]))
    if "resolution_pred" in scored:
        out.resolution = TaskPrediction(value=round(float(scored["resolution_pred"]), 1),
                                        unit="hours")
    return out
