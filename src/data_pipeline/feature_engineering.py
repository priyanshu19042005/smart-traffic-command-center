"""
feature_engineering.py
======================
**Stages 7-9 — Feature Engineering, Encoding, Scaling.**

Produces the modelling table and the reusable preprocessing transformer.

What it builds
--------------
* **Temporal**  : hour, day-of-week, month, is_weekend, rush-hour flag,
  day-segment, and *cyclical* sin/cos encodings (so 23:00 is near 00:00).
* **Geo**       : distance to city centre, coarse grid cell id (spatial bucket).
* **Severity**  : per-row ``cause_severity`` and ``veh_mass`` weights.
* **risk_score**: the **engineered regression target** (0-100) — a transparent
  weighted blend of cause severity, closure, priority, resolution time and
  vehicle mass. (No ground-truth severity exists in the source; this is the
  documented proxy label. Swap in human labels in production.)
* ``build_preprocessor`` : a scikit-learn ``ColumnTransformer`` that
  one-hot-encodes categoricals and standard-scales numerics — shared by every
  ML model so train/serve preprocessing is identical.

Run standalone::

    python -m src.data_pipeline.feature_engineering
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)

# Bengaluru city centre (approx. — used for radial distance feature).
_CITY_CENTRE = (12.9716, 77.5946)
_RUSH_HOURS = {8, 9, 10, 18, 19, 20}


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["start_datetime"]
    # Convert to IST (UTC+5:30) so "hour" reflects local traffic behaviour.
    local = ts.dt.tz_convert("Asia/Kolkata")
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek
    df["day_name"] = local.dt.day_name()
    df["month"] = local.dt.month
    df["day"] = local.dt.day
    df["week"] = local.dt.isocalendar().week.astype(int)
    df["date"] = local.dt.date.astype("datetime64[ns]")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(_RUSH_HOURS).astype(int)
    df["day_segment"] = pd.cut(
        df["hour"], bins=[-1, 5, 11, 16, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "night2"],
    ).astype(str).replace({"night2": "night"})
    # Cyclical encodings.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


# ---------------------------------------------------------------------------
# Geo
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _add_geo(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    geo = cfg.schema.geo
    df["dist_centre_km"] = _haversine_km(
        df[geo.lat], df[geo.lon], _CITY_CENTRE[0], _CITY_CENTRE[1]
    )
    # ~500m grid cell id for coarse spatial bucketing / joins.
    df["geo_cell"] = (
        df[geo.lat].round(3).astype(str) + "_" + df[geo.lon].round(3).astype(str)
    )
    return df


# ---------------------------------------------------------------------------
# Severity weights + engineered risk_score target
# ---------------------------------------------------------------------------
def _add_severity_and_risk(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = cfg.features
    sev_map = f.cause_severity.to_dict()
    mass_map = f.veh_mass_weight.to_dict()

    df["cause_severity"] = df["event_cause"].map(sev_map).fillna(0.3)
    df["veh_mass"] = df["veh_type"].map(mass_map).fillna(0.2)

    # Normalised components in [0,1].
    closure = df.get("requires_road_closure", pd.Series(False, index=df.index)).astype(float)
    prio_high = (df.get("priority", "Low").astype(str).str.lower() == "high").astype(float)

    dur = df.get("resolution_hours")
    if dur is None:
        dur_norm = pd.Series(0.3, index=df.index)
    else:
        # Robust scale by 95th percentile so a few long tails don't dominate.
        cap = np.nanpercentile(dur.dropna(), 95) if dur.notna().any() else 1.0
        dur_norm = (dur / cap).clip(0, 1).fillna(dur_norm_default(dur))

    w = f.risk_weights
    risk = (
        w.cause_severity * df["cause_severity"]
        + w.closure * closure
        + w.priority_high * prio_high
        + w.duration * dur_norm
        + w.veh_mass * df["veh_mass"]
    )
    df["risk_score"] = (100 * risk).clip(0, 100).round(2)
    return df


def dur_norm_default(dur: pd.Series) -> float:
    """Median-based fill for missing resolution durations (active events)."""
    med = dur.median()
    return 0.3 if pd.isna(med) else float(np.clip(med, 0, 1))


# ---------------------------------------------------------------------------
# Rolling spatial context (hotspot density proxy)
# ---------------------------------------------------------------------------
def _add_context(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Corridor-level event load (how busy is this corridor overall).
    if "corridor" in df.columns:
        load = df.groupby("corridor")["id"].transform("count")
        df["corridor_event_load"] = load
        df["corridor_load_norm"] = (load / load.max()).round(4)
    # Same geo-cell repeat count -> micro-hotspot signal.
    df["cell_repeat_count"] = df.groupby("geo_cell")["id"].transform("count")
    return df


# ---------------------------------------------------------------------------
# Preprocessor builder (used by all ML models)
# ---------------------------------------------------------------------------
def build_preprocessor(
    df: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
):
    """Return an unfitted ``ColumnTransformer`` (OHE + StandardScaler).

    Kept separate from the feature table so the *exact same* transform is
    fit on train and applied at serve time (no train/serve skew).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    cat = [c for c in categorical if c in df.columns]
    num = [c for c in numeric if c in df.columns]

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("cat", cat_pipe, cat), ("num", num_pipe, num)],
        remainder="drop", verbose_feature_names_out=True,
    ), cat, num


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, cfg: Optional[Config] = None,
                      persist: bool = True) -> pd.DataFrame:
    """Add all engineered features (and the risk_score target) to a clean frame."""
    cfg = cfg or load_config()
    log.info("Engineering features for %s rows ...", len(df))

    df = (
        df.pipe(_add_temporal)
          .pipe(_add_geo, cfg)
          .pipe(_add_severity_and_risk, cfg)
          .pipe(_add_context, cfg)
    )

    log.info("Feature frame: %s rows x %s cols (risk_score mean=%.1f).",
             len(df), df.shape[1], df["risk_score"].mean())
    if persist:
        out = get_path("features", cfg=cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out, index=False)
            log.info("Features -> %s", out)
        except Exception as exc:
            csv_out = out.with_suffix(".csv")
            df.to_csv(csv_out, index=False)
            log.warning("Parquet failed (%s); wrote CSV -> %s", exc, csv_out)
    return df


# Default modelling column groups (importable by models/train.py).
NUMERIC_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "dist_centre_km", "cause_severity", "veh_mass",
    "corridor_load_norm", "cell_repeat_count", "is_segment_event",
]
CATEGORICAL_FEATURES = [
    "event_type", "event_cause", "status", "veh_type",
    "corridor", "zone", "gba_identifier", "day_segment",
]


if __name__ == "__main__":  # pragma: no cover
    from src.data_pipeline.ingest import ingest
    from src.data_pipeline.clean import clean
    raw, _ = ingest(persist=False)
    feats = engineer_features(clean(raw, persist=False))
    print(feats[["event_cause", "risk_score", "hour", "is_rush_hour", "zone"]].head(10))
