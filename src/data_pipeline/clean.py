"""
clean.py
========
**Stages 3-6 — Cleaning, Missing-value handling, Dedup, Outlier handling.**

Transforms a raw ingested frame into an analysis-ready, *typed* frame.

Pipeline of operations (each is a small, testable function):
1. ``_drop_dead_columns``    — remove 100%-empty / non-informative columns.
2. ``_coerce_types``         — numeric & datetime casting (errors -> NaT/NaN).
3. ``_fix_geo``              — clamp/NULL coordinates outside the city bbox;
                               flag point vs segment events.
4. ``_fix_temporal``         — null-out impossible end times (end < start or
                               far-future), derive resolution duration.
5. ``_normalise_categoricals`` — canonicalise messy ``event_cause`` casing,
                               standardise booleans & strings.
6. ``_impute_missing``       — domain fills + KNN **spatial imputation** of
                               ``zone``/``gba_identifier``/``corridor``.
7. ``_dedupe``               — drop near-duplicate incident reports.

Run standalone::

    python -m src.data_pipeline.clean
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)

_EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# 1. Drop dead columns
# ---------------------------------------------------------------------------
def _drop_dead_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    explicit = [c for c in cfg.schema.drop_cols if c in df.columns]
    # Also drop anything that is 100% empty even if not enumerated.
    empties = [c for c in df.columns if df[c].isna().all()]
    to_drop = sorted(set(explicit) | set(empties))
    log.info("Dropping %s dead/uninformative columns.", len(to_drop))
    return df.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# 2. Type coercion
# ---------------------------------------------------------------------------
def _coerce_types(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    geo = cfg.schema.geo
    for col in [geo.lat, geo.lon, geo.end_lat, geo.end_lon]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cfg.schema.datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------
# 3. Geo fix
# ---------------------------------------------------------------------------
def _fix_geo(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    geo, bbox = cfg.schema.geo, cfg.schema.bbox

    def _clean_pair(latc: str, lonc: str, drop_outside: bool) -> None:
        if latc not in df.columns or lonc not in df.columns:
            return
        # Treat exact zeros as missing (placeholder coordinates).
        df.loc[df[latc] == 0, latc] = np.nan
        df.loc[df[lonc] == 0, lonc] = np.nan
        outside = (
            (df[latc] < bbox.lat_min) | (df[latc] > bbox.lat_max) |
            (df[lonc] < bbox.lon_min) | (df[lonc] > bbox.lon_max)
        )
        if drop_outside and outside.any():
            df.loc[outside, [latc, lonc]] = np.nan

    # Primary coords: must be valid (validation already guards these).
    _clean_pair(geo.lat, geo.lon, drop_outside=True)
    # End coords: optional; null-out garbage values.
    _clean_pair(geo.end_lat, geo.end_lon, drop_outside=True)

    # Segment flag: does the event have a valid distinct end point?
    if geo.end_lat in df.columns:
        df["is_segment_event"] = df[geo.end_lat].notna() & df[geo.end_lon].notna()
    return df


# ---------------------------------------------------------------------------
# 4. Temporal fix
# ---------------------------------------------------------------------------
def _fix_temporal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    start, end, closed = "start_datetime", "end_datetime", "closed_datetime"

    # Drop rows with no usable start time (cannot be time-indexed).
    if start in df.columns:
        before = len(df)
        df = df[df[start].notna()].copy()
        if before - len(df):
            log.info("Dropped %s rows with unparseable start_datetime.", before - len(df))

    # Null-out impossible end times.
    if end in df.columns and start in df.columns:
        bad = (df[end] < df[start]) | (df[end] > now + pd.Timedelta(days=2))
        df.loc[bad.fillna(False), end] = pd.NaT

    # Resolution duration (hours) — prefer closed, else end.
    finish = None
    if closed in df.columns:
        finish = df[closed]
    if end in df.columns:
        finish = df[end] if finish is None else finish.fillna(df[end])
    if finish is not None and start in df.columns:
        dur = (finish - df[start]).dt.total_seconds() / 3600.0
        df["resolution_hours"] = dur.where(dur.between(0, 24 * 30))  # cap at 30 days
    return df


# ---------------------------------------------------------------------------
# 5. Categorical normalisation
# ---------------------------------------------------------------------------
def _normalise_categoricals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # event_cause canonical map + lowercase fallback.
    if "event_cause" in df.columns:
        cmap = cfg.cleaning.event_cause_map.to_dict()
        df["event_cause"] = (
            df["event_cause"].map(lambda x: cmap.get(x, x) if pd.notna(x) else x)
            .str.strip().str.lower()
        )

    # Boolean-ish columns -> real booleans.
    for col in ["requires_road_closure"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.upper()
                .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
            )

    # Trim/standardise key string fields.
    for col in ["priority", "status", "event_type", "corridor", "zone",
                "gba_identifier", "veh_type", "police_station"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


# ---------------------------------------------------------------------------
# 6. Missing value handling + spatial imputation
# ---------------------------------------------------------------------------
def _knn_spatial_fill(df: pd.DataFrame, field: str, cfg: Config, k: int) -> int:
    """Fill missing *field* from the nearest labelled event (haversine BallTree).

    Returns the number of values imputed.
    """
    geo = cfg.schema.geo
    have = df[field].notna() & df[geo.lat].notna() & df[geo.lon].notna()
    need = df[field].isna() & df[geo.lat].notna() & df[geo.lon].notna()
    if need.sum() == 0 or have.sum() == 0:
        return 0
    try:
        from sklearn.neighbors import BallTree
    except Exception:  # pragma: no cover
        log.warning("sklearn BallTree unavailable; skipping spatial impute for %s.", field)
        return 0

    train_xy = np.radians(df.loc[have, [geo.lat, geo.lon]].to_numpy())
    query_xy = np.radians(df.loc[need, [geo.lat, geo.lon]].to_numpy())
    tree = BallTree(train_xy, metric="haversine")
    _, idx = tree.query(query_xy, k=min(k, have.sum()))
    nearest = df.loc[have, field].to_numpy()[idx[:, 0]]
    df.loc[need, field] = nearest
    return int(need.sum())


def _impute_missing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    clean = cfg.cleaning

    # Domain fills.
    if "priority" in df.columns:
        df["priority"] = df["priority"].fillna(clean.priority_fill)
    if "veh_type" in df.columns:
        df["veh_type"] = df["veh_type"].fillna(clean.veh_type_fill)

    # Spatial imputation for geo-group fields.
    if clean.spatial_impute.enabled:
        k = clean.spatial_impute.k_neighbors
        for field in clean.spatial_impute.fields:
            if field in df.columns:
                n = _knn_spatial_fill(df, field, cfg, k)
                if n:
                    log.info("Spatially imputed %s '%s' values.", n, field)

    # Residual categorical NaNs -> explicit 'unknown'.
    for col in cfg.schema.categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


# ---------------------------------------------------------------------------
# 7. Dedup
# ---------------------------------------------------------------------------
def _dedupe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    subset = [c for c in cfg.cleaning.dedup_subset if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        log.info("Removed %s near-duplicate incident rows.", removed)
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, cfg: Optional[Config] = None, persist: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the typed, deduped frame."""
    cfg = cfg or load_config()
    log.info("Cleaning %s raw rows ...", len(df))

    df = (
        df.pipe(_drop_dead_columns, cfg)
          .pipe(_coerce_types, cfg)
          .pipe(_fix_geo, cfg)
          .pipe(_fix_temporal, cfg)
          .pipe(_normalise_categoricals, cfg)
          .pipe(_impute_missing, cfg)
          .pipe(_dedupe, cfg)
    )

    log.info("Clean frame: %s rows x %s cols.", len(df), df.shape[1])
    if persist:
        out = get_path("processed_events", cfg=cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out, index=False)
            log.info("Clean events -> %s", out)
        except Exception as exc:
            csv_out = out.with_suffix(".csv")
            df.to_csv(csv_out, index=False)
            log.warning("Parquet failed (%s); wrote CSV -> %s", exc, csv_out)
    return df


if __name__ == "__main__":  # pragma: no cover
    from src.data_pipeline.ingest import ingest
    raw, _ = ingest(persist=False)
    out_df = clean(raw)
    print(out_df.dtypes)
    print(out_df.head(3).T)
