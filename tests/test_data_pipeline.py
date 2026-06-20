"""Tests for the data pipeline (ingest, validate, clean, feature engineering)."""
from __future__ import annotations

import pandas as pd

from src.data_pipeline.ingest import ingest
from src.data_pipeline.validate import validate, Severity
from src.data_pipeline.clean import clean
from src.data_pipeline.feature_engineering import engineer_features


def test_ingest_loads_rows(cfg):
    df, manifest = ingest(cfg, persist=False)
    assert len(df) > 1000
    assert manifest.rows == len(df)
    assert "id" in df.columns


def test_validate_returns_report(cfg):
    df, _ = ingest(cfg, persist=False)
    rep = validate(df, cfg, save=False)
    assert len(rep.checks) >= 5
    # every check has a valid severity
    assert all(c.severity in Severity for c in rep.checks)


def test_clean_removes_duplicates_and_types(cfg):
    raw, _ = ingest(cfg, persist=False)
    clean_df = clean(raw, cfg, persist=False)
    # no exact duplicate primary keys
    assert clean_df["id"].is_unique
    # datetime coerced
    assert pd.api.types.is_datetime64_any_dtype(clean_df["start_datetime"])
    # dead columns dropped
    assert "map_file" not in clean_df.columns


def test_spatial_imputation_fills_zone(cfg):
    raw, _ = ingest(cfg, persist=False)
    clean_df = clean(raw, cfg, persist=False)
    # zone starts ~58% missing; after spatial impute should be near-complete
    assert clean_df["zone"].isna().mean() < 0.05


def test_feature_engineering_creates_risk_target(features):
    assert "risk_score" in features.columns
    assert features["risk_score"].between(0, 100).all()
    for col in ["hour_sin", "hour_cos", "is_rush_hour", "cause_severity"]:
        assert col in features.columns
