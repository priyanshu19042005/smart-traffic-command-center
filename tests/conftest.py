"""Shared pytest fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils import load_config, get_path


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def features(cfg) -> pd.DataFrame:
    """Engineered feature frame; built on demand if not already persisted."""
    p = get_path("features", cfg=cfg)
    if p.exists():
        return pd.read_parquet(p)
    from src.data_pipeline.ingest import ingest
    from src.data_pipeline.clean import clean
    from src.data_pipeline.feature_engineering import engineer_features
    raw, _ = ingest(cfg, persist=False)
    return engineer_features(clean(raw, cfg, persist=False), cfg, persist=False)
