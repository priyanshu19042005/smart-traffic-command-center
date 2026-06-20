"""
Data pipeline package.

Stage order::

    ingest -> validate -> clean -> feature_engineering

Each stage is importable and runnable standalone, and is chained by
``src/run_pipeline.py``.
"""
from .ingest import ingest
from .validate import validate
from .clean import clean
from .feature_engineering import engineer_features

__all__ = ["ingest", "validate", "clean", "engineer_features"]
