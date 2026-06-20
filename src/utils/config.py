"""
config.py
=========
Loads ``config/config.yaml`` once and exposes it as a dotted-access object.

Why a wrapper instead of a raw dict?
* ``cfg.paths.models_dir`` reads better than ``cfg["paths"]["models_dir"]``.
* Centralised path resolution: every relative path in the YAML is resolved
  against the *project root* (the directory that contains ``config/``),
  so the code works no matter what the current working directory is.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml


class Config:
    """Recursive, attribute-accessible view over a nested dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"No config key '{item}'") from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, item: str) -> Any:
        value = self._data[item]
        return Config(value) if isinstance(value, dict) else value

    def get(self, item: str, default: Any = None) -> Any:
        value = self._data.get(item, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({list(self._data.keys())})"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from *start* until a folder containing ``config/`` is found."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Fallback: two levels up from this file (src/utils/ -> root)
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Adds ``PROJECT_ROOT`` to the returned config for convenience.
    """
    root = find_project_root()
    cfg_path = Path(path) if path else root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["PROJECT_ROOT"] = str(root)
    return Config(data)


def get_path(*keys: str, cfg: Config | None = None) -> Path:
    """Resolve a path from ``cfg.paths`` to an absolute :class:`Path`.

    Example
    -------
    >>> get_path("models_dir")            # -> <root>/models
    >>> get_path("processed_events")      # -> <root>/data/processed/events_clean.parquet
    """
    cfg = cfg or load_config()
    root = Path(cfg.to_dict()["PROJECT_ROOT"])
    node: Any = cfg.paths
    for key in keys:
        node = getattr(node, key)
    return (root / str(node)).resolve()
