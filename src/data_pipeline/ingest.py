"""
ingest.py
=========
**Stage 1 — Data Loading.**

Responsibilities
----------------
* Read the raw Astram CSV with an explicit, schema-locked dtype strategy
  (everything as string first; numeric/datetime coercion happens later in
  ``clean.py`` so that ingest never silently drops malformed values).
* Treat the literal token ``"NULL"`` and empty strings as missing.
* Emit a deterministic *ingest manifest* (row/col counts, file hash, load
  timestamp) for lineage/auditing.
* Persist a raw snapshot to ``data/interim`` as Parquet for fast re-runs.

Run standalone::

    python -m src.data_pipeline.ingest
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)

# Tokens that represent "missing" in the source export.
_NA_TOKENS = ["NULL", "null", "NaN", "nan", "None", ""]


@dataclass
class IngestManifest:
    """Lineage record describing one ingest run."""

    source_file: str
    sha256: str
    rows: int
    cols: int
    loaded_at_utc: str
    column_names: list[str]

    def log_summary(self) -> None:
        log.info("Ingested %s rows x %s cols from %s", self.rows, self.cols, self.source_file)
        log.debug("SHA256=%s", self.sha256)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a SHA-256 of the source file (cheap data-version fingerprint)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ingest(
    cfg: Optional[Config] = None,
    source: Optional[str | Path] = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, IngestManifest]:
    """Load the raw event CSV.

    Parameters
    ----------
    cfg : loaded :class:`Config` (loaded automatically if ``None``).
    source : override path to the CSV; defaults to ``paths.raw_csv``.
    persist : if ``True`` write a Parquet snapshot to ``data/interim``.

    Returns
    -------
    (DataFrame, IngestManifest)
        DataFrame with **all columns as ``object``/string** plus the raw
        ``id`` retained; a manifest describing the load.
    """
    cfg = cfg or load_config()
    root = Path(cfg.to_dict()["PROJECT_ROOT"])
    csv_path = Path(source) if source else (root / cfg.paths.raw_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    log.info("Loading raw CSV: %s", csv_path.name)
    # dtype=str => no silent numeric coercion; we control casting downstream.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_values=_NA_TOKENS,
        encoding="utf-8",
        on_bad_lines="warn",
    )

    manifest = IngestManifest(
        source_file=csv_path.name,
        sha256=_file_sha256(csv_path),
        rows=len(df),
        cols=df.shape[1],
        loaded_at_utc=datetime.now(timezone.utc).isoformat(),
        column_names=list(df.columns),
    )
    manifest.log_summary()

    if persist:
        out = get_path("data_interim", cfg=cfg) / "events_raw.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out, index=False)
            log.info("Raw snapshot -> %s", out)
        except Exception as exc:  # pyarrow/fastparquet may be missing
            log.warning("Parquet snapshot skipped (%s); continuing in-memory.", exc)

    return df, manifest


if __name__ == "__main__":  # pragma: no cover
    frame, mani = ingest()
    print(pd.Series(asdict(mani)).to_string())
    print(frame.head(3).T)
