"""
validate.py
===========
**Stage 2 — Data Validation (quality gates).**

Runs a battery of declarative checks driven by ``config.validation`` and
``config.schema`` and returns a structured :class:`ValidationReport`.

The report is consumed by:
* ``run_pipeline.py`` — to decide whether to abort (``fail_on_error``).
* the **Data Quality Monitoring** dashboard page.

Checks implemented
------------------
1. Schema   — required columns present.
2. Volume   — at least ``min_rows`` rows.
3. Uniqueness — primary key unique.
4. Missingness — critical columns under per-column thresholds.
5. Geo      — coordinates inside the city bounding box.
6. Domain   — categorical values within expected sets (warn-only).
7. Temporal — start_datetime parseable & not in the future.

Run standalone::

    python -m src.data_pipeline.validate
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


class Severity(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class Check:
    name: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- helpers ---------------------------------------------------------
    def add(self, name: str, severity: Severity, message: str, **details: Any) -> None:
        self.checks.append(Check(name, severity, message, details))
        getattr(log, "error" if severity is Severity.ERROR else
                "warning" if severity is Severity.WARN else "info")("[%s] %s", name, message)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.severity is Severity.ERROR]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "passed": self.passed,
            "n_errors": len(self.errors),
            "checks": [
                {"name": c.name, "severity": c.severity.value,
                 "message": c.message, "details": c.details}
                for c in self.checks
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        log.info("Validation report -> %s", path)


def validate(df: pd.DataFrame, cfg: Optional[Config] = None, save: bool = True) -> ValidationReport:
    """Validate a freshly ingested frame and return a structured report."""
    cfg = cfg or load_config()
    v = cfg.validation
    s = cfg.schema
    rep = ValidationReport()

    # 1) Schema -----------------------------------------------------------
    missing_cols = [c for c in v.required_columns if c not in df.columns]
    if missing_cols:
        rep.add("schema.required_columns", Severity.ERROR,
                f"Missing required columns: {missing_cols}", missing=missing_cols)
    else:
        rep.add("schema.required_columns", Severity.OK, "All required columns present.")

    # 2) Volume -----------------------------------------------------------
    if len(df) < v.min_rows:
        rep.add("volume.min_rows", Severity.ERROR,
                f"Only {len(df)} rows (< {v.min_rows}).", rows=len(df))
    else:
        rep.add("volume.min_rows", Severity.OK, f"{len(df)} rows.")

    # 3) Uniqueness -------------------------------------------------------
    key = v.unique_key
    if key in df.columns:
        dups = int(df[key].duplicated().sum())
        sev = Severity.ERROR if dups else Severity.OK
        rep.add("uniqueness.primary_key", sev,
                f"{dups} duplicate '{key}' values.", duplicates=dups)

    # 4) Missingness ------------------------------------------------------
    for col, max_pct in v.max_missing_pct.to_dict().items():
        if col not in df.columns:
            continue
        pct = float(df[col].isna().mean() * 100)
        sev = Severity.ERROR if pct > max_pct else Severity.OK
        rep.add(f"missing.{col}", sev,
                f"{col}: {pct:.2f}% missing (max {max_pct}%).", pct=round(pct, 2))

    # 5) Geo bounding box -------------------------------------------------
    bbox = s.bbox
    lat, lon = s.geo.lat, s.geo.lon
    if lat in df.columns and lon in df.columns:
        latf = pd.to_numeric(df[lat], errors="coerce")
        lonf = pd.to_numeric(df[lon], errors="coerce")
        out = (
            (latf < bbox.lat_min) | (latf > bbox.lat_max) |
            (lonf < bbox.lon_min) | (lonf > bbox.lon_max) | latf.isna() | lonf.isna()
        ).sum()
        pct = out / len(df) * 100
        sev = Severity.WARN if 0 < pct < 5 else Severity.ERROR if pct >= 5 else Severity.OK
        rep.add("geo.in_bbox", sev,
                f"{int(out)} ({pct:.2f}%) primary coords outside city bbox.", out_of_box=int(out))

    # 6) Domain (warn-only) ----------------------------------------------
    for col in ["priority", "status", "event_type"]:
        if col in df.columns:
            card = int(df[col].nunique(dropna=True))
            rep.add(f"domain.{col}.cardinality", Severity.OK,
                    f"{col} has {card} distinct values.", cardinality=card,
                    top=df[col].value_counts().head(5).to_dict())

    # 7) Temporal ---------------------------------------------------------
    if "start_datetime" in df.columns:
        ts = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
        unparsed = int(ts.isna().sum())
        future = int((ts > pd.Timestamp.now(tz="UTC")).sum())
        sev = Severity.WARN if (unparsed or future) else Severity.OK
        rep.add("temporal.start_datetime", sev,
                f"{unparsed} unparseable, {future} future-dated start times.",
                unparsed=unparsed, future=future)

    # ---- summary --------------------------------------------------------
    log.info("Validation complete: %s checks, %s error(s).", len(rep.checks), len(rep.errors))
    if save:
        rep.save(get_path("outputs_dir", cfg=cfg) / "validation_report.json")

    if not rep.passed and v.fail_on_error:
        raise ValueError(f"Validation failed with {len(rep.errors)} error(s); see report.")
    return rep


if __name__ == "__main__":  # pragma: no cover
    from src.data_pipeline.ingest import ingest
    frame, _ = ingest(persist=False)
    report = validate(frame)
    print(json.dumps(report.to_dict(), indent=2)[:2000])
