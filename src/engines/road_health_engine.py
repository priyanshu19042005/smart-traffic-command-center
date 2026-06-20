r"""
road_health_engine.py
=====================
**Smart Road Health Index (RHI)** — a 0-100 score per corridor and per zone
(100 = excellent, 0 = critical).

----------------------------------------------------------------------------
1. MATHEMATICAL FORMULA
----------------------------------------------------------------------------
For a road segment ``s`` (a corridor or a zone) and each *degradation cause*
``c`` (pot_holes, water_logging, road_conditions, construction,
vehicle_breakdown, accident):

    count[s,c]                         = number of events of cause c on s
    N[s]                               = total events on s
    share[s,c]  = count[s,c] / (N[s] + k)        # Laplace-smoothed prevalence
    vol[s,c]    = count[s,c]                      # absolute burden

Each cause column is min-max normalised **across all segments** so causes with
different scales are comparable:

    share_n[s,c] = minmax_s( share[s,c] )         in [0,1]
    vol_n[s,c]   = minmax_s( vol[s,c]   )         in [0,1]

A cause's contribution blends *prevalence* (how dominant the problem is on this
road) and *burden* (how much of it there is in absolute terms):

    comp[s,c]    = 0.5 * share_n[s,c] + 0.5 * vol_n[s,c]

The **degradation index** is the weighted sum over causes (weights w_c sum to 1):

    D[s] = Σ_c  w_c * comp[s,c]            ∈ [0,1]

Finally:

    RHI[s] = 100 * (1 - D[s])              ∈ [0,100]

----------------------------------------------------------------------------
2. WEIGHT ASSIGNMENT METHODOLOGY
----------------------------------------------------------------------------
Default weights live in ``config.road_health.degradation_weights`` and encode
domain priorities (potholes & water-logging hurt road health most). Two modes:

* ``weight_mode="config"`` (default) — expert/AHP weights from config.
* ``weight_mode="data"``  — data-driven: weight each cause by how strongly its
  presence co-occurs with accidents (point-biserial style), then renormalise.
  This lets the index *learn* which defects are most dangerous from history.

----------------------------------------------------------------------------
3. FEATURE NORMALISATION
----------------------------------------------------------------------------
Min-max across segments (above) + Laplace smoothing ``k`` to stabilise rates on
low-volume segments (avoids a corridor with 1 event scoring extreme).

----------------------------------------------------------------------------
4. EXPLAINABILITY LOGIC
----------------------------------------------------------------------------
Each segment row carries ``points_lost_<cause> = w_c * comp[s,c] * 100`` so the
dashboard can say *"Mysore Road lost 18 pts to potholes, 9 to water-logging"*,
plus ``top_factor`` = the single biggest contributor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


@dataclass
class RoadHealthEngine:
    """Compute the Road Health Index for a chosen grouping level."""

    cfg: Config
    weight_mode: str = "config"   # "config" | "data"

    # -- weights ---------------------------------------------------------
    def _weights(self, df: pd.DataFrame, causes: list[str]) -> dict[str, float]:
        base = {c: float(w) for c, w in
                self.cfg.road_health.degradation_weights.to_dict().items() if c in causes}
        if self.weight_mode == "config":
            total = sum(base.values()) or 1.0
            return {c: w / total for c, w in base.items()}

        # Data-driven: correlate cause-presence with accident-presence per geo cell.
        if "accident" not in df["event_cause"].unique():
            log.warning("No accidents in data; falling back to config weights.")
            return self._weights(df, causes)
        cell = df.groupby("geo_cell")["event_cause"]
        has = lambda c: cell.apply(lambda x: int((x == c).any()))  # noqa: E731
        acc = has("accident")
        corr = {}
        for c in causes:
            if c == "accident":
                corr[c] = base.get(c, 0.1)
                continue
            v = has(c)
            r = np.corrcoef(v, acc)[0, 1]
            corr[c] = max(0.0, 0.0 if np.isnan(r) else r) + 0.05
        total = sum(corr.values()) or 1.0
        weights = {c: corr[c] / total for c in causes}
        log.info("Data-driven RHI weights: %s",
                 {k: round(v, 3) for k, v in weights.items()})
        return weights

    # -- core ------------------------------------------------------------
    def score(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        """Return one row per segment with health_score, category, explainers."""
        if level not in df.columns:
            raise KeyError(f"Grouping column '{level}' not in dataframe.")
        causes = [c for c in self.cfg.road_health.degradation_weights.to_dict()
                  if c in df["event_cause"].unique()]
        k = self.cfg.road_health.exposure_smoothing
        weights = self._weights(df, causes)

        grp = df[df[level].notna() & (df[level].astype(str) != "unknown")]
        pivot = (
            grp.pivot_table(index=level, columns="event_cause",
                            values="id", aggfunc="count", fill_value=0)
        )
        for c in causes:
            if c not in pivot.columns:
                pivot[c] = 0
        n = grp.groupby(level)["id"].count().rename("total_events")

        out = pd.DataFrame(index=pivot.index)
        out["total_events"] = n
        degradation = pd.Series(0.0, index=pivot.index)

        for c in causes:
            share = pivot[c] / (n + k)
            comp = 0.5 * _minmax(share) + 0.5 * _minmax(pivot[c])
            contrib = weights[c] * comp
            degradation = degradation + contrib
            out[f"n_{c}"] = pivot[c]
            out[f"points_lost_{c}"] = (contrib * 100).round(2)

        out["degradation_index"] = degradation.clip(0, 1).round(4)
        out["health_score"] = (100 * (1 - out["degradation_index"])).round(1)
        out["health_category"] = out["health_score"].apply(self._category)
        # Top contributing defect for explainability.
        pts_cols = [f"points_lost_{c}" for c in causes]
        out["top_factor"] = out[pts_cols].idxmax(axis=1).str.replace("points_lost_", "")
        out = out.reset_index().rename(columns={level: "segment"})
        out.insert(0, "level", level)
        return out.sort_values("health_score").reset_index(drop=True)

    def _category(self, score: float) -> str:
        cats = self.cfg.road_health.categories.to_dict()
        for name in ["Critical", "Poor", "Moderate", "Good", "Excellent"]:
            if score <= cats[name]:
                return name
        return "Excellent"


def compute_road_health(
    df: pd.DataFrame,
    cfg: Optional[Config] = None,
    weight_mode: str = "config",
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute RHI for every configured grouping level (corridor, zone).

    Returns
    -------
    dict[level -> DataFrame] with columns:
        level, segment, health_score, health_category, top_factor, ...
    """
    cfg = cfg or load_config()
    engine = RoadHealthEngine(cfg=cfg, weight_mode=weight_mode)
    results: dict[str, pd.DataFrame] = {}
    for level in cfg.road_health.group_levels:
        res = engine.score(df, level)
        results[level] = res
        log.info("RHI[%s]: %s segments | mean score %.1f | %s critical",
                 level, len(res), res["health_score"].mean(),
                 (res["health_category"] == "Critical").sum())
        if persist:
            out = get_path("outputs_dir", cfg=cfg) / f"road_health_{level}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.to_csv(out, index=False)
            log.info("Road health (%s) -> %s", level, out)
    return results


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    res = compute_road_health(feats, cfg)
    for level, frame in res.items():
        print(f"\n=== {level} ===")
        print(frame[["segment", "health_score", "health_category", "top_factor"]]
              .head(8).to_string(index=False))
