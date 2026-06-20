r"""
resource_allocator.py
====================
**Resource Allocation Recommendation System** — distributes a finite fleet of
Tow Trucks, Traffic Police and Maintenance Teams across zones to minimise
unmet risk.

----------------------------------------------------------------------------
SCORING FORMULA
----------------------------------------------------------------------------
For zone ``z`` and resource ``r`` we build a **demand score** from normalised
signals (each min-max scaled across zones to [0,1]) weighted by
``config.resources.drivers[r]``::

    demand[z,r] = Σ_d  weight[r,d] · signal_n[z,d]

Signals used:
  breakdown   – count of vehicle_breakdown events
  accident    – count of accident events
  congestion  – count of congestion events
  pot_holes / water_logging – count of those causes
  risk        – mean engineered risk_score
  road_health – (100 − RHI) i.e. how unhealthy the road is
  hotspot     – number / risk of detected hotspots in the zone

----------------------------------------------------------------------------
OPTIMISATION LOGIC (apportionment)
----------------------------------------------------------------------------
Given fleet size ``F_r`` for resource ``r`` and demand shares
``p[z] = demand[z,r] / Σ_z demand``, we allocate integer units using the
**Largest-Remainder (Hamilton) method**, after first reserving the configured
``min_per_zone`` floor::

    base[z]      = min_per_zone[r]
    remaining    = F_r − Σ_z base[z]
    quota[z]     = remaining · p[z]
    alloc[z]     = floor(quota[z]) + base[z]
    leftover     = remaining − Σ floor(quota)         # distributed to the
                                                       # largest fractional parts

This guarantees Σ_z alloc[z] = F_r exactly, every zone gets its floor, and
high-demand zones get proportionally more — a transparent, auditable LP-free
allocation. (A strict LP/ILP variant is documented in ``_lp_note``.)

----------------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------------
Per-zone table: zone, tow_trucks, traffic_police, maintenance_teams,
priority_rank, demand scores, and a human-readable ``rationale``.
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
    return pd.Series(0.0, index=s.index) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def _largest_remainder(shares: pd.Series, total: int, floor: int) -> pd.Series:
    """Hamilton apportionment: integer allocation summing exactly to *total*."""
    n = len(shares)
    base = pd.Series(floor, index=shares.index)
    remaining = max(0, total - int(base.sum()))
    if shares.sum() <= 0 or remaining <= 0:
        # Spread any remainder round-robin by share order.
        alloc = base.copy()
        order = shares.sort_values(ascending=False).index
        for i in range(remaining):
            alloc[order[i % n]] += 1
        return alloc
    p = shares / shares.sum()
    quota = p * remaining
    alloc = np.floor(quota).astype(int)
    leftover = remaining - int(alloc.sum())
    frac_order = (quota - np.floor(quota)).sort_values(ascending=False).index
    for i in range(leftover):
        alloc[frac_order[i % n]] += 1
    return alloc + base


@dataclass
class ResourceAllocator:
    cfg: Config

    def _zone_signals(self, events: pd.DataFrame, rhi_zone: Optional[pd.DataFrame],
                      hotspots: Optional[pd.DataFrame], zone_col: str) -> pd.DataFrame:
        """Aggregate raw demand signals per zone."""
        ev = events[events[zone_col].astype(str) != "unknown"].copy()
        g = ev.groupby(zone_col)

        def cause_count(cause: str) -> pd.Series:
            return ev[ev["event_cause"] == cause].groupby(zone_col)["id"].count()

        sig = pd.DataFrame(index=sorted(ev[zone_col].dropna().unique()))
        sig["total_events"] = g["id"].count()
        sig["breakdown"] = cause_count("vehicle_breakdown")
        sig["accident"] = cause_count("accident")
        sig["congestion"] = cause_count("congestion")
        sig["pot_holes"] = cause_count("pot_holes")
        sig["water_logging"] = cause_count("water_logging")
        sig["risk"] = g["risk_score"].mean()
        sig = sig.fillna(0)

        # Road-health signal (higher = worse road -> more maintenance demand).
        if rhi_zone is not None and not rhi_zone.empty:
            rh = rhi_zone.set_index("segment")["health_score"]
            sig["road_health"] = (100 - rh).reindex(sig.index).fillna(0)
        else:
            sig["road_health"] = 0.0

        # Hotspot density signal.
        if hotspots is not None and not hotspots.empty and "top_zone" in hotspots:
            hz = hotspots.groupby("top_zone")["risk_score"].sum()
            sig["hotspot"] = hz.reindex(sig.index).fillna(0)
        else:
            sig["hotspot"] = 0.0
        return sig

    def allocate(self, events: pd.DataFrame, rhi_zone: Optional[pd.DataFrame] = None,
                 hotspots: Optional[pd.DataFrame] = None,
                 zone_col: str = "zone") -> pd.DataFrame:
        sig = self._zone_signals(events, rhi_zone, hotspots, zone_col)
        if sig.empty:
            log.warning("No zones available for allocation.")
            return pd.DataFrame()

        # Normalise every signal across zones.
        norm = sig.apply(_minmax)
        drivers = self.cfg.resources.drivers.to_dict()
        fleet = self.cfg.resources.fleet.to_dict()
        floors = self.cfg.resources.min_per_zone.to_dict()

        result = pd.DataFrame(index=sig.index)
        demand_cols = {}
        for resource, weight_map in drivers.items():
            demand = pd.Series(0.0, index=sig.index)
            for driver, w in weight_map.items():
                if driver in norm.columns:
                    demand = demand + float(w) * norm[driver]
            demand_cols[resource] = demand
            result[f"demand_{resource}"] = demand.round(3)
            result[resource] = _largest_remainder(
                demand, int(fleet[resource]), int(floors.get(resource, 0)))

        # Priority rank by total normalised demand.
        result["total_demand"] = sum(demand_cols.values()).round(3)
        result["priority_rank"] = result["total_demand"].rank(
            ascending=False, method="min").astype(int)
        # Attach raw signal context + rationale.
        result = result.join(sig[["total_events", "breakdown", "accident",
                                   "risk", "road_health", "hotspot"]].round(2))
        result["rationale"] = result.apply(
            lambda r: self._rationale(r), axis=1)
        result = (result.reset_index().rename(columns={"index": "zone"})
                  .sort_values("priority_rank").reset_index(drop=True))
        log.info("Allocated fleet across %s zones (tow=%s police=%s maint=%s).",
                 len(result), fleet["tow_trucks"], fleet["traffic_police"],
                 fleet["maintenance_teams"])
        return result

    @staticmethod
    def _rationale(r: pd.Series) -> str:
        bits = []
        if r.get("breakdown", 0) > 0:
            bits.append(f"{int(r['breakdown'])} breakdowns")
        if r.get("accident", 0) > 0:
            bits.append(f"{int(r['accident'])} accidents")
        if r.get("road_health", 0) > 50:
            bits.append("poor road health")
        if r.get("hotspot", 0) > 0:
            bits.append("active hotspots")
        why = ", ".join(bits) if bits else "baseline coverage"
        return (f"Tow {int(r['tow_trucks'])}, Police {int(r['traffic_police'])}, "
                f"Maint {int(r['maintenance_teams'])} - driven by {why}.")

    @staticmethod
    def _lp_note() -> str:  # documentation hook
        return ("For a strict optimisation, replace Hamilton apportionment with an "
                "ILP: maximise Σ coverage[z]*alloc[z] s.t. Σ alloc[z]=F_r, "
                "alloc[z]>=floor, using PuLP/OR-Tools.")


def allocate_resources(
    events: pd.DataFrame,
    cfg: Optional[Config] = None,
    rhi_zone: Optional[pd.DataFrame] = None,
    hotspots: Optional[pd.DataFrame] = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Run the allocation engine and persist ``outputs/resource_allocation.csv``."""
    cfg = cfg or load_config()
    alloc = ResourceAllocator(cfg).allocate(events, rhi_zone, hotspots)
    if persist and not alloc.empty:
        out = get_path("outputs_dir", cfg=cfg) / "resource_allocation.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        alloc.to_csv(out, index=False)
        log.info("Resource allocation -> %s", out)
    return alloc


if __name__ == "__main__":  # pragma: no cover
    from src.engines.road_health_engine import compute_road_health
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    rhi = compute_road_health(feats, cfg, persist=False)["zone"]
    alloc = allocate_resources(feats, cfg, rhi_zone=rhi)
    print(alloc[["zone", "tow_trucks", "traffic_police", "maintenance_teams",
                 "priority_rank", "rationale"]].to_string(index=False))
