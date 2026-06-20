r"""
hotspot_engine.py
=================
**Geospatial hotspot detection** for accidents, breakdowns, water-logging and
potholes, using **DBSCAN** (density clusters) and **KMeans** (zonal centroids).

----------------------------------------------------------------------------
MATHEMATICAL EXPLANATION
----------------------------------------------------------------------------
*DBSCAN* groups points that are densely packed. Two points are "neighbours" if
their **haversine** distance ≤ ``eps``. The earth-surface distance is

    d(p,q) = R · 2·arcsin( sqrt( sin²(Δφ/2) + cosφ_p·cosφ_q·sin²(Δλ/2) ) )

with R = 6 371 000 m. We feed DBSCAN radians + ``metric="haversine"`` and set
``eps = eps_meters / R``. A *core point* has ≥ ``min_samples`` neighbours; a
*hotspot* is a maximal density-connected set of core points. Noise (label −1)
is discarded — these are isolated, non-recurring incidents.

*KMeans* partitions the same points into ``k`` centroids minimising
Σ‖xᵢ − μ_{cluster(i)}‖². ``k`` is auto-selected by maximising the silhouette
score over ``k_range`` — used to give *every* area a "nearest command centroid",
complementing DBSCAN's density view.

----------------------------------------------------------------------------
RISK SCORE PER HOTSPOT
----------------------------------------------------------------------------
    risk = 100 · ( w_count·count_n + w_sev·severity_n )

where ``count_n`` is the min-max normalised incident count and ``severity_n``
the normalised mean ``cause_severity`` of the cluster's events.

----------------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------------
* Per-event cluster labels.
* Per-hotspot table: id, centre lat/lon, count, dominant cause, risk score.
* **GIS-ready GeoJSON** (FeatureCollection of Points) saved to ``outputs/``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)

_EARTH_RADIUS_M = 6_371_000.0


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.0, index=s.index) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


@dataclass
class HotspotEngine:
    cfg: Config

    # -- DBSCAN ----------------------------------------------------------
    def dbscan(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.cluster import DBSCAN
        geo = self.cfg.schema.geo
        d = self.cfg.hotspots.dbscan
        coords = np.radians(df[[geo.lat, geo.lon]].to_numpy())
        eps = d.eps_meters / _EARTH_RADIUS_M
        labels = DBSCAN(eps=eps, min_samples=d.min_samples,
                        metric="haversine", algorithm="ball_tree").fit_predict(coords)
        out = df.copy()
        out["cluster"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        log.info("DBSCAN: %s hotspots (%s noise points).",
                 n_clusters, int((labels == -1).sum()))
        return out

    # -- KMeans ----------------------------------------------------------
    def kmeans(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        geo = self.cfg.schema.geo
        kc = self.cfg.hotspots.kmeans
        X = df[[geo.lat, geo.lon]].to_numpy()
        seed = self.cfg.project.random_seed

        if kc.auto_k and len(df) > 50:
            best_k, best_s = kc.k, -1.0
            lo, hi = kc.k_range
            for k in range(lo, min(hi, len(df) - 1) + 1):
                km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
                s = silhouette_score(X, km.labels_)
                if s > best_s:
                    best_k, best_s = k, s
            log.info("KMeans auto-k = %s (silhouette %.3f).", best_k, best_s)
            k = best_k
        else:
            k = kc.k
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        out = df.copy()
        out["kmeans_zone"] = km.labels_
        return out

    # -- summarise clusters into hotspots --------------------------------
    def summarise(self, df: pd.DataFrame, label_col: str = "cluster",
                  cause: str = "all") -> pd.DataFrame:
        geo = self.cfg.schema.geo
        rs = self.cfg.hotspots.risk_score
        valid = df[df[label_col] != -1]
        if valid.empty:
            return pd.DataFrame()

        rows = []
        for cid, g in valid.groupby(label_col):
            dom = g["event_cause"].mode()
            rows.append({
                "hotspot_id": f"{cause}_{label_col}_{int(cid)}",
                "cause_filter": cause,
                "method": label_col,
                "center_lat": round(g[geo.lat].mean(), 6),
                "center_lon": round(g[geo.lon].mean(), 6),
                "incident_count": len(g),
                "dominant_cause": dom.iloc[0] if not dom.empty else "unknown",
                "mean_severity": round(g["cause_severity"].mean(), 3),
                "radius_m": round(self._radius_m(g, geo), 1),
                "top_corridor": g["corridor"].mode().iloc[0] if "corridor" in g else None,
                "top_zone": (g["zone"].mode().iloc[0]
                             if "zone" in g and not g["zone"].mode().empty else None),
            })
        hs = pd.DataFrame(rows)
        hs["risk_score"] = (100 * (
            rs.count_weight * _minmax(hs["incident_count"])
            + rs.severity_weight * _minmax(hs["mean_severity"])
        )).round(1)
        return hs.sort_values("risk_score", ascending=False).reset_index(drop=True)

    def _radius_m(self, g: pd.DataFrame, geo) -> float:
        lat0, lon0 = g[geo.lat].mean(), g[geo.lon].mean()
        dlat = np.radians(g[geo.lat] - lat0)
        dlon = np.radians(g[geo.lon] - lon0)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat0)) ** 2 * np.sin(dlon / 2) ** 2
        dist = _EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(a))
        return float(np.percentile(dist, 90)) if len(dist) else 0.0

    # -- GeoJSON ---------------------------------------------------------
    @staticmethod
    def to_geojson(hs: pd.DataFrame) -> dict:
        features = []
        for _, r in hs.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [r["center_lon"], r["center_lat"]]},
                "properties": {k: (None if pd.isna(v) else v)
                               for k, v in r.items()
                               if k not in ("center_lat", "center_lon")},
            })
        return {"type": "FeatureCollection", "features": features}


def detect_hotspots(
    df: pd.DataFrame,
    cfg: Optional[Config] = None,
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Detect hotspots overall and per configured cause; save CSV + GeoJSON.

    Returns ``{cause -> hotspot_table}`` including key ``"all"``.
    """
    cfg = cfg or load_config()
    engine = HotspotEngine(cfg)
    out_dir = get_path("outputs_dir", cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    targets = ["all"] + list(cfg.hotspots.causes)

    all_features = []  # collected for one combined GeoJSON
    for cause in targets:
        sub = df if cause == "all" else df[df["event_cause"] == cause]
        if len(sub) < cfg.hotspots.dbscan.min_samples:
            log.warning("Skipping '%s' hotspots (only %s events).", cause, len(sub))
            continue
        labelled = engine.dbscan(sub)
        hs = engine.summarise(labelled, "cluster", cause)
        results[cause] = hs
        if persist and not hs.empty:
            hs.to_csv(out_dir / f"hotspots_{cause}.csv", index=False)
            gj = engine.to_geojson(hs)
            (out_dir / f"hotspots_{cause}.geojson").write_text(
                json.dumps(gj, indent=2), encoding="utf-8")
            all_features.extend(gj["features"])
        log.info("Hotspots[%s]: %s clusters.", cause, len(hs))

    # KMeans command zones (overall) for resource centroids.
    if len(df) > 50:
        km = engine.kmeans(df)
        kmz = engine.summarise(km, "kmeans_zone", "all")
        results["kmeans_zones"] = kmz
        if persist and not kmz.empty:
            kmz.to_csv(out_dir / "hotspots_kmeans_zones.csv", index=False)

    if persist and all_features:
        combined = {"type": "FeatureCollection", "features": all_features}
        (out_dir / "hotspots_all_causes.geojson").write_text(
            json.dumps(combined, indent=2), encoding="utf-8")
        log.info("Combined GeoJSON -> %s", out_dir / "hotspots_all_causes.geojson")
    return results


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    res = detect_hotspots(feats, cfg)
    for cause, hs in res.items():
        if not hs.empty:
            print(f"\n=== {cause} (top 5) ===")
            print(hs[["hotspot_id", "center_lat", "center_lon", "incident_count",
                      "dominant_cause", "risk_score"]].head().to_string(index=False))
