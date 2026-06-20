"""Tests for the analytics engines (road health, hotspots, resources)."""
from __future__ import annotations

from src.engines.road_health_engine import compute_road_health
from src.engines.hotspot_engine import detect_hotspots
from src.engines.resource_allocator import allocate_resources


VALID_CATEGORIES = {"Excellent", "Good", "Moderate", "Poor", "Critical"}


def test_road_health_scores_and_categories(features, cfg):
    res = compute_road_health(features, cfg, persist=False)
    for level in ("corridor", "zone"):
        df = res[level]
        assert df["health_score"].between(0, 100).all()
        assert set(df["health_category"]).issubset(VALID_CATEGORIES)
        # explainability column present
        assert "top_factor" in df.columns


def test_hotspots_produce_geojson_ready_table(features, cfg):
    res = detect_hotspots(features, cfg, persist=False)
    assert "all" in res
    hs = res["all"]
    if not hs.empty:
        assert {"center_lat", "center_lon", "incident_count", "risk_score"}.issubset(hs.columns)
        assert hs["risk_score"].between(0, 100).all()


def test_resource_allocation_conserves_fleet(features, cfg):
    alloc = allocate_resources(features, cfg, persist=False)
    fleet = cfg.resources.fleet.to_dict()
    # Hamilton apportionment must conserve the fleet exactly.
    assert alloc["tow_trucks"].sum() == fleet["tow_trucks"]
    assert alloc["traffic_police"].sum() == fleet["traffic_police"]
    assert alloc["maintenance_teams"].sum() == fleet["maintenance_teams"]
    # police floor of >=1 per zone honoured
    assert (alloc["traffic_police"] >= 1).all()
