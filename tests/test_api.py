"""Tests for the REST API (FastAPI TestClient)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_stats(client):
    r = client.get("/api/v1/stats")
    if r.status_code == 200:
        assert r.json()["total_incidents"] > 0


def test_road_health_endpoint(client):
    r = client.get("/api/v1/road-health?level=zone")
    assert r.status_code in {200, 404}


def test_predict_endpoint(client):
    body = {"event_cause": "accident", "corridor": "Hosur Road", "zone": "South Zone 2",
            "veh_type": "heavy_vehicle", "event_type": "unplanned", "status": "active",
            "hour": 18, "latitude": 12.9081, "longitude": 77.6476}
    r = client.post("/api/v1/predict", json=body)
    # 200 if models trained, 503 in degraded mode
    assert r.status_code in {200, 503}
    if r.status_code == 200:
        data = r.json()
        assert "priority" in data and "resolution" in data


def test_predict_validation_rejects_bad_hour(client):
    body = {"event_cause": "accident", "hour": 99,
            "latitude": 12.9, "longitude": 77.6}
    r = client.post("/api/v1/predict", json=body)
    assert r.status_code == 422  # pydantic validation error


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert len(r.json()["paths"]) >= 8
