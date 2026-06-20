# API Documentation

Smart Traffic Command Center **REST API** (FastAPI). All responses are JSON.

* **Base URL:** `http://localhost:8000`
* **Interactive docs:** `GET /docs` (Swagger UI) · `GET /redoc` (ReDoc)
* **OpenAPI schema:** `GET /openapi.json`
* **Versioned prefix:** `/api/v1`

Start the server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# or:  make api      # or:  docker compose up api
```

> The API serves artifacts produced by `python -m src.run_pipeline`. Endpoints
> that need a missing artifact return **404** with a helpful message; `/predict`
> returns **503** if models aren't trained (degraded mode).

---

## Endpoint summary

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Service, model and artifact status |
| GET | `/api/v1/stats` | Headline KPIs |
| GET | `/api/v1/road-health?level=zone\|corridor` | Road Health Index table |
| GET | `/api/v1/hotspots?cause=all&limit=50` | Hotspot table |
| GET | `/api/v1/hotspots/geojson?cause=all` | GIS-ready FeatureCollection |
| GET | `/api/v1/forecast?scope=CITY` | Forecast series + horizons |
| GET | `/api/v1/resources` | Resource-allocation plan |
| GET | `/api/v1/quality` | Data-quality validation report |
| POST | `/api/v1/predict` | Score an incident (priority/closure/risk) |

---

## GET `/api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": ["priority", "closure", "risk"],
  "artifacts_available": {
    "features": true, "road_health": true, "hotspots": true,
    "forecasts": true, "resources": true
  }
}
```
`status` is `"ok"` when models are loaded, `"degraded"` otherwise.

---

## GET `/api/v1/stats`

```json
{
  "total_incidents": 7971,
  "active": 988,
  "road_closures": 595,
  "mean_risk_score": 46.09,
  "date_min": "2023-11-10",
  "date_max": "2024-04-08",
  "top_causes": {"vehicle_breakdown": 4873, "others": 636, "pot_holes": 525,
                 "water_logging": 456, "construction": 438}
}
```

---

## GET `/api/v1/road-health`

Query: `level` = `zone` (default) or `corridor`.

```bash
curl "http://localhost:8000/api/v1/road-health?level=zone"
```
```json
[
  {"level":"zone","segment":"South Zone 2","health_score":42.5,
   "health_category":"Moderate","top_factor":"pot_holes","total_events":705},
  {"level":"zone","segment":"South Zone 1","health_score":47.6,
   "health_category":"Moderate","top_factor":"pot_holes","total_events":233}
]
```

---

## GET `/api/v1/hotspots`

Query: `cause` ∈ `all | accident | vehicle_breakdown | water_logging | pot_holes`; `limit` (1–500).

```bash
curl "http://localhost:8000/api/v1/hotspots?cause=pot_holes&limit=2"
```
```json
[
  {"hotspot_id":"pot_holes_cluster_8","center_lat":12.925356,"center_lon":77.61945,
   "incident_count":71,"dominant_cause":"pot_holes","risk_score":60.0,"top_zone":"Central Zone 2"}
]
```

### GET `/api/v1/hotspots/geojson`
Returns an RFC 7946 `FeatureCollection` of `Point` features — drop straight into
Leaflet, Mapbox, QGIS or kepler.gl.

```json
{"type":"FeatureCollection","features":[
  {"type":"Feature","geometry":{"type":"Point","coordinates":[77.61945,12.925356]},
   "properties":{"hotspot_id":"pot_holes_cluster_8","incident_count":71,"risk_score":60.0}}
]}
```

---

## GET `/api/v1/forecast`

Query: `scope` = `CITY` (default) or a corridor name (e.g. `Mysore Road`).

```json
{
  "scope": "CITY",
  "horizons": {"day": 84.4, "week": 554.49, "month": 2466.57},
  "best_model": "ets",
  "forecast": [
    {"date":"2024-04-09","yhat":84.4,"yhat_lower":23.56,"yhat_upper":145.24},
    {"date":"2024-04-10","yhat":79.73,"yhat_lower":18.89,"yhat_upper":140.57}
  ]
}
```

---

## GET `/api/v1/resources`

```json
[
  {"zone":"Central Zone 2","priority_rank":1,"tow_trucks":8,"traffic_police":5,
   "maintenance_teams":5,"rationale":"Tow 8, Police 5, Maint 5 - driven by 955 breakdowns, ..."}
]
```

---

## GET `/api/v1/quality`

Returns the full validation report (pass/fail, per-check severities & messages) —
the same object the **Data Quality** dashboard page renders.

---

## POST `/api/v1/predict`

Score a hypothetical or live incident across all three models.

### Request body (`IncidentRequest`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `event_cause` | string | — (required) | e.g. `accident`, `pot_holes` |
| `corridor` | string | `Non-corridor` | |
| `zone` | string | `unknown` | |
| `veh_type` | string | `not_applicable` | e.g. `heavy_vehicle`, `bmtc_bus` |
| `event_type` | string | `unplanned` | |
| `status` | string | `active` | |
| `hour` | int | `9` | 0–23 (validated) |
| `latitude` | float | `12.9716` | 12.6–13.4 (validated) |
| `longitude` | float | `77.5946` | 77.2–77.9 (validated) |

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"event_cause":"accident","corridor":"Hosur Road","zone":"South Zone 2",
       "veh_type":"heavy_vehicle","status":"active","hour":18,
       "latitude":12.9081,"longitude":77.6476}'
```

### Response (`PredictionResponse`)

```json
{
  "priority": {"label": "High", "probability": 0.9975, "value": null},
  "closure":  {"label": "not_required", "probability": 0.0049, "value": null},
  "risk":     {"label": null, "probability": null, "value": 69.71},
  "model_versions": {"priority":"random_forest","closure":"random_forest","risk":"hist_gb"}
}
```

### Status codes
| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Validation error (e.g. `hour=99`) — Pydantic detail returned |
| 503 | Models not trained — run `python -m src.models.train` |

---

## Notes for integrators

* **CORS** is open (`*`) for demo convenience — restrict `allow_origins` in
  `src/api/main.py` for production.
* **No auth** by default — front with an API gateway / reverse proxy (see
  `docs/DEPLOYMENT.md`) and add API keys/OAuth before exposing publicly.
* The `priority`/`closure` models reflect near-deterministic business rules in
  the source data (see `docs/PROJECT_REPORT.md` §ML Quality) — treat their
  confidence accordingly.
