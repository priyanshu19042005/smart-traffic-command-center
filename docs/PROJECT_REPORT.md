# Project Report & Technical Review
### Smart Traffic Command Center + Road Health Monitoring System

**Dataset:** Astram Bengaluru traffic events — 8,173 records × 46 columns,
2023-11-10 → 2024-04-08 (150 days).
**Codebase:** ~3,900 LOC across data pipeline, analytics engines, ML layer,
REST API and an 8-page dashboard. **Tests:** 14 passing.

---

## 1. Executive Summary

The platform converts a raw, messy incident log into six operational
capabilities — **incident intelligence, road-health scoring, geospatial hotspot
discovery, forecasting, resource allocation, and ML prediction** — exposed
through both a **command-center dashboard** and a **REST API**. Every stage was
executed on the real dataset and verified end-to-end (see §10).

**Overall assessment: 7.9 / 10 — strong, production-leaning engineering with a
few honest caveats** (rule-derived classification targets, single-node compute,
and demo-grade security) that are documented rather than hidden.

| Dimension | Score | One-line verdict |
|---|---|---|
| Code Quality | 8.5 | Typed, documented, logged, modular, tested; minor broad-`except` nits. |
| Architecture | 9.0 | Clean layered design, config-driven, graceful degradation. |
| ML Quality | 7.5 | Rigorous pipeline; classification targets are rule-derived (metrics overstated). |
| Dashboard | 9.0 | 8 polished pages, maps, KPIs, downloads — all render exception-free. |
| Scalability | 6.5 | Excellent to ~10⁶ rows; needs distributed compute beyond. |
| Security | 6.0 | Safe defaults, non-root, validated inputs; no auth / open CORS by design. |
| Performance | 8.0 | Vectorised + cached; a couple of avoidable loops. |

---

## 2. Code Quality — 8.5 / 10

**Strengths**
- **Type hints + docstrings** on every public function; module headers explain
  intent and math.
- **Centralised logging** (`src/utils/logger.py`) — colourised console + rotating
  file, idempotent handler setup.
- **Config-driven** throughout via a single `config.yaml` and a dotted-access
  `Config` wrapper — no magic numbers in code.
- **Modular & runnable**: every module has a `__main__` and runs standalone
  (`python -m src.<module>`), composed by `run_pipeline.py`.
- **14 pytest tests** covering pipeline invariants, engine guarantees (fleet
  conservation, score ranges) and API contracts.

**Weaknesses / risks**
- Several **broad `except Exception`** blocks (intentional for optional-dep
  fallbacks, but they can mask real errors) → narrow them or log tracebacks.
- A few `df.get("col", default)` fallbacks assume a column exists when given a
  scalar default (e.g. `priority` in `feature_engineering`) — fragile if schema
  drifts.
- No **CI**, **linter/formatter config** (ruff/black) or **type-checker** (mypy)
  committed yet.

**Recommendations:** add `ruff` + `black` + `mypy` configs and a GitHub Actions
CI running `pytest`; replace broad excepts with targeted `ImportError`/`ValueError`.

---

## 3. Architecture Quality — 9.0 / 10

**Strengths**
- Clear **4-layer separation** (Data → Analytics → ML → Serving) with one-way
  dependencies; see `docs/ARCHITECTURE.md`.
- **Single source of truth** (`config.yaml`) drives schema, cleaning rules,
  weights, model params and fleet sizes.
- **Graceful degradation**: optional libraries (XGBoost, LightGBM, Prophet,
  MLflow, TensorFlow, Folium) are auto-detected; the platform runs fully on the
  core stack with sklearn/statsmodels/Plotly fallbacks.
- **Registry abstraction** decouples training from MLflow — never a hard blocker.
- **One image, three roles** keeps the deployment surface minimal.

**Weaknesses / risks**
- Batch orchestration only — **no streaming/queue** path for true real-time
  ingestion (acknowledged as a future extension).
- Artifacts are **file-based on a shared volume**; multi-replica serving needs
  object storage (S3/GCS).

---

## 4. ML Quality — 7.5 / 10

**Pipeline rigor (strong):** stratified split → k-fold **CV model selection**
(RF / XGBoost / LightGBM, HistGB fallback) → **`SelectFromModel`** feature
selection → **RandomizedSearchCV** tuning → held-out metrics → registry with
params, metrics and importances. Shared preprocessor prevents train/serve skew.

**Held-out results**

| Task | Type | Metric | Score |
|---|---|---|---|
| Priority | classification | F1 / ROC-AUC | 0.9995 / 1.000 |
| Closure | classification | F1 / ROC-AUC | 0.9958 / 0.9995 |
| Risk | regression | R² / MAE / RMSE | 0.709 / 5.76 / 6.67 |

> ⚠️ **Honest caveat (the most important finding).** The near-perfect
> classification scores are **not** predictive magic. In this dataset
> `priority` and `requires_road_closure` are **near-deterministic functions of
> `event_cause`** (e.g. tree-fall ⇒ closure). The models **recover the business
> rule**, which is useful for validation/encoding but **overstates real
> predictive skill**. For genuine value, retrain on operationally uncertain
> labels.
>
> `risk_score` has **no ground truth** in the source. It is an **engineered**,
> transparent weighted proxy (`config.features.risk_weights`); the regressor is
> **leak-guarded** (its direct components are excluded), giving an honest R²≈0.71.
> Replace with human-labelled severity in production.

**Other limitations**
- **150-day** series limits time-series depth (no yearly seasonality; Prophet/LSTM
  are available but ETS/seasonal-naive often win at this length).
- **SHAP** is listed but not wired into the default flow (importances are logged).
- No **drift / performance monitoring** loop yet.

**Recommendations:** source real severity labels; add population-stability/drift
checks; schedule periodic retraining; surface SHAP per-prediction in the API.

---

## 5. Dashboard Quality — 9.0 / 10

**Strengths**
- **8 pages** (Executive, Live Analytics, Road Health, Hotspots, Forecasting,
  Resources, ML Predictions, Data Quality) — **all render with zero exceptions**
  under Streamlit `AppTest`.
- **Interactive**: global date/corridor/zone/cause/priority filters; Plotly maps
  (`carto-darkmatter`), KPI cards, heatmaps, ROC/confusion/scatter diagnostics.
- **Actionable**: CSV + **GeoJSON** downloads; live ML scoring widget.
- Cohesive **Smart City Command Center** dark theme; `@st.cache_data` for speed.

**Weaknesses / risks**
- **No authentication** (suitable for demo/internal; gate before public hosting).
- Uses Plotly maps because Folium isn't installed here — Folium/Leaflet path is
  optional and documented.

---

## 6. Scalability — 6.5 / 10

| Aspect | Today | Path forward |
|---|---|---|
| Data volume | Single-node **pandas**; comfortable to ~10⁶ rows | Dask/Polars/Spark; partitioned Parquet |
| Storage | Local files + shared volume | S3/GCS + a warehouse (DuckDB/BigQuery) |
| API | **Stateless** → scales horizontally behind a load balancer | k8s HPA on CPU |
| Training | In-process | Distributed/Ray; scheduled retrain jobs |
| Clustering | DBSCAN/KMeans in-memory | Approx-NN / tiled spatial indexing |

The **serving tier already scales out** (stateless API + read-only artifacts);
the **batch/compute tier is the bottleneck** at large scale.

---

## 7. Security — 6.0 / 10

**Good**
- **No secrets** in code or config; nothing sensitive required to run.
- Container runs as a **non-root** user with a `HEALTHCHECK`.
- **Input validation** on the API via Pydantic (range-checked `hour`/lat/lon →
  422 on bad input).
- `.gitignore`/`.dockerignore` exclude local state and generated artifacts.

**Gaps (by design for a demo; harden before public exposure)**
- **No authentication/authorization** on the API or dashboard.
- **CORS is open** (`allow_origins=["*"]`).
- **No rate limiting**.
- The **raw dataset is committed** (anonymized). Confirm this is acceptable for
  your context; IDs are surrogate (`FKID*`, `FKUSR*`) but free-text
  `description` and real addresses are present.

**Recommendations:** API key/OAuth2 + reverse-proxy TLS; restrict CORS; add rate
limiting; re-evaluate shipping the dataset; add dependency scanning (Dependabot).

---

## 8. Performance — 8.0 / 10

**Strengths**
- Vectorised pandas/numpy throughout; **haversine BallTree** for O(n log n)
  spatial imputation and DBSCAN.
- Dashboard uses **cached loaders** (`@st.cache_data`); API loads models **once**
  at startup via lifespan.
- Full pipeline (excluding model tuning) runs in **~21 s**; full run incl.
  training ≈ 3 min on a laptop CPU.

**Weaknesses**
- **Recursive multi-step forecast** loops day-by-day (fine for 30-day horizons,
  not for thousands of scopes).
- **KMeans auto-k** fits k across the whole range each run — cache or use the
  elbow/MiniBatchKMeans for larger data.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Classification metrics misread as predictive skill | High | Med | Documented caveat; retrain on uncertain labels |
| Engineered `risk_score` mistaken for ground truth | Med | Med | Clearly labelled; leak-guarded; swap real labels |
| Public exposure without auth | Med | High | Add auth/TLS/CORS limits before hosting |
| Schema drift breaks `.get` fallbacks | Low | Med | Add schema contract + tests |
| Single-node compute at scale | Low (now) | High (later) | Distributed roadmap in §6 |

---

## 10. Verified Results (this dataset)

- **Data:** 8,173 → **7,971** clean rows; 18 dead columns dropped; **4,668 zones
  spatially imputed**; 86 near-duplicates removed; **12/12 validation checks pass**.
- **Road Health:** corridor mean **80.2**, zone mean **62.4**; worst zone *South
  Zone 2* (42.5, Moderate, driver = potholes, 705 events).
- **Hotspots:** 133 overall DBSCAN clusters (89 breakdown, 11 pothole, 8 water-
  logging, 1 accident); top pothole hotspot = **71 incidents**.
- **Forecast (city):** ETS selected; **~84 incidents/day** next-day projection.
- **Resources:** fleet 40/60/25 apportioned across 10 zones (exactly conserved);
  *Central Zone 2* top priority.
- **API:** 9 endpoints, all 200/healthy; example predict → High priority (0.997),
  no closure (0.005), risk 69.7.
- **Tests:** **14 passed** in ~21 s.

---

## 11. Roadmap

1. **Data realism:** source true severity & operational priority labels; add live
   ingestion (Kafka/API) for streaming.
2. **MLOps:** CI (pytest + ruff + mypy), drift monitoring, scheduled retraining,
   per-prediction SHAP in the API.
3. **Security:** auth, TLS, CORS lockdown, rate limiting, dependency scanning.
4. **Scale:** object storage + warehouse; Polars/Dask for compute; k8s autoscaling.
5. **Product:** alerting (spike anomalies → notifications), what-if resource
   simulation, mobile-friendly executive view.

---

## Appendix — File inventory

```
config/config.yaml              src/engines/road_health_engine.py
src/utils/{logger,config}.py    src/engines/hotspot_engine.py
src/data_pipeline/ingest.py     src/engines/forecast_engine.py
src/data_pipeline/validate.py   src/engines/resource_allocator.py
src/data_pipeline/clean.py      src/models/{train,predict,evaluate,registry}.py
src/data_pipeline/feature_engineering.py   src/api/{main,schemas}.py
src/run_pipeline.py             dashboard/{app,theme}.py
tests/{conftest,test_data_pipeline,test_engines,test_api}.py
Dockerfile · docker-compose.yml · Makefile · requirements.txt
docs/{ARCHITECTURE,API,DEPLOYMENT,PROJECT_REPORT}.md
```
