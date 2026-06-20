# 🛰️ Smart Traffic Command Center + Road Health Monitoring System

An enterprise, end-to-end data-science platform built on the **Astram** Bengaluru
traffic-event dataset (8,173 events, Nov 2023 – Apr 2024). It turns raw incident
logs into **incident intelligence, road-health scores, geospatial hotspots,
forecasts, resource plans, ML predictions** and an **executive command-center
dashboard**.

> **Status:** every component is implemented and verified end-to-end on the real
> dataset (see *Verified Results* below).

---

## 1. Quickstart

```bash
# 1) install (core deps are enough; optional ML libs auto-detected)
pip install -r requirements.txt

# 2) run the whole platform (ingest -> features -> RHI -> hotspots ->
#    resources -> ML train/eval -> forecasts)
python -m src.run_pipeline

# 3) launch the command center
streamlit run dashboard/app.py
```

Run a subset of stages:

```bash
python -m src.run_pipeline --only data rhi hotspots
python -m src.run_pipeline --skip forecast        # everything except forecasting
```

**Optional libraries** (XGBoost, LightGBM, Prophet, MLflow, TensorFlow, Folium)
are auto-detected. Without them the platform still runs fully using scikit-learn /
statsmodels / Plotly fallbacks — install them for maximum capability.

---

## 2. Architecture

```
Sources ──► Data Layer ──► Analytics Layer ──► ML Layer ──► Dashboard Layer
 CSV         ingest         road_health         train         8 Streamlit pages
 (API*)      validate       hotspots            predict        maps · KPIs · Plotly
             clean          resource_alloc      evaluate       downloads · filters
             features       forecast            registry
```

Full Mermaid diagram and design rationale are in the design write-up; the
single source of truth for every parameter is **`config/config.yaml`**.

---

## 3. Folder structure

```
flipkart/
├── config/config.yaml              # all paths, weights, model & engine params
├── data/
│   ├── interim/  events_raw.parquet
│   └── processed/ events_clean.parquet, events_features.parquet
├── src/
│   ├── utils/         logger.py, config.py        # logging + dotted config
│   ├── data_pipeline/ ingest.py, validate.py, clean.py, feature_engineering.py
│   ├── engines/       road_health_engine.py, hotspot_engine.py,
│   │                  forecast_engine.py, resource_allocator.py
│   ├── models/        train.py, predict.py, evaluate.py, registry.py
│   └── run_pipeline.py                            # orchestrator
├── dashboard/         app.py, theme.py            # Streamlit command center
├── models/            <task>/model.joblib + metadata + importances, registry.json
├── outputs/           road_health_*.csv, hotspots_*.{csv,geojson},
│                      forecasts.csv, resource_allocation.csv, eval_*.json,
│                      validation_report.json
└── requirements.txt
```

---

## 4. What each file does

### Data layer — `src/data_pipeline/`
| File | Responsibility |
|---|---|
| **ingest.py** | Loads the raw CSV string-typed (no silent coercion), treats `NULL`/`""` as missing, emits an **ingest manifest** (SHA-256, row/col counts, timestamp) for lineage, snapshots Parquet. |
| **validate.py** | 12 declarative **quality gates** (schema, volume, PK uniqueness, per-column missingness, geo bounding-box, domain cardinality, temporal sanity) → structured `ValidationReport` (JSON) consumed by the Data Quality page; can hard-fail the pipeline. |
| **clean.py** | Drops dead columns, casts numeric/datetime, fixes geo (zeros/garbage → NaN, bbox clamp), nulls impossible end-times & derives `resolution_hours`, normalises messy `event_cause` casing & booleans, **KNN spatial imputation** of `zone`/`gba_identifier`/`corridor` via haversine `BallTree`, drops near-duplicates. |
| **feature_engineering.py** | Temporal (hour/dow/month, rush-hour, **cyclical sin/cos**), geo (distance-to-centre, grid cell), severity weights, the engineered **`risk_score`** target, corridor/cell context, and a reusable `ColumnTransformer` (OHE + scaling) shared by every model. |

### Analytics layer — `src/engines/`
| File | Responsibility |
|---|---|
| **road_health_engine.py** | **Road Health Index 0–100** per corridor & zone. Documented formula (Laplace-smoothed prevalence + absolute burden, min-max normalised, weighted degradation), config-or-data-driven weights, **per-cause explainability** (`points_lost_*`, `top_factor`), 5 categories. |
| **hotspot_engine.py** | **DBSCAN** (haversine, metres-based `eps`) density hotspots + **KMeans** (silhouette auto-`k`) command zones per cause. Outputs hotspot id, centre, count, dominant cause, **risk score**, and **GIS-ready GeoJSON**. |
| **forecast_engine.py** | Next **day/week/month** incident forecasts, city-wide & per corridor. Back-tests **Prophet / XGBoost / LSTM / ETS / seasonal-naive** (auto-skips uninstalled), picks lowest-MAE, emits forecasts + leaderboard + STL anomaly hooks. |
| **resource_allocator.py** | Recommends **tow trucks / traffic police / maintenance teams** per zone. Demand scoring from normalised signals (breakdown, accident, risk, road-health, hotspot) × config weights, then **Largest-Remainder (Hamilton)** apportionment that exactly conserves the fleet and honours per-zone floors. |

### ML layer — `src/models/`
| File | Responsibility |
|---|---|
| **registry.py** | Backend-agnostic **model registry** — MLflow when installed, else local `joblib + JSON`. Logs params, metrics, feature importances, artifacts; maintains `registry.json` index. |
| **train.py** | For each task: stratified split → **k-fold CV model selection** (RF / XGBoost / LightGBM, with HistGB fallbacks) → optional **`SelectFromModel` feature selection** → **RandomizedSearchCV tuning** → held-out metrics → registry. Leak-guards the engineered target. |
| **predict.py** | Loads registered pipelines and scores events with **train/serve parity** (same fitted preprocessor); returns `*_pred`, `*_proba`, `*_label`. |
| **evaluate.py** | Rich diagnostics: confusion matrix, classification report, ROC/PR curves (clf), residuals & predicted-vs-actual (reg) → `eval_<task>.json` for the dashboard. |

### Presentation — `dashboard/`
`app.py` is an 8-page Streamlit **Smart City Command Center** (dark neon theme in
`theme.py`): Executive Overview · Live Incident Analytics · Road Health · Hotspots ·
Forecasting · Resource Allocation · ML Predictions · Data Quality. Interactive
filters, Plotly maps/charts, KPI cards, CSV/GeoJSON downloads, and live ML scoring.

---

## 5. ML tasks & honest notes

| Task | Type | Target | Source |
|---|---|---|---|
| **Priority** | binary clf | `priority` (High/Low) | native column |
| **Closure**  | binary clf | `requires_road_closure` | native (imbalanced ~8%) |
| **Risk**     | regression | `risk_score` (0–100) | **engineered** composite |

> ⚠️ **Read this.** The priority/closure classifiers score ~0.99 F1. That is *not*
> a magic result — in this dataset `priority` and `requires_road_closure` are
> largely **deterministic business rules of `event_cause`** (e.g. tree-fall →
> closure). The models faithfully *recover the rule*. Treat them as rule-encoders /
> validators, and replace with genuinely uncertain operational labels for true
> predictive value.
>
> `risk_score` has **no ground-truth** in the source, so the pipeline builds a
> transparent weighted proxy (config-driven) and trains a model on it (R²≈0.71
> after excluding its own direct components to avoid leakage). Swap in
> human-labelled severity for production.

---

## 6. Verified results (this dataset)

* **Data:** 8,173 → **7,971** clean rows; 18 dead cols dropped; **4,668 zones
  spatially imputed**; 86 duplicates removed; 12/12 validation checks pass.
* **Road Health:** corridor mean **80.2**, zone mean **62.4**; worst corridor
  *Non-corridor* (34.8, Poor), worst zone *South Zone 2* (42.5, Moderate).
* **Hotspots:** 133 overall DBSCAN clusters; 89 breakdown, 11 pothole, 8 water-
  logging, 1 accident; top pothole hotspot = 71 incidents.
* **Forecast (city):** ETS best; ~**84 incidents/day** next-day projection.
* **Resources:** fleet of 40/60/25 apportioned across 10 zones; *Central Zone 2*
  top priority (955 breakdowns, 34 accidents, poor road health).
* **Models:** priority F1 0.9995 · closure F1 0.996 · risk R² 0.709 (MAE 5.8).

---

## 7. Reproducibility & ops

* Single config (`config/config.yaml`); fixed `random_seed`.
* Rotating file logs in `logs/`, structured JSON reports in `outputs/`.
* Every module runs **standalone** (`python -m src.<module>`) and via the
  orchestrator. Optional deps degrade gracefully; nothing hard-crashes if a
  library is missing.
```
