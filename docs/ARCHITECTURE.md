# System Architecture

Smart Traffic Command Center + Road Health Monitoring — technical architecture.

---

## 1. System Context (C4 Level 1)

```mermaid
flowchart LR
    OPS["👮 Traffic Ops / City Officials"]
    EXEC["📊 Executives"]
    DEV["🧑‍💻 Data / ML Engineers"]
    EXT["🌐 External GIS / 3rd-party apps"]

    subgraph SYS["🛰️ Smart Traffic Command Center"]
        DASH["Streamlit Dashboard"]
        API["REST API (FastAPI)"]
        CORE["Pipeline + Engines + ML"]
    end

    SRC["📥 Astram Event Dataset (CSV)\n(future: live API / IoT sensors)"]

    SRC --> CORE
    OPS --> DASH
    EXEC --> DASH
    DEV --> API
    EXT --> API
    DASH --> CORE
    API --> CORE
```

---

## 2. Component Architecture (C4 Level 2)

```mermaid
flowchart TB
    subgraph DATA["🗄️ Data Layer — src/data_pipeline"]
        ING["ingest.py\nload • SHA-256 manifest"]
        VAL["validate.py\n12 quality gates"]
        CLN["clean.py\ndedup • types • spatial KNN impute"]
        FE["feature_engineering.py\ntemporal • geo • risk_score"]
        ING --> VAL --> CLN --> FE
    end

    subgraph ANALYTICS["📊 Analytics Layer — src/engines"]
        RHI["road_health_engine\nRHI 0-100"]
        HOT["hotspot_engine\nDBSCAN + KMeans → GeoJSON"]
        FC["forecast_engine\nProphet/XGB/LSTM/ETS"]
        RES["resource_allocator\nHamilton apportionment"]
    end

    subgraph ML["🤖 ML Layer — src/models"]
        TRN["train.py\nCV • tuning • selection"]
        REG["registry.py\nMLflow / local"]
        PRD["predict.py"]
        EVL["evaluate.py"]
        TRN --> REG --> PRD
        TRN --> EVL
    end

    subgraph SERVE["🖥️ Serving Layer"]
        API["src/api — FastAPI\n/api/v1/*"]
        DASH["dashboard/app.py\n8 pages"]
    end

    CFG["⚙️ config/config.yaml\n(single source of truth)"]
    STORE[("📦 Artifacts\ndata/ • models/ • outputs/")]

    FE --> STORE
    STORE --> RHI & HOT & FC & RES & TRN
    RHI & HOT & FC & RES --> STORE
    REG --> STORE
    STORE --> API & DASH
    PRD --> API & DASH
    EVL --> DASH
    CFG -.-> DATA & ANALYTICS & ML & SERVE

    style DATA fill:#0b3d5c,color:#fff
    style ANALYTICS fill:#1b5e20,color:#fff
    style ML fill:#4a148c,color:#fff
    style SERVE fill:#bf360c,color:#fff
```

---

## 3. Data Flow (end-to-end)

```mermaid
flowchart LR
    A["Raw CSV\n8,173 × 46"] --> B["Ingest\nstring-typed"]
    B --> C["Validate\n12 gates"]
    C --> D["Clean\n7,971 rows\n4,668 zones imputed"]
    D --> E["Features\n+risk_score\n+temporal +geo"]
    E --> F1["RHI\ncorridor/zone"]
    E --> F2["Hotspots\nGeoJSON"]
    E --> F3["Forecast\nday/week/month"]
    E --> F4["Models\npriority/closure/risk"]
    F1 & F4 --> F5["Resource\nAllocation"]
    F1 & F2 & F3 & F4 & F5 --> G["API + Dashboard"]
```

---

## 4. ML Training Pipeline

```mermaid
flowchart TB
    X["Feature table"] --> SPLIT["Stratified train/test split"]
    SPLIT --> CV["k-fold CV model selection\nRF · XGBoost · LightGBM\n(HistGB fallback)"]
    CV --> SEL["SelectFromModel\ntop-k features"]
    SEL --> TUNE["RandomizedSearchCV\nhyperparameter tuning"]
    TUNE --> FIT["Fit best estimator"]
    FIT --> METRICS["Held-out metrics\nAcc/Prec/Rec/F1/ROC-AUC · R²/MAE/RMSE"]
    FIT --> REG["Model Registry\n(MLflow / local joblib+JSON)"]
    REG --> SERVE["predict.py → API / Dashboard"]
```

---

## 5. Deployment Topology (Docker Compose)

```mermaid
flowchart TB
    subgraph HOST["🐳 Docker Host"]
        VOL[("Named volume: artifacts\ndata/ models/ outputs/")]
        P["pipeline\n(one-shot job)"]
        A["api\nuvicorn :8000"]
        D["dashboard\nstreamlit :8501"]
        P -- writes --> VOL
        VOL -- reads --> A
        VOL -- reads --> D
        P -.completes_successfully.-> A
        P -.completes_successfully.-> D
    end
    U1["Browser :8501"] --> D
    U2["API client :8000"] --> A
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Config-driven** (`config.yaml`) | Tune weights, params, fleet sizes without touching code; reproducible. |
| **String-typed ingest** | No silent numeric coercion; cleaning owns all casting → auditable. |
| **Spatial KNN imputation** | Recovers 4,668 missing zones from geography instead of dropping rows. |
| **Engineered `risk_score`** | No ground-truth severity exists; transparent weighted proxy, leak-guarded in training. |
| **Optional-dependency fallbacks** | Runs fully on sklearn/statsmodels/plotly; XGBoost/LightGBM/Prophet/MLflow/Folium enhance when present. |
| **Shared preprocessor** | Same fitted `ColumnTransformer` at train & serve → no train/serve skew. |
| **One image, three roles** | pipeline/api/dashboard from a single Dockerfile via command override → smaller surface. |
| **Registry abstraction** | MLflow when available, local JSON+joblib otherwise → never blocks. |
