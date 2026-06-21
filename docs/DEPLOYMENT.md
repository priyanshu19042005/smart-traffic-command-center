# Deployment Guide

Three ways to run the Smart Traffic Command Center: **local**, **Docker**, and
**cloud**. Pick by audience — laptop demo, reproducible container, or hosted URL.

---

## 0. Prerequisites

| | Local | Docker |
|---|---|---|
| Python | 3.10–3.12 | (bundled in image) |
| Docker | — | Docker 24+ & Compose v2 |
| RAM | ≥ 4 GB | ≥ 4 GB |
| Disk | ~1 GB | ~2 GB |

The platform is **CPU-only** — no GPU required (LSTM is optional).

---

## 1. Local (bare metal / venv)

```bash
git clone https://github.com/priyanshu19042005/smart-traffic-command-center.git
cd smart-traffic-command-center

python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_pipeline           # build all artifacts + train models (~3 min)

# Serve (two terminals, or use &):
uvicorn src.api.main:app --port 8000 # API   -> http://localhost:8000/docs
streamlit run dashboard/app.py       # UI    -> http://localhost:8501
```

Or with `make`: `make install && make pipeline && make dashboard`.

---

## 2. Docker (single image, three roles)

```bash
docker compose build                 # build the image once

docker compose run --rm pipeline     # one-shot: generate artifacts + models
docker compose up api dashboard      # serve both
#   API       -> http://localhost:8000/docs
#   Dashboard -> http://localhost:8501
```

* All three services share a **named volume** (`artifacts`) so the pipeline's
  outputs are visible to the API and dashboard.
* `api` and `dashboard` `depend_on` the pipeline completing successfully.
* The image runs as a **non-root** user and ships a `HEALTHCHECK`.

Run a single role manually:

```bash
docker build -t tcc .
docker run --rm -p 8501:8501 tcc      # dashboard (default CMD)
docker run --rm -p 8000:8000 tcc uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 3. Cloud

### 3a. Streamlit Community Cloud (zero-config demo)
The dashboard **self-bootstraps** — no pre-build step or committed artifacts needed.

1. Push the repo to GitHub (already done).
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → pick the repo/branch.
3. **Main file path:** `dashboard/app.py` · **Python:** 3.11 or 3.12 (Advanced settings).
4. Deploy. On **first load** the app detects the missing artifacts and builds them
   from the committed raw CSV — features + analytics on landing (~30 s), forecasts
   when the Forecasting page opens, and **compact models** (fast, no tuning) when
   the ML page opens. Everything is `@st.cache_resource`-cached thereafter.

> Why this matters: `data/processed`, `outputs/` and `models/` are git-ignored, so a
> fresh clone has only code + the raw CSV. Without bootstrap every page would show
> *"run `python -m src.run_pipeline`"*. The bootstrap (`dashboard/bootstrap.py`) removes
> that step. Models are trained in a memory-safe **fast mode** (`train_all(fast=True)`)
> so they fit within Community Cloud's RAM; metrics are slightly below the fully-tuned
> local run (which you still get via `python -m src.run_pipeline`).

`requirements.txt` is installed automatically. Optional heavy libs (XGBoost/Prophet/
TensorFlow) are intentionally **not** in it, so the build stays light and the platform
uses its sklearn/statsmodels fallbacks.

### 3b. Render / Railway / Fly.io (fastest hosted demo)
* **Render:** New → *Web Service* from the repo. Build: `pip install -r requirements.txt`.
  * Dashboard start: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
  * API start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
  * Add a one-off **Job**/pre-deploy running `python -m src.run_pipeline` (artifacts on a persistent disk).
* **Railway/Fly:** point at the `Dockerfile`; set the start command per service; mount a volume for `/app/outputs` and `/app/models`.

### 3c. AWS EC2 (VM)
```bash
sudo yum install -y docker git && sudo service docker start
git clone <repo> && cd smart-traffic-command-center
sudo docker compose run --rm pipeline
sudo docker compose up -d api dashboard
```
Put an **ALB / Nginx** in front; open 80/443 only; terminate TLS at the proxy.

### 3d. Kubernetes (sketch)
* One `Job` for the pipeline writing to a `PersistentVolumeClaim`.
* Two `Deployments` (api, dashboard) mounting the same PVC read-only.
* `Service` + `Ingress` per app; `readinessProbe` → `/api/v1/health` and
  `/_stcore/health`. `HorizontalPodAutoscaler` on CPU for the API.

---

## 4. Configuration & environment

* All tunables live in **`config/config.yaml`** (paths, weights, model params,
  fleet sizes). No secrets are required to run.
* Optional: set `MLFLOW_TRACKING_URI` to use a remote MLflow server (else a
  local store under `models/mlruns` is used automatically).
* Streamlit theme: `.streamlit/config.toml`. Put any future secrets in
  `.streamlit/secrets.toml` (git-ignored) — **never** commit secrets.

---

## 5. Operations

| Concern | How |
|---|---|
| **Health** | `GET /api/v1/health`, `GET /_stcore/health`; compose `HEALTHCHECK`s |
| **Logs** | Rotating file logs in `logs/` + stdout (captured by Docker/JSON driver) |
| **Refresh data** | Re-run `python -m src.run_pipeline` (or a single stage `--only forecast`) on a schedule (cron / GitHub Action / k8s CronJob) |
| **Retrain** | `python -m src.models.train`; artifacts versioned in the registry |
| **Rollback** | Models are timestamped in `models/registry.json`; restore a prior `model.joblib` |

---

## 6. Production hardening checklist

- [ ] Restrict API **CORS** `allow_origins` to known front-ends.
- [ ] Add **auth** (API key / OAuth2 / gateway) in front of the API & dashboard.
- [ ] Terminate **TLS** at a reverse proxy (Nginx/Caddy/ALB).
- [ ] Move artifacts to **object storage** (S3/GCS) for multi-replica serving.
- [ ] Add **rate limiting** and request logging at the proxy.
- [ ] Pin dependency versions / use a lockfile; enable Dependabot.
- [ ] Re-evaluate whether the **raw dataset** belongs in the image/repo for your
      context (it ships anonymized; remove via `.dockerignore`/`.gitignore` if not).
