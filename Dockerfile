# =====================================================================
#  Smart Traffic Command Center — production image
#  One image, three roles (pipeline / api / dashboard) chosen by the
#  command in docker-compose. Slim, non-root, layer-cached deps.
# =====================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libs needed by scientific wheels (lightgbm/xgboost need libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first (cached unless requirements change).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Application code.
COPY . .

# Non-root runtime user.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Pre-create writable artifact dirs.
RUN mkdir -p data/interim data/processed models outputs logs

EXPOSE 8000 8501

# Default role = dashboard (override in compose for api/pipeline).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
