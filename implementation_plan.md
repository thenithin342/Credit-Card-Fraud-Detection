# FraudGuard — Real-Time Streaming Fraud Detection Platform
## Phased Implementation Plan

> **Philosophy**: Each phase produces a working, demonstrable artifact. No phase depends on cloud resources unless explicitly stated. We build locally first, harden later.

---

## Open Questions (Answer Before Phase 1 Begins)

> [!IMPORTANT]
> **Q1 — Dataset Access**: Have you already downloaded the IEEE-CIS and ULB datasets from Kaggle, or do you need setup instructions? (Kaggle CLI vs. manual download)

> [!IMPORTANT]
> **Q2 — Python Environment**: Do you have a preferred Python version / env manager? (`conda`, `pyenv`, `uv`, plain `venv`?) We'll pin everything in a `pyproject.toml`.

> [!IMPORTANT]
> **Q3 — Cloud vs. Local-only**: For now, do you want to stay fully local (Docker Compose + kind), or do you have AWS/GCP credentials ready? This affects Phase 5+ significantly.

> [!IMPORTANT]
> **Q4 — Kafka flavor**: Redpanda (lighter, Docker-friendly, Kafka-API compatible) or full Apache Kafka? For local dev Redpanda is recommended — confirm?

> [!NOTE]
> **Q5 — Sequence model (v2)**: The spec mentions an optional LSTM/Transformer over per-user history. Should we plan a slot for it (Phase 7 extension) or skip it entirely?

---

## Phase Roadmap Overview

| Phase | Name | Duration | Deliverable |
|-------|------|----------|-------------|
| **1** | Foundation & Data Pipeline | Week 1–2 | Repo skeleton, DVC pipeline, validated datasets |
| **2** | Feature Engineering | Week 3–4 | Offline feature tables, Feast definitions |
| **3** | Model Training & Experiment Tracking | Week 5–6 | Trained models in MLflow, Optuna sweep |
| **4** | Model Serving (FastAPI) | Week 7 | Working `/predict` API with SHAP |
| **5** | Streaming Pipeline (Kafka) | Week 8–9 | End-to-end real-time scoring path |
| **6** | Containerization & Local K8s | Week 10 | Docker images + kind cluster |
| **7** | CI/CD & Monitoring | Week 11–12 | GitHub Actions, Grafana, Evidently |
| **8** | Automated Retraining Loop | Week 13–14 | Airflow DAG + registry promotion |

---

## Phase 1 — Foundation & Data Pipeline
**Goal**: Working repo, reproducible data pipeline, validated schemas.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] Project root scaffold
```
fraudguard/
├── data/raw/          ← DVC-tracked (not git)
├── data/processed/    ← DVC-tracked
├── src/
│   ├── ingestion/
│   └── validation/
├── tests/unit/
├── tests/data_validation/
├── dvc.yaml
├── params.yaml
├── pyproject.toml     ← single source of deps + tool config
├── .dvcignore
├── .gitignore
└── README.md
```

#### Key files
- **`pyproject.toml`** — pin Python 3.11, all deps (xgboost, lightgbm, feast, fastapi, mlflow, dvc, great_expectations, evidently, structlog, optuna…)
- **`dvc.yaml`** — stages: `ingest → validate → split`
- **`params.yaml`** — dataset paths, split ratios, random seeds
- **`src/ingestion/download.py`** — Kaggle CLI wrapper to pull IEEE-CIS + ULB
- **`src/validation/ge_suite.py`** — Great Expectations suite: schema check, null rates, value ranges, class distribution guard
- **`tests/data_validation/test_schema.py`** — pytest wrapper around GE suite

#### Verification
- `dvc repro` runs without errors
- GE validation passes on raw data
- `pytest tests/data_validation/` green

---

## Phase 2 — Feature Engineering + Feature Store
**Goal**: Reproducible feature tables; Feast definitions ready for online serving.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] `src/features/`
- **`feature_engineering.py`** — all transformations:
  - Velocity features (txn count in 1h/24h/7d per card)
  - Spend aggregates (mean, std, max in rolling windows)
  - Geo-distance from home location (Haversine)
  - Time-of-day / day-of-week cyclical encoding
  - Category frequency encoding + target encoding (with proper CV fold handling to avoid leakage)
- **`feast_definitions.py`** — `FeatureView`, `Entity`, `FeatureService` definitions
- **`backfill.py`** — materialize offline features to Feast's online store (Redis)

#### [NEW] `src/features/feast_repo/`
- `feature_store.yaml` (local provider, file-based offline, Redis online)
- Entity + FeatureView Python definitions

#### Updated `dvc.yaml`
Adds `feature_engineer` stage after `validate`.

#### Verification
- Feature table shape and dtype assertions pass
- No target-leakage: correlation of raw label vs. engineered features checked
- Feast `feast apply` runs; `feast materialize` populates Redis

---

## Phase 3 — Model Training & Experiment Tracking
**Goal**: Reproducible training pipeline, 3 model types tracked in MLflow, Optuna sweep, best model in registry.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] `src/training/`
- **`train.py`** — trains XGBoost / LightGBM / LogReg; handles class imbalance via `scale_pos_weight` + SMOTE comparison; logs params + metrics to MLflow
- **`tune.py`** — Optuna study, 50–100 trials, pruning via `MLflowCallback`
- **`imbalance.py`** — helpers: SMOTE, class-weight variants, threshold calibration

#### [NEW] `src/evaluation/`
- **`metrics.py`** — AUC-PR, precision@recall curve, cost-weighted loss (FN cost >> FP cost), F1 at operating threshold
- **`explain.py`** — SHAP TreeExplainer; generates waterfall + beeswarm plots saved as MLflow artifacts
- **`eval_report.py`** — consolidated report (JSON + HTML)

#### [NEW] `mlruns/` (git-ignored)
Local MLflow tracking store.

#### Updated `dvc.yaml`
Adds `train` and `evaluate` stages.

#### Verification
- All 3 models tracked; AUC-PR > baseline (random = ~0.002)
- XGBoost target: AUC-PR ≥ 0.80 on IEEE-CIS holdout
- Best model promoted to MLflow Registry `Staging`

---

## Phase 4 — Model Serving API (FastAPI)
**Goal**: Sub-100ms `/predict` endpoint with SHAP explanations; full REST API surface.
**Duration**: ~1 week

### Proposed Changes

#### [NEW] `src/serving/`
- **`app.py`** — FastAPI application
- **`schemas.py`** — Pydantic v2 request/response models (strict input validation)
- **`predictor.py`** — loads model from MLflow Registry; pulls online features from Feast/Redis; assembles feature vector; returns prediction + confidence + SHAP top-k
- **`middleware.py`** — structured logging (structlog), latency tracking, API-key auth header check
- **`health.py`** — `/health` endpoint, liveness check (model loaded, Redis reachable)

#### API surface
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single transaction scoring |
| `/predict/batch` | POST | Batch scoring (≤500 txns) |
| `/explain` | POST | Full SHAP breakdown for one transaction |
| `/health` | GET | Liveness + readiness |
| `/model/metadata` | GET | Active model version, training date, metrics |

#### [NEW] `tests/integration/test_api.py`
FastAPI `TestClient` tests for all endpoints; assert latency < 100ms on local hardware.

#### Verification
- All endpoints return correct schemas
- P99 latency < 100ms on 100 sequential local requests (Locust smoke test)

---

## Phase 5 — Streaming Pipeline (Kafka / Redpanda)
**Goal**: End-to-end: producer → stream processor → feature computation → serving API → prediction log sink.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] `src/ingestion/`
- **`producer.py`** — replays IEEE-CIS transactions at configurable TPS via Kafka topic `raw-transactions`
- **`consumer.py`** — Kafka consumer; calls `/predict`; publishes result to `prediction-logs` topic
- **`stream_features.py`** — windowed aggregates computed on-the-fly (Faust or plain confluent-kafka); writes to Redis for online feature freshness

#### [NEW] `docker-compose.yaml`
Services: `redpanda`, `redis`, `postgres`, `mlflow-server`, `fastapi-serving`

#### [NEW] `src/monitoring/log_sink.py`
Consumes `prediction-logs` topic; writes structured rows to Postgres `predictions` table.

#### Verification
- `docker compose up` → all services healthy
- Producer sends 1000 transactions → consumer processes all → Postgres row count matches
- End-to-end wall-clock latency (producer emit → prediction in DB) < 500ms

---

## Phase 6 — Containerization & Local Kubernetes
**Goal**: Production-like K8s deployment on `kind`; HPA tested.
**Duration**: ~1 week

### Proposed Changes

#### [MODIFY] `Dockerfile`
Multi-stage: `builder` (installs deps) → `runtime` (slim, no build tools). Separate `Dockerfile.training`.

#### [NEW] `k8s/`
- `namespace.yaml`
- `deployment.yaml` — FastAPI serving pod, 2 replicas, resource limits
- `service.yaml` — ClusterIP + NodePort for local access
- `hpa.yaml` — scale on CPU > 70% or custom QPS metric
- `configmap.yaml` — model version pin, Feast endpoint
- `secret.yaml` (template, never committed with real values)
- `ingress.yaml` — nginx ingress
- `cronjob.yaml` — nightly drift check job

#### Verification
- `kind create cluster` → `kubectl apply -k k8s/` → all pods `Running`
- HPA triggers on simulated load (`kubectl run` load generator)
- Rolling update (new image tag) completes with zero downtime

---

## Phase 7 — CI/CD & Monitoring
**Goal**: Automated quality gates on every PR; Grafana dashboards live; Evidently drift reports.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] `.github/workflows/ci.yaml`
Triggers: push / PR to `main`
Steps: checkout → setup Python → `ruff` lint → `black --check` → `pytest tests/unit/` → data validation smoke test → model training smoke test (10% data sample) → build Docker image (no push on PR)

#### [NEW] `.github/workflows/cd.yaml`
Triggers: merge to `main` + passing CI
Steps: build + push image to GHCR → `kubectl set image` (or Argo CD sync) → wait for rollout → smoke test `/health` → auto-rollback on failure

#### [NEW] `src/monitoring/`
- **`drift_job.py`** — loads yesterday's prediction logs from Postgres; loads reference dataset; runs Evidently `DataDriftPreset` + `ClassificationPreset`; saves HTML report; posts webhook if drift score > threshold
- **`evidently_config.yaml`** — thresholds, reference window size
- **`prometheus_config.py`** — `prometheus-fastapi-instrumentator` setup; custom metrics: `fraud_predictions_total`, `model_confidence_histogram`

#### [NEW] `grafana/dashboards/`
- `latency_qps.json` — request latency P50/P95/P99, QPS
- `fraud_rate.json` — rolling fraud prediction rate over time
- `model_health.json` — confidence distribution, model version in use

#### Verification
- CI green on a clean PR with no code changes
- Grafana dashboard shows live metrics after 5 minutes of load test
- Evidently report generated; HTML artifact saved

---

## Phase 8 — Automated Retraining Loop (Airflow)
**Goal**: Full drift → retrain → promote → deploy loop with no manual intervention.
**Duration**: ~2 weeks

### Proposed Changes

#### [NEW] `airflow/dags/`
- **`drift_check_dag.py`** — daily @06:00 UTC; runs `drift_job.py`; if drift score > threshold, triggers `retrain_dag`
- **`retrain_dag.py`** — pulls latest data window from Postgres/S3 → runs DVC pipeline stages → trains model → evaluates → if AUC-PR ≥ gate → promotes to MLflow Registry `Staging` → runs shadow deployment comparison → promotes to `Production` → triggers CD

#### [NEW] Shadow / Canary deployment support
- `src/serving/shadow.py` — routes duplicate requests to shadow model; logs both predictions; compares offline
- K8s `VirtualService` (or Argo Rollouts `Rollout`) for canary traffic split (5% → 25% → 100%)

#### [NEW] `tests/model_tests/`
- **`test_invariance.py`** — prediction shouldn't flip on irrelevant feature perturbation (e.g., change transaction ID)
- **`test_min_performance.py`** — assert AUC-PR ≥ minimum threshold before promotion gate passes
- **`load_test.py`** — Locust file; 500 concurrent users; assert P99 < 100ms

#### Verification
- Simulate drift by injecting ULB dataset as "new" production traffic
- Airflow DAG runs → drift detected → retrain triggered → new model auto-promoted
- Canary rollout completes with no latency regression

---

## Verification Plan (End-to-End)

### Automated Tests
```bash
pytest tests/unit/              # feature transform correctness
pytest tests/data_validation/   # GE schema + distributions
pytest tests/integration/       # API contract tests
pytest tests/model_tests/       # invariance + min performance
locust -f tests/load_test.py    # latency SLA
dvc repro                       # full pipeline reproducibility
```

### Manual Verification
- Kaggle dataset → `dvc repro` → MLflow UI shows 3 model runs → `/predict` returns in <100ms → Grafana dashboard live → Evidently report generated → Airflow DAG successfully triggers retrain on injected drift.

---

## What You Need to Provide / Confirm

| Item | Status |
|------|--------|
| IEEE-CIS + ULB datasets downloaded | ❓ |
| Python env manager preference | ❓ |
| Cloud (AWS/GCP) or local-only | ❓ |
| Kafka vs. Redpanda preference | ❓ |
| Sequence model (v2) in scope? | ❓ |
| GitHub repo created / Actions enabled | ❓ |

---

> [!TIP]
> **Recommended starting point**: Confirm the open questions above, then we kick off **Phase 1** — repo scaffold, `pyproject.toml`, DVC pipeline, and Great Expectations data validation. That alone is a solid, demonstrable deliverable that anchors everything else.
