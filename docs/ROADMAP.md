# Build Roadmap — FraudGuard

Work through phases in order. Do not start a phase until the previous one's
Definition of Done is fully met — partial phases are where portfolio projects
quietly die.

---

### Phase 0 — Environment & Repo Setup ✅ DONE
**Goal:** A clean, working skeleton that runs, even though it does nothing yet.

**Completed:**
- ✅ Git repo initialized, venv created, `requirements.txt` installed
- ✅ GitHub remote linked: `github.com/thenithin342/Credit-Card-Fraud-Detection`
- ✅ DVC initialized (`dvc init`), `.dvc/` committed
- ✅ `pyproject.toml` with black, ruff, mypy, pytest config
- ✅ `.gitignore`, `.env.example`, `.gitattributes` all present
- ✅ CI pipeline (`.github/workflows/ci.yaml`) — ruff + black + pytest

**Definition of Done:** ✅ `pytest tests/unit/` passes 7/7; CI green; repo on GitHub.

---

### Phase 1 — Data Pipeline & Feature Store ✅ DONE
**Goal:** Consistent, versioned features available to both training and serving.

**Completed:**
- ✅ Downloaded IEEE-CIS (590k rows) and ULB (284k rows) via Kaggle API
- ✅ Great Expectations validation suites (25/25 IEEE + 35/35 ULB expectations pass)
- ✅ Chronological train/val/test split: train=413k | val=59k | test=118k
- ✅ DVC pipeline: `ingest → validate → split` all green via `dvc repro`
- ✅ Parquet outputs in `data/processed/` (DVC-tracked)

**Definition of Done:** ✅ `dvc repro` runs all 3 stages successfully; GE validation passes.

**Next:** Feature engineering (rolling aggregates, velocity features, Redis online store)

---

### Phase 2 — Feature Engineering & Experiment Tracking
**Goal:** Consistent, versioned features + reproducible tracked model training.

**Tasks:**
- [ ] `src/features/definitions.py` — rolling aggregates (count/sum over 5m/1h/24h windows) defined once
- [ ] `src/features/offline_store.py` — write features to Postgres for training
- [ ] `src/features/online_store.py` — Redis-backed point lookups for serving
- [ ] Feature parity test: same input → identical values from both offline and online paths
- [ ] DVC stage: `feature_engineer` → `data/features/train_features.parquet`
- [ ] `src/config.py` — pydantic-settings loading all config from `.env`
- [ ] Baseline: Logistic Regression, logged to MLflow
- [ ] Primary: XGBoost/LightGBM with basic hyperparameter search, logged to MLflow
- [ ] Log PR-AUC, recall@precision, confusion matrix, PR curve as MLflow artifacts
- [ ] Register best model in MLflow Model Registry (stage: Staging)

**Definition of Done:** At least 3 tracked experiments in MLflow; one model in Staging;
parity test passes; feature parity test in CI.

**Estimate:** 5–7 days

---

### Phase 3 — Real-Time Serving API
**Goal:** A working scoring endpoint with proper engineering hygiene.

**Tasks:**
- [ ] `src/serving/main.py` — FastAPI service loading model from MLflow registry
- [ ] `/score` endpoint: transaction in → fraud probability + decision + latency
- [ ] `/health` and `/metrics` (Prometheus format) endpoints
- [ ] Input validation via pydantic; explicit error handling for missing features
- [ ] Unit + integration tests (mock the model, test the contract)
- [ ] Uncomment `api` service in `docker-compose.yml`

**Definition of Done:** p95 latency <150ms measured under `locust` load test.

**Estimate:** 4–5 days

---

### Phase 4 — Streaming Ingestion & Online Inference
**Goal:** Wire the batch-tested pieces into an actual real-time flow.

**Tasks:**
- [ ] `src/ingestion/producer.py` — replays dataset onto Redpanda at configurable speed
- [ ] `src/ingestion/consumer.py` — reads messages, calls Feature Service → Scoring Service
- [ ] Predictions + latencies logged to Postgres
- [ ] End-to-end smoke test: start stream, confirm predictions land in the log

**Definition of Done:** `docker-compose up` + one producer script → continuously scored
transactions with no manual intervention.

**Estimate:** 4–6 days

---

### Phase 5 — Monitoring, Drift Detection & Alerting
**Goal:** The system tells you when it's wrong, without you checking manually.

**Tasks:**
- [ ] `src/monitoring/drift.py` — PSI and KS-test comparing live vs. training baseline
- [ ] `src/monitoring/metrics.py` — Prometheus metrics for latency, throughput, scores
- [ ] Prometheus + Grafana dashboards (uncomment in docker-compose.yml)
- [ ] Alerting rule on threshold breach
- [ ] Controlled test: inject synthetic drift, confirm detection

**Definition of Done:** Injected-drift test reliably triggers a detectable alert; dashboards live.

**Estimate:** 5–7 days

---

### Phase 6 — Automated Retraining Orchestration
**Goal:** Close the loop from drift signal to a reviewable retrained model.

**Tasks:**
- [ ] Stand up Airflow via its official Docker Compose stack
- [ ] `src/orchestration/retrain_dag.py` — scheduled + drift-triggered retraining
- [ ] Human-approval gate: candidate lands in MLflow "Staging," not auto-promoted
- [ ] Document the promotion checklist

**Definition of Done:** A manually injected drift event results in a new, reviewable model
in MLflow within one DAG run, zero manual pipeline steps beyond final approval.

**Estimate:** 5–7 days

---

### Phase 7 — CI/CD & Containerization Polish
**Goal:** Professional-grade repo hygiene.

**Tasks:**
- [ ] GitHub Actions: lint → test → build Docker images on every PR
- [ ] Multi-stage Dockerfiles for smaller production images
- [ ] README quick-start verified on a clean clone (under 15 minutes)

**Definition of Done:** A stranger can clone and get the full stack running from only the README.

**Estimate:** 3–4 days

---

### Phase 8 — Stretch: Kubernetes, Autoscaling, Shadow Deployment
*(Optional — only if runway before interviews)*

**Estimate:** 5–8 days

---

### Phase 9 — Documentation & Demo Polish
**Goal:** Make the work legible to someone who didn't build it.

**Tasks:**
- [ ] Finalize architecture diagram and README
- [ ] Record a 3–5 min demo: stream running, dashboard live, drift triggering retraining
- [ ] Write 3–4 STAR-format interview stories from `docs/DECISIONS.md`

---

## Suggested Pace

Working part-time, Phases 0–7 take roughly **6–8 weeks**. Phase 8 is optional.
