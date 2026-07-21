# Build Roadmap

Work through phases in order. Do not start a phase until the previous one's
Definition of Done is fully met — partial phases are where portfolio projects
quietly die. Each phase lists a rough time budget assuming part-time work
alongside other commitments; adjust freely.

---

### Phase 0 — Environment & Repo Setup
**Goal:** A clean, working skeleton that runs, even though it does nothing yet.
**Tasks**
- Initialize git repo, Python virtualenv, install `requirements.txt`
- Set up `pre-commit` with black, ruff, mypy
- Flesh out `docker-compose.yml` stub services (Redpanda, Redis, Postgres, MLflow)
- Confirm `docker-compose up` brings up all containers healthy
- Set up DVC pointing at a local remote (or free-tier cloud storage bucket)
**Definition of Done:** `docker-compose up` succeeds; `pytest` runs (even if only
against the placeholder test); pre-commit hooks pass on a dummy commit.
**Estimate:** 2–3 days

### Phase 1 — Data Pipeline & Feature Store
**Goal:** Consistent, versioned features available to both training and serving.
**Tasks**
- Download and DVC-track the dataset (see `docs/DATA_SPEC.md`)
- Assign synthetic `card_id`s and engineer rolling features (count/amount over
  trailing windows) — define these once, import everywhere (no duplicated logic)
- Build offline store writer (Postgres) and online store client (Redis) sharing
  the same feature definitions module
- Write a parity test: same input → identical feature values from both paths
**Definition of Done:** Parity test passes in CI; feature definitions live in a
single shared module with no copy-pasted logic between offline/online paths.
**Estimate:** 4–6 days

### Phase 2 — Model Training & Experiment Tracking
**Goal:** Reproducible, tracked, comparable model training runs.
**Tasks**
- Time-respecting train/validation/test split (no leakage across the split)
- Baseline: Logistic Regression, logged to MLflow
- Primary: XGBoost/LightGBM with basic hyperparameter search, logged to MLflow
- Secondary: Isolation Forest or small autoencoder for unsupervised anomaly score
- Log PR-AUC, recall@precision, confusion matrix, and PR curve as MLflow artifacts
- Register the best model in the MLflow Model Registry (stage: Staging)
**Definition of Done:** At least 3 tracked experiments in MLflow; one model
promoted to "Staging" with full metric lineage visible in the UI.
**Estimate:** 5–7 days

### Phase 3 — Real-Time Serving API
**Goal:** A working scoring endpoint with proper engineering hygiene.
**Tasks**
- FastAPI service that loads the current "Production"-stage model from MLflow
- `/score` endpoint: transaction in → fraud probability + decision + latency out
- `/health` and `/metrics` endpoints (Prometheus format)
- Input validation via pydantic; explicit error handling for missing features
- Unit + integration tests (mock the model, test the contract)
**Definition of Done:** p95 latency measured locally under simple load test
(e.g. `locust` or a simple async client) meets the < 150ms target from the PRD.
**Estimate:** 4–5 days

### Phase 4 — Streaming Ingestion & Online Inference
**Goal:** Wire the batch-tested pieces into an actual real-time flow.
**Tasks**
- Producer replays dataset onto Redpanda at configurable speed
- Consumer reads messages, calls Feature Service, calls Scoring Service
- Predictions + latencies logged to Postgres
- End-to-end smoke test: start stream, confirm predictions land in the log
**Definition of Done:** Running `docker-compose up` plus one producer script
results in continuously scored transactions with no manual intervention.
**Estimate:** 4–6 days

### Phase 5 — Monitoring, Drift Detection & Alerting
**Goal:** The system tells you when it's wrong, without you checking manually.
**Tasks**
- Implement PSI and KS-test drift module comparing live vs. training baseline
- Scheduled job (or Airflow task) runs the drift check on an interval
- Prometheus + Grafana dashboards: latency, throughput, prediction distribution,
  drift score over time
- Alerting rule (Grafana alert or simple webhook) on threshold breach
- Controlled test: inject synthetic drift into a test batch, confirm detection
**Definition of Done:** Injected-drift test reliably triggers a detectable alert;
dashboards render live data end-to-end.
**Estimate:** 5–7 days

### Phase 6 — Automated Retraining Orchestration
**Goal:** Close the loop from drift signal to a reviewable retrained model.
**Tasks**
- Stand up Airflow via its official Docker Compose stack
- DAG 1: scheduled retraining (e.g. weekly)
- DAG 2: drift-triggered retraining, invoked from the Phase 5 alert
- Human-approval gate: candidate model lands in MLflow "Staging," not
  auto-promoted to "Production"
- Document the promotion checklist (what metrics must hold before promoting)
**Definition of Done:** A manually injected drift event results in a new,
reviewable model version appearing in MLflow within one DAG run, with zero
manual pipeline steps beyond the final promotion approval.
**Estimate:** 5–7 days

### Phase 7 — CI/CD & Containerization Polish
**Goal:** Professional-grade repo hygiene.
**Tasks**
- GitHub Actions: lint → test → build Docker images on every PR
- Multi-stage Dockerfiles for smaller production images
- `.env.example` fully documented; no secrets anywhere in git history
- README quick-start verified by literally following it on a clean clone
**Definition of Done:** A stranger can clone the repo and get the full stack
running by following only the README, in under 15 minutes.
**Estimate:** 3–4 days

### Phase 8 — Stretch: Kubernetes, Autoscaling, Shadow Deployment
**Goal:** Extra depth if you want to go further before interviews.
**Tasks**
- Kubernetes manifests (or Helm chart) for the scoring service, using
  `kind`/`minikube` locally
- Horizontal Pod Autoscaler based on request load
- Shadow deployment: route a copy of live traffic to a candidate model and
  compare predictions against production without affecting real decisions
**Definition of Done:** Candidate model runs in shadow mode for a defined window
with a comparison report against the production model.
**Estimate:** 5–8 days (optional)

### Phase 9 — Documentation & Demo Polish
**Goal:** Make the work legible to someone who didn't build it.
**Tasks**
- Finalize architecture diagram and README
- Record a short (3–5 min) demo screen capture: stream running, dashboard live,
  a drift event triggering retraining
- Write up 3–4 STAR-format interview stories from `docs/DECISIONS.md`
**Definition of Done:** You can walk a stranger through the entire system,
end to end, from the README alone, in under 10 minutes.
**Estimate:** 2–3 days

---

## Suggested Pace

Working part-time, Phases 0–7 (the full core deliverable) take roughly **6–8
weeks**. Phase 8 is optional and only worth doing if you have runway before you
need the project interview-ready.
