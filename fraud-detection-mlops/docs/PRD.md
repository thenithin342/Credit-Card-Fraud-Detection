# Product Requirements Document
## Real-Time Fraud & Anomaly Detection Platform

**Status:** Draft v1.0
**Owner:** Nithin
**Last updated:** 2026-07-20

---

## 1. Overview

This project is a self-contained, end-to-end MLOps platform that detects fraudulent
transactions in a simulated real-time stream. It is built to demonstrate — not just
describe — the full lifecycle of a production ML system: data ingestion, feature
computation, model training with experiment tracking, low-latency serving, drift
detection, automated retraining, and observability.

It is designed as a **portfolio project**, so every component is intentionally built
from first principles where that adds interview-defensible depth (e.g. a hand-rolled
feature store and drift detector), while using industry-standard tools where
reinventing them would add no signal (e.g. MLflow for experiment tracking).

## 2. Problem Statement

Fraud detection is one of the canonical real-time ML problems: labels are extremely
imbalanced, the data distribution shifts constantly (fraud patterns evolve), latency
requirements are strict (a transaction can't wait seconds for a verdict), and the cost
of a stale model is direct financial loss. This makes it an ideal vehicle for
demonstrating MLOps maturity, because "the model" is the easy 20% — the pipeline
around it is the hard 80%, and that's what this project is scoped to prove out.

## 3. Goals

**Portfolio / resume goals**
- Produce a repo that a FAANG or ML-startup interviewer can skim in 5 minutes and
  understand the architecture, and that you can defend in depth for 45 minutes.
- Demonstrate fluency across the full MLOps stack: streaming, feature engineering,
  experiment tracking, model registry, CI/CD, orchestration, monitoring, and
  automated retraining — not just "trained a model in a notebook."

**Technical goals**
- G1: Ingest a simulated real-time transaction stream and score transactions with
  p95 latency under 150ms.
- G2: Maintain consistent features between training (offline) and serving (online)
  paths — no train/serve skew.
- G3: Track every experiment (params, metrics, artifacts) and promote models through
  a registry with clear stage transitions (staging → production).
- G4: Detect data drift and concept drift automatically, without manual inspection.
- G5: Trigger and execute retraining automatically when drift crosses a threshold,
  with a human-approval gate before promotion to production.
- G6: Every component runs locally via Docker Compose with no paid API keys or
  managed cloud services required.

## 4. Non-Goals

- Not a compliance-grade or legally defensible fraud system.
- Not optimized for massive scale (millions of TPS); optimized for demonstrating
  the *correct architecture* at a scale that runs on a laptop.
- Not using real payment data or a proprietary internal dataset.
- Kubernetes, multi-region deployment, and A/B testing infrastructure are explicitly
  **stretch goals** (Phase 8), not part of the core deliverable.

## 5. Users & Stakeholders (Simulated Personas)

Used to frame requirements even though this is a solo project:

- **Fraud analyst** — consumes predictions and needs explainability + low false
  positive rate on legitimate transactions.
- **ML engineer (you)** — owns model quality, retraining cadence, and experiment
  history.
- **On-call / SRE** — consumes dashboards and alerts; cares about latency,
  uptime, and pipeline failures.

## 6. Success Metrics

**Model quality** (imbalanced classification — accuracy is explicitly not used):
- Precision-Recall AUC as primary metric (not ROC-AUC, given ~0.17% positive rate).
- Recall at a fixed precision operating point (e.g. recall @ 90% precision),
  chosen to reflect a realistic business tradeoff.

**System quality**
- p95 scoring latency < 150ms under simulated load.
- Feature parity check passes between offline (training) and online (serving)
  pipelines (automated test, not manual eyeballing).
- Drift detector correctly flags injected synthetic drift in a controlled test.

**Operational quality**
- Full pipeline (ingest → feature → predict → log → monitor) runs green via
  `docker-compose up` with zero manual steps.
- CI pipeline (lint, test, build) passes on every PR.
- Retraining pipeline runs end-to-end from trigger to a registered candidate model.

## 7. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Ingest a simulated streaming transaction feed (replayed historical data at configurable speed) |
| FR2 | Compute features consistently for both training (batch) and serving (real-time) |
| FR3 | Train and compare a baseline model and at least one advanced model, with all runs logged |
| FR4 | Serve real-time fraud predictions via a REST API |
| FR5 | Detect feature drift (PSI / KS-test) and performance degradation automatically |
| FR6 | Trigger a retraining pipeline on drift threshold breach or schedule |
| FR7 | Version datasets, features, and models reproducibly |
| FR8 | Run automated tests and builds on every commit (CI) |
| FR9 | Expose operational and business dashboards |
| FR10 | Alert on SLA breach, pipeline failure, or drift detection |

## 8. Non-Functional Requirements

- **Reproducibility**: any past model version must be exactly reproducible from
  versioned data + code + config.
- **Local-first**: the entire system must run on a single laptop with Docker.
- **Security basics**: no secrets committed to git; `.env`-based configuration.
- **Documentation quality**: every non-trivial architectural decision is logged in
  `docs/DECISIONS.md` with a short rationale — this becomes your interview material.

## 9. Scope & Phasing

See `docs/ROADMAP.md` for the full phased build plan. Summary:

0. Environment & repo setup
1. Data pipeline & feature store
2. Baseline + advanced model training with experiment tracking
3. Real-time serving API
4. Streaming ingestion + online inference
5. Monitoring, drift detection, alerting
6. Automated retraining orchestration
7. CI/CD & containerization polish
8. *(Stretch)* Kubernetes, autoscaling, shadow deployment / A-B testing
9. Documentation & demo polish

## 10. Assumptions

- **Dataset**: the Kaggle "Credit Card Fraud Detection" dataset (ULB/Worldline,
  ~284,807 transactions, 492 frauds, PCA-anonymized features) is used as the
  historical source, replayed as a simulated stream. See `docs/DATA_SPEC.md`.
- **Streaming**: Redpanda (Kafka-API-compatible) running in Docker stands in for a
  production Kafka cluster. The concepts transfer directly; the operational scale
  does not, and that distinction should be stated plainly in interviews.
- **Compute**: models are trained on CPU; no GPU dependency.

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Severe class imbalance makes naive accuracy misleading | Use PR-AUC and recall-at-precision as headline metrics from day one |
| Scope creep across 9 phases | Strict definition-of-done per phase in ROADMAP.md; stretch items clearly separated |
| Streaming simulation doesn't reflect true distributed systems ops | Explicitly documented as a known simplification, not hidden |
| Dataset lacks a real card/user ID for realistic feature engineering | Synthetic card ID assignment strategy documented in DATA_SPEC.md |

## 12. Deliverables

- Working Docker Compose stack covering ingestion → serving → monitoring
- MLflow-tracked experiments with a registered, promotable model
- FastAPI scoring service with automated tests
- Custom drift-detection module with dashboards
- Airflow (or equivalent) retraining DAG
- CI pipeline (GitHub Actions)
- Architecture diagram, decision log, and polished README suitable for a resume link
