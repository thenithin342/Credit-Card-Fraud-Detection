# Product Requirements Document
## FraudGuard — Real-Time Credit Card Fraud Detection Platform

**Status:** Draft v1.0  
**Owner:** Nithin  
**Last updated:** 2026-07-21

---

## 1. Overview

FraudGuard is a self-contained, end-to-end MLOps platform that detects fraudulent
transactions in a simulated real-time stream. Every component is intentionally built
to demonstrate — not just describe — the full lifecycle of a production ML system:
data ingestion, feature computation, model training with experiment tracking,
low-latency serving, drift detection, automated retraining, and observability.

It is designed as a **portfolio project**, so every component is either built from
first principles where that adds interview-defensible depth (e.g. hand-rolled feature
store and drift detector), while using industry-standard tools where reinventing adds
no signal (e.g. MLflow for experiment tracking).

## 2. Problem Statement

Fraud detection is the canonical real-time ML problem: labels are extremely imbalanced
(~0.1–3.5% positive rate), data distribution shifts constantly (fraud patterns evolve),
latency requirements are strict (<150ms), and the cost of a stale model is direct
financial loss. This makes it an ideal vehicle for demonstrating MLOps maturity —
**the model is the easy 20%; the pipeline around it is the hard 80%.**

## 3. Goals

**Portfolio / resume goals**
- Produce a repo that a FAANG or ML-startup interviewer can skim in 5 minutes and
  understand the architecture, and that you can defend in depth for 45 minutes.
- Demonstrate fluency across the full MLOps stack: streaming, feature engineering,
  experiment tracking, model registry, CI/CD, orchestration, monitoring, and automated
  retraining — not just "trained a model in a notebook."

**Technical goals**
- G1: Ingest a simulated real-time transaction stream; score transactions with p95 latency <150ms.
- G2: Maintain consistent features between training (offline) and serving (online) — no train/serve skew.
- G3: Track every experiment (params, metrics, artifacts) with clear model registry stage transitions.
- G4: Detect data drift and concept drift automatically, without manual inspection.
- G5: Trigger retraining automatically on drift, with a human-approval gate before production.
- G6: Every component runs locally via Docker Compose with no paid API keys required.

## 4. Datasets

- **IEEE-CIS Fraud Detection** (Kaggle): 590,540 transactions, rich categorical/numeric features, ~3.5% fraud rate
- **Credit Card Fraud Detection (ULB)**: 284,807 transactions, PCA-anonymized, ~0.17% fraud rate (severe imbalance)

Both datasets are DVC-tracked for full reproducibility.

## 5. Non-Goals

- Not a compliance-grade or legally defensible fraud system.
- Not optimized for massive scale (millions of TPS); optimized for demonstrating the *correct architecture*.
- Kubernetes, multi-region deployment, and A/B testing are explicitly **stretch goals (Phase 8)**.

## 6. Success Metrics

**Model quality** (accuracy is explicitly not used as primary metric):
- Precision-Recall AUC as primary metric (not ROC-AUC, given severe class imbalance).
- Recall at a fixed precision operating point (e.g. recall @ 90% precision).

**System quality**
- p95 scoring latency <150ms under simulated load.
- Feature parity check passes between offline (training) and online (serving) paths.
- Drift detector correctly flags injected synthetic drift in a controlled test.

**Operational quality**
- Full pipeline runs green via `docker-compose up` with zero manual steps.
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

- **Reproducibility**: any past model version must be exactly reproducible from versioned data + code + config.
- **Local-first**: the entire system must run on a single laptop with Docker.
- **Security basics**: no secrets committed to git; `.env`-based configuration.
- **Documentation quality**: every non-trivial architectural decision is logged in `docs/DECISIONS.md`.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Severe class imbalance makes accuracy misleading | Use PR-AUC and recall-at-precision from day one |
| Scope creep across phases | Strict definition-of-done per phase in ROADMAP.md |
| Streaming simulation doesn't reflect true distributed systems | Explicitly documented as a known simplification |
| Datasets lack real card ID (ULB) | Synthetic card ID assignment strategy documented in DATA_SPEC.md |
