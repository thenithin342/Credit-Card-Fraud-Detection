# Real-Time Fraud & Anomaly Detection Platform

An end-to-end MLOps portfolio project: streaming ingestion → hand-built feature
store → tracked model training → real-time serving → drift detection →
automated retraining. Runs entirely locally via Docker. No paid APIs required.

> Built to demonstrate that the pipeline *around* a model is the hard part —
> not just the model itself. See `docs/PRD.md` for the full rationale.

## Where to start

This repo is meant to be opened in **Claude Code**. Do this first:

1. Open this folder in Claude Code.
2. Claude Code will read `CLAUDE.md` automatically at session start — but if it
   doesn't pick it up, tell it to read `CLAUDE.md` and `docs/*.md` explicitly.
3. Paste this as your first message:

```
Read CLAUDE.md and every file in docs/ before doing anything.

We are starting from Phase 0 in docs/ROADMAP.md. Set up the repo: Python
virtualenv, install requirements.txt, configure pre-commit (black, ruff,
mypy), flesh out docker-compose.yml for the Phase 0 services (Redpanda,
Redis, Postgres, MLflow) per docs/TECH_STACK.md, and set up DVC with a local
remote.

Do not start Phase 1 work. Stop after Phase 0's Definition of Done is met,
run through the DoD checklist explicitly, and tell me what's next.
```

4. Once Phase 0 is confirmed done, move to Phase 1 the same way — tell Claude
   Code "Phase 0 is confirmed, start Phase 1 from ROADMAP.md."

## Documentation map

| File | Purpose |
|---|---|
| `CLAUDE.md` | Operating instructions for Claude Code — read first |
| `docs/PRD.md` | What we're building, why, and how success is measured |
| `docs/TECH_STACK.md` | Every tool choice and the reasoning behind it |
| `docs/ARCHITECTURE.md` | System diagram, component responsibilities, data flow |
| `docs/ROADMAP.md` | Phased build plan with explicit Definition of Done |
| `docs/DATA_SPEC.md` | Dataset details, schema, known limitations |
| `docs/DECISIONS.md` | Running architecture decision log — your interview prep |

## Repository structure

```
fraud-detection-mlops/
├── CLAUDE.md
├── README.md
├── docs/
├── data/
│   ├── raw/            # dataset lands here after Kaggle download (gitignored)
│   └── processed/      # engineered features (DVC-tracked)
├── src/
│   ├── ingestion/       # producer + stream consumer
│   ├── features/          # feature store (online + offline), single source of truth
│   ├── training/           # model training, MLflow logging
│   ├── serving/              # FastAPI scoring service
│   ├── monitoring/            # drift detection, metrics
│   └── orchestration/          # Airflow DAGs
├── infra/
│   ├── docker/                  # Dockerfiles
│   └── k8s/                       # stretch-goal manifests (Phase 8)
├── tests/
├── notebooks/                       # exploration only
├── .github/workflows/                 # CI
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick start (once past Phase 0)

```bash
cp .env.example .env
docker-compose up -d
pip install -r requirements.txt
pytest
```

## Dataset

Kaggle "Credit Card Fraud Detection" (ULB/Worldline). Free account required.
Full details, schema, and a documented limitation (no native card ID — see
`docs/DATA_SPEC.md`) are covered before you touch any code.

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Status

Follow `docs/ROADMAP.md` for current phase. Update this section as phases
complete so the README stays an honest snapshot of project state.

- [ ] Phase 0 — Environment & repo setup
- [ ] Phase 1 — Data pipeline & feature store
- [ ] Phase 2 — Model training & experiment tracking
- [ ] Phase 3 — Real-time serving API
- [ ] Phase 4 — Streaming ingestion & online inference
- [ ] Phase 5 — Monitoring, drift detection & alerting
- [ ] Phase 6 — Automated retraining orchestration
- [ ] Phase 7 — CI/CD & containerization polish
- [ ] Phase 8 — *(stretch)* Kubernetes, autoscaling, shadow deployment
- [ ] Phase 9 — Documentation & demo polish
