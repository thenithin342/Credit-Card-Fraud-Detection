# FraudGuard 🛡️
### Real-Time Credit Card Fraud Detection Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Detect fraudulent card transactions **in under 100ms** using a streaming feature pipeline, gradient boosting models, and a full MLOps loop — from data validation through automated drift-triggered retraining.

---

## Architecture

```
[Transaction Stream Producer (Redpanda)]
        │
        ▼
[Stream Processor: windowed feature computation]
        │
        ├──▶ [Feature Store (Feast + Redis online store)]
        │
        ▼
[Model Serving API (FastAPI — loads from MLflow Registry)]
        │
        ├──▶ [Prediction + SHAP explanation]
        │
        ▼
[Prediction Log Sink → Postgres]
        │
        ▼
[Monitoring: Evidently AI drift + Prometheus/Grafana]
        │
        ▼
[Airflow DAG: nightly drift check → conditional retrain → registry promotion]
```

---

## Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Foundation & Data Pipeline | ✅ **Done** | Repo scaffold, DVC, GE validation |
| 2 — Feature Engineering | 🔜 Upcoming | Feast feature store, windowed aggregates |
| 3 — Model Training | 🔜 Upcoming | XGBoost/LightGBM, Optuna, MLflow |
| 4 — Model Serving | 🔜 Upcoming | FastAPI, SHAP, sub-100ms |
| 5 — Streaming Pipeline | 🔜 Upcoming | Redpanda producer/consumer |
| 6 — Containerization & K8s | 🔜 Upcoming | Docker, kind, HPA |
| 7 — CI/CD & Monitoring | 🔜 Upcoming | GitHub Actions, Grafana, Evidently |
| 8 — Automated Retraining | 🔜 Upcoming | Airflow DAG, canary rollout |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Explainability | SHAP |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Data Validation | Great Expectations |
| Feature Store | Feast + Redis |
| Serving | FastAPI + Uvicorn |
| Streaming | Redpanda (Kafka-compatible) |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Orchestration | Apache Airflow |
| Containerization | Docker, Docker Compose |
| Orchestration (prod) | Kubernetes (kind locally) |
| CI/CD | GitHub Actions |

---

## Phase 1 Quick Start

### 1. Prerequisites

- Python 3.11+
- Git
- Kaggle account with API key

### 2. Clone & Create Virtual Environment

```bash
git clone <your-repo-url> fraudguard
cd fraudguard

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your KAGGLE_USERNAME and KAGGLE_KEY
```

Get your Kaggle API key:
1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. This downloads `kaggle.json` — you can either place it at `~/.kaggle/kaggle.json`
   OR copy the `username` and `key` values into your `.env` file

**Important**: Before downloading IEEE-CIS, you must accept the competition rules:
→ Visit [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) and click **"Join Competition"**

### 5. Initialize DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialize DVC"
```

### 6. Download Datasets

```bash
# Download both datasets (~500MB total)
python -m src.ingestion.download

# Or individually:
python -m src.ingestion.download --dataset ieee
python -m src.ingestion.download --dataset ulb

# Verify files are present:
python -m src.ingestion.download --check
```

### 7. Run Data Validation

```bash
python -m src.validation.ge_suite
```

Expected output: `all_validations_passed`

### 8. Run the Full DVC Pipeline

```bash
dvc repro
```

This runs all three stages: `ingest → validate → split`

Outputs:
- `data/raw/ieee-cis/` — raw IEEE-CIS CSV files (DVC-tracked)
- `data/raw/ulb/` — raw ULB CSV file (DVC-tracked)
- `data/processed/train.parquet` — ~400k rows
- `data/processed/val.parquet` — ~57k rows
- `data/processed/test.parquet` — ~115k rows
- `reports/validation/` — validation text reports

### 9. Run Tests

```bash
# Unit + data validation tests
pytest tests/ -v

# Data validation only (requires downloaded data)
pytest tests/data_validation/ -v
```

---

## Folder Structure

```
fraudguard/
├── data/
│   ├── raw/          ← DVC-tracked (not in git)
│   ├── processed/    ← DVC-tracked (not in git)
│   └── features/     ← DVC-tracked (Phase 2)
├── src/
│   ├── ingestion/    ← download.py, split.py
│   ├── validation/   ← ge_suite.py
│   ├── features/     ← Phase 2
│   ├── training/     ← Phase 3
│   ├── evaluation/   ← Phase 3
│   ├── serving/      ← Phase 4
│   └── monitoring/   ← Phase 7
├── tests/
│   ├── data_validation/
│   ├── unit/         ← Phase 2+
│   └── integration/  ← Phase 4+
├── reports/
│   └── validation/   ← GE reports
├── airflow/dags/     ← Phase 8
├── k8s/              ← Phase 6
├── .github/workflows/ ← Phase 7
├── dvc.yaml          ← pipeline definition
├── params.yaml       ← all parameters
├── pyproject.toml    ← dependencies + tool config
├── .env.example      ← env var template
└── README.md
```

---

## Key Design Decisions

### Why temporal split instead of random shuffle?
Transactions are sorted by `TransactionDT` before splitting. The test set always contains the **most recent** transactions. This mirrors real deployment: you train on historical data and score future data. A random shuffle would leak future information into the training set.

### Why AUC-PR instead of ROC-AUC?
With a ~0.17% fraud rate, a model predicting "never fraud" achieves 99.83% accuracy and a high ROC-AUC. AUC-PR (area under the Precision-Recall curve) focuses on performance on the **positive class** and is not inflated by the large number of true negatives.

### Why Redpanda instead of Kafka?
Redpanda is Kafka-API compatible but runs as a single binary with no JVM dependency, making local development much lighter. Switching to MSK (managed Kafka) in production requires zero code changes — only connection string updates.

---

## License

MIT — see [LICENSE](LICENSE)
