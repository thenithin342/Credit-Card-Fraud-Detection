<div align="center">

# 🛡️ FraudGuard

### Enterprise-Grade Real-Time Credit Card Fraud Detection

[![CI](https://github.com/thenithin342/Credit-Card-Fraud-Detection/actions/workflows/ci.yaml/badge.svg)](https://github.com/thenithin342/Credit-Card-Fraud-Detection/actions/workflows/ci.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**PR-AUC: 0.8143** &nbsp;|&nbsp; **ROC-AUC: 0.9215** &nbsp;|&nbsp; **P99 Latency: 31.4ms** &nbsp;|&nbsp; **40/40 Tests Passing**

</div>

---

## 📋 Overview

FraudGuard is a production-ready fraud detection system built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (~590K transactions, 394 features). It covers the complete ML lifecycle — from raw data ingestion through real-time serving — with strict latency SLAs, SHAP explainability, and a zero-Docker local runtime.

### Key Features

- 🚀 **Sub-32ms P99 scoring** via an optimised XGBoost booster + TreeSHAP pipeline
- 🧠 **Optuna hyperparameter tuning** — 50-trial study raised PR-AUC from 0.087 → **0.8143**
- 🔍 **SHAP explainability** — every prediction returns top-5 feature attributions
- 🏪 **Online feature store** — Redis/fakeredis-backed card velocity windows (5m, 1h, 24h, 7d)
- 📊 **Prometheus metrics** — latency histograms and request counters at `/metrics`
- 🧪 **40/40 unit tests** — full coverage of features, training, serving, and parity checks

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────┐
                         │         Client Application      │
                         └────────────────┬────────────────┘
                                          │  POST /v1/score
                                          ▼
                         ┌─────────────────────────────────┐
                         │      FastAPI Scoring Engine      │
                         │   /health  /v1/score  /metrics  │
                         └──────┬──────────────────┬───────┘
                                │                  │
              1. Static &       │                  │  2. Card Velocity
              Categorical       │                  │     (Redis TTL)
              Features          ▼                  ▼
                    ┌─────────────────┐   ┌──────────────────┐
                    │FeaturePreprocess│   │ OnlineFeatureStore│
                    │ Ordinal + Freq  │   │  fakeredis / Redis│
                    └────────┬────────┘   └────────┬─────────┘
                             │                     │
                             └──────────┬──────────┘
                                        │  ~312 aligned features
                                        ▼
                            ┌───────────────────────┐
                            │  Champion XGBoost Model│
                            │  + SHAP TreeExplainer  │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  ScoreResponse <150ms  │
                            │  • fraud_probability   │
                            │  • is_fraud decision   │
                            │  • top_5 SHAP features │
                            │  • latency_ms          │
                            └───────────────────────┘
```

---

## 📈 Model Performance

| Model | Val PR-AUC | Test PR-AUC | ROC-AUC | Status |
|:------|:----------:|:-----------:|:-------:|:------:|
| Logistic Regression (baseline) | 0.412 | 0.395 | 0.785 | Below target |
| LightGBM | 0.792 | 0.782 | 0.921 | ✅ Passed |
| **XGBoost + Optuna (champion)** | **0.829** | **0.8143** | **0.922** | ✅ **Production** |

### Optuna Best Hyperparameters

| Parameter | Value | Effect |
|:----------|:-----:|:-------|
| `n_estimators` | 700 | Deeper convergence under low LR |
| `max_depth` | 9 | Complex fraud interaction trees |
| `learning_rate` | 0.138 | Faster convergence |
| `scale_pos_weight` | 7.81 | Handles 3.5% fraud class imbalance |
| `min_child_weight` | 7 | Regularises rare fraud leaf splits |
| `gamma` | 0.928 | Minimum loss reduction per split |

### Latency SLA

| Percentile | Latency | SLA |
|:----------:|:-------:|:---:|
| P50 | 18.2ms | < 150ms ✅ |
| P95 | 23.8ms | < 150ms ✅ |
| P99 | **31.4ms** | < 150ms ✅ |

---

## 🗂️ Project Structure

```
fraudguard/
├── src/
│   ├── config.py                  # Pydantic settings (env-driven)
│   ├── ingestion/
│   │   ├── download.py            # Kaggle dataset downloader
│   │   └── split.py              # Temporal train/val/test split
│   ├── features/
│   │   ├── definitions.py         # Feature name contracts & constants
│   │   ├── build_features.py      # Unified feature engineering entry-point
│   │   ├── offline_store.py       # Parquet feature store builder
│   │   ├── online_store.py        # Redis/fakeredis card velocity store
│   │   ├── preprocessing.py       # Ordinal + frequency encoding
│   │   └── selection.py           # Null / variance / correlation filtering
│   ├── training/
│   │   ├── train.py               # Train LogReg + XGBoost + LightGBM
│   │   ├── tune_optuna.py         # 50-trial Optuna HPO study
│   │   ├── evaluate.py            # Test-set evaluation → test_metrics.json
│   │   └── split.py               # Train/val/test split utilities
│   ├── serving/
│   │   ├── app.py                 # FastAPI application
│   │   ├── predictor.py           # End-to-end scoring logic
│   │   ├── model_loader.py        # MLflow champion model loader
│   │   ├── explainer.py           # SHAP TreeExplainer wrapper
│   │   └── schemas.py             # Pydantic request / response schemas
│   └── evaluation/
│       └── evaluate.py            # Standalone evaluation script
├── tests/
│   ├── conftest.py                # Shared fixtures + env configuration
│   └── unit/
│       ├── test_config.py
│       ├── test_features.py
│       ├── test_parity.py         # Online/offline feature parity
│       ├── test_selection.py
│       ├── test_serving.py        # FastAPI + latency SLA tests
│       ├── test_split.py
│       └── test_training.py
├── data/                          # DVC-tracked (not in git)
├── models/
│   ├── encoders/                  # Fitted preprocessor pickle
│   └── feature_columns.json       # Ordered feature parity contract
├── reports/
│   └── evaluation/
│       └── test_metrics.json      # Champion model test metrics
├── params.yaml                    # Optuna best hyperparameters
├── dvc.yaml                       # DVC pipeline stages
├── pyproject.toml                 # Dependencies + tool config
└── review_and_improvements_report.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (`.venv`)

### 1. Clone & Setup

```bash
git clone https://github.com/thenithin342/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Download Data

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
python -m src.ingestion.download
```

### 3. Run the Full Pipeline

```bash
# Step 1 — Temporal split
python -m src.ingestion.split

# Step 2 — Build offline features
python -m src.features.build_features

# Step 3 — (Optional) Optuna hyperparameter tuning (~1 hour)
python -m src.training.tune_optuna

# Step 4 — Train all model candidates
MLFLOW_ALLOW_FILE_STORE=true python -m src.training.train

# Step 5 — Evaluate champion on test set
MLFLOW_ALLOW_FILE_STORE=true python -m src.evaluation.evaluate

# Step 6 — Launch the scoring API
MLFLOW_ALLOW_FILE_STORE=true uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### 4. Score a Transaction

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 3000000,
    "TransactionAmt": 117.50,
    "ProductCD": "W",
    "card1": 4921,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  }'
```

**Response:**
```json
{
  "transaction_id": 3000000,
  "fraud_probability": 0.0312,
  "is_fraud": false,
  "top_shap_features": [
    {"feature": "amount_zscore",    "shap_value": -0.412},
    {"feature": "txn_count_1h",     "shap_value": -0.198},
    {"feature": "TransactionAmt",   "shap_value": -0.087}
  ],
  "latency_ms": 18.4,
  "model_version": "17"
}
```

---

## 🧪 Running Tests

```bash
# Activate venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

# Run full suite (40 tests, ~9 minutes)
pytest tests/unit/ -v

# Run fast tests only (skip full training pipeline, ~40 seconds)
pytest tests/unit/ -v --ignore=tests/unit/test_training.py
```

**Latest results:**
```
================= 40 passed, 29 warnings in 552.75s =================
```

---

## 🔌 API Reference

### `GET /health`
Returns service liveness and loaded model version.

```json
{
  "status": "ok",
  "model_name": "fraud-detector",
  "model_version": "17",
  "model_stage": "Production"
}
```

### `POST /v1/score`
Score a single transaction. Returns probability, decision, and top SHAP attributions.

| Field | Type | Description |
|:------|:----:|:------------|
| `TransactionAmt` | float | Transaction amount in USD |
| `ProductCD` | string | Product category (W/H/C/S/R) |
| `card1` – `card6` | mixed | Card network identifiers |
| `P_emaildomain` | string | Purchaser email domain |
| `V*` fields | float | Vesta-engineered identity features |

### `GET /metrics`
Prometheus exposition endpoint. Exposes request latency histograms and counters.

---

## ⚙️ Configuration

All settings are driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|:---------|:-------:|:------------|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow backend URI |
| `MLFLOW_ALLOW_FILE_STORE` | `false` | Required `true` for local file store |
| `USE_FAKEREDIS` | `false` | Use in-memory Redis for testing |
| `REDIS_URL` | `redis://localhost:6379` | Production Redis connection |
| `MODEL_NAME` | `fraud-detector` | MLflow registered model name |
| `MODEL_STAGE` | `Production` | Model stage to load |

---

## 🔬 Feature Engineering

### Static Features (3)
Computed per-transaction at serving time:

| Feature | Formula |
|:--------|:--------|
| `amount_log` | `log1p(TransactionAmt)` |
| `hour_of_day` | `TransactionDT % 86400 // 3600` |
| `day_of_week` | `(TransactionDT // 86400) % 7` |

### Card Velocity Windows (10)
Retrieved from Redis (online) or computed from history (offline):

| Window | Features |
|:-------|:---------|
| 5 minutes | `txn_count_5m`, `txn_amount_sum_5m` |
| 1 hour | `txn_count_1h`, `txn_amount_sum_1h` |
| 24 hours | `txn_count_24h`, `txn_amount_sum_24h` |
| 7 days | `txn_count_7d`, `txn_amount_sum_7d` |
| All-time | `amount_zscore`, `time_since_last_txn` |

### Raw Selected Features (~287)
V*, id_*, card*, C*, D*, M*, addr*, dist* columns surviving:
- High-null filter (> 80% null → dropped)
- Low-variance filter (< 0.01 var → dropped)
- Correlation filter (ρ > 0.95 → drop one of pair)

---

## 📦 Dependencies

| Package | Version | Purpose |
|:--------|:-------:|:--------|
| `xgboost` | ≥2.0 | Champion model |
| `lightgbm` | ≥4.0 | Challenger model |
| `scikit-learn` | ≥1.4 | Preprocessing, metrics |
| `optuna` | ≥3.0 | Hyperparameter optimisation |
| `shap` | ≥0.45 | TreeSHAP explainability |
| `mlflow` | ≥2.9 | Experiment tracking & model registry |
| `fastapi` | ≥0.110 | Scoring REST API |
| `fakeredis` | ≥2.20 | In-memory Redis for testing |
| `prometheus-fastapi-instrumentator` | ≥7.0 | Prometheus metrics |
| `structlog` | ≥24.0 | Structured JSON logging |

---

## 📄 Review Report

A full audit report covering all bugs found, fixes applied, and optimisation results is available at:

📋 **[`review_and_improvements_report.md`](review_and_improvements_report.md)**

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ · Powered by XGBoost, FastAPI, MLflow & Optuna

</div>
