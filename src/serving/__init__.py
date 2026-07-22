"""src/serving package
────────────────────────────────────────────────────────────────────────
Phase 3 — Real-time scoring package for FraudGuard.

This package ships a FastAPI app that loads the champion XGBoost
model from the local MLflow registry, enriches requests with the
Redis-backed online feature store, returns a SHAP-explained
decision, and exposes Prometheus metrics.

The top-level ``app`` instance is what ``uvicorn`` (and the
``fraudguard-serving`` console script) launch.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

__all__ = ["app", "create_app", "score_transaction"]

from src.serving.app import app, create_app
from src.serving.predictor import score_transaction
