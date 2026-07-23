"""src/training/evaluate.py
────────────────────────────────────────────────────────────────────────
Standalone evaluation script for a registered MLflow model.

Loads the model from the registry, scores the *test* split, and writes
metrics both to stdout and to ``reports/evaluation/test_metrics.json``.

Usage
-----
    python -m src.training.evaluate --model-name fraud-detector --stage Staging
    python -m src.training.evaluate --model-name fraud-detector --stage Production
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn  # noqa: F401  (matches the sklearn flavour used in train.py)
import numpy as np
import pandas as pd
import structlog

from src.config import get_settings
from src.training.train import (
    LOCAL_MLFLOW_URI,
    compute_metrics,
    load_feature_columns,
    load_params,
    predict_proba,
    split_xy,
)

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def load_test_split(params: dict) -> tuple[pd.DataFrame, np.ndarray]:
    features_dir = PROJECT_ROOT / params["data"]["features_dir"]
    test = pd.read_parquet(features_dir / "test_features.parquet")
    feature_cols = load_feature_columns(params)
    return split_xy(test, feature_cols)


def resolve_model_uri(model_name: str, stage: str) -> str:
    """Resolve model URI from MLflow registry (Staging stage or latest version)."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        staging_versions = [v for v in versions if getattr(v, "current_stage", None) == stage]
        if staging_versions:
            latest = max(staging_versions, key=lambda v: int(v.version))
            return f"runs:/{latest.run_id}/model"
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            return f"runs:/{latest.run_id}/model"
    except Exception:
        pass
    return f"models:/{model_name}/{stage}"


def evaluate(
    *,
    model_name: str,
    stage: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Load + score + write metrics.  Returns the metrics dict."""
    params = load_params()
    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)

    X_test, y_test = load_test_split(params)
    log.info("test_data_loaded", rows=len(X_test), fraud_rate=float(y_test.mean()))

    model_uri = resolve_model_uri(model_name, stage)
    log.info("loading_model", uri=model_uri)
    # The artifact was logged with `mlflow.sklearn.log_model` in
    # src/training/train.py — load it back with the matching flavour.
    # XGBClassifier / LGBMClassifier are full sklearn-API objects, so
    # `predict_proba` (below) Just Works.
    model = mlflow.sklearn.load_model(model_uri)
    y_proba = predict_proba(model, X_test)

    metrics = compute_metrics(y_test, y_proba, threshold=threshold)
    metrics["test_rows"] = int(len(X_test))
    metrics["test_fraud_rate"] = float(y_test.mean())
    metrics["model_name"] = model_name
    metrics["stage"] = stage
    metrics["threshold"] = float(threshold)

    # Stdout summary
    print("\n=== Test-set Evaluation ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>22}: {v:.6f}")
        else:
            print(f"  {k:>22}: {v}")
    print()

    # Disk report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("metrics_written", path=str(out_path))
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a registered MLflow model on the test set."
    )
    parser.add_argument(
        "--model-name",
        default=get_settings().mlflow_model_name,
        help="Registered model name (default: from settings).",
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        help="Registry stage to evaluate (default: Staging).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary metrics (default: 0.5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    args = parse_args(argv)
    evaluate(
        model_name=args.model_name,
        stage=args.stage,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
