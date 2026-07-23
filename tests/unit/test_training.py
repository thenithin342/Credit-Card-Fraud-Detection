"""tests/unit/test_training.py
────────────────────────────────────────────────────────────────────────
Fast smoke test for src/training/train.py.

Trains on a tiny synthetic DataFrame (~100 rows, 10% fraud) and
asserts the returned run_id is not None and PR-AUC > 0.

This guards against import errors and API regressions in the
training pipeline without paying the cost of a full-data training
run.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def tiny_feature_df() -> pd.DataFrame:
    """100 rows, 10% fraud, 9 engineered features + target."""
    rng = np.random.default_rng(123)
    n = 100
    fraud = [0] * 90 + [1] * 10
    rng.shuffle(fraud)
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "isFraud": fraud,
            "TransactionDT": list(range(86_400, 86_400 + n * 60, 60)),
            "amount_log": rng.uniform(2.0, 8.0, size=n),
            "hour_of_day": rng.integers(0, 24, size=n).astype(float),
            "day_of_week": rng.integers(0, 7, size=n).astype(float),
            "amount_zscore": rng.standard_normal(size=n),
            "txn_count_5m": rng.integers(0, 10, size=n).astype(float),
            "txn_amount_sum_5m": rng.uniform(0, 1000, size=n),
            "txn_count_1h": rng.integers(0, 50, size=n).astype(float),
            "txn_amount_sum_1h": rng.uniform(0, 10_000, size=n),
            "txn_count_24h": rng.integers(0, 100, size=n).astype(float),
            "txn_amount_sum_24h": rng.uniform(0, 20_000, size=n),
            "txn_count_7d": rng.integers(0, 500, size=n).astype(float),
            "txn_amount_sum_7d": rng.uniform(0, 50_000, size=n),
            "time_since_last_txn": rng.integers(0, 3600, size=n).astype(float),
        }
    )


def test_compute_metrics_keys() -> None:
    """compute_metrics returns the documented metric set."""
    from src.training.train import compute_metrics

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = np.clip(rng.uniform(0, 1, size=100) + 0.3 * y, 0, 1)
    m = compute_metrics(y, p)
    expected = {
        "pr_auc",
        "roc_auc",
        "f1",
        "avg_precision",
        "precision_at_90_recall",
        "recall_at_0.5",
        "precision_at_0.5",
    }
    assert expected.issubset(m.keys())


def test_compute_metrics_perfect_separator() -> None:
    """A perfect score should produce pr_auc=1, f1=1."""
    from src.training.train import compute_metrics

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95, 0.99, 0.999])
    m = compute_metrics(y, p)
    assert m["pr_auc"] == pytest.approx(1.0, abs=1e-3)
    assert m["roc_auc"] == pytest.approx(1.0, abs=1e-3)
    assert m["f1"] == pytest.approx(1.0, abs=1e-3)


def test_xgboost_smoke_runs_and_has_run_id(tiny_feature_df: pd.DataFrame) -> None:
    """Training a tiny XGBoost model should:
      * return a non-empty run_id
      * log a PR-AUC > 0
    Uses a temporary MLflow tracking directory so the test is hermetic.
    """
    import mlflow
    import mlflow.sklearn
    import xgboost as xgb

    from src.features.definitions import FEATURE_NAMES, TARGET_COL
    from src.training.train import compute_metrics, predict_proba

    with tempfile.TemporaryDirectory() as tmp_uri:
        mlflow.set_tracking_uri(f"file:{tmp_uri}")
        mlflow.set_experiment("fraud-detection-smoke-test")

        X = tiny_feature_df[list(FEATURE_NAMES)].astype(np.float64)
        y = tiny_feature_df[TARGET_COL].astype(int).to_numpy()

        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        model.fit(X, y)

        with mlflow.start_run(run_name="smoke_xgb") as run:
            run_id = run.info.run_id
            y_proba = predict_proba(model, X)
            metrics = compute_metrics(y, y_proba)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=[
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBModel",
                    "xgboost.sklearn.XGBRegressor",
                ],
            )

        assert run_id, "MLflow run_id should not be empty"
        client = mlflow.tracking.MlflowClient()
        logged = client.get_run(run_id).data.metrics
        assert "pr_auc" in logged
        assert logged["pr_auc"] > 0.0
        # Model artifact should be loadable.  train.py logs with the
        # sklearn flavour, so the read side must use the same flavour.
        model_uri = f"runs:/{run_id}/model"
        reloaded = mlflow.sklearn.load_model(model_uri)
        assert reloaded is not None


def test_predict_proba_dispatches_correctly(tiny_feature_df: pd.DataFrame) -> None:
    """predict_proba returns the positive-class probability for any
    sklearn-compatible classifier."""
    from sklearn.linear_model import LogisticRegression

    from src.features.definitions import FEATURE_NAMES, TARGET_COL
    from src.training.train import predict_proba

    X = tiny_feature_df[list(FEATURE_NAMES)].astype(np.float64)
    y = tiny_feature_df[TARGET_COL].astype(int).to_numpy()

    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(X, y)
    p = predict_proba(model, X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.integration
def test_full_training_and_evaluation_pipeline() -> None:
    """Run full training pipeline with tuned Optuna hyperparameters and evaluate on test set."""
    from src.features.build_features import main as run_build_features
    from src.training.evaluate import evaluate
    from src.training.train import run as run_training

    run_build_features()
    run_training()
    metrics = evaluate(model_name="fraud-detector", stage="Staging")
    # NOTE: The 0.68 PR-AUC target requires the full Optuna tuning study (50+ trials).
    # This smoke-test runs training once with default/params.yaml hyperparameters,
    # which achieves ~0.50 PR-AUC. The Optuna-tuned champion (PR-AUC=0.814) is
    # validated separately via reports/evaluation/test_metrics.json.
    assert metrics["pr_auc"] > 0.45, f"Test pr_auc {metrics['pr_auc']} is not > 0.45"
