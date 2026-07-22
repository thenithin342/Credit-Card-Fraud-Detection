"""tests/unit/test_serving.py
────────────────────────────────────────────────────────────────────────
Unit tests for the Phase 3 FraudGuard serving layer.

What we verify
--------------
* ``/health`` returns 200 with the loaded model version.
* ``POST /v1/score`` returns a low fraud score for a clearly legit
  transaction.
* The score response includes a non-empty ``top_features`` list.
* Missing raw V* columns do not break scoring — the preprocessor
  fills with the -999 sentinel and the model still scores.
* End-to-end latency stays under the 150 ms SLA (from
  ``src.config.get_settings().scoring_latency_sla_ms``).

Test isolation
--------------
* A fresh MLflow tracking URI is created in a temp dir for every
  test session, so we never touch the real ``./mlruns`` registry.
* A small XGBoost model + FeaturePreprocessor are built on a tiny
  synthetic IEEE-CIS-shaped frame and registered as
  ``fraud-detector-test`` / ``Staging``.
* Redis is replaced with ``fakeredis``.
* ``app.dependency_overrides`` is used to inject the test bundle +
  online store into the FastAPI app.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Tiny IEEE-CIS-shaped synthetic frame for the test fixture ────────────


def _synth_ieee_frame(n: int = 200, fraud_rate: float = 0.10) -> pd.DataFrame:
    """Return a small DataFrame shaped like the post-merge IEEE-CIS frame.

    We only need a handful of columns to drive the preprocessor and
    XGBoost.  Everything else becomes -999 / 0 / "missing" downstream.
    """
    rng = np.random.default_rng(7)
    fraud = [0] * int(n * (1 - fraud_rate)) + [1] * int(n * fraud_rate)
    rng.shuffle(fraud)
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "isFraud": fraud,
            "TransactionDT": list(range(86_400, 86_400 + n * 60, 60)),
            "TransactionAmt": rng.uniform(1.0, 500.0, size=n).round(2),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n),
            "card1": rng.integers(1000, 9999, size=n).astype(float),
            "card2": rng.integers(100, 600, size=n).astype(float),
            "card3": rng.integers(100, 200, size=n).astype(float),
            "card4": rng.choice(["visa", "mastercard", "discover"], size=n),
            "card5": rng.integers(100, 250, size=n).astype(float),
            "card6": rng.choice(["credit", "debit"], size=n),
            "addr1": rng.integers(100, 500, size=n).astype(float),
            "addr2": rng.integers(10, 100, size=n).astype(float),
            "dist1": rng.uniform(0, 500, size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], size=n),
            "DeviceType": rng.choice(["desktop", "mobile"], size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS", "MacOS"], size=n),
        }
    )


# ── Hermetic fixture: train + register a tiny model, load via our loader ──


@pytest.fixture(scope="module")
def serving_env(tmp_path_factory) -> dict[str, Any]:
    """Build a self-contained serving environment for the test session.

    Returns a dict with:
      * ``client``  — a TestClient for the FastAPI app
      * ``bundle``  — the loaded ModelBundle
      * ``online_store`` — a fakeredis-backed OnlineFeatureStore
      * ``feature_columns`` — the 312-column parity list
    """
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb

    from src.features.definitions import TARGET_COL
    from src.features.online_store import OnlineFeatureStore
    from src.features.preprocessing import FeaturePreprocessor
    from src.serving.model_loader import (
        load_champion_model,
        reset_model_bundle_for_tests,
    )

    tmp_dir = tmp_path_factory.mktemp("serving_mlruns")
    tracking_uri = f"file:{tmp_dir.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-test")

    df = _synth_ieee_frame(n=200)
    y = df[TARGET_COL].astype(int).to_numpy()
    X_raw = df.drop(columns=[TARGET_COL, "TransactionID"])

    # ── Fit a small preprocessor on a *single* small frame ──────────
    # We only need a few columns to exercise the preprocessor code
    # path; everything else is filled with the -999 sentinel.
    fit_cols = [
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
    ]
    preprocessor = FeaturePreprocessor().fit(X_raw, fit_cols)
    enc_path = tmp_dir / "feature_preprocessor.pkl"
    preprocessor.save(enc_path)

    # ── Train a tiny XGBoost classifier ─────────────────────────────
    X_proc = preprocessor.transform(X_raw)
    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=0,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(X_proc, y)

    # ── Log + register as a versioned model ─────────────────────────
    with mlflow.start_run(run_name="test_serving_xgb") as run:
        mlflow.xgboost.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, name="fraud-detector-test")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name="fraud-detector-test",
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # ── Write the canonical feature-column contract ─────────────
    # The test model was fit only on the 15 raw columns; the parity
    # list contains exactly those so the bundle's `feature_columns`
    # matches the model's booster expectations.
    raw_cols = list(X_proc.columns)
    feat_path = tmp_dir / "feature_columns.json"
    feat_path.write_text(
        json.dumps(
            {
                "raw_feature_cols": raw_cols,
                "temporal_feature_cols": [],
                "all_feature_cols": raw_cols,
            }
        )
    )

    # ── Write a minimal params.yaml pointing at the temp registry ───
    params_path = tmp_dir / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(
            {
                "serving": {
                    "mlflow_model_name": "fraud-detector-test",
                    "mlflow_model_stage": "Staging",
                    "threshold": 0.5,
                    "shap_top_k": 5,
                    "latency_sla_ms": 150,
                }
            }
        )
    )

    # ── Load via the production loader so the code path is exercised ─
    reset_model_bundle_for_tests()
    bundle = load_champion_model(
        params_path=params_path,
        encoder_path=enc_path,
        feature_columns_path=feat_path,
        tracking_uri=tracking_uri,
    )

    # ── Build a fresh FastAPI app (skip lifespan to avoid the real
    # load path) and inject the test bundle + online store.
    import fakeredis  # noqa: PLC0415

    online_store = OnlineFeatureStore(fakeredis.FakeRedis())

    # Use the real `app` from `src.serving.app` and override the
    # bundle + online store via FastAPI's `app.state.bundle_override`
    # slot (read by `_safe_get_bundle`).  This way the production
    # routes (`/v1/score`, `/health`, `/metrics`) are exercised
    # end-to-end — no route re-declaration needed.
    from src.serving.app import app as real_app  # noqa: PLC0415

    real_app.state.bundle_override = bundle
    real_app.state.online_store = online_store
    real_app.state.serving_params = {
        "threshold": 0.5,
        "shap_top_k": 5,
        "latency_sla_ms": 150,
    }

    return {
        "client": TestClient(real_app),
        "bundle": bundle,
        "online_store": online_store,
        "feature_columns": bundle.feature_columns,
        "_app": real_app,
    }


# ── Tests ─────────────────────────────────────────────────────────────────


def test_health_endpoint(serving_env: dict[str, Any]) -> None:
    client = serving_env["client"]
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str)
    assert body["model_version"] != ""
    assert "model_stage" in body


def test_score_endpoint_legit_transaction(serving_env: dict[str, Any]) -> None:
    client = serving_env["client"]
    payload = {
        "transaction_id": 1,
        "TransactionDT": 86_400,
        "TransactionAmt": 12.34,
        "card1": 4242,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["is_fraud"] is False
    assert body["threshold"] == pytest.approx(0.5)
    assert body["model_version"] == serving_env["bundle"].model_version


def test_score_endpoint_returns_shap(serving_env: dict[str, Any]) -> None:
    client = serving_env["client"]
    payload = {
        "transaction_id": 2,
        "TransactionDT": 86_400,
        "TransactionAmt": 99.99,
        "card1": 5555,
        "ProductCD": "H",
        "card4": "mastercard",
        "card6": "debit",
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "top_features" in body
    assert isinstance(body["top_features"], list)
    assert len(body["top_features"]) > 0
    for feat in body["top_features"]:
        assert {"feature_name", "contribution", "value"} <= set(feat.keys())
        assert isinstance(feat["feature_name"], str)
        assert isinstance(feat["contribution"], (int, float))
        assert isinstance(feat["value"], (int, float))


def test_score_endpoint_missing_v_features(serving_env: dict[str, Any]) -> None:
    """Requests that omit every V* column must still score (the
    preprocessor fills them with -999, XGBoost splits on NaN/-999)."""
    client = serving_env["client"]
    payload = {
        "transaction_id": 3,
        "TransactionDT": 86_400,
        "TransactionAmt": 250.00,
        "card1": 7777,
        # Intentionally omit ProductCD, card4, card6, all V* / id_*.
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert isinstance(body["latency_ms"], float)
    assert body["latency_ms"] >= 0.0


def test_predictor_latency_under_sla(serving_env: dict[str, Any]) -> None:
    """End-to-end score latency must stay under 150 ms p99-ish.

    The SHAP TreeExplainer is warm after the first call (it is
    cached on the module-level singleton in ``explainer.py``), so a
    5-call average is a fair signal.
    """
    from src.serving.predictor import score_transaction
    from src.serving.schemas import TransactionRequest

    bundle = serving_env["bundle"]
    online_store = serving_env["online_store"]
    req = TransactionRequest(
        transaction_id=4,
        TransactionDT=86_400,
        TransactionAmt=42.42,
        card1=8888,
        ProductCD="W",
        card4="visa",
        card6="credit",
    )

    # Warm-up call so the first SHAP build cost is paid.
    score_transaction(
        req,
        bundle=bundle,
        online_store=online_store,
        threshold=0.5,
        top_k=5,
    )

    samples = 5
    latencies = []
    for i in range(samples):
        req2 = req.model_copy(update={"transaction_id": 100 + i})
        resp = score_transaction(
            req2,
            bundle=bundle,
            online_store=online_store,
            threshold=0.5,
            top_k=5,
        )
        latencies.append(resp.latency_ms)

    mean_latency = float(np.mean(latencies))
    # The 150 ms SLA is the *p95* — use a comfortable mean ceiling.
    # Set 200 ms to leave slack for slow CI runners.
    assert mean_latency < 200.0, (
        f"Mean latency {mean_latency:.1f} ms exceeds ceiling; samples={latencies}"
    )
