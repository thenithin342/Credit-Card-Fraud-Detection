"""src/serving/model_loader.py
────────────────────────────────────────────────────────────────────────
Phase 3 — Champion-model loader for the FraudGuard scoring API.

Responsibilities
----------------
* Connect to the local MLflow file backend (``./mlruns``).
* Resolve the "champion" XGBoost model from the Model Registry by
  name + stage (default name = ``fraud-detector`` / stage = ``Staging``;
  both come from ``params.yaml``).
* Load the fitted ``FeaturePreprocessor`` from
  ``models/encoders/feature_preprocessor.pkl``.
* Load the canonical feature-column list from
  ``models/feature_columns.json`` (the parity contract written by
  ``src.features.offline_store``).
* Bundle everything in an immutable ``ModelBundle`` so the serving
  path receives one self-contained object.

Thread safety
-------------
The bundle is loaded **once per process**.  We use the
double-checked-locking pattern around a module-level ``_BUNDLE`` so
the FastAPI worker (which may serve many concurrent requests) reads
the same object without re-loading from disk.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn  # noqa: F401  (registers the sklearn flavour used to load the registered model)
import structlog
import yaml

from src.features.preprocessing import FeaturePreprocessor

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoders" / "feature_preprocessor.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.json"

# Local file backend — no Docker dependency.
LOCAL_MLFLOW_URI: str = "file:./mlruns"

# Process-wide singleton state.
_BUNDLE: ModelBundle | None = None
_BUNDLE_LOCK = threading.Lock()


# ── Bundle dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelBundle:
    """Immutable bundle of everything the scoring pipeline needs.

    `feature_columns` is the *ordered* list the XGBoost model expects
    (from `models/feature_columns.json` → `all_feature_cols`).  It is
    312 columns long: 303 raw + 9 temporal.  Both the preprocessor
    output and the temporal feature block are aligned to this order
    before prediction.
    """

    model: Any  # xgboost.XGBClassifier — typed as Any to avoid a hard import.
    preprocessor: FeaturePreprocessor
    feature_columns: list[str] = field(default_factory=list)
    model_version: str = ""
    model_stage: str = ""


# ── Loader ──────────────────────────────────────────────────────────────────


def _read_serving_params(params: dict) -> tuple[str, str]:
    """Extract (model_name, model_stage) from the loaded params dict."""
    serving_cfg = params.get("serving", {}) if isinstance(params, dict) else {}
    name = str(serving_cfg.get("mlflow_model_name", "fraud-detector"))
    stage = str(serving_cfg.get("mlflow_model_stage", "Staging"))
    return name, stage


def _resolve_model_version(name: str, stage: str) -> str:
    """Return the registry version number of the model at ``stage``."""
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(name, stages=[stage])
    if not versions:
        raise RuntimeError(
            f"No registered model found for name={name!r} stage={stage!r}. "
            "Run `python -m src.training.train` first."
        )
    return str(versions[0].version)


def load_champion_model(
    *,
    params_path: Path | None = None,
    encoder_path: Path | None = None,
    feature_columns_path: Path | None = None,
    tracking_uri: str | None = None,
) -> ModelBundle:
    """Load the champion XGBoost model + preprocessor + feature list.

    All arguments are keyword-only and default to the project-standard
    locations; they exist so tests can redirect every path to a
    hermetic temp directory.

    Raises
    ------
    RuntimeError
        If the named registered model is missing at the requested
        stage, or the on-disk preprocessor / feature columns cannot
        be found.
    """
    params_path = params_path or PARAMS_FILE
    encoder_path = encoder_path or ENCODER_PATH
    feature_columns_path = feature_columns_path or FEATURE_COLUMNS_PATH

    # Allow test code to redirect the tracking URI without mutating
    # the real registry by passing ``tracking_uri=`` explicitly.
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI") in (None, "", "http://localhost:5000"):
        # Default to the local file backend so the API works without
        # an MLflow tracking server.  Tests that set MLFLOW_TRACKING_URI
        # explicitly (or pass tracking_uri=) take precedence.
        mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)

    try:
        with open(params_path, encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
    except FileNotFoundError:
        params = {}
    name, stage = _read_serving_params(params)
    log.info(
        "loading_champion_model", name=name, stage=stage, tracking_uri=mlflow.get_tracking_uri()
    )

    # ── Load the XGBoost model from the registry ───────────────────────
    # Use `models:/<name>/<stage>` so the URI is resolved via the
    # registry rather than hard-coding a version number.
    # NOTE: the artifact is persisted via `mlflow.sklearn.log_model`
    # (see src/training/train.py), so it must be rehydrated with
    # `mlflow.sklearn.load_model`. XGBClassifier is a full sklearn-API
    # object on the way back, so the rest of the serving code can keep
    # calling `predict_proba`/`predict` on it.
    model_uri = f"models:/{name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load model {model_uri!r}: {exc}. "
            "Check that 'mlflow_tracking_uri' points to the directory "
            "containing the registered model."
        ) from exc
    model_version = _resolve_model_version(name, stage)

    # ── Load the fitted preprocessor ───────────────────────────────────
    if not encoder_path.exists():
        raise RuntimeError(
            f"FeaturePreprocessor not found at {encoder_path}. "
            "Run `python -m src.features.offline_store` first."
        )
    preprocessor = FeaturePreprocessor.load(encoder_path)

    # ── Load the canonical feature-column list ────────────────────────
    if not feature_columns_path.exists():
        raise RuntimeError(
            f"Feature-column parity contract not found at {feature_columns_path}. "
            "Run `python -m src.features.offline_store` first."
        )
    with open(feature_columns_path, encoding="utf-8") as f:
        feature_contract = json.load(f)
    feature_columns: list[str] = list(feature_contract.get("all_feature_cols", []))
    if not feature_columns:
        raise RuntimeError(f"`all_feature_cols` missing from {feature_columns_path}.")

    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        model_version=model_version,
        model_stage=stage,
    )
    log.info(
        "model_loaded",
        name=name,
        stage=stage,
        version=model_version,
        n_features=len(feature_columns),
    )
    return bundle


# ── Thread-safe singleton ──────────────────────────────────────────────────


def get_model_bundle() -> ModelBundle:
    """Return the process-wide cached ``ModelBundle`` (loads on first call).

    Uses double-checked locking so concurrent FastAPI workers race to
    load exactly once and then read the cached value forever.
    """
    global _BUNDLE
    if _BUNDLE is None:
        with _BUNDLE_LOCK:
            if _BUNDLE is None:  # second check inside the lock
                _BUNDLE = load_champion_model()
    return _BUNDLE


def set_model_bundle_for_tests(bundle: ModelBundle) -> None:
    """Inject a pre-built ``ModelBundle`` (used by the test suite)."""
    global _BUNDLE
    with _BUNDLE_LOCK:
        _BUNDLE = bundle


def reset_model_bundle_for_tests() -> None:
    """Clear the cached bundle so the next call reloads from disk."""
    global _BUNDLE
    with _BUNDLE_LOCK:
        _BUNDLE = None
