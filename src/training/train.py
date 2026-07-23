"""src/training/train.py
────────────────────────────────────────────────────────────────────────
FraudGuard Phase 2B — train three classifiers and log everything to
MLflow.

Models
------
1. Logistic Regression (baseline)      — class_weight=balanced
2. XGBoost (primary)                   — scale_pos_weight from data
3. LightGBM (primary alt)              — is_unbalance=True

For each model we log:
    * params  : model type, hyperparameters, train/val row counts,
                fraud rate, feature_count
    * metrics : pr_auc, roc_auc, f1, precision_at_90_recall,
                average_precision
    * artifacts: classification_report.txt, confusion_matrix.png,
                 pr_curve.png, selected_features.json,
                 feature_importance.png (skipped for LogReg)
    * tags    : phase=2b, dataset=ieee_cis

After all three runs:
    * Pick the run with the highest `pr_auc`
    * Register that model in the MLflow Model Registry as
      `fraud-detector` at stage `Staging`.

MLflow is hard-wired to the local file backend (``./mlruns``) so the
script works with zero Docker dependencies.

Usage
-----
    python -m src.training.train
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — no DISPLAY required
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import get_settings
from src.features.definitions import TARGET_COL

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
LOCAL_MLFLOW_URI: str = (PROJECT_ROOT / "mlruns").as_uri()
REPORT_DIR = PROJECT_ROOT / "reports" / "training"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.json"

# Trust list for the skops audit that backs `mlflow.sklearn.log_model`.
# XGBClassifier / LGBMClassifier pickle references non-sklearn classes
# (xgboost.core.Booster, etc.) which skops refuses by default. Listing
# them here is the documented escape hatch — see MLflow docs for
# `mlflow.sklearn.log_model(..., skops_trusted_types=...)`.
_SKOPS_TRUSTED_TYPES: list[str] = [
    "xgboost.sklearn.XGBClassifier",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBModel",
    "xgboost.sklearn.XGBRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.basic.Booster",
    # LightGBM pickles contain a `collections.OrderedDict` (Python stdlib,
    # safe by definition) that the skops audit refuses by default.
    "collections.OrderedDict",
]

# Columns the model must NEVER see as input features.  These are
# dropped from the X matrix at training time.
_NON_FEATURE_COLS: tuple[str, ...] = ("TransactionID", "TransactionDT", TARGET_COL)


# ── Data loading ────────────────────────────────────────────────────────────


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_columns(params: dict) -> list[str]:
    """Read the parity contract from `models/feature_columns.json`.

    Falls back to ``params["features"]`` keys if the file is missing
    (defensive — should never happen in production).
    """
    # 1) prefer the on-disk contract written by offline_store.
    if FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, encoding="utf-8") as f:
            contract = json.load(f)
        cols = contract.get("all_feature_cols")
        if cols:
            return list(cols)
    # 2) Fallback: params.yaml's engineered-feature list (only 9 cols).
    log.warning("feature_columns_json_missing_using_fallback")
    return []  # caller will detect this and raise.


def load_feature_splits(params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the engineered feature parquets produced by offline_store."""
    features_dir = PROJECT_ROOT / params["data"]["features_dir"]
    train = pd.read_parquet(features_dir / "train_features.parquet")
    val = pd.read_parquet(features_dir / "val_features.parquet")
    test = pd.read_parquet(features_dir / "test_features.parquet")
    return train, val, test


def split_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a frame into (X, y) using the canonical feature column list.

    Drops the well-known non-feature columns (TransactionID,
    isFraud, TransactionDT) defensively.  Any column that is
    missing from the frame is skipped with a warning so that
    training does not crash on a stale parquet.
    """
    cols = [c for c in feature_cols if c not in _NON_FEATURE_COLS and c in df.columns]
    X = df[cols]
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the suite of metrics we log to MLflow.

    `y_proba` is the predicted probability of class=1.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # PR-AUC (area under the precision-recall curve) — primary metric.
    # Use sklearn's average_precision_score (numerically stable).
    pr_auc = float(average_precision_score(y_true, y_proba))

    roc_auc = float(roc_auc_score(y_true, y_proba))
    avg_precision = float(average_precision_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # precision at 90% recall — useful operating point for fraud:
    # if we must catch 90% of fraud, how precise can we be?
    precision_at_90_recall = float(_precision_at_recall(y_true, y_proba, target_recall=0.9))
    recall_at_threshold = float(recall_score(y_true, y_pred, zero_division=0))
    precision_at_threshold = float(precision_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "avg_precision": avg_precision,
        "precision_at_90_recall": precision_at_90_recall,
        "recall_at_0.5": recall_at_threshold,
        "precision_at_0.5": precision_at_threshold,
    }


def _precision_at_recall(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float) -> float:
    """Return the precision achievable at the lowest threshold whose
    recall is >= target_recall.  Returns 0.0 if unachievable."""
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
    if target_recall <= 0:
        return float(precision_arr[0])
    for i in range(len(recall_arr) - 1, -1, -1):
        if recall_arr[i] >= target_recall:
            return float(precision_arr[i])
    return 0.0


# ── Artifacts (plots + reports) ─────────────────────────────────────────────


def _make_confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (val, threshold=0.5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _make_pr_curve_png(y_true: np.ndarray, y_proba: np.ndarray, out_path: Path) -> None:
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall_arr, precision_arr, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (val)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _make_feature_importance_png(
    model: Any,
    feature_names: list[str],
    out_path: Path,
    top_k: int = 30,
) -> None:
    """Render a horizontal bar chart of the top ``top_k`` features by
    `model.feature_importances_`.  Skipped (no-op) for models that
    don't expose a feature_importances_ attribute.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    n = min(top_k, len(feature_names))
    # Sort by importance descending, take top-k.
    idx = np.argsort(importances)[::-1][:n]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals = importances[idx]
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.25)))
    # Plot bottom-up so the most important is on top.
    ax.barh(range(n), sorted_vals[::-1], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Model trainers ──────────────────────────────────────────────────────────


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: np.ndarray, seed: int, params: dict
) -> LogisticRegression:
    cfg = params["training"]["logreg"]
    model = LogisticRegression(
        max_iter=int(cfg.get("max_iter", 200)),
        C=float(cfg.get("C", 1.0)),
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    if len(X_train) > 50000:
        idx = np.random.default_rng(seed).choice(len(X_train), size=50000, replace=False)
        model.fit(X_train.iloc[idx], y_train[idx])
    else:
        model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    params: dict,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> Any:  # xgboost.XGBClassifier — avoid hard import for type stubs
    import xgboost as xgb

    cfg = params["training"]["xgb"]
    if "scale_pos_weight" in cfg:
        scale_pos_weight = float(cfg["scale_pos_weight"])
    else:
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    early_stopping_rounds = int(params["training"].get("early_stopping_rounds", 50))

    xgb_kwargs: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "max_depth": int(cfg.get("max_depth", 6)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "tree_method": "hist",
    }

    # Add optional tuned hyperparams if present in cfg
    for opt_key in (
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ):
        if opt_key in cfg:
            xgb_kwargs[opt_key] = type(cfg[opt_key])(cfg[opt_key])

    if X_val is not None and y_val is not None:
        xgb_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBClassifier(**xgb_kwargs)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int, params: dict) -> Any:
    import lightgbm as lgb

    cfg = params["training"]["lgbm"]
    model = lgb.LGBMClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=int(cfg.get("max_depth", 6)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        is_unbalance=True,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── MLflow helpers ──────────────────────────────────────────────────────────


def setup_mlflow(experiment_name: str) -> None:
    """Configure the local file backend.  Idempotent."""
    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)
    mlflow.set_experiment(experiment_name)


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict"):
        preds = model.predict(X)
        if isinstance(preds, pd.DataFrame):
            preds = preds.to_numpy()
        if hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] == 2:
            return preds[:, 1]
        return np.asarray(preds, dtype=float)
    raise RuntimeError(f"Model {type(model).__name__} has no predict_proba")


def log_run(
    *,
    run_name: str,
    model_type: str,
    model: Any,
    model_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Train+log one model; return the run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ── Params ──
        fraud_rate = float(np.mean(y_train))
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)
        mlflow.log_param("train_rows", int(len(X_train)))
        mlflow.log_param("val_rows", int(len(X_val)))
        mlflow.log_param("train_fraud_rate", round(fraud_rate, 6))
        mlflow.log_param("train_pos_count", n_pos)
        mlflow.log_param("train_neg_count", n_neg)
        mlflow.log_param("feature_count", int(len(feature_names)))
        if "scale_pos_weight" not in model_params and n_pos > 0:
            mlflow.log_param("scale_pos_weight", round(n_neg / n_pos, 6))

        # ── Metrics ──
        y_val_proba = predict_proba(model, X_val)
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_val, y_val_proba)
        mlflow.log_metrics(metrics)

        # ── Artifacts ──
        report_dir = REPORT_DIR / run_id
        report_dir.mkdir(parents=True, exist_ok=True)

        cls_report = classification_report(
            y_val, y_val_pred, target_names=["Legit", "Fraud"], zero_division=0
        )
        cls_path = report_dir / "classification_report.txt"
        cls_path.write_text(cls_report)
        mlflow.log_artifact(str(cls_path), artifact_path="reports")

        cm_path = report_dir / "confusion_matrix.png"
        _make_confusion_matrix_png(y_val, y_val_pred, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        pr_path = report_dir / "pr_curve.png"
        _make_pr_curve_png(y_val, y_val_proba, pr_path)
        mlflow.log_artifact(str(pr_path), artifact_path="plots")

        # Feature importance (skipped for LogReg which has no
        # .feature_importances_ attribute).
        imp_path = report_dir / "feature_importance.png"
        _make_feature_importance_png(model, feature_names, imp_path, top_k=30)
        if imp_path.exists():
            mlflow.log_artifact(str(imp_path), artifact_path="plots")

        # Selected features (the canonical feature list used to train).
        # This is a copy of models/feature_columns.json so the run
        # is self-describing.
        if FEATURE_COLUMNS_PATH.exists():
            sel_path = report_dir / "selected_features.json"
            sel_path.write_text(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
            mlflow.log_artifact(str(sel_path), artifact_path="reports")

        # Metrics json (handy for evaluation script)
        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        # ── Model ──
        # Always persist via `mlflow.sklearn.log_model`.  XGBClassifier
        # and LGBMClassifier are full sklearn-API objects, so this works
        # uniformly for all three model types AND avoids the
        # `mlflow.xgboost.log_model` TypeError on XGBoost>=2.0
        # (`_estimator_type undefined` when early_stopping_rounds is set).
        #
        # The matching load side is `mlflow.sklearn.load_model` — see
        # `src/serving/model_loader.py` and `src/training/evaluate.py`.
        # Keep the write flavour and the read flavour in sync.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=_SKOPS_TRUSTED_TYPES,
        )

        # ── Tags ──
        mlflow.set_tags(
            {
                "phase": "2b",
                "dataset": "ieee_cis",
                **(extra_tags or {}),
            }
        )

        log.info(
            "run_complete",
            run_id=run_id,
            model_type=model_type,
            pr_auc=metrics["pr_auc"],
            roc_auc=metrics["roc_auc"],
            f1=metrics["f1"],
            feature_count=len(feature_names),
        )
    return run_id


def register_best_model(
    candidate_runs: list[tuple[str, str, float]],
    model_name: str,
    stage: str = "Staging",
) -> str:
    """Pick the run with the highest pr_auc and register it.

    Parameters
    ----------
    candidate_runs : list of (model_type, run_id, pr_auc)
    model_name : registry model name
    stage : target stage (default "Staging")
    """
    best = max(candidate_runs, key=lambda r: r[2])
    model_type, run_id, pr_auc = best
    log.info(
        "registering_best",
        model_type=model_type,
        run_id=run_id,
        pr_auc=pr_auc,
        stage=stage,
    )

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=True,
    )
    log.info(
        "model_registered",
        name=model_name,
        version=mv.version,
        stage=stage,
    )
    return mv.version


# ── Entry point ─────────────────────────────────────────────────────────────


def run() -> None:
    params = load_params()
    training_cfg = params["training"]
    seed = int(training_cfg.get("random_seed", 42))
    experiment = training_cfg.get("mlflow_experiment", "fraud-detection-phase2")
    registry_name = get_settings().mlflow_model_name  # "fraud-detector"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow(experiment)

    # Load the parity contract (models/feature_columns.json).  If it
    # is missing we fail fast — training on the wrong feature set
    # is the most expensive mistake we can make.
    feature_names = load_feature_columns(params)
    if not feature_names:
        raise RuntimeError(
            "models/feature_columns.json is missing or empty.  "
            "Run `python -m src.features.offline_store` first."
        )

    train_df, val_df, _ = load_feature_splits(params)
    X_train, y_train = split_xy(train_df, feature_names)
    X_val, y_val = split_xy(val_df, feature_names)
    log.info(
        "data_loaded",
        train_rows=len(X_train),
        val_rows=len(X_val),
        feature_count=len(feature_names),
    )

    candidates: list[tuple[str, str, float]] = []

    # ── 1. Logistic Regression ──
    log.info("training_logreg")
    logreg = train_logistic_regression(X_train, y_train, seed, params)
    logreg_id = log_run(
        run_name="logistic_regression",
        model_type="logistic_regression",
        model=logreg,
        model_params={
            "max_iter": params["training"]["logreg"]["max_iter"],
            "C": params["training"]["logreg"]["C"],
            "class_weight": "balanced",
        },
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )
    candidates.append(("logistic_regression", logreg_id, _read_metric(logreg_id, "pr_auc")))

    # ── 2. XGBoost ──
    log.info("training_xgboost")
    xgb_model = train_xgboost(X_train, y_train, seed, params, X_val=X_val, y_val=y_val)
    xgb_id = log_run(
        run_name="xgboost",
        model_type="xgboost",
        model=xgb_model,
        model_params=dict(params["training"]["xgb"]),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )
    candidates.append(("xgboost", xgb_id, _read_metric(xgb_id, "pr_auc")))

    # ── 3. LightGBM ──
    log.info("training_lightgbm")
    lgbm = train_lightgbm(X_train, y_train, seed, params)
    lgbm_id = log_run(
        run_name="lightgbm",
        model_type="lightgbm",
        model=lgbm,
        model_params={
            "n_estimators": params["training"]["lgbm"]["n_estimators"],
            "max_depth": params["training"]["lgbm"]["max_depth"],
            "learning_rate": params["training"]["lgbm"]["learning_rate"],
            "is_unbalance": True,
        },
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )
    candidates.append(("lightgbm", lgbm_id, _read_metric(lgbm_id, "pr_auc")))

    # ── Register best ──
    register_best_model(candidates, model_name=registry_name, stage="Staging")

    log.info("training_pipeline_done", candidates=candidates)


def _read_metric(run_id: str, metric: str) -> float:
    """Read a single metric value back from MLflow."""
    client = mlflow.tracking.MlflowClient()
    val = client.get_run(run_id).data.metrics.get(metric)
    return float(val) if val is not None else 0.0


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    run()


if __name__ == "__main__":
    sys.exit(main() or 0)
