"""
src/training/tune_optuna.py
────────────────────────────────────────────────────────────────────────
Optuna hyperparameter optimization script for XGBoost on FraudGuard.

Features:
  - Tunes XGBoost hyperparameters using Optuna (n_trials: 50, metric: average_precision).
  - Searches scale_pos_weight in [1.0, 10.0] to calibrate class weights and prevent precision collapse.
  - Passes eval_set and early_stopping_rounds to XGBoost fit.
  - Logs the study and best trial to MLflow (file:./mlruns).
  - Updates params.yaml with the best parameters.

Usage:
  python -m src.training.tune_optuna
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import structlog
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score

from src.training.train import (
    LOCAL_MLFLOW_URI,
    load_feature_columns,
    load_feature_splits,
    load_params,
    split_xy,
)

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def save_params(params: dict) -> None:
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    log.info("params_yaml_updated", path=str(PARAMS_FILE))


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    early_stopping_rounds: int,
) -> float:
    # Hyperparameter search space
    params_trial = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = xgb.XGBClassifier(
        n_estimators=params_trial["n_estimators"],
        max_depth=params_trial["max_depth"],
        learning_rate=params_trial["learning_rate"],
        subsample=params_trial["subsample"],
        colsample_bytree=params_trial["colsample_bytree"],
        min_child_weight=params_trial["min_child_weight"],
        scale_pos_weight=params_trial["scale_pos_weight"],
        gamma=params_trial["gamma"],
        reg_alpha=params_trial["reg_alpha"],
        reg_lambda=params_trial["reg_lambda"],
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    score = float(average_precision_score(y_val, y_val_proba))
    return score


def run_optuna_tuning() -> None:
    params = load_params()
    tuning_cfg = params.get("tuning", {})
    training_cfg = params.get("training", {})

    n_trials = int(tuning_cfg.get("n_trials", 50))
    timeout = tuning_cfg.get("timeout_seconds", 3600)
    direction = tuning_cfg.get("direction", "maximize")
    seed = int(training_cfg.get("random_seed", 42))
    early_stopping_rounds = int(training_cfg.get("early_stopping_rounds", 50))
    experiment_name = training_cfg.get("mlflow_experiment", "fraud-detection-phase2")

    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    feature_names = load_feature_columns(params)
    if not feature_names:
        raise RuntimeError(
            "models/feature_columns.json is missing or empty. "
            "Run `python -m src.features.offline_store` first."
        )

    train_df, val_df, _ = load_feature_splits(params)
    X_train, y_train = split_xy(train_df, feature_names)
    X_val, y_val = split_xy(val_df, feature_names)

    log.info(
        "starting_optuna_study",
        n_trials=n_trials,
        train_rows=len(X_train),
        val_rows=len(X_val),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=direction,
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            seed,
            early_stopping_rounds,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = float(best_trial.value)

    log.info(
        "optuna_study_complete",
        best_trial_number=best_trial.number,
        best_pr_auc=best_value,
        best_params=best_params,
    )

    # Log study to MLflow
    with mlflow.start_run(run_name="optuna_tuning_xgboost"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_pr_auc", best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.set_tags(
            {
                "phase": "2b",
                "stage": "optuna_tuning",
                "model_type": "xgboost",
            }
        )

    # Update params.yaml with best XGBoost hyperparameters
    if "xgb" not in params["training"]:
        params["training"]["xgb"] = {}

    for k, v in best_params.items():
        if isinstance(v, float):
            params["training"]["xgb"][k] = round(v, 6)
        else:
            params["training"]["xgb"][k] = v

    save_params(params)
    log.info("optuna_tuning_done", best_val_pr_auc=best_value)

    # ── Execute model training and evaluation ──
    from src.training.evaluate import main as evaluate_main
    from src.training.train import run as run_training

    log.info("starting_model_training_pipeline")
    run_training()

    log.info("starting_model_evaluation_pipeline")
    evaluate_main(["--model-name", "fraud-detector", "--stage", "Staging"])


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    # Check if best XGB params are already saved in params.yaml from previous tuning
    params = load_params()
    if "scale_pos_weight" in params.get("training", {}).get("xgb", {}):
        log.info("best_params_found_in_params_yaml_proceeding_to_train_and_evaluate")
        from src.training.evaluate import main as evaluate_main
        from src.training.train import run as run_training

        log.info("starting_model_training_pipeline")
        run_training()

        log.info("starting_model_evaluation_pipeline")
        evaluate_main(["--model-name", "fraud-detector", "--stage", "Staging"])
    else:
        run_optuna_tuning()


if __name__ == "__main__":
    sys.exit(main() or 0)
