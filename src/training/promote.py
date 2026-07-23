import json
import os
from pathlib import Path

import mlflow
import structlog
import yaml
from mlflow.tracking import MlflowClient

log = structlog.get_logger()


def promote_model():
    """Reads test_metrics.json and promotes the model if it passes the threshold."""
    project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri((project_root / "mlruns").as_uri())

    client = MlflowClient()
    model_name = os.getenv("MODEL_NAME", "fraud-detector")
    metrics_path = "reports/evaluation/test_metrics.json"

    params_path = project_root / "params.yaml"
    threshold = 0.68
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        training_cfg = params.get("training") or {}
        threshold = training_cfg.get("promotion_pr_auc_threshold", 0.68)

    if not os.path.exists(metrics_path):
        log.error("metrics_not_found", path=metrics_path)
        raise FileNotFoundError(f"Metrics file {metrics_path} not found.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    pr_auc = metrics.get("pr_auc")

    log.info("checking_promotion_criteria", current_pr_auc=pr_auc, required=threshold)

    if pr_auc is None or pr_auc < threshold:
        log.warning("model_failed_promotion", pr_auc=pr_auc, threshold=threshold)
        return

    # Find latest version in Staging (or latest overall if just registered)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        log.error("no_model_versions_found", model_name=model_name)
        return

    # Sort by version number descending
    latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    log.info("promoting_model_to_production", model_name=model_name, version=latest_version.version)

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    log.info("promotion_complete")


if __name__ == "__main__":
    promote_model()
