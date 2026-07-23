import json
import os
import structlog
import mlflow
from mlflow.tracking import MlflowClient

log = structlog.get_logger()


def promote_model():
    """Reads test_metrics.json and promotes the model if it passes the threshold."""
    client = MlflowClient()
    model_name = os.getenv("MODEL_NAME", "fraud-detector")
    metrics_path = "reports/evaluation/test_metrics.json"

    if not os.path.exists(metrics_path):
        log.error("metrics_not_found", path=metrics_path)
        raise FileNotFoundError(f"Metrics file {metrics_path} not found.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    pr_auc = metrics.get("pr_auc")
    # In production, we'd pull this threshold from params.yaml. Hardcoding 0.68 for simplicity here based on gap analysis.
    threshold = 0.68

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
