"""src/monitoring/drift_monitor.py
────────────────────────────────────────────────────────────────────────
Standalone batch script for periodic data-drift monitoring.

Why batch, not an API endpoint
------------------------------
Synchronous Evidently AI drift calculation on every ``/v1/score`` request
would blow the 150 ms scoring SLA. This script is intended to be run on a
schedule (cron, GitHub Actions, Kubernetes CronJob) and exits non-zero
when drift is detected so the scheduler can alert.

What it does
------------
1. Load the *reference* dataset (the test split from
   ``params.yaml:data.features_dir/test_features.parquet``).
2. Build the *current* dataset — by default a small slice of the
   reference (self-check) until a real scoring-log capture pipeline is
   in place. Pass a path to override.
3. Run an Evidently ``DataDriftPreset`` Report.
4. Write a JSON summary to
   ``params.yaml:monitoring.report_output_dir/drift_<timestamp>.json``.
5. Exit ``0`` if drift share is below the configured threshold, else
   exit ``1``.

CLI
---
    python -m src.monitoring.drift_monitor
    python -m src.monitoring.drift_monitor --current /path/to/current.parquet
    python -m src.monitoring.drift_monitor --params /path/to/params.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── params + paths ───────────────────────────────────────────────────────


def load_params(params_path: Path) -> dict[str, Any]:
    """Load the YAML params file the same way train.py does."""
    import yaml  # local import: keep module import light

    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(params: dict[str, Any], params_path: Path) -> dict[str, Path]:
    """Resolve every on-disk path used by the monitor relative to params.yaml's
    directory. By convention ``params.yaml`` lives at the repository root, so
    ``params_path.parent`` IS the repo root.
    """
    base = params_path.resolve().parent
    features_dir = base / params["data"]["features_dir"]
    report_dir = base / params["monitoring"]["report_output_dir"]
    return {
        "base": base,
        "reference": features_dir / "test_features.parquet",
        "report_dir": report_dir,
    }


# ── Evidently wrapper ────────────────────────────────────────────────────


def compute_drift_share(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Run the Evidently DataDriftPreset and extract a JSON-safe summary."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # Evidently's DataDefinition needs to know which columns are
    # numeric vs categorical. Infer from dtype.
    numeric_cols = reference.select_dtypes(include="number").columns.tolist()
    categorical_cols = reference.select_dtypes(exclude="number").columns.tolist()
    data_def = DataDefinition(
        numerical_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )

    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)
    return json.loads(result.json())


def summarize_drift(drift_report: dict[str, Any]) -> dict[str, Any]:
    """Reduce the full Evidently JSON to the small set of numbers we care about.
    Uses defensive extraction — Evidently's JSON schema changes between versions.
    """
    metrics = drift_report.get("metrics", [])
    drift_share: float = 0.0
    drifted_count: int = 0
    per_column: list[dict[str, Any]] = []
    for m in metrics:
        metric_name = m.get("metric_name", "") or m.get("id", "")
        value = m.get("value")
        if "DriftedColumnsCount" in metric_name:
            if isinstance(value, dict):
                drift_share = float(value.get("share", 0.0))
                drifted_count = int(value.get("count", 0))
        elif "ValueDrift" in metric_name or "ColumnDrift" in metric_name:
            col = m.get("config", {}).get("column") or m.get("column")
            if col:
                per_column.append(
                    {
                        "column": col,
                        "method": m.get("config", {}).get("method"),
                        "p_value": float(value) if isinstance(value, (int, float)) else None,
                        "drifted": bool(isinstance(value, (int, float)) and value < 0.05),
                    }
                )
    if drift_share == 0.0 and not per_column:
        log.warning(
            "drift_summary_empty",
            hint="Evidently JSON schema may have changed. Check evidently version.",
            n_metrics=len(metrics),
        )
    return {
        "drift_share": drift_share,
        "drifted_columns": drifted_count,
        "per_column": per_column,
    }


# ── core runner ─────────────────────────────────────────────────────────


def run_monitor(
    *,
    params_path: Path,
    current_path: Path | None,
) -> dict[str, Any]:
    """Execute one monitoring cycle. Returns the written summary dict."""
    params = load_params(params_path)
    paths = resolve_paths(params, params_path)
    threshold = float(params["monitoring"]["drift_score_threshold"])

    if not paths["reference"].exists():
        raise FileNotFoundError(
            f"Reference parquet not found at {paths['reference']}. "
            "Run `python -m src.features.build_features` first."
        )

    log.info("loading_reference", path=str(paths["reference"]))
    reference = pd.read_parquet(paths["reference"])

    if current_path is not None:
        log.info("loading_current", path=str(current_path))
        current = pd.read_parquet(current_path)
    else:
        # No live scoring-log infra yet — self-check the reference.
        log.warning(
            "no_current_data_path_provided",
            hint="Pass --current <parquet> once scoring logs are captured.",
            fallback="using random sample of reference as 'current'",
        )
        current = reference.sample(n=min(5_000, len(reference)), random_state=0).reset_index(
            drop=True
        )

    raw = compute_drift_share(reference, current)
    summary = summarize_drift(raw)
    summary["threshold"] = threshold
    summary["drift_detected"] = summary["drift_share"] > threshold
    summary["reference_rows"] = int(len(reference))
    summary["current_rows"] = int(len(current))
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = paths["report_dir"] / f"drift_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "drift_report_written",
        path=str(out_path),
        drift_share=summary["drift_share"],
        threshold=threshold,
        drift_detected=summary["drift_detected"],
    )
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "params.yaml",
        help="Path to params.yaml (default: repo root).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Optional path to a current-window parquet for drift comparison.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero if drift > threshold (default behaviour).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    args = parse_args(argv)
    summary = run_monitor(params_path=args.params, current_path=args.current)
    if summary["drift_detected"]:
        log.warning(
            "drift_threshold_exceeded",
            drift_share=summary["drift_share"],
            threshold=summary["threshold"],
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
