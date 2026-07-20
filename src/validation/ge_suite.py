"""
src/validation/ge_suite.py
─────────────────────────────────────────────────────────────────────────────
Great Expectations 0.18.x data validation for FraudGuard.

Uses the stable legacy PandasDataset API (gx.from_pandas) which works
reliably across all GE 0.18.x releases — avoids the fluent-API churn
where context.sources / add_batch_definition_whole_dataframe changed
between patch versions.

Suites:
  1. ieee_raw_suite  — IEEE-CIS merged raw data
  2. ulb_raw_suite   — ULB creditcard.csv

Checks: row count, required columns, null rates, isFraud distribution,
        TransactionAmt/DT ranges.

Output:
  - Validation results logged via structlog
  - Plain-text report → reports/validation/
  - Exits with code 1 on any failure (DVC stage fails cleanly)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
import structlog
import yaml

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


# ── Column lists ──────────────────────────────────────────────────────────

IEEE_REQUIRED_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
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
]

ULB_REQUIRED_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]


# ── Helpers ───────────────────────────────────────────────────────────────


def _save_report(name: str, result) -> None:
    """Write a plain-text validation report to reports/validation/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stats = result.statistics
    report_path = REPORT_DIR / f"{name}_validation.txt"
    with open(report_path, "w") as f:
        f.write(f"Dataset:    {name}\n")
        f.write(f"Success:    {result.success}\n")
        f.write(f"Evaluated:  {stats['evaluated_expectations']}\n")
        f.write(f"Successful: {stats['successful_expectations']}\n")
        f.write(f"Failed:     {stats['unsuccessful_expectations']}\n\n")
        for r in result.results:
            status = "PASS" if r.success else "FAIL"
            f.write(f"  [{status}] {r.expectation_config['expectation_type']}\n")
    log.info("report_saved", path=str(report_path))


def _log_result(name: str, result) -> bool:
    """Log summary + failed expectations. Return True if all passed."""
    stats = result.statistics
    log.info(
        "validation_result",
        dataset=name,
        success=result.success,
        evaluated=stats["evaluated_expectations"],
        successful=stats["successful_expectations"],
        failed=stats["unsuccessful_expectations"],
    )
    if not result.success:
        log.error("validation_failed", dataset=name)
        for r in result.results:
            if not r.success:
                log.error(
                    "expectation_failed",
                    expectation=r.expectation_config["expectation_type"],
                    details=str(r.result),
                )
    return result.success


# ── Suite runners ─────────────────────────────────────────────────────────


def validate_ieee(df: pd.DataFrame, val_cfg: dict) -> bool:
    """Run GE expectations on the IEEE-CIS merged dataframe.

    Uses the stable PandasDataset (legacy) API that works on GE 0.18.x.
    Returns True if all expectations pass.
    """
    ge_df = gx.from_pandas(df)

    # ── Row count ──────────────────────────────────────────────────────
    ge_df.expect_table_row_count_to_be_between(min_value=val_cfg["expected_ieee_row_count_min"])

    # ── Required columns ───────────────────────────────────────────────
    for col in IEEE_REQUIRED_COLUMNS:
        ge_df.expect_column_to_exist(col)

    # ── isFraud binary ─────────────────────────────────────────────────
    ge_df.expect_column_distinct_values_to_be_in_set("isFraud", value_set=[0, 1])

    # ── Positive (fraud) rate ──────────────────────────────────────────
    ge_df.expect_column_mean_to_be_between(
        "isFraud",
        min_value=val_cfg["min_positive_rate"],
        max_value=val_cfg["max_positive_rate"],
    )

    # ── TransactionAmt positive ────────────────────────────────────────
    ge_df.expect_column_values_to_be_between(
        "TransactionAmt",
        min_value=0.01,
        max_value=None,
        mostly=0.999,
    )

    # ── TransactionDT sanity (<26M seconds ≈ 10 months) ───────────────
    ge_df.expect_column_values_to_be_between(
        "TransactionDT",
        min_value=0,
        max_value=26_000_000,
        mostly=1.0,
    )

    # ── Null rates for key columns ─────────────────────────────────────
    max_null = val_cfg["max_null_rate"]
    for col in ["TransactionAmt", "card1", "isFraud", "TransactionDT"]:
        ge_df.expect_column_values_to_not_be_null(col, mostly=1.0 - max_null)

    result = ge_df.validate()
    _log_result("ieee_cis", result)
    _save_report("ieee_cis", result)
    return result.success


def validate_ulb(df: pd.DataFrame, val_cfg: dict) -> bool:
    """Run GE expectations on the ULB creditcard.csv dataframe."""
    ge_df = gx.from_pandas(df)

    ge_df.expect_table_row_count_to_be_between(min_value=val_cfg["expected_ulb_row_count_min"])

    for col in ULB_REQUIRED_COLUMNS:
        ge_df.expect_column_to_exist(col)

    ge_df.expect_column_distinct_values_to_be_in_set("Class", value_set=[0, 1])

    ge_df.expect_column_mean_to_be_between(
        "Class",
        min_value=val_cfg["min_positive_rate"],
        max_value=val_cfg["max_positive_rate"],
    )

    ge_df.expect_column_values_to_be_between(
        "Amount",
        min_value=0.0,
        max_value=None,
        mostly=0.999,
    )

    result = ge_df.validate()
    _log_result("ulb", result)
    _save_report("ulb", result)
    return result.success


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    params = load_params()
    val_cfg = params["validation"]
    data_cfg = params["data"]

    all_passed = True

    # ── IEEE-CIS ──────────────────────────────────────────────────────────
    ieee_txn_path = PROJECT_ROOT / data_cfg["ieee_train_transactions"]
    ieee_id_path = PROJECT_ROOT / data_cfg["ieee_train_identity"]

    if ieee_txn_path.exists() and ieee_id_path.exists():
        log.info("loading_ieee_for_validation")
        txn = pd.read_csv(ieee_txn_path)
        identity = pd.read_csv(ieee_id_path)
        df_ieee = txn.merge(identity, on="TransactionID", how="left")
        all_passed = validate_ieee(df_ieee, val_cfg) and all_passed
    else:
        log.warning(
            "ieee_files_not_found",
            hint="Run: python -m src.ingestion.download",
        )

    # ── ULB ───────────────────────────────────────────────────────────────
    ulb_path = PROJECT_ROOT / data_cfg["ulb_csv"]

    if ulb_path.exists():
        log.info("loading_ulb_for_validation")
        df_ulb = pd.read_csv(ulb_path)
        all_passed = validate_ulb(df_ulb, val_cfg) and all_passed
    else:
        log.warning(
            "ulb_file_not_found",
            hint="Run: python -m src.ingestion.download",
        )

    if not all_passed:
        log.error("validation_pipeline_failed")
        sys.exit(1)

    log.info("all_validations_passed")


if __name__ == "__main__":
    main()
