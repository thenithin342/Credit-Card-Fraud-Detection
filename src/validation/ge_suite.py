"""
src/validation/ge_suite.py
─────────────────────────────────────────────────────────────────────────────
Great Expectations data validation for FraudGuard.

Runs two suites:
  1. ieee_raw_suite  — validates the IEEE-CIS merged raw data
  2. ulb_raw_suite   — validates the ULB creditcard.csv

Checks performed:
  - Row count within expected bounds
  - Required columns present
  - Null rate below threshold per column
  - isFraud distribution (positive rate within expected range)
  - No future-dated transactions (TransactionDT sanity)
  - Numeric value ranges (TransactionAmt > 0)

Output:
  - Validation results printed to stdout
  - HTML report written to reports/validation/
  - Exits with code 1 if any expectation fails (so DVC stage fails cleanly)
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


# ── IEEE-CIS Suite ────────────────────────────────────────────────────────


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


def build_ieee_suite(df: pd.DataFrame, val_cfg: dict) -> gx.core.ExpectationSuite:
    """Build and return an in-memory GE suite for the IEEE-CIS dataset."""
    context = gx.get_context(mode="ephemeral")
    ds = context.data_sources.add_pandas("ieee_source")
    da = ds.add_dataframe_asset("ieee_asset")
    batch_def = da.add_batch_definition_whole_dataframe("ieee_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.core.ExpectationSuite(name="ieee_raw_suite"))

    # Row count
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeGreaterThan(
            value=val_cfg["expected_ieee_row_count_min"]
        )
    )

    # Required columns present
    for col in IEEE_REQUIRED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # isFraud must be binary
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="isFraud", value_set=[0, 1]
        )
    )

    # Positive (fraud) rate
    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(
            column="isFraud",
            min_value=val_cfg["min_positive_rate"],
            max_value=val_cfg["max_positive_rate"],
        )
    )

    # TransactionAmt must be positive
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="TransactionAmt",
            min_value=0.01,
            max_value=None,
            mostly=0.999,
        )
    )

    # TransactionDT sanity (seconds offset; expect < 26 million seconds ≈ 10 months)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="TransactionDT",
            min_value=0,
            max_value=26_000_000,
            mostly=1.0,
        )
    )

    # Null rate checks for key columns (allow up to 50% nulls by default)
    max_null = val_cfg["max_null_rate"]
    for col in ["TransactionAmt", "card1", "isFraud", "TransactionDT"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col,
                mostly=1.0 - max_null,
            )
        )

    return suite, batch_def, context


def build_ulb_suite(df: pd.DataFrame, val_cfg: dict) -> gx.core.ExpectationSuite:
    """Build and return an in-memory GE suite for the ULB dataset."""
    context = gx.get_context(mode="ephemeral")
    ds = context.data_sources.add_pandas("ulb_source")
    da = ds.add_dataframe_asset("ulb_asset")
    batch_def = da.add_batch_definition_whole_dataframe("ulb_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.core.ExpectationSuite(name="ulb_raw_suite"))

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeGreaterThan(
            value=val_cfg["expected_ulb_row_count_min"]
        )
    )

    for col in ULB_REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="Class", value_set=[0, 1]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(
            column="Class",
            min_value=val_cfg["min_positive_rate"],
            max_value=val_cfg["max_positive_rate"],
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Amount",
            min_value=0.0,
            max_value=None,
            mostly=0.999,
        )
    )

    return suite, batch_def, context


# ── Runner ────────────────────────────────────────────────────────────────


def run_validation(
    name: str,
    df: pd.DataFrame,
    suite: gx.core.ExpectationSuite,
    batch_def,
    context,
) -> bool:
    """Run the suite, print results, save HTML report. Return True if passed."""
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    vdef = context.validation_definitions.add(
        gx.core.ValidationDefinition(
            name=f"{name}_validation",
            data=batch_def,
            suite=suite,
        )
    )
    result = vdef.run()

    success = result.success
    stats = result.statistics

    log.info(
        "validation_result",
        dataset=name,
        success=success,
        evaluated=stats["evaluated_expectations"],
        successful=stats["successful_expectations"],
        failed=stats["unsuccessful_expectations"],
    )

    if not success:
        log.error("validation_failed", dataset=name)
        for r in result.results:
            if not r.success:
                log.error(
                    "expectation_failed",
                    expectation=r.expectation_config.type,
                    details=str(r.result),
                )

    # Save HTML report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{name}_validation.html"

    # Write a simple text report (GE ephemeral context doesn't generate HTML by default)
    with open(report_path.with_suffix(".txt"), "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Evaluated: {stats['evaluated_expectations']}\n")
        f.write(f"Successful: {stats['successful_expectations']}\n")
        f.write(f"Failed: {stats['unsuccessful_expectations']}\n\n")
        for r in result.results:
            status = "✓" if r.success else "✗"
            f.write(f"  {status} {r.expectation_config.type}\n")

    log.info("report_saved", path=str(report_path.with_suffix(".txt")))
    return success


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

        suite, batch_def, context = build_ieee_suite(df_ieee, val_cfg)
        passed = run_validation("ieee_cis", df_ieee, suite, batch_def, context)
        all_passed = all_passed and passed
    else:
        log.warning(
            "ieee_files_not_found",
            hint="Run: python -m src.ingestion.download --dataset ieee",
        )

    # ── ULB ───────────────────────────────────────────────────────────────
    ulb_path = PROJECT_ROOT / data_cfg["ulb_csv"]

    if ulb_path.exists():
        log.info("loading_ulb_for_validation")
        df_ulb = pd.read_csv(ulb_path)

        suite, batch_def, context = build_ulb_suite(df_ulb, val_cfg)
        passed = run_validation("ulb", df_ulb, suite, batch_def, context)
        all_passed = all_passed and passed
    else:
        log.warning(
            "ulb_file_not_found",
            hint="Run: python -m src.ingestion.download --dataset ulb",
        )

    if not all_passed:
        log.error("validation_pipeline_failed")
        sys.exit(1)

    log.info("all_validations_passed")


if __name__ == "__main__":
    main()
