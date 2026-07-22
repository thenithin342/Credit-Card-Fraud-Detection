"""
tests/data_validation/test_schema.py
─────────────────────────────────────────────────────────────────────────────
pytest wrapper around the Great Expectations validation suite.

These tests run on the *actual downloaded data* so they are skipped
automatically if the data files haven't been downloaded yet (to keep CI green
before the download step).

Run:
    pytest tests/data_validation/ -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

# ── Helpers ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def _load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


params = _load_params()
data_cfg = params["data"]
val_cfg = params["validation"]

IEEE_TXN = PROJECT_ROOT / data_cfg["ieee_train_transactions"]
IEEE_ID = PROJECT_ROOT / data_cfg["ieee_train_identity"]
ULB_CSV = PROJECT_ROOT / data_cfg["ulb_csv"]

ieee_available = IEEE_TXN.exists() and IEEE_ID.exists()
ulb_available = ULB_CSV.exists()

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ieee_df():
    """Load and merge IEEE-CIS tables; skip if files not present."""
    if not ieee_available:
        pytest.skip(
            "IEEE-CIS data not downloaded. Run: python -m src.ingestion.download --dataset ieee"
        )
    txn = pd.read_csv(IEEE_TXN)
    identity = pd.read_csv(IEEE_ID)
    return txn.merge(identity, on="TransactionID", how="left")


@pytest.fixture(scope="module")
def ulb_df():
    """Load ULB dataset; skip if file not present."""
    if not ulb_available:
        pytest.skip("ULB data not downloaded. Run: python -m src.ingestion.download --dataset ulb")
    return pd.read_csv(ULB_CSV)


# ── IEEE-CIS Tests ────────────────────────────────────────────────────────


class TestIEEESchema:
    REQUIRED_COLS = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
    ]

    def test_row_count(self, ieee_df):
        """Dataset must have at least the expected minimum row count."""
        assert len(ieee_df) >= val_cfg["expected_ieee_row_count_min"], (
            f"Expected ≥{val_cfg['expected_ieee_row_count_min']} rows, got {len(ieee_df)}"
        )

    def test_required_columns_present(self, ieee_df):
        """All required columns must be present after merge."""
        missing = [c for c in self.REQUIRED_COLS if c not in ieee_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_target_is_binary(self, ieee_df):
        """isFraud must only contain 0 and 1."""
        unique = set(ieee_df["isFraud"].dropna().unique())
        assert unique <= {0, 1}, f"isFraud has unexpected values: {unique}"

    def test_fraud_rate_in_expected_range(self, ieee_df):
        """Positive (fraud) rate must be within expected bounds."""
        rate = ieee_df["isFraud"].mean()
        assert val_cfg["min_positive_rate"] <= rate <= val_cfg["max_positive_rate"], (
            f"Fraud rate {rate:.4f} outside expected range "
            f"[{val_cfg['min_positive_rate']}, {val_cfg['max_positive_rate']}]"
        )

    def test_transaction_amt_positive(self, ieee_df):
        """TransactionAmt should be positive (>0) for virtually all rows."""
        non_positive = (ieee_df["TransactionAmt"] <= 0).mean()
        assert non_positive < 0.001, f"{non_positive:.4%} of TransactionAmt values are non-positive"

    def test_target_no_nulls(self, ieee_df):
        """isFraud column must have no null values."""
        null_count = ieee_df["isFraud"].isnull().sum()
        assert null_count == 0, f"isFraud has {null_count} null values"

    def test_transaction_dt_sanity(self, ieee_df):
        """TransactionDT (seconds offset) must be within plausible range."""
        max_dt = ieee_df["TransactionDT"].max()
        assert max_dt < 26_000_000, f"TransactionDT max {max_dt} seems implausible"
        assert ieee_df["TransactionDT"].min() >= 0, "TransactionDT has negative values"

    def test_null_rate_key_cols(self, ieee_df):
        """Key columns must not exceed the max allowed null rate."""
        max_null = val_cfg["max_null_rate"]
        critical_cols = ["TransactionAmt", "card1", "isFraud", "TransactionDT"]
        for col in critical_cols:
            null_rate = ieee_df[col].isnull().mean()
            assert null_rate <= max_null, (
                f"Column '{col}' has null rate {null_rate:.2%}, exceeds threshold {max_null:.2%}"
            )

    def test_no_duplicate_transaction_ids(self, ieee_df):
        """TransactionID must be unique (it's the primary key)."""
        dupes = ieee_df["TransactionID"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate TransactionIDs found"


# ── ULB Tests ────────────────────────────────────────────────────────────


class TestULBSchema:
    REQUIRED_COLS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]

    def test_row_count(self, ulb_df):
        assert len(ulb_df) >= val_cfg["expected_ulb_row_count_min"], (
            f"Expected ≥{val_cfg['expected_ulb_row_count_min']} rows, got {len(ulb_df)}"
        )

    def test_required_columns_present(self, ulb_df):
        missing = [c for c in self.REQUIRED_COLS if c not in ulb_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_target_is_binary(self, ulb_df):
        unique = set(ulb_df["Class"].dropna().unique())
        assert unique <= {0, 1}, f"Class has unexpected values: {unique}"

    def test_fraud_rate_in_expected_range(self, ulb_df):
        rate = ulb_df["Class"].mean()
        assert val_cfg["min_positive_rate"] <= rate <= val_cfg["max_positive_rate"], (
            f"Fraud rate {rate:.4f} outside expected range"
        )

    def test_amount_non_negative(self, ulb_df):
        assert (ulb_df["Amount"] >= 0).all(), "ULB Amount has negative values"

    def test_no_nulls(self, ulb_df):
        """ULB dataset is clean — should have zero nulls."""
        total_nulls = ulb_df.isnull().sum().sum()
        assert total_nulls == 0, f"ULB has {total_nulls} null values"

    def test_pca_features_present(self, ulb_df):
        """All 28 PCA features (V1-V28) must be present."""
        pca_cols = [f"V{i}" for i in range(1, 29)]
        missing = [c for c in pca_cols if c not in ulb_df.columns]
        assert not missing, f"Missing PCA columns: {missing}"


# ── Integration: GE Suite ────────────────────────────────────────────────


def test_ge_suite_ieee_passes(ieee_df):
    """Run the full GE suite programmatically — must pass."""
    from src.validation.ge_suite import validate_ieee

    passed = validate_ieee(ieee_df, val_cfg)
    assert passed, "Great Expectations suite for IEEE-CIS failed"


def test_ge_suite_ulb_passes(ulb_df):
    """Run the full GE suite for ULB — must pass."""
    from src.validation.ge_suite import validate_ulb

    passed = validate_ulb(ulb_df, val_cfg)
    assert passed, "Great Expectations suite for ULB failed"
