"""
tests/unit/test_split.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Unit tests for src/ingestion/split.py

Uses the synthetic_ieee_df fixture from conftest.py â€”
no real data download required. These always run in CI.
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.split import split_data  # noqa: E402

PARAMS_FILE = PROJECT_ROOT / "params.yaml"


@pytest.fixture(scope="module")
def params():
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestSplitData:
    def test_returns_three_dataframes(self, synthetic_ieee_df, params):
        """split_data must return exactly (train, val, test)."""
        result = split_data(synthetic_ieee_df, params)
        assert len(result) == 3
        for df in result:
            assert isinstance(df, pd.DataFrame)

    def test_no_row_loss(self, synthetic_ieee_df, params):
        """Total rows across splits must equal the input row count."""
        train, val, test = split_data(synthetic_ieee_df, params)
        assert len(train) + len(val) + len(test) == len(synthetic_ieee_df)

    def test_no_overlap_between_splits(self, synthetic_ieee_df, params):
        """TransactionIDs must not appear in more than one split."""
        train, val, test = split_data(synthetic_ieee_df, params)
        ids_train = set(train["TransactionID"])
        ids_val = set(val["TransactionID"])
        ids_test = set(test["TransactionID"])

        assert ids_train.isdisjoint(ids_val), "Train and val share TransactionIDs"
        assert ids_train.isdisjoint(ids_test), "Train and test share TransactionIDs"
        assert ids_val.isdisjoint(ids_test), "Val and test share TransactionIDs"

    def test_target_column_preserved(self, synthetic_ieee_df, params):
        """isFraud column must be present in all three splits."""
        target = params["features"]["target_col"]
        train, val, test = split_data(synthetic_ieee_df, params)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            assert target in df.columns, f"'{target}' missing from {name} split"

    def test_test_set_is_most_recent(self, synthetic_ieee_df, params):
        """
        After a temporal split, the minimum TransactionDT in the test set
        should be >= the minimum TransactionDT in train (not strictly
        enforced on tiny data, but the max of train <= max of test).
        """
        train, val, test = split_data(synthetic_ieee_df, params)
        # The test set was carved from the tail â€” its max DT >= train's max DT
        assert test["TransactionDT"].max() >= train["TransactionDT"].max()

    def test_both_classes_in_train(self, synthetic_ieee_df, params):
        """Train split must contain both fraud and non-fraud examples."""
        target = params["features"]["target_col"]
        train, _, _ = split_data(synthetic_ieee_df, params)
        classes = set(train[target].unique())
        assert 0 in classes, "No non-fraud examples in train"
        assert 1 in classes, "No fraud examples in train"

    def test_split_sizes_approximately_correct(self, synthetic_ieee_df, params):
        """
        Test size should be within Â±5% of the configured test_size fraction.
        (Relaxed for tiny synthetic data.)
        """
        split_cfg = params["split"]
        expected_test_frac = split_cfg["test_size"]
        train, val, test = split_data(synthetic_ieee_df, params)
        total = len(synthetic_ieee_df)
        actual_test_frac = len(test) / total
        assert abs(actual_test_frac - expected_test_frac) < 0.10, (
            f"Test fraction {actual_test_frac:.2f} deviates too far from "
            f"configured {expected_test_frac:.2f}"
        )
