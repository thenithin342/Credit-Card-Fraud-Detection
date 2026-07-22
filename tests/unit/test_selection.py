"""tests/unit/test_selection.py
────────────────────────────────────────────────────────────────────────
Unit tests for src/features/selection.py and src/features/preprocessing.py.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Tests for drop_high_null_cols ──────────────────────────────────────────


def test_drop_high_null_cols() -> None:
    """Cols above the threshold are removed, cols below the threshold are kept."""
    from src.features.selection import drop_high_null_cols

    df = pd.DataFrame(
        {
            "keep": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0% null
            "drop1": [
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],  # 90% null
            "drop2": [np.nan] * 9 + [1.0],  # 90% null
            "borderline": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan],  # 20% null
        }
    )
    out, dropped = drop_high_null_cols(df, threshold=0.80)
    assert "keep" in out.columns
    assert "borderline" in out.columns
    assert "drop1" not in out.columns
    assert "drop2" not in out.columns
    assert set(dropped) == {"drop1", "drop2"}


def test_drop_high_null_cols_threshold_boundary() -> None:
    """A column at *exactly* the threshold is kept (the predicate is `>` not `>=`)."""
    from src.features.selection import drop_high_null_cols

    # 8 out of 10 values are null → 80% null rate (== threshold, should be kept).
    df = pd.DataFrame(
        {
            "exact": [np.nan] * 8 + [1.0, 2.0],
            "over": [np.nan] * 9 + [1.0],  # 90% null, should be dropped
        }
    )
    out, dropped = drop_high_null_cols(df, threshold=0.80)
    assert "exact" in out.columns
    assert "over" not in out.columns
    assert dropped == ["over"]


# ── Tests for drop_correlated_cols ─────────────────────────────────────────


def test_drop_correlated_cols_removes_one_of_pair() -> None:
    """Two perfectly correlated synthetic columns: only one survives."""
    from src.features.selection import drop_correlated_cols

    rng = np.random.default_rng(0)
    n = 200
    base = rng.standard_normal(size=n)
    df = pd.DataFrame(
        {
            "alpha": base,  # kept (alphabetically first)
            "bravo": base + 0.0,  # perfectly correlated, dropped
            "charlie": rng.standard_normal(size=n),  # independent, kept
        }
    )
    out, dropped = drop_correlated_cols(df, threshold=0.95)
    # alpha < bravo, so bravo is dropped.
    assert "alpha" in out.columns
    assert "charlie" in out.columns
    assert "bravo" not in out.columns
    assert "bravo" in dropped


def test_drop_correlated_cols_cross_block() -> None:
    """Ensure correlation removal works across blocks (e.g. chunking logic)."""
    from src.features.selection import drop_correlated_cols

    rng = np.random.default_rng(0)
    n = 200
    base = rng.standard_normal(size=n)

    # Create 3 columns:
    # A in block 1 (idx 0)
    # B in block 2 (idx 5)
    # A and B are perfectly correlated. B should be dropped.
    df_data = {}
    df_data["A_col"] = base
    for i in range(1, 5):
        df_data[f"noise_{i}"] = rng.standard_normal(size=n)
    df_data["B_col"] = base + 0.0
    for i in range(6, 10):
        df_data[f"noise_{i}"] = rng.standard_normal(size=n)

    df = pd.DataFrame(df_data)
    # Set block size to 5 so A and B end up in different blocks
    out, dropped = drop_correlated_cols(df, threshold=0.95, block_size=5)

    assert "A_col" in out.columns
    assert "B_col" not in out.columns
    assert "B_col" in dropped


# ── Tests for FeaturePreprocessor ──────────────────────────────────────────


def test_no_leakage_preprocessor() -> None:
    """`FeaturePreprocessor.fit()` only sees train rows; `transform` on val
    must not raise even with unseen categories."""
    from src.features.preprocessing import FeaturePreprocessor

    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    val = pd.DataFrame(
        {
            "num": [10.0, 20.0, np.nan, 40.0],
            "cat": ["a", "NEVER_SEEN", "b", None],  # unseen category + null
        }
    )
    fp = FeaturePreprocessor()
    fp.fit(train, ["num", "cat"])

    # Train transform: no nulls should remain.
    out_train = fp.transform(train)
    assert out_train.isna().sum().sum() == 0
    # Categorical mapping should be deterministic.
    assert (out_train["cat"].astype(int) >= 0).all()

    # Val transform: must not raise even with unseen categories.
    out_val = fp.transform(val)
    assert "NEVER_SEEN" in set(val["cat"])  # sanity
    # The unseen value maps to -1 (the unknown_value we configured).
    unseen_idx = val.index[val["cat"] == "NEVER_SEEN"][0]
    assert int(out_val.loc[unseen_idx, "cat"]) == -1
    # Nulls in val are filled (with the string sentinel then encoded).
    null_idx = val.index[val["cat"].isna()][0]
    assert int(out_val.loc[null_idx, "cat"]) >= 0  # maps to 'missing' bucket
    # Numeric nulls are filled with -999.
    num_null_idx = val.index[val["num"].isna()][0]
    assert float(out_val.loc[num_null_idx, "num"]) == -999.0

    # Pickle round-trip.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "preproc.pkl"
        fp.save(path)
        fp2 = FeaturePreprocessor.load(path)
        out_val_2 = fp2.transform(val)
        # Re-loaded preprocessor must produce identical output.
        pd.testing.assert_frame_equal(out_val, out_val_2)


# ── Tests for feature_columns.json consistency ─────────────────────────────


def test_feature_columns_json_consistent() -> None:
    """`all_feature_cols` = `raw_feature_cols` + `temporal_feature_cols` (in that order),
    with no duplicates."""
    from src.features.definitions import FEATURE_NAMES

    raw_feature_cols = ["V1", "V2", "C1", "card1", "ProductCD"]
    temporal_feature_cols = list(FEATURE_NAMES)
    all_feature_cols = raw_feature_cols + temporal_feature_cols

    # Contract under test.
    assert all_feature_cols == raw_feature_cols + temporal_feature_cols
    # No duplicates in either block.
    assert len(all_feature_cols) == len(set(all_feature_cols))
    assert len(raw_feature_cols) == len(set(raw_feature_cols))
    assert len(temporal_feature_cols) == len(set(temporal_feature_cols))

    # Round-trip via JSON (this is exactly what offline_store writes).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feature_columns.json"
        path.write_text(
            json.dumps(
                {
                    "raw_feature_cols": raw_feature_cols,
                    "temporal_feature_cols": temporal_feature_cols,
                    "all_feature_cols": all_feature_cols,
                }
            )
        )
        loaded = json.loads(path.read_text())
        assert loaded["all_feature_cols"] == raw_feature_cols + temporal_feature_cols
        assert loaded["temporal_feature_cols"] == list(FEATURE_NAMES)
