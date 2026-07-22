"""tests/unit/test_features.py
────────────────────────────────────────────────────────────────────────
Unit tests for src/features/definitions.py and src/features/online_store.py.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.definitions import (  # noqa: E402
    FEATURE_NAMES,
    STATIC_FEATURE_NAMES,
    TARGET_COL,
    WINDOW_FEATURE_NAMES,
    assemble_features,
    compute_static_features,
    compute_window_features,
)
from src.features.online_store import (  # noqa: E402
    KEY_PREFIX,
    OnlineFeatureStore,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_single_card() -> pd.DataFrame:
    """One card, four transactions, 5 minutes apart.  Lets us assert
    exact values for the window features."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            TARGET_COL: [0, 0, 0, 0],
            "TransactionDT": [86_400, 86_400 + 300, 86_400 + 600, 86_400 + 900],
            "TransactionAmt": [100.0, 200.0, 50.0, 75.0],
            "card1": [1234, 1234, 1234, 1234],
        }
    )


@pytest.fixture()
def synthetic_two_cards() -> pd.DataFrame:
    """Two cards, 4 transactions each, interleaved in time so we
    verify per-card isolation of the window features."""
    return pd.DataFrame(
        {
            "TransactionID": list(range(1, 9)),
            TARGET_COL: [0] * 8,
            "TransactionDT": [
                86_400,
                86_400 + 60,
                86_400 + 120,  # card A
                86_400 + 30,
                86_400 + 90,
                86_400 + 150,  # card B
                86_400 + 180,  # card A
                86_400 + 210,  # card B
            ],
            "TransactionAmt": [
                50.0,
                75.0,
                100.0,
                200.0,
                25.0,
                40.0,
                30.0,
                60.0,
            ],
            "card1": ["A", "A", "A", "B", "B", "B", "A", "B"],
        }
    )


# ── Tests for compute_static_features ──────────────────────────────────────


def test_compute_static_features_no_nulls(synthetic_single_card: pd.DataFrame) -> None:
    """No NaN after static transforms on synthetic data."""
    static = compute_static_features(synthetic_single_card)
    assert list(static.columns) == list(STATIC_FEATURE_NAMES)
    assert static.isna().sum().sum() == 0
    assert len(static) == len(synthetic_single_card)


def test_compute_static_features_values(synthetic_single_card: pd.DataFrame) -> None:
    """amount_log should be np.log1p(TransactionAmt) and hour_of_day
    should be TransactionDT % 86400 // 3600."""
    static = compute_static_features(synthetic_single_card)
    expected_log = np.log1p(synthetic_single_card["TransactionAmt"].astype(float))
    np.testing.assert_allclose(static["amount_log"].to_numpy(), expected_log.to_numpy())

    # All transactions are at DT=86400+0..900 seconds → hour_of_day = 0
    assert (static["hour_of_day"] == 0).all()
    # day_of_week = (DT // 86400) % 7 = 1 % 7 = 1
    assert (static["day_of_week"] == 1).all()


# ── Tests for compute_window_features ──────────────────────────────────────


def test_compute_window_features_shape(synthetic_single_card: pd.DataFrame) -> None:
    """Output has all expected columns and one row per input row."""
    win = compute_window_features(synthetic_single_card, card_col="card1")
    assert list(win.columns) == list(WINDOW_FEATURE_NAMES)
    assert len(win) == len(synthetic_single_card)
    assert win.isna().sum().sum() == 0


def test_compute_window_features_first_txn() -> None:
    """The very first transaction for a card has no history:
    amount_zscore == -999 sentinel, time_since_last_txn == 0, all
    counts/sums == 0."""
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "isFraud": [0],
            "TransactionDT": [86_400],
            "TransactionAmt": [42.0],
            "card1": [9999],
        }
    )
    win = compute_window_features(df, card_col="card1")
    assert win.loc[0, "amount_zscore"] == -999.0
    assert win.loc[0, "txn_count_5m"] == 0.0
    assert win.loc[0, "txn_amount_sum_5m"] == 0.0
    assert win.loc[0, "txn_count_1h"] == 0.0
    assert win.loc[0, "txn_amount_sum_1h"] == 0.0
    # First transaction for a new card → no history, use -999 sentinel.
    assert win.loc[0, "time_since_last_txn"] == -999.0


def test_compute_window_features_isolation(synthetic_two_cards: pd.DataFrame) -> None:
    """Window features for card A must not see card B's history and
    vice versa."""
    win = compute_window_features(synthetic_two_cards, card_col="card1")
    # Row 0 is the first tx for card A → no history
    assert win.loc[0, "txn_count_1h"] == 0.0
    # Row 3 is the first tx for card B → no history
    assert win.loc[3, "txn_count_1h"] == 0.0
    # Row 6 is the 4th tx for card A; by then A has 3 prior tx (rows 0,1,2)
    # all within 1h of row 6 (DT = 86400+180).
    assert win.loc[6, "txn_count_1h"] == 3.0
    # Sum of A's prior amounts: 50 + 75 + 100 = 225
    assert win.loc[6, "txn_amount_sum_1h"] == pytest.approx(225.0)


# ── Tests for assemble_features ────────────────────────────────────────────


def test_feature_names_match_definitions() -> None:
    """The columns produced by `assemble_features` exactly match
    `FEATURE_NAMES` in the canonical order."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 0, 0],
            "TransactionDT": [86_400, 86_400 + 60, 86_400 + 120],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [1, 1, 1],
        }
    )
    feats = assemble_features(df, card_col="card1")
    assert list(feats.columns) == list(FEATURE_NAMES)
    assert feats.isna().sum().sum() == 0
    assert len(feats) == 3


def test_assemble_features_preserves_index(synthetic_single_card: pd.DataFrame) -> None:
    """Output rows are aligned to the input rows by index."""
    feats = assemble_features(synthetic_single_card, card_col="card1")
    assert len(feats) == len(synthetic_single_card)
    # First tx for a new card has no history → -999 sentinel.
    # Subsequent rows are 300 s apart (5-minute spacing).
    assert feats["time_since_last_txn"].iloc[0] == -999.0
    assert feats["time_since_last_txn"].iloc[1] == pytest.approx(300.0)
    assert feats["time_since_last_txn"].iloc[2] == pytest.approx(300.0)


# ── Tests for OnlineFeatureStore (using fakeredis) ─────────────────────────


def test_online_store_set_get_roundtrip() -> None:
    """set_card_features → get_card_features returns the same values."""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    store = OnlineFeatureStore(client)

    features = {
        "amount_log": 4.5,
        "hour_of_day": 12.0,
        "amount_zscore": -0.1,
        "txn_count_5m": 3.0,
    }
    store.set_card_features("CARD-X", features, ttl_seconds=60)
    out = store.get_card_features("CARD-X")
    assert out is not None
    for k, v in features.items():
        assert float(out[k]) == pytest.approx(float(v), abs=1e-6)


def test_online_store_get_missing_returns_none() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    store = OnlineFeatureStore(client)
    assert store.get_card_features("never-seen") is None


def test_online_store_update_initialises_state() -> None:
    """update_after_transaction on a card with no prior record should
    initialise the rolling aggregates."""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    store = OnlineFeatureStore(client)
    store.update_after_transaction("CARD-NEW", amount=100.0, ts=86_400)

    feats = store.get_card_features("CARD-NEW")
    assert feats is not None
    assert feats["txn_count_1h"] == 0.0  # past-only: current tx not counted
    assert feats["txn_amount_sum_1h"] == 0.0  # past-only
    assert feats["txn_count_5m"] == 0.0  # past-only
    assert feats["time_since_last_txn"] == -999.0  # no prior tx → sentinel


def test_online_store_key_format() -> None:
    """Keys follow the fraud:features:{card_id} convention."""
    assert KEY_PREFIX == "fraud:features:"
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    store = OnlineFeatureStore(client)
    store.set_card_features("abc", {"amount_log": 1.0})
    # fakeredis stores under exactly that key
    assert client.get("fraud:features:abc") is not None
