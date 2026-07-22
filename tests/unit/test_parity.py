"""tests/unit/test_parity.py
────────────────────────────────────────────────────────────────────────
Feature-parity test — the most important correctness check in Phase 2.

Contract under test
-------------------
The online store is responsible for tracking *stateful* features
(rolling aggregates per card).  The *static* features
(amount_log, hour_of_day, day_of_week) are pure functions of the
current transaction and are computed at request time by the
serving path using `src.features.definitions.compute_static_features`.

For the parity test we therefore:

  1. Replay a card's history through the online store
     (update_after_transaction on each row).
  2. Compute the *static* features for the last row directly from
     `compute_static_features`.
  3. Read the *stateful* features for the last row from the store.
  4. Compare against the offline `assemble_features` output for the
     last row.  Tolerance: 1e-6.

This guarantees the SAME transaction processed through the offline
training pipeline and through the online serving path produces
identical feature values.
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
    WINDOW_FEATURE_NAMES,
    assemble_features,
    compute_static_features,
)
from src.features.online_store import OnlineFeatureStore  # noqa: E402

# ── Helpers ────────────────────────────────────────────────────────────────


def _build_card_history(n: int = 30, seed: int = 7) -> pd.DataFrame:
    """Build a deterministic history of *n* transactions for one card,
    spaced 30s apart, with amounts that vary so the rolling window
    has something to compute against."""
    rng = np.random.default_rng(seed)
    base_ts = 86_400  # arbitrary epoch, hour=0, day_of_week=1
    amounts = rng.uniform(10.0, 500.0, size=n).round(2)
    return pd.DataFrame(
        {
            "TransactionID": list(range(1, n + 1)),
            "isFraud": [0] * n,
            "TransactionDT": [base_ts + i * 30 for i in range(n)],
            "TransactionAmt": amounts,
            "card1": [4242] * n,
        }
    )


def _drive_online_store(df: pd.DataFrame, card_id: str = "4242") -> dict:
    """Replay a card's history through the online store exactly the
    way the streaming pipeline would.  Returns the *last* feature
    dict — i.e. the state the store holds after the last transaction.

    The returned dict only contains the *stateful* features (the ones
    the store tracks); static features are computed separately by
    the serving path from the current request's raw data.
    """
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    store = OnlineFeatureStore(client)
    for _, row in df.iterrows():
        store.update_after_transaction(
            card_id,
            amount=float(row["TransactionAmt"]),
            ts=int(row["TransactionDT"]),
        )
    out = store.get_card_features(card_id)
    assert out is not None
    return out


def _serving_view(df: pd.DataFrame, online_state: dict, card_col: str = "card1") -> dict:
    """Simulate what the serving API would produce: static features
    computed at request time + stateful features pulled from the
    online store."""
    static = compute_static_features(df).iloc[-1]
    combined = {}
    # Static features are recomputed per request, not stored.
    for name in STATIC_FEATURE_NAMES:
        combined[name] = float(static[name])
    # Stateful features come from the store.
    for name in WINDOW_FEATURE_NAMES:
        combined[name] = float(online_state.get(name, 0.0))
    return combined


# ── The parity test ────────────────────────────────────────────────────────


def test_offline_online_parity() -> None:
    """Same transactions → offline pipeline and online serving path
    produce feature values within 1e-6 tolerance."""
    history = _build_card_history(n=30)

    # ── Offline path: compute the *latest* row's features ──
    offline_features = assemble_features(history, card_col="card1")
    offline_last = offline_features.iloc[-1].to_dict()

    # ── Online path: replay history, then compose the serving view ──
    online_state = _drive_online_store(history, card_id="4242")
    online_view = _serving_view(history, online_state, card_col="card1")

    for fname in FEATURE_NAMES:
        offline_val = float(offline_last[fname])
        online_val = float(online_view[fname])
        assert abs(offline_val - online_val) < 1e-6, (
            f"Feature '{fname}' differs: offline={offline_val} vs "
            f"online={online_val} (diff={abs(offline_val - online_val):.3e})"
        )


def test_offline_online_parity_empty_history() -> None:
    """The very first transaction a card has ever made should yield
    identical defaults from both paths.

    For the offline path: a single-row DataFrame means no window
    history exists → amount_zscore=-999, all counts/sums=0,
    time_since_last_txn=-999 (no-history sentinel).

    For the online path: update_after_transaction on a fresh card
    initialises the same defaults.
    """
    single = _build_card_history(n=1)
    offline_features = assemble_features(single, card_col="card1")
    online_state = _drive_online_store(single, card_id="9999")
    online_view = _serving_view(single, online_state, card_col="card1")

    for fname in FEATURE_NAMES:
        offline_val = float(offline_features.iloc[-1][fname])
        online_val = float(online_view[fname])
        assert abs(offline_val - online_val) < 1e-6, (
            f"Empty-history feature '{fname}' differs: "
            f"offline={offline_val} vs online={online_val}"
        )


def test_offline_online_parity_window_boundaries() -> None:
    """At the 5-minute window boundary the counts should drop sharply
    in both paths.  We use a 6-transaction history spaced 60s apart
    and inspect the 6th transaction (5 minutes after tx 1)."""
    base_ts = 86_400
    history = pd.DataFrame(
        {
            "TransactionID": list(range(1, 7)),
            "isFraud": [0] * 6,
            "TransactionDT": [base_ts + i * 60 for i in range(6)],  # 0, 60, ..., 300
            "TransactionAmt": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "card1": [7777] * 6,
        }
    )
    offline = assemble_features(history, card_col="card1").iloc[-1]
    online_state = _drive_online_store(history, card_id="7777")
    online_view = _serving_view(history, online_state, card_col="card1")

    # At row 5 (ts=86400+300), the 5m window is [86400+0, 86400+300].
    # Tx at ts=86400+0 is exactly on the boundary; bisect_left with
    # cutoff=86400+0 includes it, so count_5m = 5 prior tx = 5.
    # (tx at ts=0 is strictly less than current ts=300, so it counts.)
    # 1h window has all 5 prior tx.
    expected_5m = 5.0
    expected_1h = 5.0
    assert abs(float(offline["txn_count_5m"]) - expected_5m) < 1e-6
    assert abs(float(online_view["txn_count_5m"]) - expected_5m) < 1e-6
    assert abs(float(offline["txn_count_1h"]) - expected_1h) < 1e-6
    assert abs(float(online_view["txn_count_1h"]) - expected_1h) < 1e-6
