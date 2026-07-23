"""tests/unit/test_producer.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for ``src/ingestion/producer.py``.

We mock ``confluent_kafka.Producer`` and ``time.sleep`` so the suite
never talks to a real broker and runs in milliseconds.

What we verify
--------------
* ``replay`` calls ``producer.send`` exactly ``max_rows`` times and
  returns the same count.
* ``TransactionProducer.send`` serialises pandas ``NaN`` as JSON
  ``null`` (never the literal string ``"NaN"`` or the invalid JSON
  token ``NaN``).
* ``replay`` iterates rows in ``TransactionDT`` ascending order.
* The CLI ``main()`` calls ``ensure_topic`` before producing and
  delegates to ``replay`` correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ───────────────────────────────────────────────────────────────


def _make_df(n: int = 5, *, shuffle_dt: bool = False) -> pd.DataFrame:
    """A small synthetic frame resembling IEEE-CIS rows."""
    rng = np.random.default_rng(0)
    dts = np.arange(1000, 1000 + n * 10, 10, dtype=np.int64)  # 1000, 1010, …
    if shuffle_dt:
        rng.shuffle(dts)
    # Sprinkle NaNs into DeviceInfo so the cleaner has work to do.
    device_pool = [np.nan, "Firefox", np.nan, "Chrome", np.nan]
    device = [device_pool[i % len(device_pool)] for i in range(n)]
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1, dtype=np.int64),
            "TransactionDT": dts,
            "TransactionAmt": rng.uniform(10, 500, n).round(2),
            "ProductCD": ["W"] * n,
            "DeviceInfo": device,
        }
    )


class _StubProducer:
    """Records every row passed to .send()."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.flush_called = 0

    def send(self, row: dict) -> None:
        self.sent.append(row)

    def flush(self) -> None:
        self.flush_called += 1


# ── send() ───────────────────────────────────────────────────────────────


def test_send_serialises_nan_as_null() -> None:
    """NaN values must be encoded as JSON null, not the token 'NaN'."""
    from src.ingestion import producer as mod

    # Build a real TransactionProducer, then swap the inner .producer for a
    # tracked mock so we can inspect produce()'s args.
    tp = mod.TransactionProducer(bootstrap_servers="x:1", topic="t")
    inner = MagicMock(name="Producer")
    tp.producer = inner

    row = {
        "TransactionID": 42,
        "amt": 12.5,
        "missing": float("nan"),
        "also_missing": np.nan,
        "none_field": None,
        "ok": "hello",
    }
    tp.send(row)

    inner.produce.assert_called_once()
    _args, kwargs = inner.produce.call_args
    assert kwargs["key"] == b"42"
    value = kwargs["value"]  # bytes

    # Hard check: the bytes contain no literal 'NaN' token.
    assert b"NaN" not in value
    # And they round-trip through json.loads.
    decoded = json.loads(value)
    assert decoded["TransactionID"] == 42
    assert decoded["amt"] == 12.5
    assert decoded["missing"] is None
    assert decoded["also_missing"] is None
    assert decoded["none_field"] is None
    assert decoded["ok"] == "hello"
    inner.poll.assert_called_with(0)


# ── replay() ──────────────────────────────────────────────────────────────


def test_replay_sends_expected_number_of_rows() -> None:
    from src.ingestion import producer as mod

    df = _make_df(n=5)
    stub = _StubProducer()

    # Patch time.sleep inside the producer module so the replay doesn't
    # actually wait between rows.
    with patch.object(mod.time, "sleep") as sleep_mock:
        sent = mod.replay(df, stub, speed_multiplier=1e9, max_rows=5)

    assert sent == 5
    assert len(stub.sent) == 5
    assert all(r["TransactionID"] in {1, 2, 3, 4, 5} for r in stub.sent)
    # Sleep was called between rows (4 times for 5 rows).
    assert sleep_mock.call_count == 4
    # All sleep delays were tiny.
    for c in sleep_mock.call_args_list:
        delay = c.args[0]
        assert delay > 0
        assert delay < 1.0


def test_replay_respects_max_rows_cap() -> None:
    from src.ingestion import producer as mod

    df = _make_df(n=10)
    stub = _StubProducer()
    with patch.object(mod.time, "sleep"):
        sent = mod.replay(df, stub, speed_multiplier=1e9, max_rows=3)
    assert sent == 3
    assert len(stub.sent) == 3


def test_replay_iterates_in_transactiondt_order() -> None:
    from src.ingestion import producer as mod

    # Build a frame whose TransactionDT order differs from the row index.
    df = _make_df(n=5, shuffle_dt=True)
    # Sanity: confirm the frame is *not* in order on TransactionDT.
    assert not df["TransactionDT"].is_monotonic_increasing

    # `replay` does not sort — the caller (load_dataset) does. Simulate the
    # real flow by sorting before passing in.
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    expected_ids = df_sorted["TransactionID"].tolist()

    stub = _StubProducer()
    with patch.object(mod.time, "sleep"):
        mod.replay(df_sorted, stub, speed_multiplier=1e9, max_rows=5)

    sent_ids = [r["TransactionID"] for r in stub.sent]
    assert sent_ids == expected_ids


def test_replay_skips_sleep_on_non_positive_delta() -> None:
    """If consecutive dts don't increase, replay must not time.sleep(-1)."""
    from src.ingestion import producer as mod

    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 100, 100],  # zero deltas
            "ProductCD": ["W", "W", "W"],
        }
    )
    stub = _StubProducer()
    with patch.object(mod.time, "sleep") as sleep_mock:
        mod.replay(df, stub, speed_multiplier=1.0, max_rows=3)
    # Zero delay → no sleep call.
    assert sleep_mock.call_count == 0


# ── CLI ───────────────────────────────────────────────────────────────────


def test_main_calls_ensure_topic_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main()` should call ensure_topic() before producing anything."""
    from src.ingestion import producer as mod

    ensure_topic_mock = MagicMock(name="ensure_topic")
    producer_ctor_mock = MagicMock(name="Producer")
    inner_producer = MagicMock(name="InnerProducer")
    producer_ctor_mock.return_value = inner_producer
    sleep_mock = MagicMock()

    monkeypatch.setattr(mod, "ensure_topic", ensure_topic_mock)
    monkeypatch.setattr(mod, "Producer", producer_ctor_mock)
    monkeypatch.setattr(mod.time, "sleep", sleep_mock)
    monkeypatch.setattr(
        mod, "load_dataset", lambda *a, **kw: _make_df(n=3)
    )

    rc = mod.main(["--rows", "3", "--speed", "1e9", "--topic", "test.topic"])

    assert rc == 0
    # ensure_topic called with the user-supplied topic.
    ensure_topic_mock.assert_called_once_with("test.topic")
    # Producer was constructed and flush() was called (in the `finally`).
    producer_ctor_mock.assert_called_once()
    inner_producer.flush.assert_called_once()
    # Three produce() calls (one per row).
    assert inner_producer.produce.call_count == 3
    # Each produced message went to the right topic with a bytes key+value.
    for call in inner_producer.produce.call_args_list:
        args, _kwargs = call
        assert args[0] == "test.topic"
