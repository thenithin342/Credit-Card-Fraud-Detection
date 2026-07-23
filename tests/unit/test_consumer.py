"""tests/unit/test_consumer.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for ``src/ingestion/consumer.py``.

We mock ``confluent_kafka.Consumer``, ``requests``, and the
``log_prediction`` helper so the suite never talks to a real broker,
API, or Postgres — it runs in milliseconds.

What we verify
--------------
* ``parse_message`` returns ``{}`` on malformed JSON (and on a non-dict
  top-level value) and emits a warning.
* ``score_transaction_http`` returns ``None`` on a ``requests.Timeout``
  (and on any ``RequestException``), and skips messages with missing
  required fields.
* The run loop processes exactly ``max_messages`` messages when the
  cap is set, calls ``log_prediction`` once per message, and commits
  once per success.
* A ``log_prediction`` that raises does not crash the consumer — the
  loop continues and the error counter increments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────


class _MockMsg:
    """Minimal stand-in for a confluent_kafka.Message."""

    def __init__(self, value: bytes, *, error=None) -> None:
        self._value = value
        self._error = error

    def value(self) -> bytes:
        return self._value

    def error(self):
        return self._error

    def topic(self) -> str:
        return "transactions.raw"

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return 0


class _FakeConsumer:
    """Yields a fixed list of messages from ``poll()``, then ``None``.

    Records every call to ``subscribe``, ``commit``, ``close`` so tests
    can assert on them.
    """

    def __init__(self, messages: list[_MockMsg]) -> None:
        self._messages = list(messages)
        self._index = 0
        self.subscribe_calls: list[list[str]] = []
        self.commit_calls: int = 0
        self.close_calls: int = 0

    def subscribe(self, topics: list[str]) -> None:
        self.subscribe_calls.append(list(topics))

    def poll(self, timeout: float):
        if self._index < len(self._messages):
            msg = self._messages[self._index]
            self._index += 1
            return msg
        return None

    def commit(self, asynchronous: bool = True) -> None:  # noqa: ARG002
        self.commit_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _good_response_payload() -> dict:
    """A minimally-valid ``ScoreResponse`` body."""
    return {
        "transaction_id": 1,
        "fraud_score": 0.12,
        "is_fraud": False,
        "threshold": 0.5,
        "top_features": [
            {"feature_name": "C1", "contribution": 0.03, "value": 12.5},
        ],
        "latency_ms": 42.0,
        "model_version": "champion-1",
    }


def _make_consumer_with_fake_kafka(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[_MockMsg],
):
    """Patch ``consumer_mod.Consumer`` to return a ``_FakeConsumer`` and
    import the module fresh so the patch takes effect."""
    from src.ingestion import consumer as mod

    fake = _FakeConsumer(messages)
    monkeypatch.setattr(mod, "Consumer", lambda cfg: fake)
    return mod, fake


def _make_engine_mock() -> MagicMock:
    """MagicMock that quacks like a SQLAlchemy Engine (for FraudConsumer)."""
    return MagicMock(spec=Engine)


def _make_requests_mock(mod) -> MagicMock:
    """MagicMock that retains real requests exception classes."""
    m = MagicMock()
    m.Timeout = mod.requests.Timeout
    m.HTTPError = mod.requests.HTTPError
    m.RequestException = mod.requests.RequestException
    return m


# ── parse_message ─────────────────────────────────────────────────────


def test_parse_message_returns_dict_on_valid_json() -> None:
    from src.ingestion.consumer import parse_message

    raw = json.dumps({"TransactionID": 1, "TransactionAmt": 99.9}).encode("utf-8")
    parsed = parse_message(raw)
    assert parsed == {"TransactionID": 1, "TransactionAmt": 99.9}


def test_parse_message_returns_empty_dict_on_malformed_json() -> None:
    from src.ingestion.consumer import parse_message

    # Garbage input → empty dict + warning.
    assert parse_message(b"not json at all") == {}
    # Trailing-comma invalid JSON → empty dict.
    assert parse_message(b"{,}") == {}
    # Top-level array, not a dict → empty dict.
    assert parse_message(b"[1, 2, 3]") == {}
    # None input → empty dict (TypeError).
    assert parse_message(b"") == {}


# ── score_transaction_http ────────────────────────────────────────────


def test_score_transaction_http_returns_dict_on_success() -> None:
    from src.ingestion import consumer as mod

    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    response_mock.json.return_value = _good_response_payload()

    requests_mock = _make_requests_mock(mod)
    requests_mock.post.return_value = response_mock

    with patch.object(mod, "requests", requests_mock):
        result = mod.score_transaction_http(
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 250.0,
                "card1": 12345,
            },
            "http://api:8000",
        )

    assert result is not None
    assert result["transaction_id"] == 1
    assert result["fraud_score"] == 0.12
    # The body must contain the four required fields.
    args, kwargs = requests_mock.post.call_args
    assert args[0] == "http://api:8000/v1/score"
    body = kwargs["json"]
    assert body["transaction_id"] == 1
    assert body["TransactionDT"] == 86400
    assert body["TransactionAmt"] == 250.0
    assert body["card1"] == 12345
    assert kwargs["timeout"] == 5.0


def test_score_transaction_http_returns_none_on_timeout() -> None:
    from src.ingestion import consumer as mod

    requests_mock = _make_requests_mock(mod)
    requests_mock.post.side_effect = mod.requests.Timeout("boom")

    with patch.object(mod, "requests", requests_mock):
        result = mod.score_transaction_http(
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 250.0,
                "card1": 12345,
            },
            "http://api:8000",
        )

    assert result is None
    requests_mock.post.assert_called_once()


def test_score_transaction_http_returns_none_on_5xx() -> None:
    from src.ingestion import consumer as mod

    response_mock = MagicMock()
    response_mock.raise_for_status.side_effect = mod.requests.HTTPError("503")

    requests_mock = _make_requests_mock(mod)
    requests_mock.post.return_value = response_mock

    with patch.object(mod, "requests", requests_mock):
        result = mod.score_transaction_http(
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 250.0,
                "card1": 12345,
            },
            "http://api:8000",
        )

    assert result is None


def test_score_transaction_http_skips_missing_required_fields() -> None:
    from src.ingestion import consumer as mod

    requests_mock = _make_requests_mock(mod)
    # No request must go out.
    with patch.object(mod, "requests", requests_mock):
        # Missing TransactionDT.
        result = mod.score_transaction_http(
            {"TransactionID": 1, "TransactionAmt": 250.0, "card1": 1},
            "http://api:8000",
        )
        assert result is None
        # Missing TransactionAmt.
        result = mod.score_transaction_http(
            {"TransactionID": 1, "TransactionDT": 86400, "card1": 1},
            "http://api:8000",
        )
        assert result is None
        # Explicit None values for required fields also skip.
        result = mod.score_transaction_http(
            {
                "TransactionID": 1,
                "TransactionDT": None,
                "TransactionAmt": 250.0,
                "card1": 1,
            },
            "http://api:8000",
        )
        assert result is None

    requests_mock.post.assert_not_called()


# ── run() loop ────────────────────────────────────────────────────────


def test_run_loop_processes_exactly_n_messages_with_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With max_messages=5, the loop must process exactly 5 and return 5."""
    from src.ingestion import consumer as mod

    payloads = [
        json.dumps(
            {
                "TransactionID": i,
                "TransactionDT": 86_400 + i * 60,
                "TransactionAmt": 100.0 + i,
                "card1": 1000 + i,
            }
        ).encode("utf-8")
        for i in range(1, 6)
    ]
    messages = [_MockMsg(p) for p in payloads]
    consumer_mod, fake = _make_consumer_with_fake_kafka(monkeypatch, messages)

    # requests.post returns a 2xx with a valid ScoreResponse body.
    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    response_mock.json.return_value = _good_response_payload()
    requests_mock = _make_requests_mock(consumer_mod)
    requests_mock.post.return_value = response_mock

    # log_prediction becomes a MagicMock so we can count calls.
    log_mock = MagicMock()
    monkeypatch.setattr(consumer_mod, "log_prediction", log_mock)
    monkeypatch.setattr(consumer_mod, "requests", requests_mock)

    engine_mock = _make_engine_mock()
    consumer = consumer_mod.FraudConsumer(
        bootstrap_servers="x:1",
        topic="transactions.raw",
        api_base_url="http://api:8000",
        engine=engine_mock,
    )

    processed = consumer.run(max_messages=5)

    assert processed == 5
    assert consumer.errors == 0
    assert log_mock.call_count == 5
    # 5 successful processes → 5 synchronous commits.
    assert fake.commit_calls == 5
    # Consumer subscribed to the right topic.
    assert fake.subscribe_calls == [["transactions.raw"]]
    # Consumer closed in the finally block.
    assert fake.close_calls == 1
    # One HTTP call per message.
    assert requests_mock.post.call_count == 5


def test_run_loop_commits_on_score_failure_to_skip_poison_pill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If score_transaction_http returns None, the loop commits the offset
    (poison-pill avoidance) and continues.  log_prediction must NOT be called."""
    from src.ingestion import consumer as mod

    payloads = [
        json.dumps(
            {
                "TransactionID": i,
                "TransactionDT": 86_400 + i * 60,
                "TransactionAmt": 100.0 + i,
                "card1": 1000 + i,
            }
        ).encode("utf-8")
        for i in range(1, 4)
    ]
    messages = [_MockMsg(p) for p in payloads]
    consumer_mod, fake = _make_consumer_with_fake_kafka(monkeypatch, messages)

    # All requests time out → score_transaction_http returns None.
    requests_mock = _make_requests_mock(consumer_mod)
    requests_mock.post.side_effect = mod.requests.Timeout("boom")
    monkeypatch.setattr(consumer_mod, "requests", requests_mock)

    log_mock = MagicMock()
    monkeypatch.setattr(consumer_mod, "log_prediction", log_mock)

    engine_mock = _make_engine_mock()
    consumer = consumer_mod.FraudConsumer(
        bootstrap_servers="x:1",
        topic="transactions.raw",
        api_base_url="http://api:8000",
        engine=engine_mock,
    )

    processed = consumer.run(max_messages=3)

    # processed counts all handled messages (3); errors increments for each.
    assert processed == 3
    assert consumer.errors == 3
    # Each score failure committed the offset (poison-pill skip).
    assert fake.commit_calls == 3
    # log_prediction must NOT be called when the score failed.
    log_mock.assert_not_called()


def test_run_loop_tolerates_log_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If log_prediction leaks an exception, the loop must not crash."""
    from src.ingestion import consumer as mod

    payloads = [
        json.dumps(
            {
                "TransactionID": i,
                "TransactionDT": 86_400 + i * 60,
                "TransactionAmt": 100.0 + i,
                "card1": 1000 + i,
            }
        ).encode("utf-8")
        for i in range(1, 4)
    ]
    messages = [_MockMsg(p) for p in payloads]
    consumer_mod, fake = _make_consumer_with_fake_kafka(monkeypatch, messages)

    # Successful score.
    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    response_mock.json.return_value = _good_response_payload()
    requests_mock = _make_requests_mock(consumer_mod)
    requests_mock.post.return_value = response_mock
    monkeypatch.setattr(consumer_mod, "requests", requests_mock)

    # log_prediction leaks an exception (worst case — it's supposed to
    # swallow, but we guard against future regression).
    log_mock = MagicMock(side_effect=Exception("db down"))
    monkeypatch.setattr(consumer_mod, "log_prediction", log_mock)

    engine_mock = _make_engine_mock()
    consumer = consumer_mod.FraudConsumer(
        bootstrap_servers="x:1",
        topic="transactions.raw",
        api_base_url="http://api:8000",
        engine=engine_mock,
    )

    # The run() call itself must not raise.
    processed = consumer.run(max_messages=3)

    # Loop processed 3 messages (did not crash), errors bumped to 3.
    assert processed == 3
    assert consumer.errors == 3
    assert log_mock.call_count == 3
    # Consumer closed cleanly despite the failures.
    assert fake.close_calls == 1


def test_run_loop_does_not_commit_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed message must NOT commit — let it re-deliver."""
    from src.ingestion import consumer as mod

    messages = [
        _MockMsg(b"not json at all"),
        _MockMsg(
            json.dumps(
                {
                    "TransactionID": 2,
                    "TransactionDT": 86_400,
                    "TransactionAmt": 100.0,
                    "card1": 1,
                }
            ).encode("utf-8")
        ),
    ]
    consumer_mod, fake = _make_consumer_with_fake_kafka(monkeypatch, messages)

    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    response_mock.json.return_value = _good_response_payload()
    requests_mock = _make_requests_mock(consumer_mod)
    requests_mock.post.return_value = response_mock
    monkeypatch.setattr(consumer_mod, "requests", requests_mock)

    log_mock = MagicMock()
    monkeypatch.setattr(consumer_mod, "log_prediction", log_mock)

    engine_mock = _make_engine_mock()
    consumer = consumer_mod.FraudConsumer(
        bootstrap_servers="x:1",
        topic="transactions.raw",
        api_base_url="http://api:8000",
        engine=engine_mock,
    )

    processed = consumer.run(max_messages=2)

    # Both messages polled (1 failed parse, 1 succeeded score); parse-failure committed nothing.
    assert processed == 2
    assert consumer.errors == 1
    # Exactly one commit (for the successful one).
    assert fake.commit_calls == 1
