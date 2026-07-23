"""tests/unit/test_kafka_admin.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for ``src/ingestion/kafka_admin.py``.

We mock ``confluent_kafka.admin.AdminClient`` so the suite never touches
a real broker. ``get_settings`` is also patched to return a stub
``Settings``-like object, which keeps the lru_cache singleton out of the
way and makes the bootstrap lookup path observable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ────────────────────────────────────────────────────────────


class _StubSettings:
    """Minimal stand-in for ``src.config.Settings``."""

    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self.kafka_bootstrap_servers = bootstrap_servers
        self.kafka_topic_transactions = "transactions.raw"


def _make_admin_mock(future_behavior: str | None = None) -> MagicMock:
    """Build a MagicMock standing in for an AdminClient instance.

    ``future_behavior`` is one of:
      - "exists": the future's .result() raises KafkaException wrapping
                  KafkaError(TOPIC_ALREADY_EXISTS)
      - "ok":     the future's .result() returns None (fresh creation)
      - "boom":   the future's .result() raises a generic KafkaException("boom")
    """
    from confluent_kafka import KafkaError, KafkaException

    admin = MagicMock(name="AdminClient")
    fut = MagicMock(name="Future")

    if future_behavior == "exists":
        err = KafkaError(KafkaError.TOPIC_ALREADY_EXISTS)
        fut.result.side_effect = KafkaException(err)
    elif future_behavior == "boom":
        fut.result.side_effect = KafkaException(KafkaException("boom"))
        # No embedded KafkaError — message is on the exception itself.
    else:  # "ok" / None
        fut.result.return_value = None

    admin.create_topics.return_value = {"transactions.raw": fut}
    return admin


# ── tests ──────────────────────────────────────────────────────────────


def test_ensure_topic_already_exists_succeeds() -> None:
    from src.ingestion import kafka_admin as mod

    admin_mock = _make_admin_mock("exists")

    with patch.object(mod, "AdminClient", return_value=admin_mock), patch.object(
        mod, "get_settings", return_value=_StubSettings()
    ):
        # Should not raise.
        mod.ensure_topic("transactions.raw")
        mod.AdminClient.assert_called_once_with({"bootstrap.servers": "localhost:9092"})

    admin_mock.create_topics.assert_called_once()


def test_ensure_topic_created_fresh() -> None:
    from src.ingestion import kafka_admin as mod
    from confluent_kafka.admin import NewTopic

    admin_mock = _make_admin_mock("ok")

    with patch.object(mod, "AdminClient", return_value=admin_mock), patch.object(
        mod, "get_settings", return_value=_StubSettings()
    ):
        mod.ensure_topic("transactions.raw", num_partitions=3, replication_factor=2)
        mod.AdminClient.assert_called_once_with({"bootstrap.servers": "localhost:9092"})

    # Inspect the NewTopic the module passed to create_topics.
    args, _kwargs = admin_mock.create_topics.call_args
    topics_passed = args[0]
    assert len(topics_passed) == 1
    nt = topics_passed[0]
    assert isinstance(nt, NewTopic)
    assert nt.topic == "transactions.raw"
    assert nt.num_partitions == 3
    assert nt.replication_factor == 2


def test_ensure_topic_raises_runtime_error_on_failure() -> None:
    from src.ingestion import kafka_admin as mod

    admin_mock = _make_admin_mock("boom")

    with patch.object(mod, "AdminClient", return_value=admin_mock), patch.object(
        mod, "get_settings", return_value=_StubSettings()
    ):
        with pytest.raises(RuntimeError) as excinfo:
            mod.ensure_topic("transactions.raw")

    msg = str(excinfo.value)
    assert "transactions.raw" in msg
    assert "boom" in msg


def test_ensure_topic_uses_explicit_bootstrap_servers() -> None:
    """When bootstrap_servers is supplied, get_settings() is not consulted."""
    from src.ingestion import kafka_admin as mod

    admin_mock = _make_admin_mock("ok")

    settings_stub = _StubSettings(bootstrap_servers="should-not-be-used:9092")

    with patch.object(mod, "AdminClient", return_value=admin_mock), patch.object(
        mod, "get_settings", return_value=settings_stub
    ):
        mod.ensure_topic("transactions.raw", bootstrap_servers="override:9092")
        mod.AdminClient.assert_called_once_with({"bootstrap.servers": "override:9092"})
