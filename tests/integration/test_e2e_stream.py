"""
tests/integration/test_e2e_stream.py
─────────────────────────────────────────────────────────────────────────────
End-to-end integration smoke test for streaming ingestion and online scoring.

Simulates the producer → consumer → scoring → prediction_log pipeline:
- Mocks Kafka stream with synthetic transaction dictionaries
- Uses fakeredis for the online feature store
- Uses SQLite in-memory DB for the prediction log
- Mocks score_transaction_http to return a fixed ScoreResponse JSON dict
- Runs the ingestion consumer flow for 5 synthetic transactions
- Asserts that all 5 predictions land in the prediction_log table with correct fields.
"""

from __future__ import annotations

import json
from typing import Generator
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool

from src.ingestion.consumer import (
    FraudConsumer,
    _build_record,
    parse_message,
    score_transaction_http,
)
from src.ingestion.prediction_logger import (
    create_table_if_not_exists,
    log_prediction,
)


@pytest.fixture
def fake_redis() -> Generator[fakeredis.FakeRedis, None, None]:
    client = fakeredis.FakeRedis()
    yield client
    client.flushall()


@pytest.fixture
def sqlite_engine() -> Engine:
    """Create an in-memory SQLite engine that persists data across connections."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_table_if_not_exists(engine)
    return engine


@pytest.mark.integration
def test_e2e_stream_ingestion_and_logging(
    fake_redis: fakeredis.FakeRedis,
    sqlite_engine: Engine,
) -> None:
    """End-to-end stream test: 5 synthetic transactions scored and persisted."""
    fixed_score_response = {
        "transaction_id": 1,
        "fraud_score": 0.85,
        "fraud_decision": True,
        "latency_ms": 12.0,
        "model_version": "test-1",
        "top_features": [],
    }

    synthetic_transactions = [
        {
            "TransactionID": i,
            "TransactionDT": 86400 + i * 100,
            "TransactionAmt": 50.0 + i * 10,
            "card1": 1000 + i,
        }
        for i in range(1, 6)
    ]

    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    response_mock.json.return_value = fixed_score_response

    with patch("src.ingestion.consumer.requests.post", return_value=response_mock):
        # Process each synthetic message as consumer would
        for tx in synthetic_transactions:
            raw_bytes = json.dumps(tx).encode("utf-8")
            row = parse_message(raw_bytes)
            assert row == tx

            score_res = score_transaction_http(row, "http://localhost:8000")
            assert score_res == fixed_score_response

            record = _build_record(row, score_res)
            record["transaction_id"] = tx["TransactionID"]

            log_prediction(sqlite_engine, record)

    # Verification: query prediction_log table from SQLite DB
    with sqlite_engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM prediction_log")
        ).scalar()
        assert result == 5

        rows = conn.execute(
            text(
                "SELECT id, transaction_id, fraud_score, fraud_decision, latency_ms "
                "FROM prediction_log"
            )
        ).fetchall()

        assert len(rows) == 5

        for row in rows:
            # All rows have fraud_score = 0.85
            assert row.fraud_score == 0.85
            # All rows have fraud_decision = True
            assert bool(row.fraud_decision) is True
            # latency_ms is non-null for all rows
            assert row.latency_ms is not None
            assert row.latency_ms == 12.0
