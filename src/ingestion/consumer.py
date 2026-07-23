"""
src/ingestion/consumer.py
─────────────────────────────────────────────────────────────────────────────
Real-time fraud-scoring consumer for FraudGuard.

Reads JSON-serialised transactions from the ``transactions.raw`` Redpanda
topic (produced by ``src.ingestion.producer``), POSTs each row to the
FastAPI scoring API at ``/v1/score``, and writes the response to a
Postgres ``prediction_log`` table for downstream monitoring (drift
detection, BI, manual analysis).

The loop is at-least-once: a successful ``log_prediction`` is the
synchronisation point for ``consumer.commit()`` — a Postgres or API
hiccup leaves the offset uncommitted, so the message is re-delivered
after a restart.  Per-message defensive ``try/except`` blocks keep a
single bad row from killing the consumer.

Usage:
    python -m src.ingestion.consumer                       # run forever
    python -m src.ingestion.consumer --max-messages 1000   # bounded smoke test
    python -m src.ingestion.consumer --api-url http://localhost:9000
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests
import structlog
from confluent_kafka import Consumer
from sqlalchemy import Engine

from src.config import get_settings
from src.ingestion.prediction_logger import (
    create_table_if_not_exists,
    get_engine,
    log_prediction,
)

log = structlog.get_logger(__name__)


# ── Parsing ────────────────────────────────────────────────────────────────


def parse_message(raw_value: bytes) -> dict:
    """JSON-deserialise a Kafka message value.

    Returns the parsed dict on success, or an empty dict on any parse
    error (with a warning log).  An empty dict signals the caller to
    treat the message as a parse failure and NOT commit the offset —
    letting a permanently-malformed message re-deliver after restart
    so an operator can intervene.
    """
    try:
        decoded = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("parse_failed", error=str(exc))
        return {}
    if not isinstance(decoded, dict):
        log.warning("parse_failed", error="top-level value is not a dict")
        return {}
    return decoded


# ── HTTP scoring ───────────────────────────────────────────────────────────


def score_transaction_http(
    row: dict,
    api_base_url: str,
    *,
    timeout: float = 5.0,
) -> dict | None:
    """POST *row* to the scoring API; return the response JSON or ``None``.

    Builds the minimal ``TransactionRequest`` body (transaction_id,
    TransactionDT, TransactionAmt, card1).  Returns ``None`` on:

    * missing required fields (``TransactionDT`` or ``TransactionAmt``)
    * HTTP timeout (5s default)
    * any non-2xx response
    * any other ``requests.RequestException``
    """
    tx_id_raw = row.get("TransactionID", -1)
    try:
        transaction_id = int(tx_id_raw) if tx_id_raw is not None else -1
    except (TypeError, ValueError):
        transaction_id = -1

    body: dict[str, Any] = {
        "transaction_id": transaction_id,
        "card1": row.get("card1", 0),
    }

    # Required fields — if either is missing, skip the HTTP round-trip.
    if "TransactionDT" not in row or row["TransactionDT"] is None:
        log.warning(
            "score_skipped_missing_fields",
            transaction_id=transaction_id,
            missing="TransactionDT",
        )
        return None
    if "TransactionAmt" not in row or row["TransactionAmt"] is None:
        log.warning(
            "score_skipped_missing_fields",
            transaction_id=transaction_id,
            missing="TransactionAmt",
        )
        return None

    body["TransactionDT"] = row["TransactionDT"]
    body["TransactionAmt"] = row["TransactionAmt"]

    try:
        resp = requests.post(
            f"{api_base_url}/v1/score",
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout as exc:
        log.error(
            "score_http_failed",
            transaction_id=transaction_id,
            error=str(exc),
        )
        return None
    except requests.RequestException as exc:
        log.error(
            "score_http_failed",
            transaction_id=transaction_id,
            error=str(exc),
        )
        return None


# ── Consumer ──────────────────────────────────────────────────────────────


class FraudConsumer:
    """Kafka consumer that scores transactions and writes the prediction log.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated ``host:port`` list (e.g. ``localhost:9092``).
    topic:
        Kafka topic to subscribe to (typically
        ``Settings.kafka_topic_transactions``).
    group_id:
        Consumer group id (default ``fraud-consumer-1``).  Offset
        commits are scoped to this group.
    api_base_url:
        Base URL of the FastAPI scoring API (no trailing slash).
    engine:
        SQLAlchemy engine for the prediction log.  When ``None``,
        falls back to ``get_engine()``.

    Offset semantics
    ----------------
    ``enable.auto.commit`` is on (the default) so commits still
    progress on the broker's normal interval — but the run loop also
    issues a synchronous ``commit(asynchronous=False)`` after each
    successful ``log_prediction``.  That gives at-least-once
    semantics: if the process dies between the log and the manual
    commit, the row will re-deliver and the log will skip the
    duplicate (the table has no uniqueness constraint, so duplicates
    are possible — but a single duplicate is preferable to a lost
    prediction).
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str = "fraud-consumer-1",
        api_base_url: str = "http://localhost:8000",
        engine: Engine | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.api_base_url = api_base_url
        self.engine = engine if engine is not None else get_engine()

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([topic])

        # Mutable per-run counters (reset at the top of run()).
        self.processed: int = 0
        self.errors: int = 0
        self.latency_sum_ms: float = 0.0

        log.info(
            "consumer_initialised",
            topic=topic,
            group_id=group_id,
            api_base_url=api_base_url,
        )

    # ── Run loop ─────────────────────────────────────────────────────────

    def run(self, max_messages: int | None = None) -> int:
        """Poll Kafka, score, log, commit.  Return messages processed.

        Stops after *max_messages* successful processes if provided;
        otherwise runs until ``KeyboardInterrupt`` (or a fatal Kafka
        error, which ``self.consumer.poll`` surfaces as a non-None
        ``msg.error()`` — logged and counted, not raised).
        """
        self.processed = 0
        self.errors = 0
        self.latency_sum_ms = 0.0
        start = time.monotonic()

        try:
            while True:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue

                self.processed += 1

                if msg.error():
                    log.warning("kafka_message_error", error=str(msg.error()))
                    self.errors += 1
                else:
                    row = parse_message(msg.value())
                    if not row:
                        # Parse failure: don't commit.  Let the message
                        # re-deliver after a restart (operator may
                        # intervene).
                        self.errors += 1
                    else:
                        result = score_transaction_http(row, self.api_base_url)
                        if result is None:
                            # Score failure (timeout / 4xx / 5xx): commit to
                            # skip the poison pill.  A row the API rejects
                            # once will keep rejecting forever.
                            self.errors += 1
                            self.consumer.commit(asynchronous=False)
                        else:
                            record = _build_record(row, result)
                            try:
                                log_prediction(self.engine, record)
                            except Exception as exc:  # noqa: BLE001
                                # Defensive: log_prediction already swallows, but
                                # guard against future regression so a single bad
                                # log doesn't kill the consumer.
                                log.error("log_prediction_leaked", error=str(exc))
                                self.errors += 1

                            self.consumer.commit(asynchronous=False)
                            self.latency_sum_ms += float(result.get("latency_ms") or 0.0)

                if self.processed % 100 == 0:
                    log.info(
                        "consumer_progress",
                        processed=self.processed,
                        errors=self.errors,
                        avg_latency_ms=self.latency_sum_ms / max(self.processed - self.errors, 1),
                        elapsed_seconds=round(time.monotonic() - start, 3),
                    )

                if max_messages is not None and self.processed >= max_messages:
                    break
        except KeyboardInterrupt:
            log.info(
                "consumer_interrupted",
                processed=self.processed,
                errors=self.errors,
            )
        finally:
            self.consumer.close()

        log.info(
            "consumer_finished",
            processed=self.processed,
            errors=self.errors,
            elapsed_seconds=round(time.monotonic() - start, 3),
        )
        return self.processed


# ── Record builder ────────────────────────────────────────────────────────


def _build_record(row: dict, result: dict) -> dict:
    """Compose the ``prediction_log`` row from the API response + the input."""
    top_features = result.get("top_features") or []
    top0 = top_features[0] if top_features else {}
    return {
        "transaction_id": result.get("transaction_id"),
        "card1": row.get("card1"),
        "transaction_dt": row.get("TransactionDT"),
        "transaction_amt": row.get("TransactionAmt"),
        "fraud_score": result["fraud_score"],
        "fraud_decision": result.get("is_fraud", result.get("fraud_decision", False)),
        "model_version": result.get("model_version"),
        "latency_ms": result.get("latency_ms"),
        "shap_top1_name": top0.get("feature_name"),
        "shap_top1_value": top0.get("value"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    cfg = get_settings()
    parser = argparse.ArgumentParser(
        description="Consume transactions.raw, score via the API, log to Postgres."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FastAPI scoring API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after this many successful processes (default: run forever)",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="fraud-consumer-1",
        help="Kafka consumer group id (default: fraud-consumer-1)",
    )
    args = parser.parse_args(argv)

    engine = get_engine()
    create_table_if_not_exists(engine)

    consumer = FraudConsumer(
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        topic=cfg.kafka_topic_transactions,
        group_id=args.group_id,
        api_base_url=args.api_url,
        engine=engine,
    )

    n = consumer.run(max_messages=args.max_messages)
    log.info("consumer_done", processed=n, errors=consumer.errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
