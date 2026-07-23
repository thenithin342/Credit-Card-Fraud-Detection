"""
src/ingestion/producer.py
─────────────────────────────────────────────────────────────────────────────
Real-time transaction producer for FraudGuard.

Reads the IEEE-CIS Fraud Detection dataset (transaction + identity CSVs),
merges them on ``TransactionID``, sorts by ``TransactionDT``, and replays
each row through a Kafka-compatible broker (Redpanda) in time order.
Downstream services (the FastAPI scoring API, the stream consumer)
subscribe to ``Settings.kafka_topic_transactions`` and expect a steady
flow of JSON events keyed by ``TransactionID``.

Pacing:
    Δt_real = Δt_sim / replay_speed_multiplier
so ``replay_speed_multiplier=60`` (the default) means 1 simulated hour
of the IEEE-CIS dataset per real minute.

Usage:
    python -m src.ingestion.producer                     # replay all rows
    python -m src.ingestion.producer --rows 1000         # first 1000 rows
    python -m src.ingestion.producer --speed 3600        # 1 sim-hr / real-sec
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from confluent_kafka import Producer

from src.config import get_settings
from src.ingestion.kafka_admin import ensure_topic

log = structlog.get_logger(__name__)

# Default location of the IEEE-CIS raw CSVs (mirrors
# src/ingestion/download.py:43-50).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "ieee-cis" / "train_transaction.csv"
DEFAULT_IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "ieee-cis" / "train_identity.csv"


# ── Dataset loading ───────────────────────────────────────────────────────


def load_dataset(
    transaction_path: Path | str,
    identity_path: Path | str,
) -> pd.DataFrame:
    """Load + merge + time-sort the IEEE-CIS transaction + identity frames.

    Parameters
    ----------
    transaction_path:
        Path to ``train_transaction.csv``.
    identity_path:
        Path to ``train_identity.csv``.

    Returns
    -------
    pd.DataFrame
        Merged frame, sorted by ``TransactionDT`` ascending. The
        transaction columns come first, then the identity-only columns
        (any ``TransactionID`` in identity is dropped on merge).
    """
    transaction_path = Path(transaction_path)
    identity_path = Path(identity_path)

    log.info(
        "loading_dataset",
        transaction=str(transaction_path),
        identity=str(identity_path),
    )
    txn = pd.read_csv(transaction_path)
    ident = pd.read_csv(identity_path)

    merged = txn.merge(ident, on="TransactionID", how="left")
    merged = merged.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

    log.info("dataset_loaded", rows=len(merged), cols=merged.shape[1])
    return merged


# ── Producer ──────────────────────────────────────────────────────────────


class TransactionProducer:
    """Thin wrapper around ``confluent_kafka.Producer`` for transaction rows.

    Each row is serialised to JSON with ``NaN`` values coerced to
    ``null`` (JSON has no NaN). The transaction ID is used as the
    Kafka message key so all events for a given transaction land on the
    same partition when partitioning is keyed.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": bootstrap_servers})
        log.info(
            "producer_initialised",
            topic=topic,
            bootstrap_servers=bootstrap_servers,
        )

    @staticmethod
    def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
        """Coerce NaN/NaT/``None`` to JSON-``null`` and stringify anything else."""
        cleaned: dict[str, Any] = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
                cleaned[k] = None
            else:
                # numpy scalars / pandas Timestamps / Decimals: fall through to str.
                cleaned[k] = v
        return cleaned

    def send(self, row: dict[str, Any]) -> None:
        """Serialise *row* to JSON and enqueue it on the configured topic."""
        cleaned = self._clean_row(row)
        key = str(cleaned["TransactionID"]).encode("utf-8")
        value = json.dumps(cleaned, default=str).encode("utf-8")
        self.producer.produce(self.topic, key=key, value=value)
        # Service any delivery callbacks without blocking the hot loop.
        self.producer.poll(0)

    def flush(self) -> None:
        """Block until all queued messages are delivered (or 30s timeout)."""
        self.producer.flush(timeout=30)


# ── Replay loop ───────────────────────────────────────────────────────────


def replay(
    df: pd.DataFrame,
    producer: TransactionProducer,
    speed_multiplier: float,
    max_rows: int | None = None,
) -> int:
    """Replay *df* row-by-row in ``TransactionDT`` order, paced to real time.

    The delay between consecutive events is
    ``Δt_real = Δt_sim / speed_multiplier`` where ``Δt_sim`` is the
    difference in the dataset's ``TransactionDT`` column (a
    delta-seconds field). Non-positive deltas are skipped (the IEEE-CIS
    dataset has a handful near the start), and the very first row is
    sent immediately.

    Parameters
    ----------
    df:
        Time-sorted frame. ``TransactionDT`` is required.
    producer:
        A :class:`TransactionProducer` (or any object exposing
        ``send(dict)``).
    speed_multiplier:
        ``Settings.replay_speed_multiplier`` (default 60.0).
    max_rows:
        Optional cap on rows sent; ``None`` means the full frame.

    Returns
    -------
    int
        Number of rows actually sent.
    """
    if max_rows is None or max_rows > len(df):
        max_rows = len(df)

    dts = df["TransactionDT"].to_numpy()
    rows = df.to_dict(orient="records")

    start = time.monotonic()
    sent = 0
    prev_dt: int | float | None = None

    log.info(
        "replay_starting",
        rows=max_rows,
        speed_multiplier=speed_multiplier,
        total_rows=len(df),
    )

    for i in range(max_rows):
        dt = dts[i]
        if prev_dt is not None:
            delay = (dt - prev_dt) / speed_multiplier
            if delay > 0:
                time.sleep(delay)
        prev_dt = dt

        producer.send(rows[i])
        sent += 1

        if sent % 1000 == 0:
            log.info(
                "replay_progress",
                rows_sent=sent,
                elapsed_seconds=round(time.monotonic() - start, 3),
            )

    log.info(
        "replay_finished",
        rows_sent=sent,
        elapsed_seconds=round(time.monotonic() - start, 3),
    )
    return sent


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
        description="Replay the IEEE-CIS dataset through Redpanda."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="How many transactions to send (default: all)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=cfg.replay_speed_multiplier,
        help=(
            "Replay speed multiplier: Δt_real = Δt_sim / speed. "
            f"Default from settings: {cfg.replay_speed_multiplier}"
        ),
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=cfg.kafka_topic_transactions,
        help=f"Target Kafka topic (default from settings: {cfg.kafka_topic_transactions})",
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_TRANSACTION_PATH,
        help="Path to train_transaction.csv",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=DEFAULT_IDENTITY_PATH,
        help="Path to train_identity.csv",
    )
    args = parser.parse_args(argv)

    # Bootstrap: make sure the topic exists before we start producing.
    ensure_topic(args.topic)

    df = load_dataset(args.transactions, args.identity)
    producer = TransactionProducer(cfg.kafka_bootstrap_servers, args.topic)

    try:
        n = replay(df, producer, args.speed, max_rows=args.rows)
    finally:
        producer.flush()

    log.info("replay_done", rows=n, topic=args.topic, speed=args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
