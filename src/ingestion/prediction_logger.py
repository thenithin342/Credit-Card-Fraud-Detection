"""
src/ingestion/prediction_logger.py
─────────────────────────────────────────────────────────────────────────────
Postgres-backed prediction log for the FraudGuard stream consumer.

Each scored transaction is written to the ``prediction_log`` table so
downstream consumers (the Evidently drift monitor, manual analysis,
BI dashboards) can inspect what the model actually predicted over time.

Public API
----------
``get_engine(dsn=None)``
    Cached SQLAlchemy engine.  ``None`` dsn → ``get_settings().postgres_dsn``.

``create_table_if_not_exists(engine)``
    Idempotent DDL — safe to call on every consumer boot.

``log_prediction(engine, record)``
    INSERT one row, never raises (swallows and logs).  A Postgres hiccup
    must not crash the consumer loop.

The DDL is intentionally narrow (one table, no ORM) — the rest of the
codebase does not use SQLAlchemy, and dragging in a ``Base``/``Session``
model would add a maintenance burden for zero benefit.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import structlog
from sqlalchemy import Engine, create_engine, text

from src.config import get_settings

log = structlog.get_logger(__name__)

# ── Module-level engine cache ──────────────────────────────────────────────
#
# The cache is keyed by the DSN string so a config change (env var
# override, test fixture) rebuilds the engine.  Tests inject a mock
# engine directly into ``FraudConsumer`` and never touch this cache.

_engine: Engine | None = None
_engine_dsn: str | None = None

# ── DDL ────────────────────────────────────────────────────────────────────

_DDL: str = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id               SERIAL PRIMARY KEY,
    transaction_id   BIGINT       NOT NULL,
    card1            BIGINT,
    transaction_dt   BIGINT,
    transaction_amt  FLOAT,
    fraud_score      FLOAT        NOT NULL,
    fraud_decision   BOOLEAN      NOT NULL,
    model_version    TEXT,
    latency_ms       FLOAT,
    shap_top1_name   TEXT,
    shap_top1_value  FLOAT,
    created_at       TIMESTAMPTZ  DEFAULT NOW()
)
""".strip()

# Columns the INSERT binds — order matches the table schema above.
_INSERT_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "card1",
    "transaction_dt",
    "transaction_amt",
    "fraud_score",
    "fraud_decision",
    "model_version",
    "latency_ms",
    "shap_top1_name",
    "shap_top1_value",
)


# ── Engine factory ────────────────────────────────────────────────────────


def get_engine(dsn: str | None = None) -> Engine:
    """Return a cached SQLAlchemy engine for *dsn* (or settings default).

    Parameters
    ----------
    dsn:
        SQLAlchemy-compatible Postgres URL.  When ``None`` (the default),
        falls back to ``Settings.postgres_dsn``.

    Returns
    -------
    Engine
        Process-wide cached engine, rebuilt only when the DSN changes.
    """
    global _engine, _engine_dsn

    if dsn is None:
        dsn = get_settings().postgres_dsn

    if _engine is None or _engine_dsn != dsn:
        log.info("creating_postgres_engine", dsn=dsn)
        _engine = create_engine(dsn, pool_pre_ping=True, future=True)
        _engine_dsn = dsn

    return _engine


_SQLITE_DDL: str = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id   BIGINT       NOT NULL,
    card1            BIGINT,
    transaction_dt   BIGINT,
    transaction_amt  FLOAT,
    fraud_score      FLOAT        NOT NULL,
    fraud_decision   BOOLEAN      NOT NULL,
    model_version    TEXT,
    latency_ms       FLOAT,
    shap_top1_name   TEXT,
    shap_top1_value  FLOAT,
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
)
""".strip()


# ── Schema bootstrap ──────────────────────────────────────────────────────


def create_table_if_not_exists(engine: Engine) -> None:
    """Create ``prediction_log`` if it does not already exist (idempotent)."""
    log.info("ensuring_prediction_log_table")
    ddl = _SQLITE_DDL if engine.dialect.name == "sqlite" else _DDL
    with engine.begin() as conn:
        conn.execute(text(ddl))
    log.info("prediction_log_table_ready")


# ── Row writer ────────────────────────────────────────────────────────────


def log_prediction(engine: Engine, record: dict) -> None:
    """INSERT one *record* into ``prediction_log``; never raises.

    *record* keys map 1:1 to ``prediction_log`` columns.  Missing keys
    become SQL NULL via ``dict.get(col, None)``.  Any exception is
    caught and logged — a failed write must not crash the consumer
    loop, which would block the partition.

    Parameters
    ----------
    engine:
        A SQLAlchemy ``Engine`` (typically the cached module-level one).
    record:
        Dict with at minimum ``transaction_id``, ``fraud_score``,
        ``fraud_decision`` (the NOT-NULL columns).  All other columns
        are optional and default to NULL when absent.
    """
    insert = text(
        "INSERT INTO prediction_log ("
        + ", ".join(_INSERT_COLUMNS)
        + ") VALUES ("
        + ", ".join(f":{c}" for c in _INSERT_COLUMNS)
        + ")"
    )
    try:
        params = {col: record.get(col) for col in _INSERT_COLUMNS}
        with engine.begin() as conn:
            conn.execute(insert, params)
    except Exception as exc:  # noqa: BLE001
        # Swallow + log: a Postgres hiccup must not crash the consumer
        # loop, which would block the partition.
        log.error(
            "log_prediction_failed",
            error=str(exc),
            transaction_id=record.get("transaction_id"),
        )
