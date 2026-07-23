"""
src/ingestion/kafka_admin.py
─────────────────────────────────────────────────────────────────────────────
Kafka/Redpanda topic administration for FraudGuard.

Provides a single idempotent helper, ``ensure_topic``, that creates a topic
if it does not already exist and is a no-op when it does. This is the
canonical bootstrap step before any producer or consumer can be wired up
against the ``fraud-redpanda`` broker defined in ``docker-compose.yml``.

Usage:
    python -m src.ingestion.kafka_admin            # create transactions.raw

Programmatic:
    from src.ingestion.kafka_admin import ensure_topic
    ensure_topic("transactions.raw", num_partitions=3, replication_factor=1)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)


def ensure_topic(
    topic_name: str,
    num_partitions: int = 1,
    replication_factor: int = 1,
    bootstrap_servers: str | None = None,
) -> None:
    """Create *topic_name* on the configured broker if it does not exist.

    Idempotent: if the topic is already present, logs an info message and
    returns silently. On any genuine failure, raises ``RuntimeError`` with
    a clear message that includes the original exception.

    Parameters
    ----------
    topic_name:
        Name of the topic to ensure.
    num_partitions:
        Partition count for newly created topics (ignored if topic exists).
    replication_factor:
        Replication factor for newly created topics (ignored if topic exists).
    bootstrap_servers:
        Comma-separated ``host:port`` list. When ``None`` (the default),
        falls back to ``Settings.kafka_bootstrap_servers``.
    """
    if bootstrap_servers is None:
        bootstrap_servers = get_settings().kafka_bootstrap_servers

    log.info(
        "ensuring_topic",
        topic=topic_name,
        partitions=num_partitions,
        replication_factor=replication_factor,
        bootstrap_servers=bootstrap_servers,
    )

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    new_topic = NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
    futures = admin.create_topics([new_topic])

    for _topic, fut in futures.items():
        try:
            fut.result(timeout=10)
        except KafkaException as exc:
            # The broker reports an already-existing topic as a KafkaException
            # whose embedded KafkaError has code KafkaError.TOPIC_ALREADY_EXISTS.
            # Treat that as success (idempotent bootstrap) and re-raise anything else.
            err = exc.args[0] if exc.args else None
            if isinstance(err, KafkaError) and err.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                log.info("topic_already_exists", topic=topic_name)
                return
            log.error("topic_create_failed", topic=topic_name, error=str(exc))
            raise RuntimeError(
                f"Failed to create Kafka topic {topic_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            log.error("topic_create_failed", topic=topic_name, error=str(exc))
            raise RuntimeError(
                f"Failed to create Kafka topic {topic_name!r}: {exc}"
            ) from exc

    log.info(
        "topic_created",
        topic=topic_name,
        partitions=num_partitions,
        replication_factor=replication_factor,
    )


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    cfg = get_settings()
    ensure_topic(cfg.kafka_topic_transactions)
