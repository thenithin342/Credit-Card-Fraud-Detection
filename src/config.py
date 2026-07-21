"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised, type-safe configuration for FraudGuard.

All settings are read from environment variables (or a .env file via
python-dotenv).  Import and use ``get_settings()`` everywhere — do NOT
read os.environ directly in application code.

Usage:
    from src.config import get_settings

    cfg = get_settings()
    print(cfg.redis_host)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All FraudGuard runtime configuration, loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Kafka / Redpanda ──────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated list of Kafka/Redpanda broker addresses.",
    )
    kafka_topic_transactions: str = Field(
        default="transactions.raw",
        description="Topic name for raw transaction events.",
    )

    # ── Redis (online feature store) ──────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # ── Postgres (offline store + prediction log) ─────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="fraud_mlops")
    postgres_user: str = Field(default="fraud_mlops")
    postgres_password: str = Field(default="change_me_locally")

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy-compatible Postgres connection string."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_model_name: str = Field(default="fraud-detector")

    # ── Serving API ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    scoring_latency_sla_ms: int = Field(
        default=150,
        description="p95 latency SLA in milliseconds (from PRD).",
    )

    # ── Streaming simulation ──────────────────────────────────────────────
    replay_speed_multiplier: float = Field(
        default=60.0,
        description="Replay speed: 60 = 1 simulated hour per real minute.",
    )

    # ── Drift detection ───────────────────────────────────────────────────
    drift_psi_threshold: float = Field(
        default=0.2,
        description="PSI threshold: >0.1 minor drift, >0.2 major drift.",
    )
    drift_check_interval_minutes: int = Field(default=15)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache means the .env file is read only once per process,
    and tests can clear the cache with ``get_settings.cache_clear()``.
    """
    return Settings()
