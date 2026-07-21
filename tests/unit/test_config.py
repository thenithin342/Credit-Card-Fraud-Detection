"""tests/unit/test_config.py — verifies src/config.py loads correctly."""

from src.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Settings should load with sensible defaults even without a .env file."""
    get_settings.cache_clear()
    cfg = Settings()
    assert cfg.redis_port == 6379
    assert cfg.api_port == 8000
    assert cfg.scoring_latency_sla_ms == 150
    assert cfg.kafka_topic_transactions == "transactions.raw"
    assert cfg.drift_psi_threshold == 0.2


def test_postgres_dsn_format() -> None:
    """DSN property should produce a valid SQLAlchemy URI."""
    cfg = Settings()
    dsn = cfg.postgres_dsn
    assert dsn.startswith("postgresql+psycopg2://")
    assert cfg.postgres_user in dsn
    assert cfg.postgres_db in dsn


def test_get_settings_is_cached() -> None:
    """get_settings() should return the same object on repeated calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
