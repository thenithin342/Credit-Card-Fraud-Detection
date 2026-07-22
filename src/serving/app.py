"""src/serving/app.py
────────────────────────────────────────────────────────────────────────
Phase 3 — FastAPI scoring service for FraudGuard.

Endpoints
---------
* ``GET  /health``    — liveness + the loaded model version/stage.
* ``POST /v1/score``  — score a single transaction.  Returns a
                        ``ScoreResponse`` with the fraud probability,
                        decision, top-k SHAP, and end-to-end latency.
* ``GET  /metrics``   — Prometheus exposition (provided by
                        ``prometheus-fastapi-instrumentator``).

Startup
-------
A lifespan context loads the ``ModelBundle`` and SHAP ``TreeExplainer``
exactly once per process.  Tests can pre-load a fake bundle via
``app.dependency_overrides`` (see ``tests/unit/test_serving.py``).

The Redis client is created lazily on first request so the API can
boot even when Redis is unreachable in dev environments.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
import yaml
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import ValidationError

from src.serving.explainer import get_explainer, reset_explainers_for_tests
from src.serving.model_loader import (
    ModelBundle,
    get_model_bundle,
    reset_model_bundle_for_tests,
)
from src.serving.predictor import score_transaction
from src.serving.schemas import ScoreResponse, TransactionRequest

log = structlog.get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PARAMS_FILE = os.path.join(PROJECT_ROOT, "params.yaml")


# ── Lightweight lazy Redis wrapper so the API boots without Redis ──────────


class _LazyRedisStore:
    """Defers the real Redis connection until the first call.

    The production code path uses ``OnlineFeatureStore``; this wrapper
    is here so unit tests (and dev runs without Redis) can patch
    ``app.state.online_store`` before the first request.
    """

    def __init__(self) -> None:
        self._impl: Any = None

    def _ensure(self) -> Any:
        if self._impl is None:
            # Local import so importing this module never requires redis.
            try:
                from src.config import get_settings

                cfg = get_settings()
                use_fake_env = os.environ.get("USE_FAKEREDIS")
                use_fake = (
                    use_fake_env.lower() in ("true", "1", "yes")
                    if use_fake_env is not None
                    else getattr(cfg, "use_fakeredis", True)
                )
            except Exception:  # noqa: BLE001
                use_fake = True

            if use_fake:
                import fakeredis  # type: ignore[import-untyped]

                from src.features.online_store import OnlineFeatureStore

                self._impl = OnlineFeatureStore(fakeredis.FakeRedis())
                return self._impl

            try:
                import redis  # type: ignore[import-untyped]

                from src.features.online_store import OnlineFeatureStore

                host = os.environ.get("REDIS_HOST", "localhost")
                port = int(os.environ.get("REDIS_PORT", "6379"))
                client = redis.Redis(host=host, port=port)
                client.ping()
                self._impl = OnlineFeatureStore(client)
            except Exception as exc:  # noqa: BLE001
                log.warning("redis_unavailable_falling_back_to_fakeredis", error=str(exc))
                import fakeredis  # type: ignore[import-untyped]

                from src.features.online_store import OnlineFeatureStore

                self._impl = OnlineFeatureStore(fakeredis.FakeRedis())
        return self._impl

    def get_card_features(self, card_id: str) -> dict | None:
        return self._ensure().get_card_features(card_id)

    def update_after_transaction(self, card_id: str, amount: float, ts: int) -> None:
        return self._ensure().update_after_transaction(card_id, amount, ts)


def _read_serving_params() -> dict[str, Any]:
    """Read the ``serving`` block from ``params.yaml``.

    Falls back to sensible defaults if the file is missing (e.g. when
    the API is deployed as a wheel with no params.yaml alongside).
    """
    defaults: dict[str, Any] = {
        "threshold": 0.5,
        "shap_top_k": 5,
        "latency_sla_ms": 150,
    }
    try:
        with open(PARAMS_FILE, encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return defaults
    serving = params.get("serving", {}) if isinstance(params, dict) else {}
    merged = {**defaults, **{k: serving[k] for k in defaults if k in serving}}
    return merged


# ── Lifespan: load bundle + explainer once per process ─────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model and warm the SHAP explainer on startup."""
    params = _read_serving_params()
    app.state.serving_params = params
    app.state.online_store = _LazyRedisStore()

    try:
        bundle: ModelBundle = get_model_bundle()
    except Exception as exc:  # noqa: BLE001
        # We log but do NOT raise — endpoints that need the bundle
        # will surface a clear 500.  This keeps ``/health`` alive
        # even when MLflow is unreachable.
        log.error("model_load_failed_at_startup", error=str(exc))
        bundle = None  # type: ignore[assignment]

    if bundle is not None:
        # Building TreeExplainer is expensive; warm it here so the
        # first request doesn't pay the cost.
        try:
            import numpy as np
            import pandas as pd
            explainer = get_explainer(bundle.model)
            dummy_X = pd.DataFrame(
                np.zeros((1, len(bundle.feature_columns))), columns=bundle.feature_columns
            )
            explainer.top_features(dummy_X, feature_names=bundle.feature_columns, k=1)
        except Exception as exc:  # noqa: BLE001
            log.warning("shap_warmup_failed", error=str(exc))
        log.info(
            "serving_started",
            model_version=bundle.model_version,
            model_stage=bundle.model_stage,
            n_features=len(bundle.feature_columns),
        )

    try:
        yield
    finally:
        # No teardown needed — MLflow clients and TreeExplainer are
        # stateless.  Reset module-level caches so test isolation is
        # easy.
        reset_model_bundle_for_tests()
        reset_explainers_for_tests()


# ── App factory ────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="FraudGuard Scoring API",
        version="0.1.0",
        description=(
            "Real-time fraud-detection scoring service.  Loads the "
            "champion XGBoost model from the local MLflow registry, "
            "enriches each request with the Redis-backed online "
            "feature store, and returns a SHAP-explained decision."
        ),
        lifespan=lifespan,
    )

    # Prometheus /metrics endpoint.  `expose` registers the route
    # AND the middleware that records http_requests_total etc.
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    # ── Health ───────────────────────────────────────────────────────────
    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        bundle = _safe_get_bundle(request)
        payload: dict[str, Any] = {"status": "ok"}
        if bundle is not None:
            payload["model_version"] = str(bundle.model_version)
            payload["model_stage"] = str(bundle.model_stage)
            payload["n_features"] = len(bundle.feature_columns)
        else:
            payload["status"] = "degraded"
        return payload

    # ── Score ────────────────────────────────────────────────────────────
    @app.post("/v1/score", response_model=ScoreResponse)
    def score(req: TransactionRequest, request: Request) -> ScoreResponse:
        bundle = _safe_get_bundle(request)
        if bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Champion model is not loaded.  Check /health and MLflow config.",
            )
        params = getattr(request.app.state, "serving_params", {}) or {}
        threshold = float(params.get("threshold", 0.5))
        top_k = int(params.get("shap_top_k", 5))
        online_store = request.app.state.online_store
        try:
            return score_transaction(
                req,
                bundle=bundle,
                online_store=online_store,
                threshold=threshold,
                top_k=top_k,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("scoring_failed", transaction_id=req.transaction_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _safe_get_bundle(request: Request) -> ModelBundle | None:
    """Return the process-wide ModelBundle, or None on any failure.

    Honours a ``request.app.state.bundle_override`` slot if a test
    (or local dev runner) has injected one.  Production never sets
    that slot, so the real ``get_model_bundle`` call runs as usual.
    """
    override = getattr(request.app.state, "bundle_override", None)
    if override is not None:
        return override

    # If it failed to load previously, don't keep retrying and spamming MLflow.
    if getattr(request.app.state, "bundle_load_failed", False):
        return None

    try:
        return get_model_bundle()
    except Exception as exc:  # noqa: BLE001
        log.warning("bundle_unavailable", error=str(exc))
        request.app.state.bundle_load_failed = True
        return None


# Module-level app for `uvicorn src.serving.app:app`.
app = create_app()


def main() -> None:
    """Run the API via uvicorn (used by the ``fraudguard-serving`` console script)."""
    import uvicorn  # local import keeps top-of-file light

    params = _read_serving_params()
    host = os.environ.get("FRAUDGUARD_API_HOST", params.get("host", "0.0.0.0"))
    port = int(os.environ.get("FRAUDGUARD_API_PORT", params.get("port", 8000)))
    uvicorn.run("src.serving.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
