"""src/features/online_store.py
────────────────────────────────────────────────────────────────────────
Redis-backed online feature store for real-time serving (Phase 3+).

The online store mirrors the *latest* stateful features per card in
Redis so that a serving API can look them up in O(1) per prediction.

Feature parity contract
-----------------------
The online store is responsible for tracking *stateful* features
(rolling aggregates per card).  The *static* features
(amount_log, hour_of_day, day_of_week) are pure functions of the
current transaction and are computed at request time by the serving
path using `src.features.definitions.compute_static_features`.

Both paths share `src.features.definitions` so values are identical
within 1e-6 (see `tests/unit/test_parity.py`).

Key format
----------
``fraud:features:{card_id}``  →  JSON-encoded dict of feature values.

Public API
----------
OnlineFeatureStore(redis_client)
    .set_card_features(card_id, features, ttl_seconds=3600)
    .get_card_features(card_id) -> dict | None
    .update_after_transaction(card_id, amount, ts)
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import bisect
import json
from typing import Any

import structlog

from src.features.definitions import (
    _AMOUNT_ROLLING_WINDOW,
    _NUMERIC_NULL_FILL,
    _WINDOW_1H_SEC,
    _WINDOW_5M_SEC,
    _WINDOW_7D_SEC,
    _WINDOW_24H_SEC,
)

log = structlog.get_logger(__name__)

# ── Key + TTL defaults ──────────────────────────────────────────────────────

KEY_PREFIX: str = "fraud:features:"
DEFAULT_TTL_SECONDS: int = 3600


class OnlineFeatureStore:
    """Thin Redis-backed feature store keyed by card_id.

    The store keeps, per card, the *strictly-past* transactions
    needed to compute trailing-window aggregates.  A transaction is
    only added to the rolling history **after** its window counts
    have been computed for it — so the current transaction is
    never counted in its own trailing aggregates, exactly matching
    the offline `compute_window_features` semantics.

    Parameters
    ----------
    redis_client
        Any object exposing ``get(key) -> bytes | None``,
        ``set(key, value, ex=ttl)``, and ``delete(key)`` — by
        convention a ``redis.Redis`` instance or a ``fakeredis``
        fake in tests.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(card_id: str | int) -> str:
        return f"{KEY_PREFIX}{card_id}"

    @staticmethod
    def _serialize(features: dict[str, Any]) -> bytes:
        return json.dumps(features, default=str).encode("utf-8")

    @staticmethod
    def _deserialize(raw: bytes) -> dict[str, Any]:
        decoded = json.loads(raw.decode("utf-8"))
        out: dict[str, Any] = {}
        for k, v in decoded.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    @staticmethod
    def _window_aggregates(history: list[tuple[int, float]], current_ts: int) -> dict[str, float]:
        """Compute the trailing-window aggregates for ``current_ts``
        using *only* the past history.  Mirrors the offline logic in
        `compute_window_features` exactly: the current transaction
        is **not** included in the window.

        ``history`` is a chronologically-ordered list of
        ``(ts, amount)`` tuples for prior transactions of this card.
        """
        cutoff_7d = current_ts - _WINDOW_7D_SEC
        i7 = bisect.bisect_left(history, (cutoff_7d, -float("inf")))
        in_7d = history[i7:]
        amounts_7d = [a for _, a in in_7d]

        cutoff_24h = current_ts - _WINDOW_24H_SEC
        i24 = bisect.bisect_left(in_7d, (cutoff_24h, -float("inf")))
        in_24h = in_7d[i24:]
        amounts_24h = [a for _, a in in_24h]

        cutoff_1h = current_ts - _WINDOW_1H_SEC
        i1 = bisect.bisect_left(in_24h, (cutoff_1h, -float("inf")))
        in_1h = in_24h[i1:]
        amounts_1h = [a for _, a in in_1h]

        cutoff_5m = current_ts - _WINDOW_5M_SEC
        i5 = bisect.bisect_left(in_1h, (cutoff_5m, -float("inf")))
        in_5m = in_1h[i5:]
        amounts_5m = [a for _, a in in_5m]

        return {
            "txn_count_5m": float(len(in_5m)),
            "txn_amount_sum_5m": float(sum(amounts_5m)),
            "txn_count_1h": float(len(in_1h)),
            "txn_amount_sum_1h": float(sum(amounts_1h)),
            "txn_count_24h": float(len(in_24h)),
            "txn_amount_sum_24h": float(sum(amounts_24h)),
            "txn_count_7d": float(len(in_7d)),
            "txn_amount_sum_7d": float(sum(amounts_7d)),
        }

    @staticmethod
    def _zscore(history: list[tuple[int, float]], amount: float) -> float:
        """Compute amount_zscore vs. the previous rolling window.

        Matches the offline semantics: reference = last
        ``_AMOUNT_ROLLING_WINDOW`` amounts from ``history``
        (i.e. excluding the current transaction).  Returns the
        -999 sentinel if there is no prior history.
        """
        if not history:
            return _NUMERIC_NULL_FILL
        ref_amounts = [a for _, a in history[-_AMOUNT_ROLLING_WINDOW:]]
        if not ref_amounts:
            return _NUMERIC_NULL_FILL
        n = len(ref_amounts)
        mean = sum(ref_amounts) / n
        var = sum((a - mean) ** 2 for a in ref_amounts) / n
        std = var**0.5
        if std == 0.0:
            return 0.0
        return (amount - mean) / std

    # ── Public API ─────────────────────────────────────────────────────────

    def set_card_features(
        self,
        card_id: str,
        features: dict[str, float | int],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Write a card's latest feature values to Redis."""
        if not card_id:
            raise ValueError("card_id must be a non-empty string")
        self._redis.set(
            self._key(card_id),
            self._serialize(features),
            ex=ttl_seconds,
        )
        log.debug("online_features_set", card_id=card_id, ttl=ttl_seconds)

    def get_card_features(self, card_id: str) -> dict | None:
        """Fetch a card's latest features.  Returns None if not found."""
        if not card_id:
            return None
        raw = self._redis.get(self._key(card_id))
        if raw is None:
            return None
        return self._deserialize(raw)

    def delete_card_features(self, card_id: str) -> None:
        """Remove a card's features from Redis (admin / GC helper)."""
        self._redis.delete(self._key(card_id))

    def update_after_transaction(
        self,
        card_id: str,
        amount: float,
        ts: int,
    ) -> None:
        """Update rolling aggregates after a new transaction is observed.

        Reads the current state (history of past transactions),
        computes the new feature values from the *past* history
        (the current transaction is NOT counted in its own trailing
        aggregates — this matches the offline pipeline exactly),
        then appends the new transaction to the history for the
        next call.

        The stored feature dict is the *serving-time* view: when the
        next request comes in, the serving path will read these
        values directly and combine them with freshly-computed
        static features for the new request's transaction.
        """
        current = self.get_card_features(card_id)
        if current is None:
            history: list[tuple[int, float]] = []
            last_ts: int | None = None
        else:
            raw_history = current.get("_hist", [])
            history = [(int(t), float(a)) for t, a in raw_history]
            last_ts_raw = current.get("_last_ts")
            last_ts = int(last_ts_raw) if last_ts_raw is not None else None

        # ── Compute aggregates from strictly-past history ──
        aggs = self._window_aggregates(history, current_ts=int(ts))
        z = self._zscore(history, amount=float(amount))

        # time_since_last_txn: use the -999 sentinel when there is no
        # prior transaction — matches the offline pipeline exactly.
        tsl = _NUMERIC_NULL_FILL if last_ts is None else float(int(ts) - last_ts)

        # ── Compose the new state and append the current tx to history ──
        # Prune history to the 7d window.
        cutoff = int(ts) - _WINDOW_7D_SEC
        pruned_history = [[int(t), float(a)] for (t, a) in history if t >= cutoff]
        pruned_history.append([int(ts), float(amount)])

        new_state: dict[str, Any] = {
            "amount_zscore": float(z),
            "txn_count_5m": aggs["txn_count_5m"],
            "txn_amount_sum_5m": aggs["txn_amount_sum_5m"],
            "txn_count_1h": aggs["txn_count_1h"],
            "txn_amount_sum_1h": aggs["txn_amount_sum_1h"],
            "txn_count_24h": aggs["txn_count_24h"],
            "txn_amount_sum_24h": aggs["txn_amount_sum_24h"],
            "txn_count_7d": aggs["txn_count_7d"],
            "txn_amount_sum_7d": aggs["txn_amount_sum_7d"],
            "time_since_last_txn": tsl,
            # Bookkeeping for the next call — bounded to 7d window.
            "_hist": pruned_history,
            "_last_ts": int(ts),
        }

        self.set_card_features(card_id, new_state)
        log.debug(
            "online_features_updated",
            card_id=card_id,
            ts=int(ts),
            count_5m=aggs["txn_count_5m"],
            count_1h=aggs["txn_count_1h"],
        )
