"""src/serving/predictor.py
────────────────────────────────────────────────────────────────────────
Phase 3 — End-to-end scoring pipeline for the FraudGuard API.

The single public entry point is ``score_transaction``, which takes
a validated ``TransactionRequest`` and returns a ``ScoreResponse``:

  1. Coerce the request payload into a 1-row DataFrame.
  2. Enrich with the 3 static engineered features (amount_log,
     hour_of_day, day_of_week) computed by
     ``src.features.definitions.compute_static_features``.
  3. Look up the 6 window features from the Redis-backed
     ``OnlineFeatureStore`` (default 0 / -999 sentinel for unseen
     cards — matches offline semantics).
  4. Apply the fitted ``FeaturePreprocessor`` to the 303 raw columns.
  5. Concatenate the 9 temporal columns onto the preprocessed block
     in the exact order written to ``feature_columns.json``.
  6. ``model.predict_proba`` to get the fraud probability.
  7. SHAP top-``k`` explanation against the same 312-column frame.
  8. Write-back the current transaction to the online store so the
     *next* request for this card sees accurate window aggregates.
  9. Return the ``ScoreResponse`` with the round-trip latency.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb

from src.features.definitions import (
    FEATURE_NAMES,
    NUMERIC_NULL_FILL,
    WINDOW_FEATURE_NAMES,
    compute_static_features,
)
from src.features.online_store import OnlineFeatureStore
from src.serving.explainer import get_explainer
from src.serving.model_loader import ModelBundle
from src.serving.schemas import ScoreResponse, TopFeature, TransactionRequest

log = structlog.get_logger(__name__)


# Sentinel used when a card has no prior history in the online store
# — keeps serving consistent with the offline ``NUMERIC_NULL_FILL``
# convention and ensures the XGBoost model can still split on it.
_DEFAULT_WINDOW_FALLBACK: dict[str, float] = {
    "amount_zscore": NUMERIC_NULL_FILL,
    "txn_count_5m": 0.0,
    "txn_amount_sum_5m": 0.0,
    "txn_count_1h": 0.0,
    "txn_amount_sum_1h": 0.0,
    "txn_count_24h": 0.0,
    "txn_amount_sum_24h": 0.0,
    "txn_count_7d": 0.0,
    "txn_amount_sum_7d": 0.0,
    "time_since_last_txn": NUMERIC_NULL_FILL,
}


def _get_card_id(card1: Any) -> str | None:
    """Return a clean string representation of card1 if valid, or None if missing/null/0.

    Prevents key collisions in Redis (e.g. 'fraud:features:0') when card1 is missing.
    """
    if card1 is None or pd.isna(card1):
        return None
    try:
        val = float(card1)
        if val == 0.0 or pd.isna(val):
            return None
        if val.is_integer():
            return str(int(val))
        return str(val)
    except (ValueError, TypeError):
        s = str(card1).strip()
        if not s or s in ("0", "0.0", "nan", "None"):
            return None
        return s


# ── Helpers ────────────────────────────────────────────────────────────────


def _coerce_request_to_dataframe(req: TransactionRequest) -> pd.DataFrame:
    """Flatten a ``TransactionRequest`` (incl. extra fields) into a
    1-row DataFrame.  IEEE-CIS nullable numerics are cast to float so
    the preprocessor can fill them with the -999 sentinel.
    """
    payload: dict[str, Any] = req.model_dump()
    row = pd.DataFrame([payload])

    # Cast the required + common fields explicitly.  We do NOT cast
    # other columns blindly — strings (e.g. ProductCD, card4, M1)
    # must remain strings for the OrdinalEncoder.
    for col in (
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
    ):
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")

    return row


def _build_temporal_block(
    req: TransactionRequest,
    raw: pd.DataFrame,
    online_store: OnlineFeatureStore,
) -> pd.DataFrame:
    """Compute the 9 temporal columns for a single request.

    Static features (3) come from ``compute_static_features``; window
    features (6) come from the online store (or the -999 / 0
    fallback when the card is unseen).  The result is a 1-row
    DataFrame in ``FEATURE_NAMES`` order.
    """
    static = compute_static_features(raw)

    card_id = _get_card_id(req.card1)
    if card_id:
        try:
            prior = online_store.get_card_features(card_id) or {}
        except Exception as exc:
            log.warning("redis_read_failed", error=str(exc))
            prior = {}
    else:
        prior = {}
    window_values = {**_DEFAULT_WINDOW_FALLBACK, **prior}
    # Drop bookkeeping keys the online store also stores; they are
    # never model inputs.
    for k in list(window_values):
        if k not in WINDOW_FEATURE_NAMES:
            window_values.pop(k, None)
    window_df = pd.DataFrame([window_values], columns=list(WINDOW_FEATURE_NAMES))

    out = pd.concat([static, window_df], axis=1)
    out = out[list(FEATURE_NAMES)]
    return out.astype(np.float64, copy=False)


def _align_to_model_columns(
    preprocessed: pd.DataFrame,
    temporal: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Concatenate the preprocessed 303 columns with the 9 temporal
    columns and reorder to the exact ``feature_columns`` list.

    Both inputs are 1-row frames.  Output is the 1-row, 312-column
    frame the model expects.
    """
    # The preprocessor is fit only on the 303 raw columns.  The
    # temporal block is a separate 9-column frame.  Concatenate,
    # then reindex to the canonical order so missing columns are
    # filled with NaN (the model sees NaN as a missing branch).
    merged = pd.concat(
        [preprocessed.reset_index(drop=True), temporal.reset_index(drop=True)], axis=1
    )
    # Reindex to the canonical feature order.  Any column the
    # preprocessor dropped (e.g. a V* field the client omitted) will
    # be NaN here; XGBoost handles that natively.
    aligned = merged.reindex(columns=feature_columns)
    return aligned


# ── Public entry point ─────────────────────────────────────────────────────


def score_transaction(
    req: TransactionRequest,
    *,
    bundle: ModelBundle,
    online_store: OnlineFeatureStore,
    threshold: float = 0.5,
    top_k: int = 5,
) -> ScoreResponse:
    """Score a single transaction end-to-end.

    Parameters
    ----------
    req
        Validated request payload (raw IEEE-CIS fields + identity).
    bundle
        Process-wide model bundle (model, preprocessor, columns).
    online_store
        Redis-backed feature store; receives ``update_after_transaction``
        once the score is computed.
    threshold
        Fraud decision cutoff in [0, 1].  ``is_fraud = fraud_score >= threshold``.
    top_k
        Number of SHAP top features to return.

    Returns
    -------
    ScoreResponse
        Decision payload (score, threshold, top-k SHAP, latency, model version).
    """
    t0 = time.perf_counter()

    raw = _coerce_request_to_dataframe(req)
    temporal = _build_temporal_block(req, raw, online_store)

    # Reindex to the preprocessor's fit-time column list so missing
    # raw columns are NaN (not silently dropped).  The preprocessor
    # will then fill numerics with the -999 sentinel and pass
    # categoricals through the OrdinalEncoder (which knows NaN
    # means "missing").  Without this, a request that omits a
    # categorical the encoder was fit on would crash with a
    # column-count mismatch.
    expected_cols = list(bundle.preprocessor.numeric_cols_) + list(
        bundle.preprocessor.categorical_cols_
    )
    raw = raw.reindex(columns=expected_cols)

    preprocessed = bundle.preprocessor.transform(raw)
    X = _align_to_model_columns(preprocessed, temporal, bundle.feature_columns)

    if isinstance(bundle.model, xgb.Booster):
        dtest = xgb.DMatrix(X)
        proba = float(bundle.model.predict(dtest)[0])
    else:
        proba = float(bundle.model.predict_proba(X)[0, 1])

    explainer = get_explainer(bundle.model)
    top = explainer.top_features(X, bundle.feature_columns, k=top_k)

    # Write-back.  Failures here must not break the response — the
    # card just sees slightly stale window aggregates on the next
    # request.  The online store itself logs the underlying error.
    card_id = _get_card_id(req.card1)
    if card_id:
        try:
            online_store.update_after_transaction(
                card_id=card_id,
                amount=float(req.TransactionAmt),
                ts=int(req.TransactionDT),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("online_writeback_failed", error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    log.info(
        "scored_transaction",
        transaction_id=req.transaction_id,
        fraud_score=proba,
        latency_ms=latency_ms,
        model_version=bundle.model_version,
    )

    return ScoreResponse(
        transaction_id=int(req.transaction_id),
        fraud_score=proba,
        is_fraud=bool(proba >= threshold),
        threshold=float(threshold),
        top_features=[TopFeature(**f.to_dict()) for f in top],
        latency_ms=float(latency_ms),
        model_version=str(bundle.model_version),
    )
