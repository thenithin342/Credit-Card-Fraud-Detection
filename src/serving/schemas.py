"""src/serving/schemas.py
────────────────────────────────────────────────────────────────────────
Pydantic request / response schemas for the FraudGuard scoring API.

Design notes
------------
* `TransactionRequest` declares the *minimum* required fields up front
  (transaction_id, TransactionDT, TransactionAmt, card1) and uses
  `extra="allow"` so callers can include any of the 300+ raw IEEE-CIS
  fields the model was trained on.  Unknown fields are silently kept
  by the request handler and forwarded to the preprocessor — which
  drops anything it doesn't recognise.
* `ScoreResponse` mirrors the contract called out in the Phase 3 PRD:
  fraud probability, the threshold applied, top SHAP contributions,
  end-to-end latency, and the model version that produced the score.
* The IEEE-CIS field names are intentionally left as snake_case /
  mixed-case identifiers (e.g. `TransactionDT`, `P_emaildomain`) so
  the wire format matches the parquet schema produced by the
  ingestion pipeline.  No aliasing is needed.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TopFeature(BaseModel):
    """One row of the SHAP top-k explanation.

    `contribution` is the SHAP value for the feature on the
    *positive* (fraud) class.  `value` is the feature's preprocessed
    numeric value at scoring time.
    """

    feature_name: str
    contribution: float
    value: float


class TransactionRequest(BaseModel):
    """A single IEEE-CIS-shaped transaction to be scored.

    The model was trained on 312 columns (303 raw + 9 temporal).
    Every raw column is optional; the preprocessor fills missing
    entries with the -999 sentinel and the model still scores.
    """

    model_config = ConfigDict(extra="allow")

    transaction_id: int = Field(
        default=-1,
        description="Caller-supplied transaction identifier (echoed back in the response). Use -1 if not provided.",
    )
    TransactionDT: int = Field(
        ...,
        description="Seconds since the IEEE-CIS reference epoch (required for temporal features).",
    )
    TransactionAmt: float = Field(..., description="Transaction amount in USD.")
    card1: int | float = Field(
        default=0,
        description="Surrogate card identifier used for online feature lookup.",
    )


class ScoreResponse(BaseModel):
    """The decision returned by POST /v1/score."""

    transaction_id: int
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold: float
    top_features: list[TopFeature]
    latency_ms: float
    model_version: str
