"""src/serving/explainer.py
────────────────────────────────────────────────────────────────────────
Phase 3 — SHAP explainer for the FraudGuard scoring API.

Wraps ``shap.TreeExplainer`` in a small class with a single
``top_features`` entry point that returns the *k* most influential
features for a single scoring row, sorted by absolute SHAP value.

Thread safety
-------------
A module-level singleton keyed on model identity is used so the
underlying TreeExplainer (which is expensive to build) is built
once per process.  The first call to ``get_explainer(model)`` for
a given model id materialises the explainer; subsequent calls
return the cached instance.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
import structlog

log = structlog.get_logger(__name__)


@dataclass
class FeatureContribution:
    """One row of the top-k SHAP explanation.

    `contribution` is the SHAP value for the feature on the positive
    (fraud) class — its sign indicates *direction* (push toward fraud
    vs. push toward legit) and its magnitude indicates *strength*.
    `value` is the feature's numeric value at scoring time.
    """

    feature_name: str
    contribution: float
    value: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "feature_name": self.feature_name,
            "contribution": float(self.contribution),
            "value": float(self.value),
        }


class ShapExplainer:
    """Thin wrapper over ``shap.TreeExplainer``.

    Built once per model and reused across requests.  Call
    ``top_features`` with a single-row DataFrame to get the k most
    influential features for that prediction.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._explainer = shap.TreeExplainer(model)

    def top_features(
        self,
        X_row: pd.DataFrame,
        feature_names: list[str],
        k: int = 5,
    ) -> list[FeatureContribution]:
        """Return the top-``k`` features by |SHAP| for a single row.

        Parameters
        ----------
        X_row
            A single-row DataFrame whose columns are aligned with
            ``feature_names`` (same order, same length).
        feature_names
            Ordered list of feature names the model was trained on.
        k
            Number of features to return (clipped to the number of
            columns actually present).

        Notes
        -----
        For binary XGBoost models, ``shap_values`` returns either a
        ``(1, n_features)`` matrix (older shap) or a list with one
        element (newer shap).  We normalise both shapes to a flat
        1-D vector of length ``n_features``.
        """
        if len(X_row) != 1:
            raise ValueError(f"X_row must contain exactly 1 row, got {len(X_row)}.")

        shap_values = self._explainer.shap_values(X_row)

        # shap >=0.42 returns a list[ndarray] for some models.  For
        # XGBoost binary classifiers the "positive" class is index 0
        # of the list, so we pick that.  Otherwise fall back to the
        # 2-D array as-is.
        if isinstance(shap_values, list):
            # For binary classifiers, the last element is always the positive class
            sv = np.asarray(shap_values[-1])
        else:
            sv = np.asarray(shap_values)

        if sv.ndim == 2:
            sv = sv[0]
        if sv.ndim != 1:
            sv = sv.flatten()

        if sv.shape[0] != len(feature_names):
            raise RuntimeError(
                f"SHAP vector length {sv.shape[0]} does not match "
                f"feature_names length {len(feature_names)}."
            )

        row_values = X_row.iloc[0].to_numpy(dtype=float, copy=True)
        abs_sv = np.abs(sv)
        top_k = min(k, len(feature_names))
        top_idx = np.argsort(-abs_sv)[:top_k]

        return [
            FeatureContribution(
                feature_name=str(feature_names[i]),
                contribution=float(sv[i]),
                value=float(row_values[i]),
            )
            for i in top_idx
        ]


# ── Thread-safe singleton keyed on model id ──────────────────────────────

_EXPLAINERS: dict[int, ShapExplainer] = {}
_EXPLAINERS_LOCK = threading.Lock()


def get_explainer(model: Any) -> ShapExplainer:
    """Return (or build + cache) the ``ShapExplainer`` for ``model``.

    Keyed on ``id(model)`` so two different model objects get their
    own explainer.  The MLflow-loaded model is a singleton inside
    ``ModelBundle``, so this collapses to one explainer per bundle.
    """
    key = id(model)
    cached = _EXPLAINERS.get(key)
    if cached is not None:
        return cached
    with _EXPLAINERS_LOCK:
        cached = _EXPLAINERS.get(key)
        if cached is None:
            log.info("building_shap_explainer", model_id=key)
            cached = ShapExplainer(model)
            _EXPLAINERS[key] = cached
    return cached


def reset_explainers_for_tests() -> None:
    """Drop all cached explainers (test fixture helper)."""
    global _EXPLAINERS
    with _EXPLAINERS_LOCK:
        _EXPLAINERS = {}
