"""tests/unit/_skops_compat.py
────────────────────────────────────────────────────────────────────────
Shim that lets the test suite call `mlflow.sklearn.log_model` with the
`skops_trusted_types=` kwarg on *either* old or new mlflow releases.

The CI image pins an mlflow version that predates the
`skops_trusted_types` parameter, while the local dev image has a newer
mlflow that performs a skops audit on sklearn pickles.  Trying to pass
the kwarg unconditionally blows up on CI with:

    TypeError: log_model() got an unexpected keyword argument
               'skops_trusted_types'

So we probe the signature once and return the kwarg dict only when the
installed mlflow actually understands it.  This mirrors the helper
in `src/training/train.py` and is kept under `tests/unit/` because only
tests need this — production code already calls the helper directly.
"""

from __future__ import annotations

import inspect
from typing import Any

import mlflow.sklearn

_SKOPS_TRUSTED_TYPES: list[str] = [
    "xgboost.sklearn.XGBClassifier",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBModel",
    "xgboost.sklearn.XGBRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.basic.Booster",
    "collections.OrderedDict",
]


def skops_log_kwargs() -> dict[str, Any]:
    """Return ``{"skops_trusted_types": [...]}`` if mlflow supports it."""
    try:
        params = inspect.signature(mlflow.sklearn.log_model).parameters
    except (TypeError, ValueError):
        return {}
    if "skops_trusted_types" in params:
        return {"skops_trusted_types": _SKOPS_TRUSTED_TYPES}
    return {}
