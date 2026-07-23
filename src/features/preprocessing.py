"""src/features/preprocessing.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Missing value imputation and categorical encoding for FraudGuard.

`FeaturePreprocessor` is fit **only on the training split** and then
applied identically to val and test.  Pickled instances live at
``models/encoders/feature_preprocessor.pkl`` and are the canonical
reference for both the offline training pipeline and (Phase 3) the
online serving path.

What it does
------------
* Numeric columns:  nulls are filled with the sentinel ``-999``
  (XGBoost and LightGBM treat this as a valid split value).
* Categorical columns: an `OrdinalEncoder` is fit on the union of
  unique values in train (plus a synthetic ``"missing"`` row so
  that nulls at transform time always map to a known bucket).
  Categories unseen at fit time are mapped to ``-1`` (the
  ``unknown_value`` we configure).

Why ordinal (not one-hot)
-------------------------
The IEEE-CIS dataset has high-cardinality categoricals (id_30 has 71
unique domains, id_33 has 202, DeviceInfo has 1584).  One-hot
encoding would balloon the feature space and slow training.  XGBoost
and LightGBM can use ordinal-encoded integers directly because they
split on thresholds, not on category identity.
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import OrdinalEncoder

log = structlog.get_logger(__name__)

# Explicit list of known categorical columns in IEEE-CIS / FraudGuard
CATEGORICAL_COLS: Final[set[str]] = (
    {
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
    }
    | {f"M{i}" for i in range(1, 10)}
    | {f"id_{i}" for i in range(12, 39)}
)

# Numeric nulls are replaced with a value XGBoost / LightGBM treat
# as a valid split boundary.  ``-999`` is far from any plausible
# real value of the IEEE-CIS numeric features.
NUMERIC_NULL_FILL: float = -999.0

# String used to fill nulls in categorical columns *before* they are
# passed to the OrdinalEncoder (the encoder cannot handle NaN).
CATEGORICAL_NULL_FILL: str = "missing"

# Encoded value used by sklearn for genuinely unseen categories.
# Set to -1 so it stays distinguishable from 0..N-1 learned buckets.
_UNKNOWN_VALUE: int = -1


def _normalize_cat_series(s: pd.Series) -> pd.Series:
    """Normalize categorical values to strings consistently across int, float, str, and nulls."""
    if pd.api.types.is_float_dtype(s):
        valid_mask = s.notna()
        is_int_mask = valid_mask & (s % 1 == 0)
        str_s = pd.Series(CATEGORICAL_NULL_FILL, index=s.index, dtype=object)
        str_s.loc[is_int_mask] = s.loc[is_int_mask].astype(np.int64).astype(str)
        str_s.loc[valid_mask & ~is_int_mask] = s.loc[valid_mask & ~is_int_mask].astype(str)
        return str_s
    elif pd.api.types.is_integer_dtype(s):
        valid_mask = s.notna()
        str_s = pd.Series(CATEGORICAL_NULL_FILL, index=s.index, dtype=object)
        str_s.loc[valid_mask] = s.loc[valid_mask].astype(np.int64).astype(str)
        return str_s
    else:
        s_obj = s.astype(object).fillna(CATEGORICAL_NULL_FILL)
        str_s = s_obj.astype(str)
        has_dot_zero = str_s.str.endswith(".0")
        if has_dot_zero.any():
            str_s.loc[has_dot_zero] = str_s.loc[has_dot_zero].str[:-2]
        return str_s


class FeaturePreprocessor:
    """Fit on train, apply consistently to val / test / online.

    The class stores:
        * ``numeric_cols_``  — list of column names treated as numeric
        * ``categorical_cols_`` — list of column names treated as categorical
        * ``encoder_`` — fitted ``OrdinalEncoder`` for the categoricals
        * ``freq_maps_`` — dictionary mapping cat columns to their frequencies
        * ``numeric_fill_`` — value used to fill numeric nulls (default -999)

    Pickling this object preserves all of the above so that the
    serving path can call ``transform`` without re-fitting.
    """

    def __init__(self) -> None:
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.encoder_: OrdinalEncoder | None = None
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.numeric_fill_: float = NUMERIC_NULL_FILL
        self._is_fitted: bool = False

    # ── Fit / transform API ─────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> FeaturePreprocessor:
        """Identify numeric vs categorical columns and fit the encoder + frequency maps.

        Parameters
        ----------
        df : pd.DataFrame
            The training frame.  Must contain every column in
            ``feature_cols``.
        feature_cols : list[str]
            The columns to treat as features.  Order is preserved
            in the output of ``transform``.

        Returns
        -------
        FeaturePreprocessor
            ``self`` so the call can be chained.
        """
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns missing from training frame: {missing[:5]}")

        # Split features into categorical vs numeric.
        # Any column explicitly in CATEGORICAL_COLS or with non-numeric dtype is categorical.
        self.categorical_cols_ = [
            c
            for c in feature_cols
            if (c in CATEGORICAL_COLS or not pd.api.types.is_numeric_dtype(df[c]))
        ]
        self.numeric_cols_ = [c for c in feature_cols if c not in self.categorical_cols_]

        log.info(
            "preprocessor_fit",
            n_numeric=len(self.numeric_cols_),
            n_categorical=len(self.categorical_cols_),
        )

        self.freq_maps_ = {}
        if self.categorical_cols_:
            cat_dict = {col: _normalize_cat_series(df[col]) for col in self.categorical_cols_}
            cat_data = pd.DataFrame(cat_dict, index=df.index)

            for col in self.categorical_cols_:
                counts = cat_data[col].value_counts(normalize=True)
                self.freq_maps_[col] = counts.to_dict()

            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=_UNKNOWN_VALUE,
                dtype=np.int64,
            )
            n_cat = len(self.categorical_cols_)
            missing_row = np.full((1, n_cat), CATEGORICAL_NULL_FILL, dtype=object)
            augmented = np.vstack([cat_data.to_numpy(), missing_row])
            self.encoder_.fit(augmented)
            try:
                cats_per_col = [int(len(c)) for c in self.encoder_.categories_]
            except Exception:
                cats_per_col = []
            log.info("ordinal_encoder_fitted", n_cols=n_cat, cats_per_col=cats_per_col)
        else:
            self.encoder_ = None

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing to a DataFrame.

        The output is a *new* DataFrame with the same index as the
        input and columns the preprocessor was fit on (including frequency-encoded
        categorical features), in a stable order.

        Parameters
        ----------
        df : pd.DataFrame
            Frame to transform.  May be val, test, or a single
            online request.

        Returns
        -------
        pd.DataFrame
            Transformed frame.
        """
        if not self._is_fitted:
            raise RuntimeError("FeaturePreprocessor.transform called before .fit()")

        out = pd.DataFrame(index=df.index)

        # Numeric path: fill nulls, preserve dtype (cast to float64 so
        # downstream models see a consistent type).  Build a single
        # DataFrame for the numeric block to avoid fragmentation.
        present_num = [c for c in self.numeric_cols_ if c in df.columns]
        if present_num:
            num_frame = df[present_num].astype(np.float64).fillna(self.numeric_fill_)
            out = pd.concat([out, num_frame], axis=1)

        # Categorical path: fill nulls with sentinel, ordinal-encode + frequency-encode
        if self.categorical_cols_ and self.encoder_ is not None:
            # Always build all categorical columns (fill missing ones with sentinel)
            cat_dict = {
                col: _normalize_cat_series(df[col]) if col in df.columns
                else pd.Series(CATEGORICAL_NULL_FILL, index=df.index)
                for col in self.categorical_cols_
            }
            cat_data = pd.DataFrame(cat_dict, index=df.index)
            encoded = self.encoder_.transform(cat_data.to_numpy())
            cat_frame = pd.DataFrame(
                encoded.astype(np.int64),
                columns=self.categorical_cols_,
                index=df.index,
            )
            freq_dict = {}
            for col in self.categorical_cols_:
                freq_map = self.freq_maps_.get(col, {})
                freq_dict[f"{col}_freq"] = (
                    cat_data[col].map(freq_map).fillna(0.0).astype(np.float64)
                )
            freq_frame = pd.DataFrame(freq_dict, index=df.index)
            out = pd.concat([out, cat_frame, freq_frame], axis=1)

        # Re-order columns to the fit-time order so the consumer
        # always sees a stable column layout.  Columns not in the
        # fit set are dropped here.
        ordered_cat_cols = []
        for col in self.categorical_cols_:
            if col in out.columns:
                ordered_cat_cols.append(col)
            freq_col = f"{col}_freq"
            if freq_col in out.columns:
                ordered_cat_cols.append(freq_col)
        ordered = [c for c in (self.numeric_cols_ + ordered_cat_cols) if c in out.columns]
        return out[ordered]

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Convenience: fit on ``df`` then transform it.  Returns the
        transformed DataFrame."""
        self.fit(df, feature_cols)
        return self.transform(df)

    # â”€â”€ Persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save(self, path: Path) -> None:
        """Pickle ``self`` to ``path``.  Parents are created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("preprocessor_saved", path=str(path), bytes=path.stat().st_size)

    @classmethod
    def load(cls, path: Path) -> FeaturePreprocessor:
        """Load a pickled FeaturePreprocessor from ``path``."""
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FeaturePreprocessor; got {type(obj).__name__}")
        log.info("preprocessor_loaded", path=str(path))
        return obj
