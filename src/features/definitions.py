"""src/features/definitions.py
────────────────────────────────────────────────────────────────────────
Single source of truth for FraudGuard feature names and computation
logic.

Both the *offline* training pipeline (`src.features.offline_store`) and
the *online* serving path (`src.features.online_store`) import from
this module so that the SAME feature definitions produce IDENTICAL
values regardless of where they are called.  Never duplicate a
computation anywhere else in the codebase.

Exports
-------
FEATURE_NAMES
    Ordered list of every engineered feature column.
TARGET_COL
    The label column we predict.
ID_COLUMNS
    Columns the model must NOT see at training time.
STATIC_FEATURE_NAMES
    Subset of FEATURE_NAMES computed row-wise (no group context).
WINDOW_FEATURE_NAMES
    Subset of FEATURE_NAMES that depend on trailing history per card.
compute_static_features(df)
    Row-level transforms (log amount, hour-of-day, day-of-week, ...).
compute_window_features(df, card_col)
    Trailing-window aggregates per card (no leakage: looks at history
    strictly *before* each row's TransactionDT).

Notes
-----
* Null handling: numeric nulls in static features are filled with -999
  (XGBoost / LightGBM treat -999 as a valid split value).  Categorical
  nulls are filled with the sentinel string "missing".
* Window features are computed in O(N) by iterating per card in
  chronological order.  This is intentional — it matches the exact
  semantics of the online store, where at serving time we only know the
  *past* transactions for a card.
* `time_since_last_txn` is -999.0 for the very first transaction a card
  ever makes (it is the first observation in the window, representing a numeric null fill).
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import bisect
from typing import Final

import numpy as np
import pandas as pd

# ── Column name constants ───────────────────────────────────────────────────

TARGET_COL: Final[str] = "isFraud"

# Columns the model must never see as input features.
ID_COLUMNS: Final[tuple[str, ...]] = (
    "TransactionID",
    "TransactionDT",  # raw timestamp → replaced by hour_of_day / day_of_week
)

# Engineered static (row-level) feature names.
STATIC_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "amount_log",
    "hour_of_day",
    "day_of_week",
)

# Engineered window (per-card history) feature names.
WINDOW_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "amount_zscore",
    "txn_count_5m",
    "txn_amount_sum_5m",
    "txn_count_1h",
    "txn_amount_sum_1h",
    "txn_count_24h",
    "txn_amount_sum_24h",
    "txn_count_7d",
    "txn_amount_sum_7d",
    "time_since_last_txn",
)

# The full ordered list of feature columns the model consumes.
FEATURE_NAMES: Final[tuple[str, ...]] = STATIC_FEATURE_NAMES + WINDOW_FEATURE_NAMES

# Sentinel used for nulls in numeric engineered columns.
_NUMERIC_NULL_FILL: Final[float] = -999.0

# Trailing window sizes in seconds, matching params.yaml `features.window_minutes`.
_WINDOW_5M_SEC: Final[int] = 5 * 60
_WINDOW_1H_SEC: Final[int] = 60 * 60
_WINDOW_24H_SEC: Final[int] = 24 * 60 * 60
_WINDOW_7D_SEC: Final[int] = 7 * 24 * 60 * 60

# Seconds in a day / week used for time-derived features.
_SECONDS_PER_DAY: Final[int] = 86_400
_DAYS_PER_WEEK: Final[int] = 7

# Rolling-mean / rolling-std window used to compute the card-level
# amount_zscore.  We use a trailing window of the *previous* 20
# transactions for the same card (fits comfortably in memory for
# IEEE-CIS card1 which has ~12k unique cards across ~400k rows).
_AMOUNT_ROLLING_WINDOW: Final[int] = 20


# ── Static features (no group context) ─────────────────────────────────────


def compute_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the static engineered features only.

    The output contains exactly the columns in `STATIC_FEATURE_NAMES`,
    in that order, with no nulls.  All inputs are expected to be
    already-present raw columns; rows are not reordered.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TransactionAmt`` and ``TransactionDT``.

    Returns
    -------
    pd.DataFrame
        New DataFrame indexed exactly like ``df`` with only the
        engineered static columns.
    """
    out = pd.DataFrame(index=df.index)

    # log1p(TransactionAmt) — compresses the heavy right tail of
    # transaction amounts (which span ~$0.25 to ~$30k in IEEE-CIS).
    out["amount_log"] = np.log1p(df["TransactionAmt"].astype(float)).fillna(_NUMERIC_NULL_FILL)

    # hour_of_day : [0, 23]  — TransactionDT is seconds since an
    # arbitrary epoch; mod by 86_400 then integer-divide by 3_600.
    dt = df["TransactionDT"].astype(float)
    out["hour_of_day"] = (dt % _SECONDS_PER_DAY // 3_600).fillna(_NUMERIC_NULL_FILL)

    # day_of_week : [0, 6]  — useful proxy for weekly seasonality.
    out["day_of_week"] = ((dt // _SECONDS_PER_DAY) % _DAYS_PER_WEEK).fillna(_NUMERIC_NULL_FILL)

    return out


# ── Window features (per-card history, chronological, no leakage) ──────────


def compute_window_features(
    df: pd.DataFrame,
    card_col: str = "card1",
) -> pd.DataFrame:
    """Return a DataFrame with the per-card window-aggregated features.

    The function is **deterministic** and **leakage-free** for any
    DataFrame sorted (or sortable) by ``TransactionDT``: each row's
    features are computed only from rows for the same ``card_col``
    whose ``TransactionDT`` is **strictly less** than the current
    row's ``TransactionDT``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TransactionDT``, ``TransactionAmt``, and the
        column named by ``card_col``.
    card_col : str, default "card1"
        Surrogate card identifier column.

    Returns
    -------
    pd.DataFrame
        New DataFrame indexed exactly like ``df`` with columns in
        `WINDOW_FEATURE_NAMES` (in that order), no nulls.
    """
    if card_col not in df.columns:
        raise KeyError(f"card_col '{card_col}' not found in DataFrame columns")
    if "TransactionDT" not in df.columns:
        raise KeyError("'TransactionDT' not found in DataFrame columns")
    if "TransactionAmt" not in df.columns:
        raise KeyError("'TransactionAmt' not found in DataFrame columns")

    n = len(df)
    out = pd.DataFrame(
        {
            "amount_zscore": np.full(n, _NUMERIC_NULL_FILL, dtype=np.float64),
            "txn_count_5m": np.zeros(n, dtype=np.float64),
            "txn_amount_sum_5m": np.zeros(n, dtype=np.float64),
            "txn_count_1h": np.zeros(n, dtype=np.float64),
            "txn_amount_sum_1h": np.zeros(n, dtype=np.float64),
            "txn_count_24h": np.zeros(n, dtype=np.float64),
            "txn_amount_sum_24h": np.zeros(n, dtype=np.float64),
            "txn_count_7d": np.zeros(n, dtype=np.float64),
            "txn_amount_sum_7d": np.zeros(n, dtype=np.float64),
            # -999 sentinel: "no prior transactions" is meaningfully different
            # from 0 seconds (which indicates an active velocity attack).
            "time_since_last_txn": np.full(n, _NUMERIC_NULL_FILL, dtype=np.float64),
        },
        index=df.index,
    )

    # Sort by card then time so we can iterate per-card in chronological
    # order.  We track the *original* positional index so we can write
    # results back into `out` at the right place.
    work = df[[card_col, "TransactionDT", "TransactionAmt"]].copy()
    card_s = work[card_col].astype(object)
    null_mask = (
        card_s.isna()
        | (card_s == -1)
        | (card_s == 0)
        | (card_s == "-1")
        | (card_s == "0")
        | (card_s == "missing")
    )
    if null_mask.any():
        null_indices = np.where(null_mask)[0]
        unique_missing = [f"__missing_{i}" for i in range(len(null_indices))]
        card_s.iloc[null_indices] = unique_missing
    work[card_col] = card_s
    work["_orig_idx"] = np.arange(n)
    work = work.sort_values([card_col, "TransactionDT"], kind="mergesort")

    # Pre-extract numpy arrays for speed — pure-Python loop, but with
    # only 400k rows and ~12k unique cards it finishes in ~3s.
    cards = work[card_col].to_numpy()
    times = work["TransactionDT"].to_numpy(dtype=np.float64)
    amts = work["TransactionAmt"].to_numpy(dtype=np.float64)
    orig_idx = work["_orig_idx"].to_numpy()

    current_card: object = None
    hist_ts: list[float] = []  # strictly-increasing timestamps
    hist_amt: list[float] = []  # aligned with hist_ts
    rolling_amt_window: list[float] = []  # last _AMOUNT_ROLLING_WINDOW amounts

    for i in range(n):
        card = cards[i]
        ts = times[i]
        amt = amts[i]

        if card != current_card:
            # New card → reset history.
            current_card = card
            hist_ts = []
            hist_amt = []
            rolling_amt_window = []

        # ── amount_zscore (vs. the card's previous rolling window) ──
        if rolling_amt_window:
            mean = float(np.mean(rolling_amt_window))
            std = float(np.std(rolling_amt_window))
            if std > 0:
                out.iat[orig_idx[i], 0] = (amt - mean) / std
            else:
                out.iat[orig_idx[i], 0] = 0.0
        # else: leave at the -999 sentinel (first transaction for this card)

        # ── count / sum over trailing windows ──
        # 5-minute window
        cutoff_5m = ts - _WINDOW_5M_SEC
        j = bisect.bisect_left(hist_ts, cutoff_5m)
        out.iat[orig_idx[i], 1] = float(len(hist_ts) - j)
        out.iat[orig_idx[i], 2] = float(sum(hist_amt[j:]))

        # 1-hour window
        cutoff_1h = ts - _WINDOW_1H_SEC
        k = bisect.bisect_left(hist_ts, cutoff_1h)
        out.iat[orig_idx[i], 3] = float(len(hist_ts) - k)
        out.iat[orig_idx[i], 4] = float(sum(hist_amt[k:]))

        # 24-hour window
        cutoff_24h = ts - _WINDOW_24H_SEC
        m = bisect.bisect_left(hist_ts, cutoff_24h)
        out.iat[orig_idx[i], 5] = float(len(hist_ts) - m)
        out.iat[orig_idx[i], 6] = float(sum(hist_amt[m:]))

        # 7-day window
        cutoff_7d = ts - _WINDOW_7D_SEC
        p = bisect.bisect_left(hist_ts, cutoff_7d)
        out.iat[orig_idx[i], 7] = float(len(hist_ts) - p)
        out.iat[orig_idx[i], 8] = float(sum(hist_amt[p:]))

        # ── time_since_last_txn (seconds since previous tx for this card) ──
        if hist_ts:
            out.iat[orig_idx[i], 9] = float(ts - hist_ts[-1])
        # else: leave at -999.0 (first transaction for this card)

        # Append current transaction to history.
        hist_ts.append(ts)
        hist_amt.append(amt)
        rolling_amt_window.append(amt)
        if len(rolling_amt_window) > _AMOUNT_ROLLING_WINDOW:
            rolling_amt_window.pop(0)

    return out


# ── Public assembly helper (used by both offline + online paths) ───────────


def assemble_features(
    df: pd.DataFrame,
    card_col: str = "card1",
) -> pd.DataFrame:
    """Compute static + window features and return one combined frame.

    The output has exactly the columns in `FEATURE_NAMES` (in order)
    and is indexed like the input.  Null values are not expected; if
    any sneak in they are filled with `_NUMERIC_NULL_FILL`.
    """
    static = compute_static_features(df)
    window = compute_window_features(df, card_col=card_col)
    out = pd.concat([static, window], axis=1)
    # Re-order to the canonical FEATURE_NAMES order.
    out = out[list(FEATURE_NAMES)]
    out = out.fillna(_NUMERIC_NULL_FILL)
    return out
