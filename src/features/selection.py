"""src/features/selection.py
────────────────────────────────────────────────────────────────────────
Feature selection utilities — null dropping, variance filtering, and
correlation pruning.

Fitted **only** on the training split.  Applied identically to val
and test to prevent leakage.  The full pipeline is exposed as
`select_features()` which returns a report dict describing what
was dropped and why.

Conventions
-----------
* `drop_high_null_cols`    — drops columns whose null rate exceeds the
                             threshold (default 0.80).
* `drop_low_variance_cols` — drops numeric columns whose non-null
                             standard deviation is below the threshold
                             (default 0.01).
* `drop_correlated_cols`   — drops one column from each pair whose
                             absolute Pearson correlation exceeds the
                             threshold (default 0.95).  When the block
                             is large (e.g. 339 V* columns) we process
                             in blocks of 50 to avoid OOM on the
                             n×n covariance matrix.
* `select_features`        — composes the three above.  Always returns
                             a *new* list of selected column names
                             plus a report dict.

Important
---------
These utilities are **fit-only on train**.  Applying them to val/test
will silently use the train-time decision boundaries and would
introduce subtle leakage via the thresholding step.  The intended
usage in `offline_store.run()` is:

    selected_cols, report = select_features(train_df)
    # now apply the SAME column-drop mask to val/test
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Default thresholds (overridable per call).
HIGH_NULL_THRESHOLD: float = 0.80  # drop cols with >80% missing
LOW_VARIANCE_STD: float = 0.01  # drop numeric cols with std < this
CORRELATION_THRESHOLD: float = 0.95  # drop one of any pair with |r| > this

# Block size for the V* correlation sweep.  Each block is B×B which
# stays well under the 339×339 OOM ceiling on 16 GB boxes.
_CORR_BLOCK_SIZE: int = 50


def drop_high_null_cols(
    df: pd.DataFrame,
    threshold: float = HIGH_NULL_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where the null rate strictly exceeds ``threshold``.

    The check uses **strict greater-than** so a column whose null
    rate equals the threshold is kept; call sites that need
    *greater-than-or-equal* semantics can pass ``threshold - 1e-9``.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Not mutated.
    threshold : float
        Drop any column whose fraction of null values is greater than
        this value.  Default 0.80.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The filtered frame and the names of the columns that were
        dropped (in original order).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    null_rate = df.isna().mean()
    dropped_mask = null_rate > threshold
    dropped_cols = null_rate[dropped_mask].index.tolist()
    if dropped_cols:
        log.info(
            "dropping_high_null_cols",
            threshold=threshold,
            n_dropped=len(dropped_cols),
            sample=sorted(dropped_cols)[:5],
        )
    return df.drop(columns=dropped_cols), dropped_cols


def drop_low_variance_cols(
    df: pd.DataFrame,
    threshold: float = LOW_VARIANCE_STD,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop numeric columns whose non-null standard deviation is below
    ``threshold``.

    Non-numeric columns (object dtype, etc.) are left untouched
    because variance is not defined for them; callers should encode
    categoricals first if they want to drop low-cardinality strings.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Not mutated.
    threshold : float
        Drop any numeric column with std on non-null values strictly
        less than this value.  Default 0.01.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The filtered frame and the dropped column names.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    dropped: list[str] = []
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        if len(non_null) == 0:
            # All-null cols are already caught by the null step; we
            # skip them here to avoid a std-on-empty RuntimeWarning.
            continue
        std = float(non_null.std())
        if std < threshold:
            dropped.append(col)
    if dropped:
        log.info(
            "dropping_low_variance_cols",
            threshold=threshold,
            n_dropped=len(dropped),
            sample=sorted(dropped)[:5],
        )
    return df.drop(columns=dropped), dropped


def drop_correlated_cols(
    df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    block_size: int = _CORR_BLOCK_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one column from each pair whose |Pearson r| > threshold.

    For a pair (A, B) with |r| > threshold, the column whose name
    sorts *later* alphabetically is dropped.  In other words we keep
    the column that appears **first** alphabetically.  This is a
    deterministic, leakage-free tie-breaker.

    Implementation notes
    --------------------
    Computing the full 339×339 correlation matrix on 400k rows is
    borderline OOM on a 16 GB machine (≈ 0.9 GB of float64).  We
    therefore process the columns in overlapping blocks of
    ``block_size`` (default 50) and accumulate the drop list.  Each
    block only needs a 50×50 correlation matrix regardless of
    the full column count.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Only numeric columns participate.  Not mutated.
    threshold : float
        Drop if |r| > threshold.  Default 0.95.
    block_size : int
        Block size for the rolling correlation sweep.  Default 50.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The filtered frame and the dropped column names.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) <= 1:
        return df.copy(), []

    dropped: set[str] = set()
    n_blocks = (len(numeric_cols) + block_size - 1) // block_size

    for i in range(n_blocks):
        start_i = i * block_size
        end_i = min(start_i + block_size, len(numeric_cols))
        block_i_cols = [c for c in numeric_cols[start_i:end_i] if c not in dropped]
        if not block_i_cols:
            continue

        for j in range(i, n_blocks):
            start_j = j * block_size
            end_j = min(start_j + block_size, len(numeric_cols))
            block_j_cols = [c for c in numeric_cols[start_j:end_j] if c not in dropped]
            if not block_j_cols:
                continue

            if i == j:
                # Same block: check upper triangle
                corr = df[block_i_cols].corr(min_periods=50).abs()
                for r_idx in range(len(block_i_cols)):
                    col_r = block_i_cols[r_idx]
                    if col_r in dropped:
                        continue
                    for c_idx in range(r_idx + 1, len(block_i_cols)):
                        col_c = block_i_cols[c_idx]
                        if col_c in dropped:
                            continue
                        r = corr.iloc[r_idx, c_idx]
                        if pd.isna(r):
                            continue
                        if r > threshold:
                            if col_r < col_c:
                                dropped.add(col_c)
                            else:
                                dropped.add(col_r)
                                break
            else:
                # Cross-block: check combinations of block i and block j
                combined_cols = block_i_cols + block_j_cols
                corr = df[combined_cols].corr(min_periods=50).abs()

                for r_idx, col_r in enumerate(block_i_cols):
                    if col_r in dropped:
                        continue
                    for c_idx, col_c in enumerate(block_j_cols):
                        if col_c in dropped:
                            continue
                        r = corr.iloc[r_idx, len(block_i_cols) + c_idx]
                        if pd.isna(r):
                            continue
                        if r > threshold:
                            if col_r < col_c:
                                dropped.add(col_c)
                            else:
                                dropped.add(col_r)

    dropped_cols = sorted(dropped)
    if dropped_cols:
        log.info(
            "dropping_correlated_cols",
            threshold=threshold,
            n_dropped=len(dropped_cols),
            sample=dropped_cols[:5],
        )
    return df.drop(columns=dropped_cols), dropped_cols


def select_features(
    train_df: pd.DataFrame,
    null_threshold: float = HIGH_NULL_THRESHOLD,
    var_threshold: float = LOW_VARIANCE_STD,
    corr_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], dict[str, Any]]:
    """Run the full selection pipeline on the training frame.

    IMPORTANT: fit ONLY on ``train_df``, never on val/test.  Applying
    selection separately to each split would use different threshold
    boundaries and silently leak val/test statistics.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training frame.  Must already be the result of any
        upstream cleaning (e.g. column renaming).  This function
        does NOT remove the target column — callers are expected
        to exclude it from the returned ``selected_cols`` if they
        do not want it as a feature.
    null_threshold, var_threshold, corr_threshold : float
        See `drop_high_null_cols`, `drop_low_variance_cols`,
        `drop_correlated_cols` for the semantics of each threshold.

    Returns
    -------
    tuple[list[str], dict[str, Any]]
        * ``selected_cols`` — the column names that survived all
          three filters, in their original order.
        * ``selection_report`` — a dict with keys
          ``dropped_high_null``, ``dropped_low_variance``,
          ``dropped_correlated``, and ``final_count``.
    """
    log.info("feature_selection_start", rows=len(train_df), cols=len(train_df.columns))

    df, dropped_high_null = drop_high_null_cols(train_df, threshold=null_threshold)
    df, dropped_low_variance = drop_low_variance_cols(df, threshold=var_threshold)
    df, dropped_correlated = drop_correlated_cols(df, threshold=corr_threshold)

    selected_cols = df.columns.tolist()
    report: dict[str, Any] = {
        "dropped_high_null": dropped_high_null,
        "dropped_low_variance": dropped_low_variance,
        "dropped_correlated": dropped_correlated,
        "final_count": len(selected_cols),
    }
    log.info(
        "feature_selection_done",
        n_dropped_high_null=len(dropped_high_null),
        n_dropped_low_variance=len(dropped_low_variance),
        n_dropped_correlated=len(dropped_correlated),
        final_count=report["final_count"],
    )
    return selected_cols, report
