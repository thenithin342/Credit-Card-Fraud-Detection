"""src/features/offline_store.py
────────────────────────────────────────────────────────────────────────
Offline feature-store builder for FraudGuard (Phase 2B).

Reads the temporal splits produced by `src.ingestion.split`, then
constructs a *combined* feature matrix:

    Final features
        = selected raw columns (V* + id_* + card* + ProductCD + M* + emails
          + DeviceInfo + C* + D* + addr* + dist*)
        + 9 temporal features from `src.features.definitions` (FEATURE_NAMES)

Raw columns are first passed through `src.features.selection` to drop
high-null / low-variance / highly-correlated columns, then through
`src.features.preprocessing.FeaturePreprocessor` for null imputation
and ordinal encoding of categoricals.  Both stages are fit on the
*training* split only and applied identically to val and test.

Outputs
-------
data/features/{train,val,test}_features.parquet
models/encoders/feature_preprocessor.pkl
models/feature_columns.json           — exact ordered feature list (parity contract)

Usage
-----
    python -m src.features.offline_store
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

from src.features.definitions import (
    FEATURE_NAMES,
    TARGET_COL,
    assemble_features,
)
from src.features.preprocessing import NUMERIC_NULL_FILL, FeaturePreprocessor
from src.features.selection import (
    CORRELATION_THRESHOLD,
    HIGH_NULL_THRESHOLD,
    LOW_VARIANCE_STD,
    select_features,
)

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# Columns the model must NEVER see as input features.  They are
# carried through to the output parquets for traceability but
# dropped from the X matrix at training time.
ID_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "TransactionDT",
    TARGET_COL,
)


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_split_paths(params: dict) -> dict[str, Path]:
    """Map split names → input parquet paths under data/processed/."""
    processed_dir = PROJECT_ROOT / params["data"]["processed_dir"]
    return {name: processed_dir / f"{name}.parquet" for name in ("train", "val", "test")}


def _resolve_output_paths(params: dict) -> dict[str, Path]:
    """Map split names → output feature-parquet paths under data/features/."""
    features_dir = PROJECT_ROOT / params["data"]["features_dir"]
    return {name: features_dir / f"{name}_features.parquet" for name in ("train", "val", "test")}


def _coerce_card1_to_int(df: pd.DataFrame, card_col: str) -> pd.DataFrame:
    """Assign unique missing identifiers per row for missing/null card1 to prevent velocity pooling."""
    if card_col in df.columns:
        df = df.copy()
        mask = df[card_col].isna() | (df[card_col] == -1) | (df[card_col] == 0)
        if mask.any():
            s = df[card_col].astype(object)
            missing_indices = np.where(mask)[0]
            unique_missing = [f"__missing_{i}" for i in range(len(missing_indices))]
            s.iloc[missing_indices] = unique_missing
            df[card_col] = s
    return df


def build_temporal_features(full_df: pd.DataFrame, card_col: str) -> pd.DataFrame:
    """Compute the 9 engineered temporal features for ``full_df``,
    in place of the 9 columns from `FEATURE_NAMES`.

    Returns a frame with columns [TransactionID, *FEATURE_NAMES] in
    that order, indexed like ``full_df``.
    """
    df = _coerce_card1_to_int(full_df, card_col)
    features = assemble_features(df, card_col=card_col)
    out = pd.concat(
        [
            df[["TransactionID"]].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
    for col in FEATURE_NAMES:
        if col not in out.columns:
            out[col] = 0.0
    return out[["TransactionID", *FEATURE_NAMES]]


def run(params: dict | None = None) -> None:
    """Build features for train / val / test and write outputs.

    Pipeline
    --------
    1. Load all 3 splits from ``data/processed/``.
    2. Combine chronologically and compute 9 temporal window features
       on the full sequence (so val/test cards carry their training
       history forward).
    3. Run `select_features` on **train only** to determine which raw
       columns survive the null / variance / correlation filters.
    4. Drop ID_COLUMNS from the selected raw list.
    5. Fit `FeaturePreprocessor` on the **train** raw columns.
    6. Transform train / val / test with the same preprocessor.
    7. Concatenate the processed raw columns with the temporal features
       (temporal features come LAST — see Critical Rule #3).
    8. Save `models/feature_columns.json` (the parity contract for
       Phase 3 serving).
    9. Write `data/features/{split}_features.parquet` for each split.
    10. Log the selection report via structlog.
    """
    if params is None:
        params = load_params()

    card_col = params["features"].get("card_id_col", "card1")
    features_cfg = params["features"]
    null_thr = float(features_cfg.get("null_drop_threshold", HIGH_NULL_THRESHOLD))
    var_thr = float(features_cfg.get("variance_threshold", LOW_VARIANCE_STD))
    corr_thr = float(features_cfg.get("correlation_threshold", CORRELATION_THRESHOLD))
    encoder_path = PROJECT_ROOT / features_cfg.get(
        "encoder_path", "models/encoders/feature_preprocessor.pkl"
    )
    feature_columns_path = PROJECT_ROOT / features_cfg.get(
        "feature_columns_path", "models/feature_columns.json"
    )

    inputs = _resolve_split_paths(params)
    outputs = _resolve_output_paths(params)

    # ── Step 1: load all three splits ──────────────────────────────────────
    log.info("loading_all_splits")
    split_dfs: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        split_dfs[name] = pd.read_parquet(inputs[name])
        log.info("split_loaded", split=name, rows=len(split_dfs[name]))

    # ── Step 2: combine chronologically and compute temporal features ─────
    full_df = pd.concat(
        [split_dfs["train"], split_dfs["val"], split_dfs["test"]],
        ignore_index=True,
    )
    full_df = full_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    log.info("combined_dataset", total_rows=len(full_df))

    log.info("computing_temporal_features")
    full_temporal = build_temporal_features(full_df, card_col=card_col)

    # ── Step 3: feature selection on train only ──────────────────────────
    log.info(
        "running_feature_selection",
        null_thr=null_thr,
        var_thr=var_thr,
        corr_thr=corr_thr,
    )
    selected_raw_cols, selection_report = select_features(
        split_dfs["train"],
        null_threshold=null_thr,
        var_threshold=var_thr,
        corr_threshold=corr_thr,
    )
    # Strip ID columns from the feature list — they are kept in the
    # output parquets for traceability but the model never sees them.
    selected_raw_cols = [c for c in selected_raw_cols if c not in ID_COLUMNS]
    log.info(
        "selected_raw_cols",
        n=len(selected_raw_cols),
        sample=sorted(selected_raw_cols)[:5],
    )

    # ── Step 4-6: fit preprocessor on train, transform all splits ────────
    log.info("fitting_preprocessor")
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(split_dfs["train"], selected_raw_cols)

    log.info("transforming_splits")
    processed: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        processed[name] = preprocessor.transform(split_dfs[name])
        log.info(
            "split_processed",
            split=name,
            rows=len(processed[name]),
            cols=len(processed[name].columns),
        )

    # ── Step 7: concat raw processed + temporal features ─────────────────
    # Re-attach the ID columns (TransactionID, isFraud, TransactionDT)
    # from the source splits so the output parquets stay self-describing.
    log.info("concatenating_features")
    for name in ("train", "val", "test"):
        split_ids = split_dfs[name][["TransactionID", TARGET_COL, "TransactionDT"]].reset_index(
            drop=True
        )
        temporal_rows = full_temporal[
            full_temporal["TransactionID"].isin(split_ids["TransactionID"])
        ].reset_index(drop=True)
        # Re-order temporal rows to match the split's row order.
        temporal_rows = (
            temporal_rows.set_index("TransactionID").loc[split_ids["TransactionID"]].reset_index()
        )
        feats = pd.concat(
            [
                split_ids,
                processed[name].reset_index(drop=True),
                temporal_rows[list(FEATURE_NAMES)].fillna(NUMERIC_NULL_FILL).reset_index(drop=True),
            ],
            axis=1,
        )
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(outputs[name], index=False, compression="snappy")
        log.info(
            "features_written",
            split=name,
            path=str(outputs[name]),
            rows=len(feats),
            cols=len(feats.columns),
        )

    # ── Step 8: save preprocessor + feature_columns.json parity contract ─
    preprocessor.save(encoder_path)
    log.info("encoder_saved", path=str(encoder_path))

    raw_feature_cols = list(processed["train"].columns)
    temporal_feature_cols = list(FEATURE_NAMES)
    all_feature_cols = raw_feature_cols + temporal_feature_cols

    feature_columns = {
        "raw_feature_cols": raw_feature_cols,
        "temporal_feature_cols": temporal_feature_cols,
        "all_feature_cols": all_feature_cols,
    }
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
    log.info(
        "feature_columns_saved",
        path=str(feature_columns_path),
        n_raw=len(raw_feature_cols),
        n_temporal=len(temporal_feature_cols),
        n_total=len(all_feature_cols),
    )

    # ── Step 9: log selection report ─────────────────────────────────────
    log.info(
        "selection_report",
        dropped_high_null=len(selection_report["dropped_high_null"]),
        dropped_low_variance=len(selection_report["dropped_low_variance"]),
        dropped_correlated=len(selection_report["dropped_correlated"]),
        final_raw_count=selection_report["final_count"],
    )


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    run()
    log.info("offline_feature_store_done")


if __name__ == "__main__":
    sys.exit(main() or 0)
