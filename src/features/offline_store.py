"""src/features/offline_store.py
────────────────────────────────────────────────────────────────────────
Offline feature-store builder for FraudGuard.

Reads the temporal splits produced by `src.ingestion.split` and writes
engineered features (one parquet per split) to `data/features/`.

This module is the *batch* counterpart of the online store.  Both
share `src.features.definitions` so the feature values are
guaranteed identical.

Usage
-----
    python -m src.features.offline_store

Inputs
------
data/processed/{train,val,test}.parquet   (DVC-tracked)

Outputs
-------
data/features/{train,val,test}_features.parquet
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog
import yaml

from src.features.definitions import (
    FEATURE_NAMES,
    TARGET_COL,
    assemble_features,
)

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


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
    """Coerce card1 to a plain Python int64 column (IEEE-CIS sometimes
    loads it as float when nulls are present).  Returns a copy."""
    if card_col in df.columns and df[card_col].dtype != "int64":
        df = df.copy()
        df[card_col] = df[card_col].fillna(-1).astype("int64")
    return df


def build_features_for_split(
    df: pd.DataFrame,
    card_col: str = "card1",
) -> pd.DataFrame:
    """Compute engineered features for one split and preserve the
    identifying columns + target.

    Returns a DataFrame with the columns:

        TransactionID, isFraud, TransactionDT, <FEATURE_NAMES...>
    """
    df = _coerce_card1_to_int(df, card_col)
    features = assemble_features(df, card_col=card_col)

    out = pd.concat(
        [
            df[["TransactionID", TARGET_COL, "TransactionDT"]].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    # Guarantee exact column order + presence.
    ordered = ["TransactionID", TARGET_COL, "TransactionDT", *FEATURE_NAMES]
    for col in FEATURE_NAMES:
        if col not in out.columns:
            out[col] = 0.0
    out = out[ordered]
    return out


def run(params: dict | None = None) -> None:
    """Build features for train / val / test and write parquets.

    IMPORTANT — single chronological pass:
    Window features (rolling aggregates per card) are computed on the
    *full* combined dataset in chronological order so that val and test
    cards carry their training-period history forward.  Processing each
    split independently would reset the per-card state to zero at the
    start of val/test, destroying the temporal signal and producing
    misleadingly low PR-AUC scores (~8%).
    """
    if params is None:
        params = load_params()

    card_col = params["features"].get("card_id_col", "card1")

    inputs = _resolve_split_paths(params)
    outputs = _resolve_output_paths(params)

    # ── Step 1: load all three splits ──────────────────────────────────
    log.info("loading_all_splits")
    split_dfs: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        split_dfs[name] = pd.read_parquet(inputs[name])
        log.info("split_loaded", split=name, rows=len(split_dfs[name]))

    # ── Step 2: combine chronologically and compute features once ───────
    # Concatenate all rows and sort by TransactionDT so window features
    # are computed in the correct temporal order.  The split membership
    # (train/val/test) is not relevant to per-card history — a card that
    # appeared in training will carry its transaction history into val.
    full_df = pd.concat(
        [split_dfs["train"], split_dfs["val"], split_dfs["test"]],
        ignore_index=True,
    )
    full_df = full_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    log.info("combined_dataset", total_rows=len(full_df))

    log.info("computing_features_full_pass")
    full_feats = build_features_for_split(full_df, card_col=card_col)

    # ── Step 3: split feature rows back using TransactionID membership ──
    for name in ("train", "val", "test"):
        out_path = outputs[name]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        split_ids = set(split_dfs[name]["TransactionID"].tolist())
        feats = full_feats[full_feats["TransactionID"].isin(split_ids)].copy()

        feats.to_parquet(out_path, index=False, compression="snappy")

        null_counts = feats.isna().sum().sum()
        log.info(
            "features_written",
            split=name,
            path=str(out_path),
            rows=len(feats),
            cols=len(feats.columns),
            null_count=int(null_counts),
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
