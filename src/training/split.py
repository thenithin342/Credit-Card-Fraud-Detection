"""
src/training/split.py
────────────────────────────────────────────────────────────────────────
Temporal train / val / test split for the IEEE-CIS dataset.

Reads:
    data/raw/ieee-cis/train_transaction.csv
    data/raw/ieee-cis/train_identity.csv (merged on TransactionID)

Writes:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet

Design decisions:
  - Strict temporal splitting based on TransactionDT sequence to prevent
    data leakage from future transactions into past training windows.
  - Transactions are sorted chronologically by TransactionDT.
  - Split sequentially:
    - Train: earliest (1 - test_size - val_size) fraction
    - Val: middle val_size fraction
    - Test: latest test_size fraction
  - Saves to Parquet format for fast loading and space efficiency.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
import yaml

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ieee(params: dict) -> pd.DataFrame:
    """Merge transaction + identity tables on TransactionID."""
    data_cfg = params["data"]
    root = PROJECT_ROOT

    txn_path = root / data_cfg["ieee_train_transactions"]
    id_path = root / data_cfg["ieee_train_identity"]

    log.info("loading_transactions", path=str(txn_path))
    txn = pd.read_csv(txn_path)

    log.info("loading_identity", path=str(id_path))
    identity = pd.read_csv(id_path)

    log.info(
        "merging",
        txn_rows=len(txn),
        identity_rows=len(identity),
    )
    merged = txn.merge(identity, on="TransactionID", how="left")
    log.info("merged", total_rows=len(merged), total_cols=len(merged.columns))
    return merged


def split_data(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pure temporal split based on TransactionDT sequence.
    Sorts by TransactionDT and splits sequentially:
    - Train: earliest (1 - test_size - val_size) fraction
    - Val: middle val_size fraction
    - Test: latest test_size fraction

    Prevents temporal data leakage between train, val, and test.
    """
    split_cfg = params["split"]
    features_cfg = params["features"]
    target_col = features_cfg["target_col"]

    # Sort by time to ensure temporal sequence order
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)

    test_size = float(split_cfg.get("test_size", 0.20))
    val_size = float(split_cfg.get("val_size", 0.10))

    test_count = int(n * test_size)
    val_count = int(n * val_size)
    train_count = n - test_count - val_count

    train = df_sorted.iloc[:train_count].copy()
    val = df_sorted.iloc[train_count : train_count + val_count].copy()
    test = df_sorted.iloc[train_count + val_count :].copy()

    log.info(
        "temporal_split_complete",
        total_rows=n,
        train=len(train),
        val=len(val),
        test=len(test),
        train_fraud_rate=round(float(train[target_col].mean()), 4) if target_col in train else None,
        val_fraud_rate=round(float(val[target_col].mean()), 4) if target_col in val else None,
        test_fraud_rate=round(float(test[target_col].mean()), 4) if target_col in test else None,
    )
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, params: dict) -> None:
    processed_dir = PROJECT_ROOT / params["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        out = processed_dir / f"{name}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        log.info("saved_parquet", split=name, path=str(out), rows=len(df))


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    params = load_params()

    df = load_ieee(params)
    train, val, test = split_data(df, params)
    save_splits(train, val, test, params)

    log.info("split_stage_done")


if __name__ == "__main__":
    main()
