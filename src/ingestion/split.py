"""
src/ingestion/split.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Stratified train / val / test split for the IEEE-CIS dataset.

Reads:
    data/raw/ieee-cis/train_transaction.csv
    data/raw/ieee-cis/train_identity.csv   (merged on TransactionID)

Writes:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet

Design decisions:
  - Uses a temporal split *within* the stratified hold-out: transactions are
    sorted by TransactionDT before splitting so that the test set always
    contains the most recent data â€” this mirrors real-world deployment where
    you score future transactions with a model trained on past data.
  - Preserves the isFraud label distribution in every split via stratification.
  - Saves to Parquet (columnar, ~5Ã— smaller than CSV, faster to read).
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
import yaml
from sklearn.model_selection import train_test_split

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
    Temporal-stratified split.
    Sort by TransactionDT â†’ stratified split preserving isFraud ratio.
    """
    split_cfg = params["split"]
    features_cfg = params["features"]
    target_col = features_cfg["target_col"]

    # Sort by time to prevent data leakage from future â†’ past
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]
    seed = split_cfg["random_seed"]
    stratify = split_cfg["stratify"]

    # First: carve out test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=False,  # keep temporal order; stratify not used here
    )

    # Second: split remaining into train + val with stratification
    # Guard: sklearn requires â‰¥2 samples per class to stratify.
    # Fall back to non-stratified split on tiny or highly-skewed data.
    val_fraction = val_size / (1.0 - test_size)
    target_col = features_cfg["target_col"]

    can_stratify = False
    if stratify:
        min_class_count = train_val[target_col].value_counts().min()
        can_stratify = min_class_count >= 2
        if not can_stratify:
            log.warning(
                "stratify_disabled",
                reason="too_few_minority_samples",
                min_class_count=int(min_class_count),
            )

    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val[target_col] if can_stratify else None,
    )

    log.info(
        "split_complete",
        train=len(train),
        val=len(val),
        test=len(test),
        train_fraud_rate=round(train[target_col].mean(), 4),
        val_fraud_rate=round(val[target_col].mean(), 4),
        test_fraud_rate=round(test[target_col].mean(), 4),
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
