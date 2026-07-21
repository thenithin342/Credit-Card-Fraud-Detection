# Data Specification — FraudGuard

## 1. Source Datasets

### IEEE-CIS Fraud Detection (Primary)
- **Source**: Kaggle competition `ieee-fraud-detection`
- **Size**: 590,540 transactions, 434 columns after merge
- **Fraud rate**: ~3.5% (realistic imbalance)
- **Features**: Transaction table (amount, card info, product, email domains, C/D/M/V columns) + Identity table (device type, browser, OS, screen size)
- **Download**: `kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-cis`

### Credit Card Fraud Detection (ULB)
- **Source**: Kaggle dataset `mlg-ulb/creditcardfraud`
- **Size**: 284,807 transactions
- **Fraud rate**: ~0.172% (severe imbalance — primary benchmark dataset)
- **Features**: `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class`
- **Download**: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ulb --unzip`

Both datasets are DVC-tracked — never committed directly to git.

## 2. Known Limitations & Mitigations

**IEEE-CIS**: Categorical columns have high cardinality and many nulls (~50% in some identity columns). Mitigation: frequency encoding + careful null imputation in feature engineering.

**ULB**: No card/user identifier, which is unrealistic for a real fraud system (fraud detection lives and dies on per-entity behavioral history). Mitigation: assign synthetic `card_id` by randomly assigning each transaction to one of N=5,000 synthetic cards with a fixed seed.

> **Interview talking point**: *"The dataset doesn't include entity IDs, so I simulated them to make the online feature store meaningful — in a real system these would come from the account/card table."* State this upfront; it reads as maturity, not a flaw.

## 3. Streaming Simulation

The producer (`src/ingestion/producer.py`) replays rows **sorted by time**, publishing each as a JSON message to the `transactions.raw` Redpanda topic. Replay speed is configurable via `REPLAY_SPEED_MULTIPLIER` (e.g. 60 = 1 simulated hour per real minute).

Example message schema:
```json
{
  "transaction_id": "uuid4",
  "card_id": "synthetic_id",
  "time": 86400,
  "amount": 149.62,
  "v_features": {"V1": -1.359, "V2": -0.072, "...": "...", "V28": -0.021},
  "event_timestamp": "2026-07-21T16:00:00Z"
}
```

## 4. Feature Schema

Defined **once** in `src/features/definitions.py` — imported by both offline writer and online store client. This single-source-of-truth is what the Phase 2 parity test verifies.

| Feature | Description | Source |
|---|---|---|
| `v1`...`v28` | Raw PCA components (ULB) / raw numeric cols (IEEE) | Raw dataset |
| `amount` | Raw transaction amount | Raw dataset |
| `amount_zscore` | Z-score of amount vs. this card's rolling history | Engineered |
| `txn_count_5m` | Transaction count for this card in trailing 5 min | Engineered |
| `txn_amount_sum_5m` | Sum of amounts for this card in trailing 5 min | Engineered |
| `txn_count_1h` | Transaction count for this card in trailing 1 hour | Engineered |
| `txn_amount_sum_1h` | Sum of amounts for this card in trailing 1 hour | Engineered |
| `time_since_last_txn` | Seconds since this card's previous transaction | Engineered |

## 5. Splitting Rules

- **Respect chronological order**: train on earlier `Time`, validate and test on strictly later portions. Random shuffling is explicitly disallowed (it leaks future information).
- **Splits**: 70% train / 10% val / 20% test by chronological order.
- **Current split** (IEEE-CIS): train=413,378 | val=59,054 | test=118,108
- **Fraud rate preserved**: train=3.51% | val=3.51% | test=3.44% ✅
- Log exact split boundaries as an MLflow parameter for every run.

## 6. DVC Pipeline

```
dvc repro
├── ingest    → downloads raw data to data/raw/
├── validate  → Great Expectations: 25/25 IEEE + 35/35 ULB checks
└── split     → data/processed/train.parquet, val.parquet, test.parquet
```

## 7. Validation Results (Phase 1)

| Dataset | Expectations | Passed | Failed |
|---------|-------------|--------|--------|
| IEEE-CIS | 25 | 25 | 0 |
| ULB | 35 | 35 | 0 |

Reports: `reports/validation/ieee_cis_validation.txt`, `reports/validation/ulb_validation.txt`
