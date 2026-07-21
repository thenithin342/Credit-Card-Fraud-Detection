# Data Specification

## 1. Source Dataset

**Kaggle: Credit Card Fraud Detection** (ULB Machine Learning Group / Worldline)
- ~284,807 transactions made by European cardholders over two days in September 2013
- 492 labeled frauds (~0.172% positive rate — severe imbalance, by design)
- Features: `Time` (seconds elapsed since the first transaction), `V1`–`V28`
  (PCA-anonymized features, original meaning not disclosed for confidentiality),
  `Amount`, and `Class` (1 = fraud, 0 = legitimate)

Download via the Kaggle CLI:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```
(Requires a free Kaggle account and API token in `~/.kaggle/kaggle.json`.)

Immediately DVC-track the raw file so every experiment is reproducible against an
exact data version:
```bash
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc .gitignore
git commit -m "data: track raw dataset with DVC"
```

## 2. Known Limitation & Mitigation

The dataset has **no card/user identifier**, which is unrealistic for a real fraud
system (fraud detection lives and dies on per-entity behavioral history). To make
the feature engineering meaningful:

- Assign a synthetic `card_id` by clustering transactions using `Amount` and a
  rolling time-window heuristic, **or** — simpler and defensible — randomly assign
  each transaction to one of N synthetic cards (e.g. N=5,000) with a fixed seed,
  documented explicitly as a simulation choice.
- State this limitation plainly in the README and in interviews: *"the dataset
  doesn't include entity IDs, so I simulated them to make the online feature store
  meaningful — in a real system these would come from the account/card table."*
  This kind of caveat, stated upfront, reads as maturity, not as a flaw to hide.

## 3. Streaming Simulation

The producer (`src/ingestion`) replays rows **sorted by `Time`**, publishing each
as a JSON message to the `transactions.raw` Redpanda topic. Replay speed is
configurable via an environment variable (`REPLAY_SPEED_MULTIPLIER`), so a
two-day dataset can be replayed in minutes for demo purposes while still
preserving realistic relative timing/order.

Example message schema published to `transactions.raw`:
```json
{
  "transaction_id": "uuid4",
  "card_id": "synthetic id, see above",
  "time": 0,
  "amount": 149.62,
  "v_features": { "V1": -1.359, "V2": -0.072, "...": "...", "V28": -0.021 },
  "event_timestamp": "ISO 8601, wall-clock time of publish"
}
```

## 4. Feature Schema

Defined **once** in `src/features/definitions.py` and imported by both the offline
writer and the online store client — this single-source-of-truth is what the
Phase 1 parity test verifies.

| Feature | Description | Source |
|---|---|---|
| `v1`...`v28` | Raw PCA components, passed through | Raw dataset |
| `amount` | Raw transaction amount | Raw dataset |
| `amount_zscore` | Z-score of amount vs. this card's rolling history | Engineered (online + offline) |
| `txn_count_5m` | Count of transactions for this card in trailing 5 min (simulated time) | Engineered (online + offline) |
| `txn_amount_sum_5m` | Sum of transaction amounts for this card in trailing 5 min | Engineered (online + offline) |
| `time_since_last_txn` | Seconds since this card's previous transaction | Engineered (online + offline) |

## 5. Label Handling & Splitting

- Respect chronological order: train on the earlier portion of `Time`, validate
  and test on strictly later portions. Random shuffling would leak future
  information into training and is explicitly disallowed.
- Because of the ~0.17% positive rate, do **not** rebalance the training set with
  naive oversampling alone — use `scale_pos_weight` (XGBoost) or class weighting,
  and evaluate exclusively with PR-AUC / recall-at-precision, never raw accuracy.
- Log the exact split boundaries (timestamps) as an MLflow parameter for every run
  so results are always traceable to a specific data cut.
