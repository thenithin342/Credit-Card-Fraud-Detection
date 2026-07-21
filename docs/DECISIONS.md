# Architecture Decision Log — FraudGuard

Use this log as your interview material. Every entry is a potential 5-minute answer.

---

## Template

**Decision:** One-line description  
**Date:** YYYY-MM-DD  
**Status:** Accepted | Superseded | Deprecated  
**Context:** What forced this decision?  
**Choice:** What was chosen and why?  
**Alternatives considered:** What was rejected and why?  
**Consequences:** What trade-offs does this introduce?

---

## ADR-001: Great Expectations over Pandera for data validation

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Need schema + distribution validation at ingest time that integrates with DVC pipeline and fails CI on bad data.  
**Choice:** Great Expectations 0.18.x with the legacy PandasDataset API (`gx.from_pandas()`). Runs as a DVC stage; exits with code 1 on any failure.  
**Alternatives considered:** Pandera (lighter, cleaner API); custom assertions (no standardization).  
**Consequences:** GE's API changed significantly between 0.18.x minor versions (data_sources vs sources vs from_pandas). Had to pin to the stable legacy API. GE is heavier than Pandera but more recognizable in enterprise ML pipelines.

---

## ADR-002: Redpanda over Apache Kafka for local streaming

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Need a Kafka-compatible message broker that runs in a single Docker container without ZooKeeper.  
**Choice:** Redpanda — single binary, Kafka-protocol-compatible, runs on a laptop.  
**Alternatives considered:** Apache Kafka (requires ZooKeeper, more complex Compose setup), Redis Streams (simpler but not Kafka-compatible, less resume-recognized).  
**Consequences:** All producer/consumer code uses the standard `kafka-python` client — directly portable to a real Kafka cluster at any employer.

---

## ADR-003: Custom feature store over Feast

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Need consistent features between training (offline) and serving (online) paths — no train/serve skew.  
**Choice:** Hand-built: Redis for online (point lookups by card_id with TTLs) + Postgres for offline (append-only feature log). Feature definitions live in a single `src/features/definitions.py` module imported by both paths.  
**Alternatives considered:** Feast (managed, faster to stand up). Rejected because "I used Feast" is a much weaker interview answer than "I built what Feast does — here's the Redis schema, here's why TTLs matter, here's the parity test."  
**Consequences:** More code to maintain; the parity test is the safety net.

---

## ADR-004: Custom drift detector over Evidently AI

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Need drift detection that is explainable at the statistics level.  
**Choice:** Hand-built PSI (Population Stability Index) + KS-test module. PSI chosen because it's the industry standard in banking/fintech; KS-test for statistical rigor.  
**Alternatives considered:** Evidently AI (good tool, well-documented, faster). Rejected for the same reason as Feast — building this yourself is the higher-signal interview choice for a fraud detection portfolio.  
**Consequences:** Must implement and validate the statistics correctly. The reward is a genuine ability to explain PSI thresholds (0.1 = minor, 0.2 = major) and why KS-test p-value matters.

---

## ADR-005: MLflow over Weights & Biases for experiment tracking

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Need experiment tracking + model registry that runs locally with no external accounts.  
**Choice:** MLflow with SQLite backend locally. No internet dependency, no API keys, Postgres backend available for production.  
**Alternatives considered:** W&B (excellent UI, strong community, but requires an account and internet connection — violates the local-first constraint).  
**Consequences:** MLflow UI is less polished than W&B but the concepts (runs, params, metrics, artifacts, registry stages) are exactly what interviewers ask about.

---

## ADR-006: DVC over Git-LFS for data versioning

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** 590k+ row datasets and parquet outputs cannot be committed to git directly.  
**Choice:** DVC with a local cache (`.dvc/cache`). `dvc.yaml` defines the full ingest→validate→split pipeline as a DAG; `dvc repro` runs only changed stages.  
**Alternatives considered:** Git-LFS (simpler but no pipeline semantics — just storage), storing paths in .gitignore and sharing via manual download (not reproducible).  
**Consequences:** Every collaborator must run `dvc pull` after `git pull`. The pipeline DAG in `dvc.yaml` is the single source of truth for data provenance.

---

## ADR-007: Chronological split over random split

**Date:** 2026-07-20  
**Status:** Accepted  
**Context:** Fraud patterns evolve over time (concept drift). Random splitting would leak future fraud patterns into training, producing overly optimistic validation metrics.  
**Choice:** Strict chronological split: 70% train (earliest) / 10% val / 20% test (latest). Class ratio preserved via stratified sampling within chronological windows.  
**Alternatives considered:** Random split (easier, standard in many Kaggle solutions — explicitly disallowed here because it doesn't reflect production reality).  
**Consequences:** Test set is the "hardest" split — model must generalize to unseen time periods. This is the correct production analog and worth emphasizing in interviews.
