# Tech Stack & Rationale

Every choice below favors tools that are either (a) genuinely industry-standard and
worth having on a resume as-is, or (b) deliberately hand-built because building them
yourself is what proves depth in an interview. The "Why" column is written so you can
lift it almost directly into an interview answer.

| Layer | Tool | Why | Alternative considered |
|---|---|---|---|
| Language | Python 3.11 | Standard for ML/data infra; type hints + modern async support | — |
| Streaming ingestion | **Redpanda** (Kafka API-compatible) | Single Docker container, no ZooKeeper, but you write and reason about real Kafka-protocol producers/consumers | Kafka (heavier local setup), Redis Streams (lighter but less resume-recognized) |
| Online feature store | **Custom-built**: Redis (low-latency key-value) + thin FastAPI/Python service layer | Building this yourself proves you understand *what* a feature store actually does (point lookups, TTLs, online/offline consistency) rather than just "I used Feast" | Feast (managed, faster to stand up, less differentiating) |
| Offline feature store | PostgreSQL | Durable historical feature log for training + audit + reproducibility | Parquet files on disk (simpler, less realistic) |
| Experiment tracking & model registry | **MLflow** | Genuinely the industry standard; no value in reinventing this | Weights & Biases (great, but adds an external account dependency) |
| Orchestration | **Apache Airflow** | Most recognized name on a resume for retraining DAGs; official Docker Compose stack is well documented | Prefect (lighter, more Pythonic, good if Airflow setup friction is too high) |
| Baseline model | Logistic Regression (scikit-learn) | Establishes an interpretable floor before adding complexity | — |
| Primary supervised model | XGBoost / LightGBM | Handles tabular + imbalanced data well; gives you feature importance + calibration talking points | Random Forest |
| Unsupervised anomaly signal | Isolation Forest or a small autoencoder (PyTorch, CPU) | Adds the "anomaly detection" half of the project — catches fraud patterns with no labeled precedent | One-Class SVM (slower, less scalable) |
| Real-time serving | **FastAPI** + Uvicorn, Dockerized | Async, fast, typed, minimal boilerplate; the de facto standard for ML serving APIs | Flask (older, less async-native), TorchServe/Triton (overkill for tabular models) |
| Drift detection | **Custom-built**: PSI + KS-test module, scheduled batch job | Same rationale as the feature store — shows you understand the statistics behind "drift," not just a library call | Evidently AI (good tool, but writing this yourself is the higher-signal choice for this project) |
| Metrics & dashboards | Prometheus + Grafana | Standard observability stack; transfers directly to any backend/infra role | Datadog (paid, not local-first) |
| CI/CD | GitHub Actions | Free, ubiquitous, directly portable to any employer's workflow | GitLab CI, Jenkins |
| Containerization | Docker + Docker Compose | Local-first, reproducible, no cloud dependency for the core deliverable | — |
| Orchestration at scale *(stretch)* | Kubernetes (kind/minikube) + basic manifests | Only if you want the Phase 8 stretch goal; not required for the core resume story | — |
| Data & model versioning | DVC + Git | Reproducibility requirement (PRD §8); pairs naturally with MLflow artifacts | Git-LFS alone (weaker semantics) |
| Config management | pydantic-settings + `.env` | Type-safe config, no secrets in code | python-decouple |
| Testing | pytest, pytest-cov | Standard; required for CI gate | — |
| Formatting / linting | black, ruff, mypy | Signals code quality discipline in the repo | — |

## Design principle behind these choices

Two tools in this stack are deliberately **not** the "obvious managed choice"
(Feast, Evidently). That's intentional: in an interview, "I used library X" is a
much weaker answer than "I built the core of what library X does, here's the
statistic behind it, and here's why I'd reach for the managed version in a real
company with more scale and less time." Both framings are included in
`docs/DECISIONS.md` so you have the tradeoff articulated, not just the code.
