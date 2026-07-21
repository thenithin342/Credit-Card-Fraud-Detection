# Tech Stack & Rationale — FraudGuard

Every choice below favors tools that are either (a) genuinely industry-standard and
worth having on a resume as-is, or (b) deliberately hand-built because building them
yourself is what proves depth in an interview.

| Layer | Tool | Why | Alternative considered |
|---|---|---|---|
| Language | Python 3.11 | Standard for ML/data infra; type hints + modern async | — |
| Streaming ingestion | **Redpanda** (Kafka API-compatible) | Single Docker container, no ZooKeeper, but you write real Kafka-protocol producers/consumers | Kafka (heavier), Redis Streams (less resume-recognized) |
| Online feature store | **Custom-built**: Redis + Python service layer | Proves you understand *what* a feature store actually does — not just "I used Feast" | Feast (managed, faster, less differentiating) |
| Offline feature store | PostgreSQL | Durable historical feature log for training + audit + reproducibility | Parquet files on disk (simpler, less realistic) |
| Data versioning | **DVC** + Git | Reproducibility — datasets, processed features, and model artifacts are tracked | Git-LFS alone (weaker semantics) |
| Data validation | **Great Expectations 0.18.x** | Schema + distribution checks at ingest time; CI-friendly exit code | Pandera (lighter but less enterprise-recognized) |
| Experiment tracking | **MLflow** | Genuinely the industry standard | W&B (great, but adds external account dependency) |
| Orchestration | **Apache Airflow** | Most recognized name for retraining DAGs | Prefect (lighter, more Pythonic) |
| Baseline model | Logistic Regression (scikit-learn) | Interpretable floor before adding complexity | — |
| Primary model | XGBoost / LightGBM | Handles tabular + imbalanced data; feature importance talking points | Random Forest |
| Unsupervised signal | Isolation Forest or small PyTorch autoencoder | Catches fraud patterns with no labeled precedent | One-Class SVM (slower) |
| Serving | **FastAPI** + Uvicorn, Dockerized | Async, typed, minimal boilerplate; de facto standard for ML APIs | Flask (less async-native) |
| Drift detection | **Custom-built**: PSI + KS-test | Shows you understand the statistics behind "drift" — not just a library call | Evidently AI (good, but writing this is the higher-signal choice) |
| Metrics & dashboards | Prometheus + Grafana | Standard observability stack | Datadog (paid, not local-first) |
| CI/CD | GitHub Actions | Free, ubiquitous, directly portable to any employer's workflow | GitLab CI |
| Containerization | Docker + Docker Compose | Local-first, no cloud dependency | — |
| Config management | pydantic-settings + `.env` | Type-safe config, no secrets in code | python-decouple |
| Testing | pytest, pytest-cov | Standard; CI gate | — |
| Formatting / linting | black, ruff, mypy | Signals code quality discipline | — |
| Kubernetes *(stretch)* | kind/minikube + manifests | Phase 8 stretch goal only | — |

## Design Principle

Two tools are deliberately **not** the "obvious managed choice" (Feast, Evidently).
That's intentional: in an interview, "I used library X" is a much weaker answer than
"I built the core of what library X does, here's the statistic behind it, and here's
why I'd reach for the managed version in a real company with more scale and less time."
