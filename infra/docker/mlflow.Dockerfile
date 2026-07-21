# syntax=docker/dockerfile:1
# ── FraudGuard MLflow Server ──────────────────────────────────────────────
# Uses the official mlflow image; backend store is Postgres (configured at
# runtime via docker-compose environment). Artifacts stored in /mlflow/artifacts.

FROM ghcr.io/mlflow/mlflow:v2.13.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.9

EXPOSE 5000

# CMD is supplied by docker-compose.yml to allow runtime env var substitution
# for the backend store URI (Postgres credentials come from environment).
