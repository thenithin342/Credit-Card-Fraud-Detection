# Minimal MLflow tracking server image.
# There is no official MLflow image, so we build a small one.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary

WORKDIR /mlflow

EXPOSE 5000

# Backend store and artifact root are kept simple (local) for Phase 0/2.
# Revisit to point the backend store at Postgres once docs/ROADMAP.md Phase 2
# needs multi-run querying at scale.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
