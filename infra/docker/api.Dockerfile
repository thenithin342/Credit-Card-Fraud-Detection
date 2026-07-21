# syntax=docker/dockerfile:1
# ── FraudGuard API — Production Dockerfile ──────────────────────────────
# Multi-stage build: builder installs deps, final image only has runtime.

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --user --prefer-binary -r requirements.txt

# ── Final stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

WORKDIR /app

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy source code
COPY src/ ./src/
COPY params.yaml .

# Use non-root user
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
