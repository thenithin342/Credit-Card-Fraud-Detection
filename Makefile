.PHONY: install up down logs test lint format dvc-pull clean help

PYTHON := .venv/Scripts/python
PIP    := .venv/Scripts/pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Create venv, install deps, install pre-commit hooks
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install --prefer-binary -r requirements.txt
	.venv/Scripts/pre-commit install

up: ## Start all Docker services (Redpanda, Redis, Postgres, MLflow)
	docker compose up -d

down: ## Stop all Docker services
	docker compose down

logs: ## Tail logs from all Docker services
	docker compose logs -f

test: ## Run unit tests (no real data required)
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-all: ## Run all tests including data_validation (needs downloaded data)
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

format: ## Auto-format with black, then fix with ruff
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

dvc-pull: ## Pull DVC-tracked data (requires configured remote)
	.venv/Scripts/dvc pull

dvc-repro: ## Re-run the full DVC pipeline (ingest → validate → split)
	.venv/Scripts/dvc repro

clean: ## Remove Python caches and build artefacts
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv/*" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
