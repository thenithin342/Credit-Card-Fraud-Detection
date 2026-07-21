.PHONY: up down logs test lint format install

install:
	pip install -r requirements.txt
	pre-commit install

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check .
	mypy src

format:
	black .
	ruff check --fix .
