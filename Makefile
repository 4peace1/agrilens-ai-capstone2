.PHONY: install install-dev lint format test test-cov up down build clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install ruff

lint:
	ruff check .

format:
	ruff check . --fix

test:
	pytest -v

test-cov:
	pytest --cov=app --cov=worker --cov=inference --cov-report=term-missing

up:
	docker compose up --build

down:
	docker compose down -v

build:
	docker build -f Dockerfile -t agrilens-gateway:local .
	docker build -f worker/Dockerfile -t agrilens-worker:local .
	docker build -f inference/model_server/Dockerfile -t agrilens-inference:local .
	docker build -f notification/Dockerfile -t agrilens-notification:local .

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
