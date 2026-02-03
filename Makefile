.PHONY: help install dev test lint format fix typecheck clean run

# Default target
help:
	@echo "Toolbridge - Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install    Install production dependencies"
	@echo "  make dev        Install development dependencies"
	@echo "  make run        Run the proxy locally"
	@echo ""
	@echo "Quality:"
	@echo "  make test       Run tests"
	@echo "  make coverage   Run tests with coverage report"
	@echo "  make lint       Run linter (ruff)"
	@echo "  make format     Format code (ruff)"
	@echo "  make fix        Format code with unsafe fixes"
	@echo "  make typecheck  Run type checker (mypy)"
	@echo "  make check      Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      Remove build artifacts"

# Install production dependencies
install:
	uv pip install -e .

# Install development dependencies
dev:
	uv pip install -e ".[dev]"

# Run the proxy locally
run:
	python3 toolbridge.py

# Run tests
test:
	python3 -m pytest tests/ -v

# Run tests with coverage
coverage:
	python3 -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Run linter
lint:
	ruff check toolbridge.py tests/

# Format code
format:
	ruff check --fix toolbridge.py tests/
	ruff format toolbridge.py tests/

# Format code with unsafe fixes
fix:
	ruff check --fix --unsafe-fixes toolbridge.py tests/
	ruff format toolbridge.py tests/

# Run type checker
typecheck:
	mypy toolbridge.py

# Run all checks
check: lint typecheck test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
