# Makefile for StatsTest From Scratch
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev install-poetry install-uv test test-cov lint format type-check clean clean-build clean-pyc clean-test pre-commit-install pre-commit-run all

# Default target
.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)StatsTest From Scratch - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@grep -E '^install.*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@grep -E '^(test|lint|format|type-check|pre-commit).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@grep -E '^(clean|lock).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation Targets
# ============================================================================

install: ## Install package with pip (production dependencies only)
	pip install -e .

install-dev: ## Install package with pip including dev dependencies
	pip install -e ".[dev]"

install-poetry: ## Install package using Poetry
	poetry install --with dev

install-uv: ## Install package using uv
	uv pip install -e ".[dev]"

# ============================================================================
# Testing Targets
# ============================================================================

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=StatsTest --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

test-verbose: ## Run tests in verbose mode
	pytest -v

test-failed: ## Re-run only failed tests
	pytest --lf

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw

# ============================================================================
# Code Quality Targets
# ============================================================================

lint: ## Run linting checks with ruff
	ruff check .

lint-fix: ## Run linting and auto-fix issues
	ruff check --fix .

format: ## Format code with ruff
	ruff format .

format-check: ## Check code formatting without modifying files
	ruff format --check .

type-check: ## Run type checking with mypy
	mypy StatsTest --ignore-missing-imports

# ============================================================================
# Pre-commit Targets
# ============================================================================

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	pre-commit autoupdate

# ============================================================================
# Combined Targets
# ============================================================================

check: format-check lint type-check test ## Run all code quality checks and tests

ci: format-check lint test ## Run CI pipeline (format, lint, test)

all: clean install-dev pre-commit-install test-cov ## Full setup: clean, install, configure, test

# ============================================================================
# Maintenance Targets
# ============================================================================

clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr .ruff_cache

lock-poetry: ## Update poetry.lock file
	poetry lock

lock-check: ## Verify poetry.lock is up to date
	poetry check

# ============================================================================
# Build Targets
# ============================================================================

build: clean ## Build source and wheel distributions
	python -m build

build-poetry: clean ## Build using Poetry
	poetry build

# ============================================================================
# Documentation Targets
# ============================================================================

docs-serve: ## Serve documentation locally (if docs exist)
	@echo "$(YELLOW)Documentation serving not yet implemented$(NC)"

# ============================================================================
# Utility Targets
# ============================================================================

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version | cut -d' ' -f1-2)"
	@command -v poetry >/dev/null 2>&1 && echo "Poetry: $$(poetry --version)" || echo "Poetry: not installed"
	@command -v uv >/dev/null 2>&1 && echo "uv: $$(uv --version)" || echo "uv: not installed"
	@echo ""
	@echo "$(BLUE)Installed Packages$(NC)"
	@pip list | grep -E "(numpy|scipy|pandas|statsmodels|pytest|ruff|mypy)"

deps-update: ## Show outdated dependencies
	pip list --outdated

# ============================================================================
# Quick Development Workflow
# ============================================================================

dev: install-dev pre-commit-install ## Quick setup for development
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Run 'make test' to run tests"
	@echo "  - Run 'make lint' to check code quality"
	@echo "  - Run 'make format' to format code"
