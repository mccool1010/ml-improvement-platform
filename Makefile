# Convenience targets. Every command also appears in the README, because `make`
# is not guaranteed to be installed on Windows development machines.

.PHONY: help install data validate baseline candidate test lint format typecheck check clean

help:
	@echo "install    create the venv and install dependencies"
	@echo "data       download and checksum-verify the register"
	@echo "validate   run schema validation and report split composition"
	@echo "baseline   train the baseline model"
	@echo "candidate  train the candidate model"
	@echo "test       run the test suite"
	@echo "check      lint, format check, typecheck and test"

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

data:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

baseline:
	python scripts/train.py --model baseline

candidate:
	python scripts/train.py --model candidate

test:
	pytest

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy

check: lint typecheck test
	ruff format --check src tests scripts

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
