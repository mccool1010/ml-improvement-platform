# Convenience targets. Every command also appears in the README, because `make`
# is not guaranteed to be installed on Windows development machines.

.PHONY: help install data validate baseline candidate reproduce test lint format typecheck check verify clean

help:
	@echo "install    create the venv and install dependencies"
	@echo "data       download and checksum-verify the register"
	@echo "validate   run schema validation and report split composition"
	@echo "baseline   train the baseline model"
	@echo "candidate  train the candidate model"
	@echo "reproduce  retrain both and verify against the locked M1 reference"
	@echo "test       run the test suite"
	@echo "check      lint, format check, typecheck and test"

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

data:
	python -m ml_platform download

validate:
	python -m ml_platform validate

baseline:
	python -m ml_platform train --model baseline

candidate:
	python -m ml_platform train --model candidate

reproduce:
	python -m ml_platform reproduce --profile strict

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

# Full gate: static checks, tests, and a verified reproduction of the M1 metrics.
verify: check reproduce

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
