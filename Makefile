UV ?= uv
PYTHON ?= 3.11

.PHONY: format check tests coverage build prod pre-commit

format:
	$(UV) run --python $(PYTHON) ruff format src tests
	$(UV) run --python $(PYTHON) ruff check --fix src tests

check:
	$(UV) run --python $(PYTHON) ruff check src tests
	$(UV) run --python $(PYTHON) ruff format --check src tests
	$(UV) run --python $(PYTHON) ty check src tests
	$(UV) run --python $(PYTHON) basedpyright src tests

tests:
	$(UV) run --python $(PYTHON) pytest -q

coverage:
	$(UV) run --python $(PYTHON) pytest --cov=src/kedi_autobench --cov-branch --cov-report=term-missing -q

build:
	$(UV) build

prod: check coverage build

pre-commit:
	$(UV) run --python $(PYTHON) pre-commit run --all-files
