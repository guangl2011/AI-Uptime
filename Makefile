.PHONY: demo seed test check install frontend-install clean

# The package lives under src/ and is installed editable into .venv; make
# bootstrap the venv from pyproject.toml on first use so a fresh clone runs
# `make demo` / `make check` / `make test` with no manual setup.
VENV := .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

$(VENV)/bin/python: pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip
	$(VENV)/bin/pip install --quiet -e '.[dev]'

install:
	$(MAKE) $(VENV)/bin/python

# Demo runbook: regenerate the seed facility, boot the backend and hit its
# liveness probe, then build the frontend. Phase 0 stub — later phases add
# topology confirmation and the assessment report to this same target.
demo: frontend-install $(VENV)/bin/python
	$(PYTHON) seed/generate_seed.py
	$(PYTHON) -m uptime_api.demo
	npm --prefix frontend run build

seed: $(VENV)/bin/python
	$(PYTHON) seed/generate_seed.py

frontend-install:
	npm --prefix frontend ci --no-audit --no-fund --silent

test: $(VENV)/bin/python
	$(PYTHON) -m pytest -q

check: $(VENV)/bin/python
	$(RUFF) check src tests seed
	$(MYPY) src
	$(PYTHON) -m pytest -q

clean:
	rm -rf $(VENV)
