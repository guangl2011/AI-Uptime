.PHONY: demo seed eval sim invariants test check install frontend-install clean

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

# Full evaluation harness (evals/): seeded synthetic corpus -> assessor ->
# metric gates. Writes eval-report.md/.json under eval-results/ (gitignored);
# non-zero exit when any gate fails. Mirrors the sibling repo's `make eval`.
eval: $(VENV)/bin/python
	$(PYTHON) -m evals --output-dir eval-results

# Simulation suite (evals/sim/): seeded discrete-event scenarios + corpus
# fault sweeps -> adjudicated-trace gates with a determinism digest. Writes
# sim-report.{md,json} under eval-results/sim/ (gitignored); non-zero exit
# when any gate fails. Runs alongside `make eval`, not instead of it —
# eval scores single-snapshot answers, sim scores traces and invariants.
sim: $(VENV)/bin/python
	$(PYTHON) -m evals.sim --output-dir eval-results/sim

# Invariant sweep (evals/sim/invariant_sweep): the plan's 1,000-seed gate for
# the Category 3-4 properties (monotonicity, repair recovery, determinism,
# replay folds, non-clairvoyance, recovery state). Writes
# invariant-sweep.{md,json} under eval-results/invariants/ (gitignored);
# non-zero exit with the failing seed + repro when any gate fails. Nightly
# cadence — push CI runs the Hypothesis property seeds via `make test`.
invariants: $(VENV)/bin/python
	$(PYTHON) -m evals.sim.invariant_sweep --output-dir eval-results/invariants

frontend-install:
	npm --prefix frontend ci --no-audit --no-fund --silent

test: $(VENV)/bin/python
	$(PYTHON) -m pytest -q

check: $(VENV)/bin/python
	$(RUFF) check src tests seed evals
	$(MYPY) src evals
	$(PYTHON) -m pytest -q
	$(PYTHON) -m evals --output-dir eval-results

clean:
	rm -rf $(VENV)
