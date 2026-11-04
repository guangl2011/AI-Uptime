.PHONY: demo seed test check frontend-install

# Demo runbook: regenerate the seed facility, boot the backend and hit its
# liveness probe, then build the frontend. Phase 0 stub — later phases add
# topology confirmation and the assessment report to this same target.
demo: frontend-install
	python seed/generate_seed.py
	python -m uptime_api.demo
	npm --prefix frontend run build

seed:
	python seed/generate_seed.py

frontend-install:
	npm --prefix frontend ci --no-audit --no-fund --silent

test:
	python -m pytest -q

check:
	ruff check src tests seed
	mypy src
	python -m pytest -q
