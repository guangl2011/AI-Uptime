# AI-Uptime

AI-Uptime assesses uptime risk in mission-critical facilities — **cold chain
first, then AIDC data centers, then hospitals** — by modeling a site as a
versioned asset/service dependency graph with provenance, and producing an
evidence-cited risk assessment with explicit abstention where data is
insufficient. Nothing in this system writes to facility control systems:
integrations are read-only, agents propose, humans execute.

> Phase 0 scaffold — the codebase is a stub baseline for the phases that
> follow (graph model, assessor, report) per the MVP specification.

## Quickstart

```bash
pip install -e '.[dev]'   # 1. install the backend + dev tools
make demo                 # 2. load the seed facility, boot the backend,
                          #    check liveness, build the frontend
make test                 # 3. the test suite (same thresholds CI runs)
make check                # 4. ruff + mypy + pytest, as CI runs them
```

## Stack

- **Backend:** Python 3.11, FastAPI, SQLAlchemy (SQLite for MVP)
- **Frontend:** React + Vite (TypeScript)
- **CI:** GitHub Actions — ruff + mypy + pytest (backend), lint + build (frontend)
- **Hooks:** pre-commit (ruff, ruff-format, hygiene, gitleaks secret scanning) —
  enable with `pre-commit install`

## Layout

```
src/uptime_api/   backend layers (ingestion, graph, assessor, report land in later phases)
frontend/         React + Vite + TypeScript console
seed/             seed facility (Phase 0 stub; synthetic corpus generator follows)
tests/            mirrors src/uptime_api/
docs/             testing plan and project docs
```

## Testing

The authoritative testing plan lives in the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)**. A repo-local
[`docs/testing-plan.md`](docs/testing-plan.md) placeholder tracks what is
actually implemented; the two are reconciled as phases land.

## Governance posture (binding on every phase)

- Zero autonomous control — no write path to facility systems, ever.
- Every recommendation is human-reviewed and traceable to evidence.
- Integrations are read-only and least-privilege; no secrets are committed.
