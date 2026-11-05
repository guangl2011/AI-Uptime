# AI-Uptime

AI-Uptime assesses uptime risk in mission-critical facilities — **cold chain
first, then AIDC data centers, then hospitals** — by modeling a site as a
versioned asset/service dependency graph with provenance, and producing an
evidence-cited risk assessment with explicit abstention where data is
insufficient. Nothing in this system writes to facility control systems:
integrations are read-only, agents propose, humans execute.

> **Phase 5 — calibrated WP2 forecast landed:** the scanner flow (inventory →
> topology confirm → assessment report) works for **cold chain, AIDC, and
> hospital** synthetic facilities with no per-facility code changes; the eval
> harness gates every metric per segment and overall; hospital life-safety
> isolation is a reportable finding. WP2 is now a **calibrated forecast**
> behind the feature flag (default OFF, informational only): posterior-mean
> probabilities with uncertainty bounds and stated abstention on conflicting
> or sparse evidence, fitted on a frozen canonical table and scored on a
> facility-level holdout — AUROC ≥ 0.80 and ≥ 15% relative Brier improvement
> vs the flat-prior baseline are **merge-blocking gates**, reported
> synthetic-only and never executed on. Governance posture unchanged:
> read-only integrations, zero autonomous control, evidence-cited findings
> with explicit abstention.

## Quickstart

```bash
pip install -e '.[dev]'   # 1. install the backend + dev tools
make demo                 # 2. load the seed facility through the API,
                          #    verify provenance, build the frontend
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
seed/             seed facility (cold-chain example with provenance; synthetic corpus generator follows)
tests/            mirrors src/uptime_api/
docs/             testing plan and project docs
```

## API surface (Phase 1)

- `POST /facilities` · `GET /facilities` · `GET /facilities/{fid}`
- `POST /facilities/{fid}/graph/versions` — commit a full graph snapshot as a new immutable version
- `GET /facilities/{fid}/graph/versions` — version history
- `GET /facilities/{fid}/graph?version=N` — read a pinned version (default: latest), provenance intact
- `POST /facilities/{fid}/inventory` — CSV/JSON equipment-inventory import (first version only; ambiguous rows are flagged, never guessed)
- `GET /healthz`

## Testing

The authoritative testing plan lives in the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)**. The repo-local
[`docs/testing-plan.md`](docs/testing-plan.md) tracks what is actually
implemented (three-segment corpus, per-segment gates, fault-injection
scenarios, WP2 forecast pass) and is reconciled with that artifact as
phases land. Run the full evaluation with `make eval`.

## Development setup

```bash
# Backend — http://localhost:8000, docs at /docs
pip install -e '.[dev]'
uvicorn uptime_api.app:app --reload

# Frontend — http://localhost:5173, proxies /healthz and /facilities
# to :8000 with no CORS changes
npm --prefix frontend install
npm --prefix frontend run dev

# Hooks (ruff, hygiene, gitleaks secret scanning)
pre-commit install
```

## Governance posture (binding on every phase)

- Zero autonomous control — no write path to facility systems, ever.
- Every recommendation is human-reviewed and traceable to evidence.
- Integrations are read-only and least-privilege; no secrets are committed.
