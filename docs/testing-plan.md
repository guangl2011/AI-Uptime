# AI-Uptime Testing Plan (placeholder)

The authoritative testing plan is the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)** (synthetic evaluation and CI
strategy, AUROC / Brier / calibration / leakage-split policy). This file
tracks what is *implemented in the repo* and will be reconciled with that
artifact as phases land.

## Implemented (Phase 0)

| Check | Where | Runs in CI |
|---|---|---|
| Liveness probe smoke test | `tests/test_health.py` | yes — pytest |
| Demo runbook end-to-end (seed loads, backend boots, healthz OK) | `tests/test_demo_smoke.py`, `make demo` | pytest covers the demo module; `make demo` is local |
| Seed fixture sanity + cold-chain-first market sequence guard | `tests/test_demo_smoke.py` | yes — pytest |
| Static checks | `ruff check src tests seed`, `mypy src` (strict) | yes — backend job |
| Frontend lint + typecheck build | `npm run lint`, `npm run build` | yes — frontend job |
| Secret scanning | gitleaks pre-commit hook | pre-commit |

## Not yet implemented (later phases, per art_XaCiEWwo and the agent-team plan)

- Synthetic corpus with adjudicated labels; `make eval` harness
- Topology-recovery accuracy, criticality-ranking quality, AUROC ≥ 0.80 gate
- Calibration / Brier reporting, confidence intervals
- No-leakage time-split evaluation
- Reproducibility check: same assessment on a pinned graph version is
  byte-for-byte identical (lands with the versioned graph in Phase 1+)
