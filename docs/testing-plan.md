# AI-Uptime Testing Plan

The authoritative testing plan is the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)** (synthetic evaluation and CI
strategy, AUROC / Brier / calibration / leakage-split policy). This file
tracks what is *implemented in the repo* and is reconciled with that
artifact as phases land.

## Current state (Phase 3 — eval harness hardening)

### Eval harness (`evals/`, run via `make eval`)

The harness scores the deterministic synthetic cold-chain corpus
(40 facilities by default, ~573 labeled single-fault cases, seeded and
reproducible — `CORPUS_SEED` in `evals/corpus.py`). It drives the real API
(InventoryStep → graph commit → assessment), then reports and gates:

| Metric | Definition | Gate |
|---|---|---|
| Topology recovery | Inventory-imported nodes vs generator intent (missing / extra / attribute mismatches) | exact recovery |
| Exposure correctness | Predicted exposed-service sets vs by-construction labels | 100% exact |
| SPOF detection | Set precision / recall / false-positive rate vs constructed SPOFs | recall 1.00, FPR ≤ 0.02 |
| Ranking agreement | Mean Spearman τ between adjudicated impact size and assessor scores | τ ≥ 0.60 |
| AUROC | Fault-interruption classification over labeled single-fault cases (Mann–Whitney, ties half-credit), 95% paired bootstrap CI | ≥ 0.80 (DOE go/no-go) |
| Brier score | Mean (p − y)² vs flat-prior baseline, 95% paired bootstrap CI on the improvement | Brier ≤ 0.10 |
| Calibration reporting | Reliability table (10 bins) + ECE — **reported, not gated**: calibrated probabilities are a WP2 deliverable; the MVP asserts topological exposure, not calibrated risk | table covers all cases |
| No-leakage | Feature windows vs label windows for every replay fold | 0 violations |
| Fault-injection scenarios | 4 named, reviewable scenario files run as known answers | 4/4 pass |

Sparse-label honesty: the report states explicitly that every number is
synthetic (labels are adjudicated from topology construction, not observed
outages) and makes no claim about real facilities. Gates can be tuned via
`python -m evals --facilities N --output-dir DIR`.

### No-leakage discipline (enforced in code — `evals/splits.py`)

Each replay fold fixes a feature window `[fault − 7d, fault)` and a label
window `[fault, repair]`. `feature_events` raises `LeakageError` on any
event at or after the repair inside a feature window; `check_no_leakage`
aggregates per-fold checks and reports counts (features used,
post-repair events excluded, violations). The harness fails its gate on
any violation — the discipline is code, not convention.

### Fault-injection scenarios (`evals/scenarios/*.json`)

Four named, reviewable facilities with hand-adjudicated expected answers:
single-compressor failure under N+1 (absorbed) and under N (interruption),
utility-feed loss under 2N (absorbed), and a rack failure cascading to
unserved cold rooms. Each is committed through the real API and scored
against its expectations; the tests also verify the runner fails when an
expectation is tampered with or a redundancy declaration is degraded.

### Unit / API tests (mirror the module layout)

| Check | Where |
|---|---|
| Metrics math (AUROC ties, Brier, baseline, reliability bins, ECE, Spearman, bootstrap determinism) | `tests/evals/test_metrics.py` |
| Corpus determinism, label-vs-construction integrity, CSV round-trip through the real parser | `tests/evals/test_corpus.py` |
| Fold geometry, leakage guards, cross-seed leak-free replay folds | `tests/evals/test_splits.py` |
| Scenario file integrity + known-answer runs + tamper detection | `tests/evals/test_scenarios.py` |
| Inventory parsing / graph provenance / scoring / SPOF / API surface | `tests/ingestion/`, `tests/graph/`, `tests/assessor/`, `tests/api/` |
| Read-only posture (no control-write surface) | `tests/test_readonly_posture.py` |

## CI

- **On every PR and push to `main`** (`.github/workflows/ci.yml`): ruff +
  mypy (strict) + pytest + eval gates over the full corpus; the eval report
  uploads as a workflow artifact even on gate failure. Frontend job: lint +
  build.
- **Nightly** (`.github/workflows/eval-nightly.yml`): full eval on schedule
  (03:17 UTC) + manual dispatch; report uploaded as a per-run artifact —
  never committed to the repo (`eval-results/` is gitignored).

## Reproducibility

Assessments pin a graph version; the corpus is seeded and deterministic
(same seed → identical graphs, CSVs, labels, and event logs, verified by
test). Re-running `make eval` reproduces the same metric values.

## Not yet implemented (later phases, per art_XaCiEWwo)

- Historical replay on pilot data (WP4): the point where AUROC / Brier /
  lead-time numbers start to mean anything about a real facility.
- Calibration *thresholds* and probability-bearing outputs (WP2): gated on
  labeled data; the MVP reports exposure and data gaps instead.
- Ranked-prioritization improvement (≥ 20%) and actionable lead time:
  metrics machinery (rank agreement) is live; the targets stay claimable
  only after replay data exists.

## Known gaps, named honestly

All current metrics are synthetic-only — they prove the machinery, not
field performance. Until the first pilot's history exists, AUROC, Brier,
and lead-time targets can only be exercised on constructed ground truth.
The market-pain figures cited in the DOE proposal (e.g. Uptime Institute
outage costs) are not re-verified here and must be re-checked from current
public sources before any customer-facing claim.
