# AI-Uptime Testing Plan

The authoritative testing plan is the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)** (synthetic evaluation and CI
strategy, AUROC / Brier / calibration / leakage-split policy). This file
tracks what is *implemented in the repo* and is reconciled with that
artifact as phases land.

## Current state (Phase 4 — AIDC + hospital segments, WP2 groundwork)

### Eval harness (`evals/`, run via `make eval`)

The harness scores the deterministic synthetic corpus across the market
sequence — cold chain → AIDC data centers → hospitals — 40 cold-chain,
20 AIDC, and 20 hospital facilities by default (~1,250 labeled single-fault
cases), seeded and reproducible (`CORPUS_SEED` in `evals/corpus/`; each
generates in an independent RNG namespace so resizing one segment never
shifts another's). It drives the real API (InventoryStep → graph commit →
assessment), then reports and gates the same metric set **per segment and
overall** — every segment gate uses the same threshold as the overall gate,
so a regression in one market is visible and blocking, not averaged away.

| Metric | Definition | Gate |
|---|---|---|
| Topology recovery | Inventory-imported nodes vs generator intent (missing / extra / attribute mismatches) | exact recovery |
| Exposure correctness | Predicted exposed-service sets vs by-construction labels | 100% exact |
| SPOF detection | Set precision / recall / false-positive rate vs constructed SPOFs | recall 1.00, FPR ≤ 0.02 |
| Ranking agreement | Mean Spearman τ between the adjudicated priority ordering (exposure-first, then total reach: `1000×exposed + impacted`, construction-derived — never the assessor's own score) and asset scores | τ ≥ 0.60 |
| AUROC | Fault-interruption classification over labeled single-fault cases (Mann–Whitney, ties half-credit), 95% paired bootstrap CI | ≥ 0.80 (DOE go/no-go) |
| Brier score | Mean (p − y)² vs flat-prior baseline, 95% paired bootstrap CI on the improvement | Brier ≤ 0.10 |
| Calibration reporting | Reliability table (10 bins) + ECE — **reported, not gated**: calibrated probabilities are a WP2 deliverable; the MVP asserts topological exposure, not calibrated risk | table covers all cases |
| No-leakage | Feature windows vs label windows for every replay fold | 0 violations |
| Fault-injection scenarios | 6 named, reviewable scenario files run as known answers — 4 cold-chain (Phase 3) plus an AIDC lone-chiller cooling-train case and a hospital life-safety isolation case (which must raise the isolation finding) | 6/6 pass |

### Scoring model (changed in Phase 4 — calibration discipline)

Phase 3 scored assets `40 × exposed + 10 × tolerated` services. On the
deeper AIDC/hospital chains that tolerated-service credit assigned real
scores to faults the redundancy fully absorbs (a 2N source pair's utility
feed scored ~90 while interrupting nothing), which the Brier gate correctly
catches as miscalibration. The model is now **`60 × exposed` services,
capped at 100; a fully absorbed fault scores 0** — the deterministic
semantics say it interrupts no service, so a nonzero score would assert
risk that is not there. Absorbed impact is still reported in
`tolerated_services` and the driver lines. The ranking ground truth moved
with it: priority is exposure-first (the product surfaces unprotected
damage first) with total reach as the tiebreak.

### WP2 forecasting pass (feature-flagged; reported, never gated)

The `uptime_api.forecasting` package holds the WP2 groundwork: a
`ForecastModel` interface returning `RiskForecast` estimates with
uncertainty bounds, a flat-prior baseline estimator, and a flag resolver
(default OFF). The flag exists only so the module can be evaluated: the
harness installs the flat-prior model for an extra assessment pass and
scores its `p_fault` against the same by-construction labels — AUROC,
Brier, and per-segment tables are **reported, not gated** (calibrated
probabilities stay a WP2 deliverable per the locked testing plan).
Forecasts are informational estimates; nothing in the codebase acts on
them, and the read-only posture test (`tests/test_readonly_posture.py`)
runs unchanged.

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

Six named, reviewable facilities with hand-adjudicated expected answers:
single-compressor failure under N+1 (absorbed) and under N (interruption),
utility-feed loss under 2N (absorbed), a rack failure cascading to unserved
cold rooms, an AIDC lone-chiller loss whose cascade the N+1 pump pair
absorbs at the CDU, and a hospital med-gas feed wired from the general
distribution panel — a red-line life-safety isolation violation the
assessor must report as a finding. Each is committed through the real API
and scored against its expectations (exposed services, per-service fault
tolerance, and — optionally per scenario — required findings); the tests
also verify the runner fails when an expectation is tampered with, a
redundancy declaration is degraded, or an isolation violation goes
unreported.

### Unit / API tests (mirror the module layout)

| Check | Where |
|---|---|
| Metrics math (AUROC ties, Brier, baseline, reliability bins, ECE, Spearman, bootstrap determinism) | `tests/evals/test_metrics.py` |
| Corpus determinism, label-vs-construction integrity, CSV round-trip through the real parser (all three segments) | `tests/evals/test_corpus.py` |
| Fold geometry, leakage guards, cross-seed leak-free replay folds | `tests/evals/test_splits.py` |
| Scenario file integrity + known-answer runs + tamper detection | `tests/evals/test_scenarios.py` |
| Inventory parsing / graph provenance / scoring / SPOF / API surface | `tests/ingestion/`, `tests/graph/`, `tests/assessor/`, `tests/api/` |
| Read-only posture (no control-write surface) | `tests/test_readonly_posture.py` |

Frontend component tests (Vitest + RTL) are still a named gap: the console
is type-checked (tsc), linted, and built in CI, and the harness exercises
the API contracts it consumes, but findings-citation rendering is not yet
asserted by a component test.

## CI

- **On every PR and push to `main`** (`.github/workflows/ci.yml`): ruff +
  mypy (strict) + pytest + eval gates over the full corpus; the eval report
  uploads as a workflow artifact even on gate failure. Simulation gates run
  in the same job (see below). Frontend job: lint + build.
- **Nightly** (`.github/workflows/eval-nightly.yml`): full eval + simulation
  suite on schedule (03:17 UTC) + manual dispatch; reports uploaded as
  per-run artifacts — never committed to the repo (`eval-results/` is
  gitignored).

## Reproducibility

Assessments pin a graph version; the corpus is seeded and deterministic
(same seed → identical graphs, CSVs, labels, and event logs, verified by
test). Re-running `make eval` reproduces the same metric values.

## Simulation suite (Sim M1)

`make sim` (`python -m evals.sim`, code under `evals/sim/`) runs a seeded,
deterministic, discrete-event simulator over graph snapshots from the same
corpus — it drives the real assessment API and writes nothing outside its
own throwaway databases and report artifacts. It scores *traces*, not
single snapshots: ordered fault / repair / redundancy-change / load-shift
events, with the full assessment state recorded at every step. What-if
markers are M3 scope; the event union already names them so they can be
added without a format change.

Gates (reported per segment, same convention as the eval report):

- **Adjudicated sequenced scenarios** — one hand-adjudicated multi-step
  trace per segment (`evals/sim/scenarios/`, JSON DSL extending the static
  scenario style with event ordering): cold-chain compressor failover and
  redundancy upgrade, AIDC chiller cascade with pump-pair degradation, and
  hospital life-safety ride-through with a med-gas manifold upgrade. Lost
  services, tolerance flags, and single-point-of-failure sets must match
  the adjudicated trace exactly — 100% agreement, no tolerance; a mismatch
  is treated as a bug (in the scenario or the engine), not a difference to
  wave off.
- **Corpus fault sweeps** — fault every asset of a reduced corpus
  (reproducible from the corpus seed) and check the observed cascade loss
  against the construction-derived labels: exact agreement with the
  what-if exposure where the labels agree; otherwise bounded between the
  what-if exposure and the full-impact label. The two labels disagree
  exactly when a cascade eats a redundant group's reserve — the stateful
  behavior the static eval cannot see.
- **Determinism** — two isolated passes (fresh app instances, same seed,
  same corpus) must produce an identical trace digest; any drift fails the
  gate.

CI: the sim gates run in the `backend` job on every PR and push to `main`
(a superset of the plan's assessor/graph/evals trigger set — the suite is
cheap, and running it unconditionally keeps it merge-blocking under the
existing required-check contexts), and in the nightly workflow. A gate
failure uploads the failing seed, trace step, and diff as the workflow
artifact — a repro case ships with the failure.

Honest semantics note: the simulator's binary states are stateful, not
timed — redundancy group absorption follows the degraded state (a dead
group member stops protecting its peers), but transfer durations and
thermal ride-through are not modeled; that stays deferred to the
physics-modeling trajectory (WP4). SPOF sets in traces are therefore
state-conditional ("the surviving pump is a SPOF while its partner is
down"), which is the intended behavior, not drift.

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
