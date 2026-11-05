# AI-Uptime Testing Plan

The authoritative testing plan is the project artifact
**AI Uptime — Testing Plan (art_XaCiEWwo)** (synthetic evaluation and CI
strategy, AUROC / Brier / calibration / leakage-split policy). This file
tracks what is *implemented in the repo* and is reconciled with that
artifact as phases land.

## Current state (Phase 5 — calibrated WP2 forecast, gated on the holdout)

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
| Calibration reporting | Reliability table (10 bins) + ECE — **reported, not gated**: the assessor's exposure score is not a probability; the forecast's own calibration is gated separately below | table covers all cases |
| Forecast AUROC (WP2) | Held-out fault-interruption classification by the calibrated forecaster, 95% paired bootstrap CI | ≥ 0.80 (DOE go/no-go) |
| Forecast Brier improvement (WP2) | Relative Brier improvement of the calibrated forecaster vs the flat-prior baseline on the same holdout, 95% paired bootstrap CI on the improvement | ≥ 15% relative and CI > 0 (DOE go/no-go) |
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

### WP2 calibrated forecast (Phase 5 — feature-flagged, gated on the holdout)

The placeholder flat-prior estimator is replaced by a calibrated
forecaster (`uptime_api.forecasting`): a `ForecastModel` interface
returning `RiskForecast` estimates with uncertainty bounds, plus a
deterministic flag resolver (default OFF).

**Model.** Features are per-asset protection statuses derived
deterministically from the graph snapshot (`features.py`): whether every
supply edge the asset feeds is covered by an explicit redundant group
(N+1 / 2N with ≥ 2 members), whether it also has unprotected edges, and
whether the asset feeds nothing at all. Declared redundancy that its
member groups cannot support (an N+1/2N edge whose group has one member,
or names no group) is a **conflict**: the snapshot contradicts itself, so
the forecaster abstains with a stated reason and falls back to the
flat-prior value — never a confident guess from contradictory data.

**Calibration.** A per-status Beta-Binomial table maps protection status
→ P(service-interruption exposure within a 90-day horizon): posterior
mean as the point estimate, ±2 posterior sigma as the uncertainty bounds
(clipped into [0.01, 0.99] and always containing the point estimate). The
canonical table is frozen in `forecasting/canonical.py` — fitted once
from the canonical corpus through the real API-loading path on a
facility-level 75/25 split (60 train / 20 eval facilities; seed
`CORPUS_SEED`), never re-fit implicitly. A drift-pin test refits from the
corpus and fails if the frozen constants and the documented counts (507 /
341 / 106 training cases) no longer match, so corpus or feature changes
force a deliberate regeneration. Cells with < 30 training cases abstain
with a stated reason (visible when a hand-shrunken corpus is evaluated);
abstained values carry the flat-prior fallback and full-width bounds, so
abstention can never score better than the baseline it defers to.

**Evaluation — the WP2 go/no-go gates, now enforced.** The forecast pass
(`evals/harness.py`) fits on the training facilities only, then scores
held-out facilities through the real assessment surface. Gates (overall
and per scored segment; a missing forecast pass fails the overall gate
rather than skipping):

- **Forecast AUROC ≥ 0.80** (DOE go/no-go) on the held-out cases, with a
  95% paired bootstrap CI.
- **Forecast Brier ≥ 15% better than the flat-prior baseline** (DOE
  go/no-go): relative improvement `(baseline − model) / baseline` with a
  95% paired bootstrap CI whose lower bound must be > 0.
- **Calibration reporting** — reliability table + ECE over the holdout,
  gated on table coverage (every scored case lands in the table), with
  the ECE value reported, not thresholded.

Canonical-corpus results (verified 2026-11-05, explicitly synthetic-only):
321 held-out cases from 20 facilities, forecast AUROC 1.000, Brier 0.0000
vs baseline 0.2487 (+100% relative, CI > 0), ECE 0.003, no abstentions.
See "Known gaps" for why these perfect numbers are degenerate rather
than impressive.

**Posture (unchanged and binding).** The flag stays default-off; when
enabled the forecasts are informational annotations on the report —
uncertainty bounds on every point estimate, abstention with a stated
reason on conflicts or sparse calibration data, and no action or control
path anywhere (the read-only posture test runs unchanged). The console
renders abstained rows as "abstained — <reason>", never as a bare
fallback number.

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
| Metrics math (AUROC ties, Brier, baseline, relative Brier improvement, reliability bins, ECE, Spearman, bootstrap determinism) | `tests/evals/test_metrics.py` |
| Corpus determinism, label-vs-construction integrity, CSV round-trip through the real parser (all three segments) | `tests/evals/test_corpus.py` |
| Fold geometry, leakage guards, cross-seed leak-free replay folds | `tests/evals/test_splits.py` |
| Scenario file integrity + known-answer runs + tamper detection | `tests/evals/test_scenarios.py` |
| Forecast pass on the facility-level holdout: gates, abstention accounting, report sections (JSON + Markdown) | `tests/evals/test_harness_forecast.py` |
| Protection features, Beta-Binomial calibration, facility-level split, abstention semantics, flag resolution | `tests/forecasting/` |
| Canonical calibration table drift pin (frozen constants == fresh refit) | `tests/forecasting/test_canonical.py` |
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
  suite + 1,000-seed invariant sweep + forecast-under-dynamics seed sweep
  on schedule (03:17 UTC) + manual dispatch; reports uploaded as
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
markers arrived in M3 (see "What-if mode" below); the event union
already named them, so they were added without a format change.

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

## Simulation invariants (Sim M2)

`make invariants` (`python -m evals.sim.invariant_sweep`, code under
`evals/sim/invariants.py`) runs the plan's Category 3-4 invariants as
mathematical properties over generated seeds — one facility per segment
per seed, no expected outputs to wave off: a violation is a bug in the
engine or the corpus, and the failure output carries the failing seed plus
a minimal repro (regeneration recipe and the events to replay).

Gates (zero tolerance, reported per invariant):

- **Redundancy monotonicity** — adding redundancy (a standby member via
  the N → N+1 → 2N ladder) never increases exposure: for the intact state
  and every single-asset fault, the upgraded dead set is a subset of the
  original's, and the standby itself never dies.
- **Repair recovery** — a repair never re-exposes a dead service, and
  repairing every event fault restores the intact graph exactly (no stale
  exposure, no torn edges; upgrades persist, faults must not).
- **Trace determinism** — the same seed regenerates the facility
  identically and replays to an identical materialized trace.
- **Replay folds** — the existing `evals.splits` discipline (LeakageError)
  holds on the corpus replay cases: no post-fault event in the feature
  window; the fault's work order and its repair inside the label window.
- **Non-clairvoyance** — pre-repair trace steps are identical with and
  without the future repair in the replay input.
- **Recovery state** — the constructed interruption is visible in the
  pre-repair state, and the post-repair state equals the intact baseline —
  at the engine level, and through the real assessment API (the API
  comparison keeps evidence claims and drops per-version provenance
  pointers, which are re-issued by every commit by design).

CI: the invariants run as Hypothesis property tests
(`tests/evals/sim/test_invariants.py`) inside the existing pytest run on
every PR and push to `main` — push CI runs the property seeds only, so a
push stays under a few minutes. The full 1,000-seed sweep runs in the
nightly workflow (`make invariants`), with `invariant-sweep.{md,json}`
uploaded as the per-run artifact, same convention as eval and sim. The
nightly run states the plan's seed-count gate in its own report.

Honest semantics note: the invariants check the engine's own semantics —
the corpus, the engine, and the replay cases share authors, so these are
consistency properties, not independent field validation (the shared
blind-spot caveat of art_non0bSFL carries). Generated corpora may contain
parallel duplicate edges; they are benign for cascade closure and are
excluded from load-shift candidates, which the engine defines on
single-edge dependencies.

## What-if mode (Sim M3)

Category 5's counterfactual surface. The sequenced-scenario DSL gains
`what_if` markers — `remove_asset` (a component is deleted from the
graph) or `upgrade_redundancy` (a component's redundancy class moves up
the N → N+1 → 2N ladder, introducing a standby when needed). A marker is
not a timeline event: it occupies a trace step that re-assesses the
*unchanged* state, while the counterfactual branch materializes the
intervention separately and re-assesses through the same real API
surface. Nothing in the branch writes anywhere — the read-only posture
holds for counterfactuals exactly as for traces.

Gates (both merge-blocking inside `make sim`):

- **`whatif_deltas`** — the observed delta between the baseline report
  and the branch report (lost/restored services, SPOFs added/removed,
  first-fault-tolerance changes) must equal the adjudicated
  `expected_delta` exactly, on every shipped what-if case per segment.
  A field the adjudication does not declare fails the gate; so does a
  declared field the assessor does not produce.
- **`whatif_spof_path`** — for upgrade markers, any single point of
  failure the baseline found on the upgraded path must be *removed* by
  the upgrade (the structural claim an N → N+1 upgrade makes), or the
  runner must abstain with a stated reason. Abstention is the honest
  answer when the verdict is structurally undecidable on that state —
  for example, when the primary is already faulted and fewer than two
  live members remain. A surviving SPOF without abstention is the
  silent wrong answer this gate exists to catch.

Marker purity and counterfactual monotonicity are checked as a matter
of course: the marker step's report must equal the previous step's byte
for byte (minus volatile fields), and the `whatif_counterfactual`
invariant (Sim M2 sweep) proves upgrades never increase exposure and
removals cascade exactly like the equivalent fault. Constructed what-if
cases run inside the existing pytest run; the sim report gains a
Category 5 section (deltas and SPOF-path verdicts per marker) in the
same nightly artifact — no new workflow, no new nightly budget.

Honest semantics note: a what-if delta validates the graph semantics —
that the assessor's answer moves exactly as the dependency graph says
it should. It is not a validated predictor of facility outcomes; that
claim waits for pilot replay (WP4). The abstention path exists because
state-conditional redundancy can make a path's verdict genuinely
undecidable at the binary-state layer (transfer timing and
ride-through stay out of scope, per the M1 note above).

## Forecast under dynamics (Sim M4)

The plan's final milestone: the Phase 5 calibrated forecast is evaluated
across simulated dynamics. The sequenced-scenario DSL's fault / repair /
redundancy-change / load-shift events materialize graph states exactly as
in M1, and at every state the forecast runs through the same real API path
(the canonical frozen model is installed on the throwaway app for the pass
duration and restored after — the product default stays OFF and
informational, with no action path). What is under test is the forecast's
*behavior across state changes* — signs, restoration, abstention identity,
bound behavior, determinism — not its accuracy and not the event timing.

Category 6 gates (per segment, merge-blocking inside `make sim`):

- **`forecast_degradation_sign`** — a fault that removes protection must
  never *decrease* the forecast of a surviving stable feed relative to its
  pre-fault value (expected degradation moves with the right sign), and a
  fault that silently clears an abstention (a redundancy loss the model
  never registered) fails the gate. Scope is deliberately narrow: feeds
  whose own supply edges are unchanged between the two states, so a
  path-loss confound (the asset's forecast moving because its *own* feed
  died) is not misread as a sign violation.
- **`forecast_repair_restores`** — after every fault is repaired, the
  report's forecast rows must equal a fresh assessment of the intact
  engine state of the same (post-upgrade) topology, byte for byte. The
  only sanctioned difference is the canonical-table drift pin (a refit
  changes shipped constants deliberately and announces itself).
- **`forecast_abstention_identity`** — the report's forecast rows at
  *every* materialized state — baseline, every event step, and the intact
  reference — must equal the static path's forecast on the same snapshot
  exactly: same probabilities, same bounds, same abstention status and
  stated reason. Replay may not change the answer an on-demand assessment
  would give.
- **`forecast_bounds_monotone`** — across added fault events on unchanged
  supply paths, uncertainty intervals may widen or hold, never silently
  shrink: information loss cannot tighten a bound. Upgrade and repair
  steps are out of scope by construction (`is_fault_step`) — a redundancy
  upgrade *adds* protection, so a legitimately tighter bound there is not
  a violation.
- **Determinism (Category 6 (e))** — folded into the existing M1
  determinism gate: two same-seed passes must produce an identical
  forecast-trace digest (canonical JSON over every state's forecast rows)
  alongside the unchanged M1–M3 assessment-trace digest.

CI: the canonical pass runs inside `make sim` on every PR and push (80
facilities, 693 forecast states, ~46 s — cheap enough to stay
merge-blocking). The nightly workflow gains the seed sweep
(`make forecast-sweep`, 100 seeds by default, one facility per segment per
seed) with `forecast-dynamics.{md,json}` uploaded as the per-run artifact;
the property-level determinism check runs inside the existing pytest run.

Honest semantics note — the degenerate-separation caveat, now
load-bearing: these gates assert *structural* behavior only. No accuracy
claim (AUROC, Brier, calibration) is made under dynamics, because on the
synthetic corpus the labels are a deterministic function of the same
protection facts the forecast features read — an accuracy metric across
dynamics would be tautological, not evidence of skill. What M4 validates
is mechanism behavior under state change (does the forecast move, restore,
abstain, and hold its uncertainty the way the graph says it must), not
real-world forecast accuracy, real event timing, or facility outcomes;
those stay gated on pilot replay (WP4). Sweep runtime and per-segment
coverage are stated in each sweep report.

## Not yet implemented (later phases, per art_XaCiEWwo)

- Historical replay on pilot data (WP4): the point where AUROC / Brier /
  lead-time numbers start to mean anything about a real facility.
- Calibrated probabilities against *real* labels (WP2): the calibrated
  forecaster and its go/no-go gates are live on synthetic data; every
  probability-bearing claim stays gated on labeled field data.
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

Forecast-specific gaps: the by-construction labels are a deterministic
function of the same topology facts the protection features read, so the
forecaster separates the holdout perfectly — AUROC 1.000 and Brier 0.0000
are degenerate (the labels are recoverable from the graph), not evidence
of predictive skill. The gates bind only once real, noisier labels exist;
the shared-blind-spot risk applies (features and labels come from the
same construction), and the abstention path is exercised by constructed
conflicts and thin-cell corpora, not yet by real operator data. The same
caveat governs the Sim M4 dynamics gates: they are structural properties
(signs, restoration, abstention identity, bound behavior, determinism),
never accuracy claims under dynamics, and the sweep's scope rules
(stable unchanged feeds, fault steps only) are the documented boundary of
what the gates can and cannot see.
