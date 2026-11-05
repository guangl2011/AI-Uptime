"""Sim M4 — forecast-under-dynamics: Category 6 honesty gates over the corpus.

The first three sim milestones tested the *assessor* across dynamics; this
one tests the *forecast* (the Phase 5 calibrated model) across the same
dynamics. The sequenced-scenario event engine faults, repairs,
re-redundancies, and load-shifts the corpus facilities exactly as before;
after every materialized graph state the pass assesses that state through
the real API with the canonical frozen forecast model installed, and
records the per-asset forecast trace.

What is under test is forecast *behavior across state changes* — signs,
restoration, abstention identity, bound behavior, determinism — never
forecast accuracy. The plan's degenerate-separation caveat is load-bearing
here: the synthetic labels are a deterministic function of the same
protection facts the forecast features read, so any accuracy metric
(AUROC, Brier) evaluated under dynamics would be tautological on synthetic
labels. Category 6 therefore asserts structural properties only; accuracy
language stays out of the gates and out of this module's outputs.

Category 6 gates (zero-tolerance, per segment; determinism folds into the
existing M1 determinism gate — see ``runner.evaluate_sim_gates``):

- ``forecast_degradation_sign`` — across every fault step, for every
  consumer live on both sides of the step and every surviving direct feed
  whose own outgoing edge set is unchanged (the *unchanged-path scope*):
  a non-abstained forecast never decreases, an abstention only ever
  *appears* carrying the conflict trigger the features declare (never
  clears), and a protecting group that drops below two live members is
  registered as a conflict abstention on the surviving feeds it protected.
- ``forecast_repair_restores`` — after the final seeded event, repairing
  every still-faulted asset and re-assessing yields forecast rows exactly
  equal, asset for asset, to a fresh assessment of the engine-intact state
  of the same (post-upgrade) topology. No tolerance is added: the
  calibration table is frozen and drift-pinned (tests/forecasting/
  test_canonical.py), so equality is exact by construction.
- ``forecast_abstention_identity`` — at every state, the report's forecast
  rows are exactly what the canonical model computes on that state's
  committed snapshot (the static path), and every abstained row states its
  reason. Replay cannot invent, drop, or reword an abstention the static
  path does not have.
- ``forecast_bounds_monotone`` — across every fault step on unchanged-path
  feeds, the interval never shrinks: information loss widens or holds the
  bounds, never tightens them silently. (Repairs and upgrades may narrow
  bounds — redundancy or evidence was *added* — so they are out of scope,
  matching the plan's "across added fault events".)

Scope honesty: the unchanged-path restriction is deliberate, not a
weakening. When a fault kills a *second* service an asset also fed, that
asset's own supply edges change and its point estimate legitimately moves
in either direction; the assessor's loss behavior on those states is
already gated by Categories 1–4. The forecast gate follows the plan's
claim — the forecast of a downstream service must not improve when
redundancy is removed — along the paths that did not themselves change.

Governance (binding, unchanged): the pass drives the assessment/forecast
engine through a throwaway local app only — nothing external, nothing
outside the repo's own eval/sim outputs, no facility corpus changes. The
forecaster is installed per-run on the throwaway app; the product default
stays OFF and informational with no action path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeGuard, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from evals.corpus import CORPUS_SEED, SEGMENTS, FacilityCorpus, generate_corpus
from evals.harness import load_facility
from evals.sim.engine import (
    TOLERANT_CONFIGS,
    FaultEvent,
    RepairEvent,
    SimEvent,
    SimState,
    Topology,
    apply_event,
    dead_nodes,
    describe_event,
    initial_state,
    materialize,
)
from evals.sim.invariants import (
    InvariantResult,
    InvariantViolation,
    repro_recipe,
    seeded_events,
)
from evals.sim.runner import SIM_AUTHOR, GateResult, _commit_and_assess
from uptime_api.app import create_app
from uptime_api.forecasting.calibrated import CalibratedForecastModel
from uptime_api.forecasting.features import asset_protection_features
from uptime_api.forecasting.schema import ForecastStatus, RiskForecast
from uptime_api.graph.enums import NodeType
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut

SIGN = "forecast_degradation_sign"
REPAIR = "forecast_repair_restores"
ABSTENTION_IDENTITY = "forecast_abstention_identity"
BOUNDS = "forecast_bounds_monotone"

FORECAST_PROPERTIES: tuple[str, ...] = (SIGN, REPAIR, ABSTENTION_IDENTITY, BOUNDS)

SIM_NOTES_PREFIX = "sim forecast-dynamics"


# ---------------------------------------------------------------------------
# Recorded shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicsStep:
    """One materialized state: its event label, snapshot, and forecast rows."""

    step: int
    event: str
    version_no: int
    snapshot: GraphSnapshotOut
    forecasts: dict[str, RiskForecast]  # asset_key -> the report's row
    dead: frozenset[str]


@dataclass(frozen=True)
class FacilityForecastTrace:
    """The forecast-across-dynamics trace of one facility."""

    facility_id: str
    segment: str
    steps: tuple[DynamicsStep, ...]
    checks: dict[str, int]  # property name -> check count
    observations: dict[str, int]  # descriptive counters (registrations, confounds)
    violations: tuple[InvariantViolation, ...]


@dataclass(frozen=True)
class ForecastDynamicsResult:
    """Everything one forecast-under-dynamics pass produced."""

    seed: int
    facilities: int
    facilities_by_segment: dict[str, int]
    states_assessed: int
    runtime_seconds: float
    traces: tuple[FacilityForecastTrace, ...]

    def checks_for(self, invariant: str, segment: str) -> int:
        return sum(
            trace.checks.get(invariant, 0)
            for trace in self.traces
            if trace.segment == segment
        )

    def violations_for(
        self, invariant: str, segment: str
    ) -> tuple[InvariantViolation, ...]:
        return tuple(
            violation
            for trace in self.traces
            if trace.segment == segment
            for violation in trace.violations
            if violation.invariant == invariant
        )

    def observations_for(self, segment: str) -> dict[str, int]:
        totals: dict[str, int] = {}
        for trace in self.traces:
            if trace.segment != segment:
                continue
            for key, value in trace.observations.items():
                totals[key] = totals.get(key, 0) + value
        return totals


@dataclass(frozen=True)
class SignObservations:
    """Descriptive counters from one fault step (report material, not gates)."""

    conflict_registrations: int = 0  # non-abstained -> abstained-with-conflict
    path_loss_confounds: int = 0  # surviving feeds excluded by the unchanged-path scope


# ---------------------------------------------------------------------------
# The unchanged-path scope (Category 6 (a)/(d) comparison set)
# ---------------------------------------------------------------------------


class StableFeed(NamedTuple):
    consumer: str
    asset_key: str
    edge: EdgeOut  # the surviving post-step edge; same triple on the pre side


@dataclass(frozen=True)
class StablePathScope:
    """The unchanged-path comparison scope for one fault step."""

    pairs: tuple[StableFeed, ...]
    surviving_feeds: int  # all surviving direct feeds, before the restriction


def _asset_keys(snapshot: GraphSnapshotOut) -> set[str]:
    return {
        node.node_key for node in snapshot.nodes if node.node_type is NodeType.ASSET
    }


def _outgoing_signatures(
    snapshot: GraphSnapshotOut, asset_key: str
) -> frozenset[tuple[str, str | None, str | None]]:
    """The asset's full supply signature: every edge it emits, with config/group."""
    return frozenset(
        (edge.to_node, edge.redundancy_config, edge.member_group)
        for edge in snapshot.edges
        if edge.from_node == asset_key
    )


def _direct_feeds(
    snapshot: GraphSnapshotOut, consumer: str, asset_keys: set[str]
) -> dict[str, EdgeOut]:
    """Live direct asset feeds of one consumer (materialized edges only)."""
    return {
        edge.from_node: edge
        for edge in snapshot.edges
        if edge.to_node == consumer and edge.from_node in asset_keys
    }


def _group_sizes(snapshot: GraphSnapshotOut) -> dict[tuple[str, str], int]:
    """Live member count per (consumer, member group) — materialized edges only."""
    sizes: dict[tuple[str, str], int] = {}
    for edge in snapshot.edges:
        if edge.member_group is not None:
            key = (edge.to_node, edge.member_group)
            sizes[key] = sizes.get(key, 0) + 1
    return sizes


def _stable_path_feeds(pre: DynamicsStep, post: DynamicsStep) -> StablePathScope:
    """Surviving unchanged-path direct feeds per consumer, plus the full count.

    A feed survives the step when both endpoints are live on both sides; it
    is *unchanged-path* when the feed asset's own outgoing edge set is
    identical across the step (a consumer it also fed did not die — the
    path-loss confound Categories 1–4 already own).
    """
    pre_assets = _asset_keys(pre.snapshot)
    post_assets = _asset_keys(post.snapshot)
    pre_nodes = {node.node_key for node in pre.snapshot.nodes}
    post_nodes = {node.node_key for node in post.snapshot.nodes}
    pairs: list[StableFeed] = []
    surviving = 0
    for consumer in sorted(pre_nodes & post_nodes):
        pre_feeds = _direct_feeds(pre.snapshot, consumer, pre_assets)
        post_feeds = _direct_feeds(post.snapshot, consumer, post_assets)
        surviving += len(set(pre_feeds) & set(post_feeds))
        for asset_key in sorted(set(pre_feeds) & set(post_feeds)):
            if _outgoing_signatures(pre.snapshot, asset_key) == _outgoing_signatures(
                post.snapshot, asset_key
            ):
                pairs.append(StableFeed(consumer, asset_key, post_feeds[asset_key]))
    return StablePathScope(pairs=tuple(pairs), surviving_feeds=surviving)


# ---------------------------------------------------------------------------
# Category 6 property checks (pure functions over recorded steps)
# ---------------------------------------------------------------------------


def _violation(
    invariant: str,
    facility: FacilityCorpus,
    seed: int,
    events: Sequence[SimEvent],
    detail: str,
) -> InvariantViolation:
    return InvariantViolation(
        invariant=invariant,
        seed=seed,
        segment=facility.segment,
        facility_id=facility.facility_id,
        detail=detail,
        repro=repro_recipe(facility, seed, events),
    )


def is_fault_step(event: SimEvent) -> TypeGuard[FaultEvent]:
    """Whether sign/bounds gates apply to this step.

    Only faults REMOVE information. Repairs restore it and redundancy
    upgrades add it — either may legitimately tighten a bound or lower a
    forecast, so those steps are out of scope by construction.
    """
    return isinstance(event, FaultEvent)


def check_degradation_sign(
    facility: FacilityCorpus,
    seed: int,
    pre: DynamicsStep,
    post: DynamicsStep,
    scope: StablePathScope,
    events: Sequence[SimEvent],
) -> InvariantResult:
    """Category 6 (a): removing redundancy never lowers a downstream forecast."""
    violations: list[InvariantViolation] = []
    checks = 0
    post_features = {
        feature.asset_key: feature
        for feature in asset_protection_features(post.snapshot)
    }
    group_sizes_pre = _group_sizes(pre.snapshot)
    group_sizes_post = _group_sizes(post.snapshot)

    def fail(detail: str) -> None:
        violations.append(_violation(SIGN, facility, seed, events, detail))

    for feed in scope.pairs:
        pre_row = pre.forecasts.get(feed.asset_key)
        post_row = post.forecasts.get(feed.asset_key)
        if pre_row is None or post_row is None:
            checks += 1
            fail(
                f"asset {feed.asset_key} feeding {feed.consumer}: forecast row missing "
                f"(pre {pre_row is not None}, post {post_row is not None}) at fault "
                f"step {post.step} — every live asset must carry a forecast row"
            )
            continue
        post_conflict = post_features[feed.asset_key].conflict is not None
        if post_row.status is ForecastStatus.ABSTAINED:
            checks += 1
            if not post_conflict:
                fail(
                    f"asset {feed.asset_key} feeding {feed.consumer}: abstained at "
                    f"fault step {post.step} without the declared conflict trigger "
                    f"(reason {post_row.abstention_reason!r}) — under the canonical "
                    "calibration table only a conflicting-evidence state may abstain"
                )
            continue
        if pre_row.status is ForecastStatus.ABSTAINED:
            checks += 1
            fail(
                f"asset {feed.asset_key} feeding {feed.consumer}: abstention cleared "
                f"across fault step {post.step} on an unchanged supply path "
                f"({pre_row.abstention_reason!r} -> non-abstained "
                f"{post_row.p_fault:.4f}) — conflicting evidence did not resolve "
                "by removing redundancy"
            )
            continue
        checks += 1
        if post_row.p_fault < pre_row.p_fault:
            fail(
                f"asset {feed.asset_key} feeding {feed.consumer}: forecast p_fault "
                f"DECREASED across fault step {post.step} on an unchanged supply "
                f"path: {pre_row.p_fault:.4f} -> {post_row.p_fault:.4f} "
                f"({pre_row.status.value} -> {post_row.status.value}) — a fault "
                "that removes redundancy must never lower the forecast of a "
                "downstream service"
            )
        config = feed.edge.redundancy_config
        group = feed.edge.member_group
        if config is not None and config in TOLERANT_CONFIGS and group is not None:
            key = (feed.consumer, group)
            pre_size = group_sizes_pre.get(key, 0)
            post_size = group_sizes_post.get(key, 0)
            if pre_size >= 2 and post_size < 2:
                checks += 1
                # post_row is non-abstained here (abstained rows continued
                # above): a group that drops below two live members declares
                # conflicting evidence, so a non-abstained row means the
                # redundancy loss went unregistered.
                fail(
                    f"asset {feed.asset_key} feeding {feed.consumer}: the "
                    f"protecting group {group!r} dropped from {pre_size} to "
                    f"{post_size} live members across fault step {post.step} "
                    f"(conflict trigger declared: {post_conflict}), but the "
                    f"forecast stayed non-abstained ({post_row.status.value} "
                    f"{post_row.p_fault:.4f}) — the redundancy loss was not "
                    "registered as conflicting evidence"
                )
    return InvariantResult(SIGN, checks, tuple(violations))


def check_bounds_monotone(
    facility: FacilityCorpus,
    seed: int,
    pre: DynamicsStep,
    post: DynamicsStep,
    scope: StablePathScope,
    events: Sequence[SimEvent],
) -> InvariantResult:
    """Category 6 (d): added faults never tighten an unchanged path's interval."""
    violations: list[InvariantViolation] = []
    checks = 0

    for feed in scope.pairs:
        pre_row = pre.forecasts.get(feed.asset_key)
        post_row = post.forecasts.get(feed.asset_key)
        if pre_row is None or post_row is None:
            continue  # the missing-row case is the sign gate's finding
        checks += 1
        if post_row.lower > pre_row.lower or post_row.upper < pre_row.upper:
            violations.append(
                _violation(
                    BOUNDS,
                    facility,
                    seed,
                    events,
                    f"asset {feed.asset_key} feeding {feed.consumer}: forecast "
                    f"interval SHRANK across fault step {post.step} on an unchanged "
                    f"supply path: [{pre_row.lower:.4f}, {pre_row.upper:.4f}] -> "
                    f"[{post_row.lower:.4f}, {post_row.upper:.4f}] — information "
                    "loss widens or holds bounds, never tightens them silently",
                )
            )
    return InvariantResult(BOUNDS, checks, tuple(violations))


def check_abstention_identity(
    facility: FacilityCorpus,
    seed: int,
    step: DynamicsStep,
    model: CalibratedForecastModel,
    events: Sequence[SimEvent],
) -> InvariantResult:
    """Category 6 (c): the report's forecast rows are the static path's, exactly."""
    violations: list[InvariantViolation] = []
    checks = 1
    static = {row.asset_key: row for row in model.forecast(step.snapshot)}
    if static != step.forecasts:
        differing = sorted(
            key
            for key in set(static) | set(step.forecasts)
            if static.get(key) != step.forecasts.get(key)
        )
        first = differing[0] if differing else "<none>"
        violations.append(
            _violation(
                ABSTENTION_IDENTITY,
                facility,
                seed,
                events,
                f"at step {step.step} ({step.event}), the report's forecast rows "
                f"differ from the canonical model on the committed snapshot — "
                f"first difference at asset {first!r}: report "
                f"{step.forecasts.get(first)!r} vs static {static.get(first)!r}",
            )
        )
    for asset_key, row in sorted(static.items()):
        if row.status is ForecastStatus.ABSTAINED and not row.abstention_reason:
            checks += 1
            violations.append(
                _violation(
                    ABSTENTION_IDENTITY,
                    facility,
                    seed,
                    events,
                    f"at step {step.step}, asset {asset_key} abstained without a "
                    "stated reason — the abstention contract requires the reason "
                    "to survive replay",
                )
            )
    return InvariantResult(ABSTENTION_IDENTITY, checks, tuple(violations))


def check_repair_restores(
    facility: FacilityCorpus,
    seed: int,
    repaired: DynamicsStep,
    intact: DynamicsStep,
    events: Sequence[SimEvent],
) -> InvariantResult:
    """Category 6 (b): post-repair forecasts equal the intact-state baseline."""
    violations: list[InvariantViolation] = []
    if repaired.forecasts != intact.forecasts:
        differing = sorted(
            key
            for key in set(repaired.forecasts) | set(intact.forecasts)
            if repaired.forecasts.get(key) != intact.forecasts.get(key)
        )
        first = differing[0] if differing else "<none>"
        violations.append(
            _violation(
                REPAIR,
                facility,
                seed,
                events,
                f"post-repair forecast rows (step {repaired.step}) do not equal the "
                f"intact-state baseline (step {intact.step}) for asset {first!r}: "
                f"repaired {repaired.forecasts.get(first)!r} vs intact "
                f"{intact.forecasts.get(first)!r} — repair must restore the "
                "intact-state forecast exactly (the canonical table is frozen, so "
                "no tolerance applies)",
            )
        )
    return InvariantResult(REPAIR, 1, tuple(violations))


def _fault_step_observations(
    scope: StablePathScope, pre: DynamicsStep, post: DynamicsStep
) -> SignObservations:
    """Descriptive counters for the report: registrations and excluded confounds."""
    registrations = 0
    for feed in scope.pairs:
        pre_row = pre.forecasts.get(feed.asset_key)
        post_row = post.forecasts.get(feed.asset_key)
        if (
            pre_row is not None
            and post_row is not None
            and pre_row.status is not ForecastStatus.ABSTAINED
            and post_row.status is ForecastStatus.ABSTAINED
        ):
            registrations += 1
    return SignObservations(
        conflict_registrations=registrations,
        path_loss_confounds=scope.surviving_feeds - len(scope.pairs),
    )


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def _forecast_rows(report: dict[str, object]) -> dict[str, RiskForecast]:
    """The report's forecast rows keyed by asset (validated into the schema)."""
    raw = report.get("forecasts", [])
    result: dict[str, RiskForecast] = {}
    for row in raw if isinstance(raw, list) else []:
        parsed = RiskForecast.model_validate(row)
        result[parsed.asset_key] = parsed
    return result


def _fetch_snapshot(
    client: TestClient, facility_id: str, version_no: int
) -> GraphSnapshotOut:
    response = client.get(
        f"/facilities/{facility_id}/graph", params={"version": version_no}
    )
    if response.status_code != 200:
        raise AssertionError(
            f"graph snapshot fetch failed ({response.status_code}): {response.text[:2000]}"
        )
    return GraphSnapshotOut.model_validate(response.json())


def _commit_step(
    client: TestClient,
    facility: FacilityCorpus,
    state: SimState,
    step: int,
    event: str,
) -> DynamicsStep:
    """Materialize one state, commit + assess it through the real API, record it."""
    payload = materialize(
        state,
        dead_nodes(state),
        author=SIM_AUTHOR,
        notes=f"{SIM_NOTES_PREFIX} {facility.facility_id} step {step}: {event}",
    )
    version_no, report = _commit_and_assess(client, facility.facility_id, payload)
    return DynamicsStep(
        step=step,
        event=event,
        version_no=version_no,
        snapshot=_fetch_snapshot(client, facility.facility_id, version_no),
        forecasts=_forecast_rows(report),
        dead=dead_nodes(state),
    )


def _baseline_step(
    client: TestClient,
    facility: FacilityCorpus,
    state: SimState,
    report: dict[str, object],
) -> DynamicsStep:
    """Step 0: the onboarding baseline (version 2, assessed by load_facility)."""
    version_no = report["graph_version"]
    assert isinstance(version_no, int)
    return DynamicsStep(
        step=0,
        event="intact baseline (onboarding)",
        version_no=version_no,
        snapshot=_fetch_snapshot(client, facility.facility_id, version_no),
        forecasts=_forecast_rows(report),
        dead=dead_nodes(state),
    )


def _trace_facility(
    client: TestClient,
    facility: FacilityCorpus,
    model: CalibratedForecastModel,
    seed: int,
    events: Sequence[SimEvent],
) -> FacilityForecastTrace:
    topology = Topology.from_payload(facility.graph)
    state = initial_state(topology)
    checks: dict[str, int] = {}
    observations: dict[str, int] = {}
    violations: list[InvariantViolation] = []

    def add(result: InvariantResult) -> None:
        checks[result.invariant] = checks.get(result.invariant, 0) + result.checks
        violations.extend(result.violations)

    def observe(signs: SignObservations) -> None:
        for key, value in (
            ("conflict_registrations", signs.conflict_registrations),
            ("path_loss_confounds", signs.path_loss_confounds),
        ):
            observations[key] = observations.get(key, 0) + value

    baseline_report = load_facility(client, facility)
    baseline = _baseline_step(client, facility, state, baseline_report)
    steps: list[DynamicsStep] = [baseline]
    add(check_abstention_identity(facility, seed, baseline, model, events[:0]))

    outstanding: set[str] = set()
    previous = baseline
    for index, event in enumerate(events, start=1):
        state = apply_event(state, event)
        label = describe_event(event)
        record = _commit_step(client, facility, state, index, label)
        steps.append(record)
        add(check_abstention_identity(facility, seed, record, model, events[:index]))
        if is_fault_step(event):
            scope = _stable_path_feeds(previous, record)
            add(
                check_degradation_sign(
                    facility, seed, previous, record, scope, events[:index]
                )
            )
            add(
                check_bounds_monotone(
                    facility, seed, previous, record, scope, events[:index]
                )
            )
            observe(_fault_step_observations(scope, previous, record))
            outstanding.add(event.asset_key)
        elif isinstance(event, RepairEvent):
            outstanding.discard(event.asset_key)
        previous = record

    # Category 6 (b): repair everything still faulted, then compare against a
    # fresh assessment of the engine-intact state of the same (post-upgrade)
    # topology — the intact reference the repaired state must return to.
    for key in sorted(outstanding):
        state = apply_event(state, RepairEvent(key))
    if outstanding:
        repaired = _commit_step(
            client, facility, state, len(steps), "repair all remaining faults"
        )
        steps.append(repaired)
        restoration_events: list[SimEvent] = [*events]
        restoration_events.extend(RepairEvent(key) for key in sorted(outstanding))
        add(
            check_abstention_identity(
                facility, seed, repaired, model, restoration_events
            )
        )
    else:
        repaired = steps[-1]
    intact_state = SimState(state.topology, frozenset())
    intact = _commit_step(
        client,
        facility,
        intact_state,
        len(steps),
        "intact reference (post-repair baseline)",
    )
    steps.append(intact)
    add(check_abstention_identity(facility, seed, intact, model, events))
    add(check_repair_restores(facility, seed, repaired, intact, events))

    return FacilityForecastTrace(
        facility_id=facility.facility_id,
        segment=facility.segment,
        steps=tuple(steps),
        checks=checks,
        observations=observations,
        violations=tuple(violations),
    )


def run_forecast_dynamics(
    client: TestClient,
    *,
    seed: int = CORPUS_SEED,
    facilities_per_segment: dict[str, int] | None = None,
) -> ForecastDynamicsResult:
    """Run the forecast-across-dynamics pass over the corpus (Sim M4).

    The canonical frozen forecast model is installed on the throwaway app for
    the duration of the pass and restored after; the product default
    (forecasting OFF, informational) is untouched.
    """
    corpus = generate_corpus(seed, facilities_per_segment)
    app = cast(FastAPI, client.app)
    model = CalibratedForecastModel()
    previous = app.state.forecaster
    app.state.forecaster = model
    started = time.monotonic()
    try:
        traces = tuple(
            _trace_facility(client, facility, model, seed, seeded_events(facility, seed))
            for facility in corpus.facilities
        )
    finally:
        app.state.forecaster = previous
    counts: dict[str, int] = {}
    for trace in traces:
        counts[trace.segment] = counts.get(trace.segment, 0) + 1
    return ForecastDynamicsResult(
        seed=seed,
        facilities=len(traces),
        facilities_by_segment=dict(sorted(counts.items())),
        states_assessed=sum(len(trace.steps) for trace in traces),
        runtime_seconds=time.monotonic() - started,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Determinism digest (Category 6 (e)) and gates
# ---------------------------------------------------------------------------


def _digest_step(step: DynamicsStep) -> dict[str, object]:
    return {
        "step": step.step,
        "event": step.event,
        "forecasts": {
            asset_key: {
                "status": row.status.value,
                "p_fault": row.p_fault,
                "lower": row.lower,
                "upper": row.upper,
                "basis": row.basis,
                "abstention_reason": row.abstention_reason,
            }
            for asset_key, row in sorted(step.forecasts.items())
        },
    }


def forecast_trace_digest(traces: tuple[FacilityForecastTrace, ...]) -> str:
    """Canonical digest of the full forecast-across-dynamics trace."""
    payload = {
        "facilities": [
            {
                "facility_id": trace.facility_id,
                "segment": trace.segment,
                "steps": [_digest_step(step) for step in trace.steps],
            }
            for trace in traces
        ]
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def evaluate_forecast_gates(
    result: ForecastDynamicsResult,
) -> tuple[GateResult, ...]:
    """Per-segment Category 6 gate rows, each with its evidence line.

    Zero-tolerance: any violation fails the gate; so does a zero-check run
    (a silent zero-check run fails, mirroring the sweep discipline). The
    determinism half of Category 6 folds into the existing M1 determinism
    gate (``runner.evaluate_sim_gates``), not here.
    """
    gates: list[GateResult] = []
    for invariant in FORECAST_PROPERTIES:
        for segment in SEGMENTS:
            checks = result.checks_for(invariant, segment)
            violations = result.violations_for(invariant, segment)
            detail = (
                f"{checks} checks across {result.facilities_by_segment.get(segment, 0)} "
                f"facilities: {len(violations)} violations"
            )
            if violations:
                first = violations[0]
                detail += f"; first: {first.detail} (repro: {first.repro})"
            gates.append(
                GateResult(
                    name=f"{segment}:{invariant}",
                    passed=not violations and checks > 0,
                    detail=detail,
                )
            )
    return tuple(gates)


# ---------------------------------------------------------------------------
# JSON artifact rendering (the pass's own section, shared with make sim)
# ---------------------------------------------------------------------------


def _step_json(step: DynamicsStep) -> dict[str, object]:
    return {
        "step": step.step,
        "event": step.event,
        "version_no": step.version_no,
        "abstained": sorted(
            key
            for key, row in step.forecasts.items()
            if row.status is ForecastStatus.ABSTAINED
        ),
    }


def forecast_dynamics_json(result: ForecastDynamicsResult) -> dict[str, object]:
    """The structured forecast-under-dynamics section for the sim report."""
    return {
        "facilities": result.facilities,
        "facilities_by_segment": result.facilities_by_segment,
        "states_assessed": result.states_assessed,
        "runtime_seconds": round(result.runtime_seconds, 2),
        "per_property": {
            invariant: {
                segment: {
                    "checks": result.checks_for(invariant, segment),
                    "violations": len(result.violations_for(invariant, segment)),
                }
                for segment in SEGMENTS
            }
            for invariant in FORECAST_PROPERTIES
        },
        "observations": {
            segment: result.observations_for(segment) for segment in SEGMENTS
        },
        "violations": [
            violation.render()
            for trace in result.traces
            for violation in trace.violations
        ],
        "traces": [
            {
                "facility_id": trace.facility_id,
                "segment": trace.segment,
                "steps": [_step_json(step) for step in trace.steps],
            }
            for trace in result.traces
        ],
    }


# ---------------------------------------------------------------------------
# Nightly sweep CLI: multi-seed forecast-dynamics (bounded API budget)
# ---------------------------------------------------------------------------


def run_forecast_sweep(
    *,
    seed_count: int,
    start_seed: int,
    facilities_per_segment: int,
    output_dir: Path,
) -> int:
    """Multi-seed forecast-dynamics sweep: one facility per segment per seed.

    Each seed runs the full event replay on a throwaway app. The canonical
    corpus (make sim, every PR) covers the canonical seed; this sweep varies
    the seed to shake out corpus-shape luck. Category 6 has no 1,000-seed
    gate — the seed budget is calibrated to keep the nightly job bounded.
    """
    started = time.monotonic()
    traces: list[FacilityForecastTrace] = []
    states = 0
    facilities_by_segment: dict[str, int] = {segment: 0 for segment in SEGMENTS}
    for seed in range(start_seed, start_seed + seed_count):
        facilities = {segment: facilities_per_segment for segment in SEGMENTS}
        with tempfile.TemporaryDirectory(prefix="aiuptime-forecast-sweep-") as tmp:
            app = create_app(f"sqlite:///{Path(tmp) / 'forecast-sweep.db'}")
            with TestClient(app) as client:
                result = run_forecast_dynamics(
                    client, seed=seed, facilities_per_segment=facilities
                )
        traces.extend(result.traces)
        states += result.states_assessed
        for segment, count in result.facilities_by_segment.items():
            facilities_by_segment[segment] = (
                facilities_by_segment.get(segment, 0) + count
            )
    combined = ForecastDynamicsResult(
        seed=start_seed,
        facilities=len(traces),
        facilities_by_segment=dict(sorted(facilities_by_segment.items())),
        states_assessed=states,
        runtime_seconds=time.monotonic() - started,
        traces=tuple(traces),
    )
    gates = evaluate_forecast_gates(combined)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forecast-dynamics.md").write_text(
        _render_sweep_markdown(seed_count, combined, gates), encoding="utf-8"
    )
    (output_dir / "forecast-dynamics.json").write_text(
        json.dumps(_render_sweep_json(combined, gates), indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(gate.passed for gate in gates) else 1


def _render_sweep_markdown(
    seed_count: int, combined: ForecastDynamicsResult, gates: tuple[GateResult, ...]
) -> str:
    lines = [
        "# Forecast-under-dynamics sweep (Sim M4, Category 6)",
        "",
        f"Seeds: {seed_count} (one facility per segment per seed) · "
        f"Facilities traced: {combined.facilities} · "
        f"States assessed: {combined.states_assessed} · Runtime: "
        f"{combined.runtime_seconds:.0f}s",
        "",
        "Structural behavior gates only — never accuracy: the synthetic labels are a "
        "deterministic function of the same protection facts the forecast features "
        "read, so accuracy under dynamics would be tautological. This validates "
        "mechanism behavior under state change, not real-world forecast accuracy or "
        "event timing.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        marker = "PASS" if gate.passed else "FAIL"
        lines.append(f"| `{gate.name}` | {marker} | {gate.detail} |")
    violations = [
        violation.render() for trace in combined.traces for violation in trace.violations
    ]
    if violations:
        lines += ["", "## Failures (repro detail)", ""]
        lines += [f"- {violation}" for violation in violations]
    return "\n".join(lines) + "\n"


def _render_sweep_json(
    combined: ForecastDynamicsResult, gates: tuple[GateResult, ...]
) -> dict[str, object]:
    return {
        "facilities": combined.facilities,
        "facilities_by_segment": combined.facilities_by_segment,
        "states_assessed": combined.states_assessed,
        "runtime_seconds": round(combined.runtime_seconds, 2),
        "gates": [
            {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
            for gate in gates
        ],
        "per_property": {
            invariant: {
                segment: {
                    "checks": combined.checks_for(invariant, segment),
                    "violations": len(combined.violations_for(invariant, segment)),
                }
                for segment in SEGMENTS
            }
            for invariant in FORECAST_PROPERTIES
        },
        "observations": {
            segment: combined.observations_for(segment) for segment in SEGMENTS
        },
        "violations": [
            violation.render()
            for trace in combined.traces
            for violation in trace.violations
        ],
        "determinism_note": (
            "one pass per seed; the canonical two-pass determinism gate runs in "
            "make sim on the canonical corpus"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.sim.forecast_dynamics",
        description="Run the Sim M4 forecast-under-dynamics seed sweep.",
    )
    parser.add_argument(
        "--seeds", type=int, default=100, help="number of seeds to sweep (default: 100)"
    )
    parser.add_argument(
        "--start-seed", type=int, default=1, help="first seed (default: 1)"
    )
    parser.add_argument(
        "--facilities-per-segment",
        type=int,
        default=1,
        help="facilities per segment per seed (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval-results/forecast-dynamics"),
        help="where forecast-dynamics.{md,json} land",
    )
    args = parser.parse_args(argv)
    return run_forecast_sweep(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        facilities_per_segment=args.facilities_per_segment,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
