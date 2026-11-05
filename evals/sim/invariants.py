"""Sim M2 invariant checks: the simulation plan's Categories 3 and 4.

Category 3 — mathematical properties over generated seeds, stated as
properties rather than expected outputs (run by Hypothesis in
``tests/evals/sim/test_invariants.py`` on every push, and by the
1,000-seed sweep in ``evals.sim.invariant_sweep`` nightly):

- ``redundancy_monotonicity`` — adding redundancy (a standby member via the
  N → N+1 → 2N ladder) never increases exposure: for the intact state and
  every single-asset fault, the upgraded dead set is a subset of the
  original's, and the standby itself never dies.
- ``repair_recovery`` — a repair never re-exposes a dead service (the dead
  set only shrinks through repairs), and repairing every event fault
  restores the intact graph exactly — no stale exposure, no torn edges.
- ``trace_determinism`` — the same seed regenerates the facility
  identically and replays to an identical materialized trace.

Category 4 — recovery state on replay folds, reusing the existing
``evals.splits`` discipline (LeakageError) rather than a parallel mechanism:

- ``replay_no_leakage`` — every fold built from a corpus replay case keeps
  post-repair events out of the pre-repair feature window, and each label
  window contains the fault's work order and its repair.
- ``replay_no_clairvoyance`` — truncating the episode (dropping the repair)
  leaves every pre-repair trace step identical: pre-repair states are a
  function of pre-repair events only.
- ``recovery_state`` — post-repair reflects restored redundancy: at the
  engine level the post-repair snapshot equals the intact baseline, and
  through the real API the post-repair report equals the baseline report
  except for the pinned version, while the constructed interruption is
  visible in the pre-repair state.

Category 3 over what-if interventions (M3, ``whatif_counterfactual``) —
the same property discipline applied to the counterfactual branches:
upgrades applied to any seeded state never increase exposure and never kill
the standby; a removal's cascade equals the same asset's fault cascade; and
every branch payload replays identically.

Every violation carries the failing seed and a minimal repro (how to
regenerate the facility and which events to replay) — a repro case ships
with the failure, per the plan's CI rule.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from fastapi.testclient import TestClient

from evals.corpus import FacilityCorpus, generate_segment_corpus
from evals.harness import FEATURE_WINDOW, load_facility
from evals.sim.engine import (
    NODE_TYPE_ASSET,
    FaultEvent,
    LoadShiftEvent,
    RedundancyChangeEvent,
    RemoveAssetEvent,
    RepairEvent,
    SimEvent,
    SimEventError,
    SimState,
    Topology,
    WhatIfIntervention,
    apply_event,
    apply_intervention,
    dead_nodes,
    describe_event,
    describe_intervention,
    initial_state,
    materialize,
)

# The runner's commit+assess helper is the canonical real-API path — reused
# here rather than duplicated, so invariants exercise the exact M1 route.
from evals.sim.runner import SIM_AUTHOR, _commit_and_assess, lost_services
from evals.splits import (
    LeakageError,
    build_folds,
    feature_events,
    label_window_events,
)

_T = TypeVar("_T")

# One name per invariant, stable across the property tests, the sweep CLI,
# and the report artifacts (gates aggregate by this name).
INVARIANT_NAMES: tuple[str, ...] = (
    "redundancy_monotonicity",
    "repair_recovery",
    "trace_determinism",
    "replay_no_leakage",
    "replay_no_clairvoyance",
    "recovery_state",
    "whatif_counterfactual",
)

# Tolerance ladder: upgrade targets per current config, each strictly
# non-decreasing in tolerance. Downgrades are not "added redundancy" and are
# out of scope for the monotonicity property.
UPGRADE_TARGETS: dict[str, tuple[str, ...]] = {
    "N": ("N+1", "2N"),
    "N+1": ("2N",),
    "2N": ("2N",),
}


@dataclass(frozen=True)
class InvariantViolation:
    """One violated invariant, with the failing seed and a minimal repro."""

    invariant: str
    seed: int
    segment: str
    facility_id: str
    detail: str  # expected vs observed
    repro: str  # regeneration recipe + events to replay

    def render(self) -> str:
        return (
            f"[{self.invariant}] seed={self.seed} facility={self.facility_id} "
            f"({self.segment}): {self.detail} | repro: {self.repro}"
        )


@dataclass(frozen=True)
class InvariantResult:
    """The outcome of one invariant check over one facility."""

    invariant: str
    checks: int
    violations: tuple[InvariantViolation, ...] = ()


# ---------------------------------------------------------------------------
# Shared machinery: upgrade construction, seeded events, payload traces
# ---------------------------------------------------------------------------


def upgradeable_groups(topology: Topology) -> list[tuple[str, str]]:
    """(member_group, current_config) per single-consumer group, sorted.

    A group whose edges name several consumers is ambiguous for a standby
    join (the engine rejects it too), so it is not upgradeable here.
    """
    consumers: dict[str, set[str]] = {}
    configs: dict[str, str] = {}
    for edge in topology.edges:
        if edge.member_group is None or edge.redundancy_config is None:
            continue
        consumers.setdefault(edge.member_group, set()).add(edge.to_node)
        configs[edge.member_group] = edge.redundancy_config
    return sorted(
        (group, configs[group]) for group, targets in consumers.items() if len(targets) == 1
    )


def build_upgrade(topology: Topology, group: str, target_config: str) -> RedundancyChangeEvent:
    """A redundancy-change event adding one standby member to ``group``.

    The standby payload is cloned from a group member (sorted-first, for
    determinism) with a fresh node key and its own provenance ref, so the
    payload validates through the real API should a caller commit it.
    """
    member_edge = sorted(
        (edge for edge in topology.edges if edge.member_group == group),
        key=lambda edge: edge.from_node,
    )[0]
    member_node = topology.nodes[member_edge.from_node]

    suffix = 1
    standby_key = f"{member_edge.from_node}-SB{suffix}"
    while standby_key in topology.nodes:
        suffix += 1
        standby_key = f"{member_edge.from_node}-SB{suffix}"

    node_raw = dict(member_node.raw)
    node_raw["node_key"] = standby_key
    node_raw["name"] = f"{node_raw.get('name', standby_key)} (standby)"
    raw_provenance = node_raw.get("provenance")
    if isinstance(raw_provenance, dict):
        provenance = dict(raw_provenance)
        provenance["source_ref"] = f"{provenance.get('source_ref', '')}/standby"
        node_raw["provenance"] = provenance

    edge_raw = dict(member_edge.raw)
    edge_raw["from_node"] = standby_key
    edge_raw["redundancy_config"] = target_config
    return RedundancyChangeEvent(
        member_group=group,
        redundancy_config=target_config,
        standby_node=node_raw,
        standby_edge=edge_raw,
    )


def seeded_events(facility: FacilityCorpus, seed: int, max_steps: int = 6) -> tuple[SimEvent, ...]:
    """A legal event sequence derived deterministically from (facility, seed).

    ``random.Random`` over a string seeds via SHA-512 — not affected by
    PYTHONHASHSEED — so the sequence is stable across processes and runs.
    Kinds: faults, repairs, one redundancy upgrade per group, load shifts;
    each candidate is legal against the current state by construction, so
    an apply failure is a generator bug and propagates.
    """
    rng = random.Random(f"{facility.facility_id}:{seed}")
    state = initial_state(Topology.from_payload(facility.graph))
    events: list[SimEvent] = []
    upgraded: set[str] = set()

    for _ in range(max_steps):
        available: list[str] = []
        unfaulted = sorted(
            key
            for key, spec in state.topology.nodes.items()
            if spec.node_type == NODE_TYPE_ASSET and key not in state.faulted
        )
        if unfaulted:
            available.append("fault")
        if state.faulted:
            available.append("repair")
        if any(group not in upgraded for group, _ in upgradeable_groups(state.topology)):
            available.append("redundancy_change")
        if _load_shift_candidates(state):
            available.append("load_shift")
        if not available:
            break

        event: SimEvent
        match rng.choice(available):
            case "fault":
                event = FaultEvent(asset_key=rng.choice(unfaulted))
            case "repair":
                event = RepairEvent(asset_key=rng.choice(sorted(state.faulted)))
            case "redundancy_change":
                groups = [
                    (group, config)
                    for group, config in upgradeable_groups(state.topology)
                    if group not in upgraded
                ]
                group, config = rng.choice(groups)
                target = rng.choice(UPGRADE_TARGETS[config])
                event = build_upgrade(state.topology, group, target)
                upgraded.add(group)
            case "load_shift":
                consumer, source, destination = rng.choice(_load_shift_candidates(state))
                event = LoadShiftEvent(consumer=consumer, from_node=source, to_node=destination)
            case other:
                raise SimEventError(f"seeded_events picked unknown kind {other!r}")
        state = apply_event(state, event)
        events.append(event)
    return tuple(events)


def _load_shift_candidates(state: SimState) -> list[tuple[str, str, str]]:
    """(consumer, from_node, to_node) shifts legal against the current state.

    Mirrors the engine's load-shift preconditions exactly: exactly one
    dependency edge per (source, consumer) pair — counting all edges,
    including member-group ones — and that edge must not be a redundant
    group member (M1 shifts unprotected dependencies only). Generated
    corpora can carry parallel duplicate edges; they are benign for
    cascade closure but disqualify a pair from shifting.
    """
    edge_counts: dict[tuple[str, str], int] = {}
    group_pairs: set[tuple[str, str]] = set()
    for edge in state.topology.edges:
        pair = (edge.from_node, edge.to_node)
        edge_counts[pair] = edge_counts.get(pair, 0) + 1
        if edge.member_group is not None:
            group_pairs.add(pair)
    return [
        (consumer, source, destination)
        for (source, consumer), count in sorted(edge_counts.items())
        if count == 1 and (source, consumer) not in group_pairs
        for destination in sorted(state.topology.nodes)
        if destination not in (source, consumer)
    ]


def payload_trace(topology: Topology, events: Sequence[SimEvent]) -> tuple[dict[str, object], ...]:
    """Materialized snapshot payloads: the baseline, then one per event."""
    state = initial_state(topology)
    payloads = [_snapshot(state, 0, None)]
    for index, event in enumerate(events, start=1):
        state = apply_event(state, event)
        payloads.append(_snapshot(state, index, event))
    return tuple(payloads)


def _snapshot(state: SimState, step: int, event: SimEvent | None) -> dict[str, object]:
    label = "baseline" if event is None else describe_event(event)
    return materialize(
        state,
        dead_nodes(state),
        author=SIM_AUTHOR,
        notes=f"sim invariant step {step}: {label}",
    )


def _graph(payload: dict[str, object]) -> tuple[object, object]:
    """The committed-graph part of a materialized payload (drops author/notes)."""
    return payload["nodes"], payload["edges"]


def strip_record_ids(value: _T) -> _T:
    """A comparable copy of a report body with provenance pointers removed.

    Every graph commit writes fresh provenance rows (WP1's per-version
    model), so an evidence citation's ``record_id`` points into the pinned
    version rather than naming a claim. Recovery comparisons keep the
    claims (kind, text, source) and drop the pointers.
    """
    stripped: object = value
    if isinstance(stripped, dict):
        stripped = {
            key: strip_record_ids(item) for key, item in stripped.items() if key != "record_id"
        }
    elif isinstance(stripped, list):
        stripped = [strip_record_ids(item) for item in stripped]
    return cast(_T, stripped)


def _facility_index(facility: FacilityCorpus) -> int:
    """The facility's generation index, recovered from its id suffix."""
    match = re.search(r"(\d+)$", facility.facility_id)
    return int(match.group(1)) if match else 0


def _event_json(event: SimEvent) -> dict[str, object]:
    """One event as DSL-style JSON (the repro's replay list)."""
    if isinstance(event, FaultEvent):
        return {"event": "fault", "asset_key": event.asset_key}
    if isinstance(event, RepairEvent):
        return {"event": "repair", "asset_key": event.asset_key}
    if isinstance(event, RedundancyChangeEvent):
        return {
            "event": "redundancy_change",
            "member_group": event.member_group,
            "redundancy_config": event.redundancy_config,
        }
    if isinstance(event, LoadShiftEvent):
        return {
            "event": "load_shift",
            "consumer": event.consumer,
            "from_node": event.from_node,
            "to_node": event.to_node,
        }
    raise SimEventError(f"unrecognized event {event!r}")


def repro_recipe(facility: FacilityCorpus, seed: int, events: Sequence[SimEvent]) -> str:
    """Minimal repro: how to regenerate the facility and what to replay."""
    index = _facility_index(facility)
    event_list = json.dumps(
        [_event_json(event) for event in events], sort_keys=True, separators=(",", ":")
    )
    return (
        f"generate_segment_corpus({facility.segment!r}, seed={seed}, "
        f"n_facilities={index + 1})[-1] (facility {facility.facility_id}); "
        f"replay events: {event_list}"
    )


# ---------------------------------------------------------------------------
# Category 3: monotonicity, repair, determinism
# ---------------------------------------------------------------------------


def check_monotonicity(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 3a: added redundancy never increases exposure.

    For every upgradeable group, every upgrade target, and the intact state
    plus every single-asset fault: the upgraded dead set must be a subset of
    the original's, and the standby itself must never die.
    """
    topology = Topology.from_payload(facility.graph)
    node_keys = frozenset(topology.nodes)
    asset_keys = sorted(
        key for key, spec in topology.nodes.items() if spec.node_type == NODE_TYPE_ASSET
    )
    fault_sets = [frozenset(), *(frozenset({key}) for key in asset_keys)]

    violations: list[InvariantViolation] = []
    checks = 0
    for group, current in upgradeable_groups(topology):
        for target in UPGRADE_TARGETS[current]:
            event = build_upgrade(topology, group, target)
            upgraded = apply_event(initial_state(topology), event).topology
            standby_key = str(event.standby_node["node_key"])
            fault_events = [FaultEvent(asset_key=key) for key in sorted(asset_keys)]
            for faulted in fault_sets:
                checks += 1
                before = dead_nodes(SimState(topology, faulted))
                after = dead_nodes(SimState(upgraded, faulted))
                if (after & node_keys) <= before and standby_key not in after:
                    continue
                gained = sorted((after & node_keys) - before)
                fault_prefix = [fault for fault in fault_events if fault.asset_key in faulted]
                violations.append(
                    InvariantViolation(
                        invariant="redundancy_monotonicity",
                        seed=seed,
                        segment=facility.segment,
                        facility_id=facility.facility_id,
                        detail=(
                            f"upgrading group {group!r} to {target} (standby {standby_key}) "
                            f"with faulted {sorted(faulted)}: dead set grew by {gained} "
                            f"(before {sorted(before)}, after {sorted(after)})"
                        ),
                        repro=repro_recipe(facility, seed, [*fault_prefix, event]),
                    )
                )
    return InvariantResult("redundancy_monotonicity", checks, tuple(violations))


def check_repair_recovery(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 3b: repair never re-exposes; full repair restores the intact graph.

    Walks the seeded event sequence: after every repair event the dead set
    must be a subset of the pre-repair one. Then repairing every remaining
    event fault must return the working topology to its intact state — no
    stale exposure, no torn edges (upgrades persist; faults must not).
    """
    topology = Topology.from_payload(facility.graph)
    events = seeded_events(facility, seed)
    violations: list[InvariantViolation] = []
    checks = 0

    state = initial_state(topology)
    dead_previous = dead_nodes(state)
    outstanding: set[str] = set()
    for index, event in enumerate(events, start=1):
        state = apply_event(state, event)
        dead = dead_nodes(state)
        if isinstance(event, RepairEvent):
            checks += 1
            if not dead <= dead_previous:
                reexposed = sorted(dead - dead_previous)
                violations.append(
                    InvariantViolation(
                        invariant="repair_recovery",
                        seed=seed,
                        segment=facility.segment,
                        facility_id=facility.facility_id,
                        detail=(
                            f"repair {event.asset_key} re-exposed {reexposed} "
                            f"(dead before {sorted(dead_previous)}, after {sorted(dead)})"
                        ),
                        repro=repro_recipe(facility, seed, events[:index]),
                    )
                )
            outstanding.discard(event.asset_key)
        elif isinstance(event, FaultEvent):
            outstanding.add(event.asset_key)
        dead_previous = dead

    restoration = [RepairEvent(asset_key=key) for key in sorted(outstanding)]
    if restoration:
        checks += 1
        final = state
        for event in restoration:
            final = apply_event(final, event)
        intact = initial_state(final.topology)
        final_dead = dead_nodes(final)
        final_payload = _snapshot(final, len(events) + len(restoration), None)
        intact_payload = _snapshot(intact, 0, None)
        if final_dead or _graph(final_payload) != _graph(intact_payload):
            violations.append(
                InvariantViolation(
                    invariant="repair_recovery",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"full repair does not restore the intact graph: dead services "
                        f"{sorted(final_dead & final.topology.service_keys())}, "
                        f"snapshot drift {_graph(final_payload) != _graph(intact_payload)}"
                    ),
                    repro=repro_recipe(facility, seed, [*events, *restoration]),
                )
            )
    return InvariantResult("repair_recovery", checks, tuple(violations))


def check_trace_determinism(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 3c: same seed regenerates the facility and replays identically."""
    violations: list[InvariantViolation] = []
    checks = 0

    index = _facility_index(facility)
    regenerated = generate_segment_corpus(facility.segment, seed, index + 1)[-1]
    checks += 1
    if regenerated != facility:
        violations.append(
            InvariantViolation(
                invariant="trace_determinism",
                seed=seed,
                segment=facility.segment,
                facility_id=facility.facility_id,
                detail="same seed does not regenerate the facility identically",
                repro=repro_recipe(facility, seed, ()),
            )
        )

    events = seeded_events(facility, seed)
    checks += 1
    first = payload_trace(Topology.from_payload(facility.graph), events)
    second = payload_trace(Topology.from_payload(facility.graph), events)
    if first != second:
        step = next(
            (i for i, (a, b) in enumerate(zip(first, second, strict=False)) if a != b),
            -1,
        )
        violations.append(
            InvariantViolation(
                invariant="trace_determinism",
                seed=seed,
                segment=facility.segment,
                facility_id=facility.facility_id,
                detail=f"trace step {step} differs between two same-seed runs",
                repro=repro_recipe(facility, seed, events),
            )
        )
    return InvariantResult("trace_determinism", checks, tuple(violations))


# ---------------------------------------------------------------------------
# Category 3 over what-if interventions (M3): counterfactual properties
# ---------------------------------------------------------------------------


def _seeded_interventions(
    facility: FacilityCorpus, seed: int
) -> tuple[SimState, tuple[WhatIfIntervention, ...]]:
    """The seeded post-sequence state plus its counterfactual interventions.

    Same RNG discipline as ``seeded_events`` (``random.Random`` over a
    namespaced string seed): replay the seeded event sequence, then derive
    one upgrade of a random upgradeable group (random target) and up to two
    removals of live assets. Branches apply these to the post-sequence
    state — never the timeline, never the intact baseline alone.
    """
    rng = random.Random(f"{facility.facility_id}:whatif:{seed}")
    topology = Topology.from_payload(facility.graph)
    state = initial_state(topology)
    for event in seeded_events(facility, seed):
        state = apply_event(state, event)
    interventions: list[WhatIfIntervention] = []
    groups = upgradeable_groups(state.topology)
    if groups:
        group, current = rng.choice(groups)
        interventions.append(
            build_upgrade(state.topology, group, rng.choice(UPGRADE_TARGETS[current]))
        )
    live_assets = sorted(
        key
        for key, spec in state.topology.nodes.items()
        if spec.node_type == NODE_TYPE_ASSET and key not in state.faulted
    )
    for key in rng.sample(live_assets, min(2, len(live_assets))):
        interventions.append(RemoveAssetEvent(asset_key=key))
    return state, tuple(interventions)


def _whatif_repro(
    facility: FacilityCorpus, seed: int, intervention: WhatIfIntervention
) -> str:
    """Minimal repro for a what-if violation: the seeded state, then the
    intervention, in DSL-style JSON (removals are what_if markers)."""
    base = repro_recipe(facility, seed, ())
    if isinstance(intervention, RemoveAssetEvent):
        ix_json = {"event": "what_if", "type": "remove_asset", "asset_key": intervention.asset_key}
    else:
        ix_json = {
            "event": "what_if",
            "type": "upgrade_redundancy",
            "member_group": intervention.member_group,
            "redundancy_config": intervention.redundancy_config,
        }
    return f"{base}; then apply what-if intervention: {json.dumps(ix_json, sort_keys=True)}"


def check_whatif_counterfactual(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 3 properties over what-if interventions (M3).

    From the seeded post-sequence state (its own fault set and any seeded
    upgrades included), every intervention applies on a branch:

    - an upgrade's counterfactual dead set is a subset of the baseline's —
      from the state itself and from every single additional fault on top —
      and the standby never dies (monotonicity, multi-fault states);
    - a removal's dead set equals the same asset's fault cascade (the
      counterfactual loss propagates exactly like a fault's);
    - every branch's materialized snapshot replays identically (seeded and
      replayable, as the Category 5 runner gates require).
    """
    violations: list[InvariantViolation] = []
    checks = 0
    state, interventions = _seeded_interventions(facility, seed)
    base_dead = dead_nodes(state)
    node_keys = frozenset(state.topology.nodes)

    additional_faults = [
        frozenset({key})
        for key in sorted(state.topology.nodes)
        if key not in state.faulted and state.topology.nodes[key].node_type == NODE_TYPE_ASSET
    ]

    for intervention in interventions:
        cf_state = apply_intervention(state, intervention)
        cf_dead = dead_nodes(cf_state)
        repro = _whatif_repro(facility, seed, intervention)

        if isinstance(intervention, RedundancyChangeEvent):
            standby_key = str(intervention.standby_node["node_key"])
            gained = sorted((cf_dead & node_keys) - base_dead)
            checks += 1
            if gained or standby_key in cf_dead:
                violations.append(
                    InvariantViolation(
                        invariant="whatif_counterfactual",
                        seed=seed,
                        segment=facility.segment,
                        facility_id=facility.facility_id,
                        detail=(
                            f"what-if upgrade {describe_intervention(intervention)} "
                            f"(standby {standby_key}) from the seeded state: dead set "
                            f"grew by {gained} (before {sorted(base_dead)}, "
                            f"after {sorted(cf_dead)})"
                        ),
                        repro=repro,
                    )
                )
            for extra in additional_faults:
                faulted = state.faulted | extra
                before = dead_nodes(SimState(state.topology, faulted))
                after = dead_nodes(SimState(cf_state.topology, faulted))
                checks += 1
                gained = sorted((after & node_keys) - before)
                if gained or standby_key in after:
                    violations.append(
                        InvariantViolation(
                            invariant="whatif_counterfactual",
                            seed=seed,
                            segment=facility.segment,
                            facility_id=facility.facility_id,
                            detail=(
                                f"what-if upgrade {describe_intervention(intervention)} with "
                                f"faulted {sorted(faulted)}: dead set grew by {gained} "
                                f"(before {sorted(before)}, after {sorted(after)})"
                            ),
                            repro=repro,
                        )
                    )

        if isinstance(intervention, RemoveAssetEvent):
            faulted_with_key = state.faulted | {intervention.asset_key}
            fault_equivalent = dead_nodes(SimState(state.topology, faulted_with_key))
            checks += 1
            if cf_dead != fault_equivalent:
                violations.append(
                    InvariantViolation(
                        invariant="whatif_counterfactual",
                        seed=seed,
                        segment=facility.segment,
                        facility_id=facility.facility_id,
                        detail=(
                            f"what-if removal of {intervention.asset_key}: dead set "
                            f"{sorted(cf_dead)} differs from the same asset's fault "
                            f"cascade {sorted(fault_equivalent)}"
                        ),
                        repro=repro,
                    )
                )

        # Seeded and replayable: the branch materialization is a pure
        # function of (state, intervention) — two renders are identical.
        checks += 1
        cf_dead_two = dead_nodes(apply_intervention(state, intervention))
        first_payload = materialize(
            cf_state,
            cf_dead,
            author=SIM_AUTHOR,
            notes="whatif determinism probe",
        )
        second_payload = materialize(
            cf_state,
            cf_dead_two,
            author=SIM_AUTHOR,
            notes="whatif determinism probe",
        )
        if _graph(first_payload) != _graph(second_payload):
            violations.append(
                InvariantViolation(
                    invariant="whatif_counterfactual",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"what-if {describe_intervention(intervention)}: two same-seed "
                        "branch materializations differ"
                    ),
                    repro=repro,
                )
            )

    return InvariantResult("whatif_counterfactual", checks, tuple(violations))


# ---------------------------------------------------------------------------
# Category 4: replay folds, non-clairvoyance, recovery state
# ---------------------------------------------------------------------------


def check_replay_folds(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 4: replay folds held to the existing splits discipline.

    Builds the same folds the eval harness builds (``evals.splits`` — the
    LeakageError mechanism, not a parallel one) and asserts both invariants:
    the feature window admits no post-fault event (``feature_events`` raises
    on post-repair leakage), and the label window contains the fault's work
    order and its repair.
    """
    violations: list[InvariantViolation] = []
    checks = 0
    folds = build_folds(facility.replay_cases, FEATURE_WINDOW)
    for case, fold in zip(facility.replay_cases, folds, strict=True):
        episode: tuple[SimEvent, ...] = (
            FaultEvent(asset_key=case.asset_key),
            RepairEvent(asset_key=case.asset_key),
        )
        repro = repro_recipe(facility, seed, episode)

        checks += 1
        try:
            features = feature_events(facility.events, fold)
        except LeakageError as error:
            violations.append(
                InvariantViolation(
                    invariant="replay_no_leakage",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=f"feature_events raised for case {case.case_id}: {error}",
                    repro=repro,
                )
            )
            continue
        leaked = sorted(
            event.event_id for event in features if event.occurred_at >= fold.fault_time
        )
        if leaked:
            violations.append(
                InvariantViolation(
                    invariant="replay_no_leakage",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: feature window admits post-fault events {leaked}"
                    ),
                    repro=repro,
                )
            )

        checks += 1
        label_kinds = sorted(event.kind for event in label_window_events(facility.events, fold))
        if "work_order" not in label_kinds or "repair" not in label_kinds:
            violations.append(
                InvariantViolation(
                    invariant="replay_no_leakage",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: label window [{fold.fault_time}, "
                        f"{fold.repair_time}] lacks the fault/repair events "
                        f"(found kinds {label_kinds})"
                    ),
                    repro=repro,
                )
            )
    return InvariantResult("replay_no_leakage", checks, tuple(violations))


def check_replay_recovery(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 4: pre-repair states are a function of pre-repair events only.

    Replays each replay case's episode twice — with and without the repair —
    and asserts the pre-repair steps are identical (truncation invariance):
    a state built without knowledge of the future repair cannot differ from
    one built with it.
    """
    topology = Topology.from_payload(facility.graph)
    violations: list[InvariantViolation] = []
    checks = 0
    for case in facility.replay_cases:
        episode: tuple[SimEvent, ...] = (
            FaultEvent(asset_key=case.asset_key),
            RepairEvent(asset_key=case.asset_key),
        )
        full = payload_trace(topology, episode)
        truncated = payload_trace(topology, episode[:1])
        checks += 1
        if full[:2] != truncated:
            step = next(
                (i for i, (a, b) in enumerate(zip(full, truncated, strict=False)) if a != b),
                -1,
            )
            violations.append(
                InvariantViolation(
                    invariant="replay_no_clairvoyance",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: pre-repair trace step {step} changes when "
                        f"the future repair is added to the replay input"
                    ),
                    repro=repro_recipe(facility, seed, episode),
                )
            )
    return InvariantResult("replay_no_clairvoyance", checks, tuple(violations))


def check_recovery_state(facility: FacilityCorpus, seed: int) -> InvariantResult:
    """Category 4 (engine level): post-repair reflects restored redundancy.

    Per replay case: the constructed interruption is visible in the faulted
    (pre-repair) state, and the post-repair snapshot equals the intact
    baseline — no stale exposure, no torn edges.
    """
    topology = Topology.from_payload(facility.graph)
    violations: list[InvariantViolation] = []
    checks = 0
    baseline = payload_trace(topology, ())[0]
    for case in facility.replay_cases:
        episode: tuple[SimEvent, ...] = (
            FaultEvent(asset_key=case.asset_key),
            RepairEvent(asset_key=case.asset_key),
        )
        trace = payload_trace(topology, episode)
        repro = repro_recipe(facility, seed, episode)

        checks += 1
        dead_faulted = dead_nodes(SimState(topology, frozenset({case.asset_key})))
        dead_services = dead_faulted & topology.service_keys()
        if case.interrupted_service not in dead_services:
            violations.append(
                InvariantViolation(
                    invariant="recovery_state",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: constructed interruption "
                        f"{case.interrupted_service!r} is not exposed in the "
                        f"pre-repair state (dead services {sorted(dead_services)})"
                    ),
                    repro=repro,
                )
            )

        checks += 1
        if _graph(trace[2]) != _graph(baseline):
            violations.append(
                InvariantViolation(
                    invariant="recovery_state",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: post-repair snapshot differs from the "
                        f"intact baseline (stale exposure or torn edges)"
                    ),
                    repro=repro,
                )
            )
    return InvariantResult("recovery_state", checks, tuple(violations))


def check_recovery_via_api(
    client: TestClient, facility: FacilityCorpus, seed: int
) -> InvariantResult:
    """Category 4 through the real assessor: fault, assess, repair, re-assess.

    Drives the same TestClient route the eval harness uses: the constructed
    interruption must be visible in the pre-repair report, and the
    post-repair report must equal the baseline report except for the pinned
    version number (no stale exposure in any assessed field).
    """
    baseline = load_facility(client, facility)
    topology = Topology.from_payload(facility.graph)
    violations: list[InvariantViolation] = []
    checks = 0
    for case in facility.replay_cases:
        repro_events: tuple[SimEvent, ...] = (
            FaultEvent(asset_key=case.asset_key),
            RepairEvent(asset_key=case.asset_key),
        )
        repro = repro_recipe(facility, seed, repro_events)

        faulted_state = SimState(topology, frozenset({case.asset_key}))
        fault_payload = materialize(
            faulted_state,
            dead_nodes(faulted_state),
            author=SIM_AUTHOR,
            notes=f"sim invariant: fault {case.asset_key} (pre-repair state)",
        )
        _, fault_report = _commit_and_assess(client, facility.facility_id, fault_payload)
        checks += 1
        if case.interrupted_service not in lost_services(fault_report):
            violations.append(
                InvariantViolation(
                    invariant="recovery_state",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: constructed interruption "
                        f"{case.interrupted_service!r} missing from the pre-repair "
                        f"report (observed unserved {sorted(lost_services(fault_report))})"
                    ),
                    repro=repro,
                )
            )

        intact = SimState(topology, frozenset())
        repair_payload = materialize(
            intact,
            dead_nodes(intact),
            author=SIM_AUTHOR,
            notes="sim invariant: post-repair state",
        )
        _, repaired_report = _commit_and_assess(client, facility.facility_id, repair_payload)
        checks += 1
        body_keys = (set(baseline) | set(repaired_report)) - {"graph_version"}
        baseline_body = strip_record_ids(baseline)
        repaired_body = strip_record_ids(repaired_report)
        differing = sorted(
            key for key in body_keys if baseline_body.get(key) != repaired_body.get(key)
        )
        if differing:
            violations.append(
                InvariantViolation(
                    invariant="recovery_state",
                    seed=seed,
                    segment=facility.segment,
                    facility_id=facility.facility_id,
                    detail=(
                        f"case {case.case_id}: post-repair report differs from baseline "
                        f"in fields {differing} (stale exposure)"
                    ),
                    repro=repro,
                )
            )
    return InvariantResult("recovery_state", checks, tuple(violations))
