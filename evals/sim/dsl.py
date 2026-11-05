"""Sequenced-scenario DSL: adjudicated event sequences with expected traces.

Extends the static fault-injection scenario style (``evals/scenarios/``)
with an ordered ``events`` list and a per-step ``expected_trace``. Same
topology payload shape, same adjudication discipline — expectations come
from construction intent, never from running the assessor. A scenario is a
named, reviewable regression gate: Category 1 (failover semantics) and
Category 2 (cascade dynamics) of the simulation testing plan, hand-checked
step by step.

The event union is closed and validated by kind:

    {"event": "fault",             "asset_key": "COMP-1-1"}
    {"event": "repair",            "asset_key": "COMP-1-1"}
    {"event": "redundancy_change", "member_group": "rack_compressors-1",
     "redundancy_config": "N+1", "standby_node": {...}, "standby_edge": {...}}
    {"event": "load_shift",        "consumer": "SVC-REFRIG-0",
     "from_node": "RACK-1", "to_node": "RACK-2"}
    {"event": "what_if",           "label": "remove-condenser",
     "intervention": {"type": "remove_asset", "asset_key": "COND-1"},
     "expected_delta": {...}}

Array order is the execution order; M1 models no wall-clock timing (failover
transfer durations are a WP4 physics question, per the plan).

What-if markers (M3, Category 5) do not advance the timeline: a marker saves
the state at its position, applies its ``intervention`` to a branch copy,
assesses both through the real API, and the observed assessment delta is
compared to the marker's ``expected_delta`` — declared fields exact, same
partial-map discipline as the trace steps. Interventions: ``remove_asset``
(the component leaves the facility; its loss cascades like a fault) and
``upgrade_redundancy`` (the N → N+1 upgrade — reclassify plus a standby
member, the timeline's own transition). An upgrade marker additionally
_carries a structural verdict_: every live group member that was a SPOF of
the fed service at baseline must have lost that finding in the counterfactual,
or the evaluator abstains with a stated reason (accepted; a residual SPOF is
a silent wrong answer and fails). Unknown kinds are rejected with the
supported list named so a forward-ported file fails loudly instead of
silently doing less.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.corpus.common import SEGMENTS
from evals.sim.engine import (
    VALID_CONFIGS,
    FaultEvent,
    LoadShiftEvent,
    RedundancyChangeEvent,
    RemoveAssetEvent,
    RepairEvent,
    SimDslError,
    SimError,
    SimEvent,
    Topology,
    apply_event,
    apply_intervention,
    describe_intervention,
    initial_state,
)

SIM_SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
FACILITY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")

# The closed event union — what_if markers joined it in M3.
EVENT_KINDS = ("fault", "repair", "redundancy_change", "load_shift", "what_if")

INTERVENTION_TYPES = ("remove_asset", "upgrade_redundancy")


@dataclass(frozen=True)
class ExpectedStep:
    """Adjudicated assertions for one trace step.

    ``lost_services`` must match exactly (a tolerance would defeat the
    gate). The per-service maps are partial: they pin only the services the
    adjudication names, and every named key must exist in the topology.
    """

    step: int
    lost_services: frozenset[str]
    service_first_fault_tolerated: dict[str, bool]
    service_spofs: dict[str, frozenset[str]]


@dataclass(frozen=True)
class ExpectedDelta:
    """Adjudicated counterfactual delta for one what-if marker (partial).

    Same discipline as ``ExpectedStep``: declared fields match exactly, and
every named key must exist. ``service_first_fault_tolerated`` declares the
counterfactual's absolute value per service (the state the graph should be
in after the intervention), not a from/to pair. ``expect_spof_path_abstention``
pins the SPOF-path verdict's shape when the adjudication says the verdict
cannot be established (``True`` — the evaluator must abstain) or must be
established (``False``); ``None`` leaves it unpinned and only recorded.
    """

    lost_services: frozenset[str]
    restored_services: frozenset[str]
    service_spofs_added: dict[str, frozenset[str]]
    service_spofs_removed: dict[str, frozenset[str]]
    service_first_fault_tolerated: dict[str, bool]
    expect_spof_path_abstention: bool | None


@dataclass(frozen=True)
class WhatIfMarker:
    """One counterfactual marker: intervention plus adjudicated delta.

    The marker does not advance the timeline; its baseline is the state at
    its position in the event list.
    """

    label: str
    intervention: RemoveAssetEvent | RedundancyChangeEvent
    expected_delta: ExpectedDelta

    @property
    def described(self) -> str:
        return describe_intervention(self.intervention)


@dataclass(frozen=True)
class _RawWhatIf:
    """Phase-1 carrier for a what_if event: intervention parsed, delta raw.

    Delta validation needs the scenario-wide asset-key set (base assets plus
    every standby any event introduces, including later ones), so markers
    are finalized in a second pass once all events are parsed.
    """

    label: str
    intervention: RemoveAssetEvent | RedundancyChangeEvent
    raw_delta: dict[str, Any]


@dataclass(frozen=True)
class SimScenario:
    """One sequenced scenario: topology, ordered events, adjudicated trace."""

    scenario_id: str
    title: str
    description: str
    market: str
    facility_id: str
    topology_payload: dict[str, object]
    events: tuple[SimEvent | WhatIfMarker, ...]
    expected_trace: tuple[ExpectedStep, ...]
    adjudicated_by: str

    @property
    def topology(self) -> Topology:
        """The parsed base topology (step 0's snapshot)."""
        return Topology.from_payload(self.topology_payload)


def load_sim_scenarios(directory: Path = SIM_SCENARIO_DIR) -> list[SimScenario]:
    """Every sequenced scenario, sorted by id (deterministic run order)."""
    scenarios = [
        parse_scenario(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]
    return sorted(scenarios, key=lambda scenario: scenario.scenario_id)


def parse_scenario(raw: dict[str, Any]) -> SimScenario:
    """Parse and validate one sequenced-scenario payload."""
    scenario_id = _required_str(raw, "scenario_id")
    title = _required_str(raw, "title")
    description = _required_str(raw, "description")
    market = _required_str(raw, "market")
    facility_id = _required_str(raw, "facility_id")
    adjudicated_by = _required_str(raw, "adjudicated_by")
    if market not in SEGMENTS:
        raise SimDslError(f"{scenario_id}: market {market!r} is not one of {SEGMENTS}")
    if not FACILITY_ID_PATTERN.fullmatch(facility_id):
        raise SimDslError(f"{scenario_id}: facility_id {facility_id!r} fails the API pattern")
    if not facility_id.startswith("sim-"):
        raise SimDslError(
            f"{scenario_id}: facility_id {facility_id!r} must use the sim- namespace "
            "(simulator-created facilities stay distinguishable from corpus ids)"
        )

    topology_payload = raw.get("topology")
    if not isinstance(topology_payload, dict):
        raise SimDslError(f"{scenario_id}: 'topology' must be an object")
    topology = Topology.from_payload(topology_payload)

    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise SimDslError(f"{scenario_id}: 'events' must be a non-empty list")
    parsed_events = tuple(
        _parse_event(scenario_id, index, raw_event, topology)
        for index, raw_event in enumerate(raw_events)
    )
    asset_keys = _scenario_asset_keys(parsed_events, topology)
    events = tuple(
        _finalize_event(scenario_id, event, topology, asset_keys)
        for event in parsed_events
    )
    _require_distinct_labels(scenario_id, events)

    expected = raw.get("expected_trace")
    if not isinstance(expected, list):
        raise SimDslError(f"{scenario_id}: 'expected_trace' must be a list")
    if len(expected) != len(events) + 1:
        raise SimDslError(
            f"{scenario_id}: expected_trace has {len(expected)} steps for "
            f"{len(events)} events — expected {len(events) + 1} (baseline + one per event)"
        )
    trace = tuple(
        _parse_step(scenario_id, index, raw_step, topology, asset_keys)
        for index, raw_step in enumerate(expected)
    )

    scenario = SimScenario(
        scenario_id=scenario_id,
        title=title,
        description=description,
        market=market,
        facility_id=facility_id,
        topology_payload=topology_payload,
        events=events,
        expected_trace=trace,
        adjudicated_by=adjudicated_by,
    )
    check_state_legality(scenario)
    return scenario


def _required_str(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise SimDslError(f"scenario field {field!r} must be a non-empty string")
    return value


def _parse_event(
    scenario_id: str, index: int, raw_event: object, topology: Topology
) -> SimEvent | _RawWhatIf:
    if not isinstance(raw_event, dict):
        raise SimDslError(f"{scenario_id}: event {index} is not an object")
    kind = raw_event.get("event")
    if kind not in EVENT_KINDS:
        raise SimDslError(
            f"{scenario_id}: event {index} has unknown kind {kind!r} — supported kinds: "
            f"{', '.join(EVENT_KINDS)}"
        )
    # Reference checks run here, against the base topology, so a scenario
    # file with a typo'd key fails at load; state-level legality (double
    # fault, repairing an unfaulted asset, a marker whose intervention is
    # inapplicable at its position) is checked against the walked state in
    # check_state_legality, at load time too.
    match kind:
        case "fault":
            key = _ref(raw_event, "asset_key", scenario_id, index)
            _asset_exists(topology, key, scenario_id, index, "fault")
            return FaultEvent(asset_key=key)
        case "repair":
            key = _ref(raw_event, "asset_key", scenario_id, index)
            _asset_exists(topology, key, scenario_id, index, "repair")
            return RepairEvent(asset_key=key)
        case "redundancy_change":
            group = _ref(raw_event, "member_group", scenario_id, index)
            if not any(edge.member_group == group for edge in topology.edges):
                raise SimDslError(
                    f"{scenario_id}: event {index} targets unknown member group {group!r}"
                )
            config = raw_event.get("redundancy_config")
            if not isinstance(config, str):
                raise SimDslError(
                    f"{scenario_id}: event {index} needs a redundancy_config string"
                )
            standby_node = raw_event.get("standby_node")
            standby_edge = raw_event.get("standby_edge")
            if not isinstance(standby_node, dict) or not isinstance(standby_edge, dict):
                raise SimDslError(
                    f"{scenario_id}: event {index} needs standby_node and standby_edge objects"
                )
            return RedundancyChangeEvent(
                member_group=group,
                redundancy_config=config,
                standby_node=standby_node,
                standby_edge=standby_edge,
            )
        case "load_shift":
            consumer = _ref(raw_event, "consumer", scenario_id, index)
            from_node = _ref(raw_event, "from_node", scenario_id, index)
            to_node = _ref(raw_event, "to_node", scenario_id, index)
            for key in (consumer, to_node):
                if key not in topology.nodes:
                    raise SimDslError(
                        f"{scenario_id}: event {index} references unknown node {key!r}"
                    )
            if not any(
                edge.from_node == from_node and edge.to_node == consumer
                for edge in topology.edges
            ):
                raise SimDslError(
                    f"{scenario_id}: event {index} shifts dependency "
                    f"{from_node!r} -> {consumer!r}, which the topology does not declare"
                )
            return LoadShiftEvent(consumer=consumer, from_node=from_node, to_node=to_node)
        case "what_if":
            return _parse_what_if(scenario_id, index, raw_event, topology)
    raise SimDslError(f"{scenario_id}: event {index} kind {kind!r} not parsed")  # pragma: no cover


def _finalize_event(
    scenario_id: str,
    event: SimEvent | _RawWhatIf,
    topology: Topology,
    asset_keys: frozenset[str],
) -> SimEvent | WhatIfMarker:
    """Phase 2: finalize what-if markers (parse their deltas); pass through
    timeline events untouched."""
    if not isinstance(event, _RawWhatIf):
        return event
    return WhatIfMarker(
        label=event.label,
        intervention=event.intervention,
        expected_delta=_parse_expected_delta(
            scenario_id, event.label, event.raw_delta, topology.service_keys(), asset_keys
        ),
    )


def _require_distinct_labels(
    scenario_id: str, events: tuple[SimEvent | WhatIfMarker, ...]
) -> None:
    """Marker labels name the adjudicated case — duplicates would make the
    report's per-marker rows ambiguous."""
    labels = [event.label for event in events if isinstance(event, WhatIfMarker)]
    if len(labels) != len(set(labels)):
        raise SimDslError(f"{scenario_id}: what_if labels must be distinct (got {labels})")


def _parse_what_if(
    scenario_id: str, index: int, raw_event: dict[str, Any], topology: Topology
) -> _RawWhatIf:
    label = _ref(raw_event, "label", scenario_id, index)
    raw_intervention = raw_event.get("intervention")
    if not isinstance(raw_intervention, dict):
        raise SimDslError(f"{scenario_id}: what_if {label!r} needs an intervention object")
    ix_type = raw_intervention.get("type")
    if ix_type not in INTERVENTION_TYPES:
        raise SimDslError(
            f"{scenario_id}: what_if {label!r} has unknown intervention type {ix_type!r} — "
            f"supported types: {', '.join(INTERVENTION_TYPES)}"
        )
    intervention: RemoveAssetEvent | RedundancyChangeEvent
    match ix_type:
        case "remove_asset":
            key = _ref(raw_intervention, "asset_key", scenario_id, index)
            _asset_exists(topology, key, scenario_id, index, "remove")
            intervention = RemoveAssetEvent(asset_key=key)
        case "upgrade_redundancy":
            # Same payload shape and validation as the timeline's
            # redundancy_change — a what-if upgrade is that transition,
            # applied to a branch state.
            group = _ref(raw_intervention, "member_group", scenario_id, index)
            if not any(edge.member_group == group for edge in topology.edges):
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} targets unknown member group {group!r}"
                )
            config = raw_intervention.get("redundancy_config")
            if not isinstance(config, str) or config not in VALID_CONFIGS:
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} needs a redundancy_config in "
                    f"{sorted(VALID_CONFIGS)}"
                )
            standby_node = raw_intervention.get("standby_node")
            standby_edge = raw_intervention.get("standby_edge")
            if not isinstance(standby_node, dict) or not isinstance(standby_edge, dict):
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} needs standby_node and standby_edge objects"
                )
            intervention = RedundancyChangeEvent(
                member_group=group,
                redundancy_config=config,
                standby_node=standby_node,
                standby_edge=standby_edge,
            )
        case _:
            raise SimDslError(
                f"{scenario_id}: what_if {label!r} has unknown intervention type "
                f"{ix_type!r} — supported types: {', '.join(INTERVENTION_TYPES)}"
            )
    raw_delta = raw_event.get("expected_delta")
    if not isinstance(raw_delta, dict):
        raise SimDslError(f"{scenario_id}: what_if {label!r} needs an expected_delta object")
    return _RawWhatIf(label=label, intervention=intervention, raw_delta=raw_delta)


def _parse_expected_delta(
    scenario_id: str,
    label: str,
    raw_delta: dict[str, Any],
    services: frozenset[str],
    asset_keys: frozenset[str],
) -> ExpectedDelta:
    """Validate one marker's adjudicated delta (partial maps, keys exist)."""

    def _service_set(field: str) -> frozenset[str]:
        value = raw_delta.get(field, [])
        if not isinstance(value, list) or not all(isinstance(key, str) for key in value):
            raise SimDslError(f"{scenario_id}: what_if {label!r} {field} must be a list of keys")
        unknown = set(value) - services
        if unknown:
            raise SimDslError(
                f"{scenario_id}: what_if {label!r} {field} names non-service keys {sorted(unknown)}"
            )
        return frozenset(value)

    def _spof_map(field: str) -> dict[str, frozenset[str]]:
        raw_map = raw_delta.get(field, {})
        if not isinstance(raw_map, dict):
            raise SimDslError(f"{scenario_id}: what_if {label!r} {field} must be an object")
        parsed: dict[str, frozenset[str]] = {}
        for service_key, spof_keys in raw_map.items():
            if service_key not in services:
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} {field} names unknown service "
                    f"{service_key!r}"
                )
            if not isinstance(spof_keys, list) or not all(
                isinstance(item, str) for item in spof_keys
            ):
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} {field} {service_key} expects a list of keys"
                )
            unknown_spofs = set(spof_keys) - asset_keys
            if unknown_spofs:
                raise SimDslError(
                    f"{scenario_id}: what_if {label!r} {field} {service_key} names unknown "
                    f"SPOF assets {sorted(unknown_spofs)}"
                )
            parsed[service_key] = frozenset(spof_keys)
        return parsed

    fft = raw_delta.get("service_first_fault_tolerated", {})
    if not isinstance(fft, dict):
        raise SimDslError(
            f"{scenario_id}: what_if {label!r} service_first_fault_tolerated must be an object"
        )
    fft_parsed: dict[str, bool] = {}
    for service_key, value in fft.items():
        if service_key not in services:
            raise SimDslError(
                f"{scenario_id}: what_if {label!r} names unknown service {service_key!r}"
            )
        if not isinstance(value, bool):
            raise SimDslError(
                f"{scenario_id}: what_if {label!r} {service_key} expects a boolean"
            )
        fft_parsed[service_key] = value

    abstain = raw_delta.get("expect_spof_path_abstention")
    if abstain is not None and not isinstance(abstain, bool):
        raise SimDslError(
            f"{scenario_id}: what_if {label!r} expect_spof_path_abstention must be a boolean"
        )
    return ExpectedDelta(
        lost_services=_service_set("lost_services"),
        restored_services=_service_set("restored_services"),
        service_spofs_added=_spof_map("service_spofs_added"),
        service_spofs_removed=_spof_map("service_spofs_removed"),
        service_first_fault_tolerated=fft_parsed,
        expect_spof_path_abstention=abstain,
    )


def _ref(raw_event: dict[str, Any], field: str, scenario_id: str, index: int) -> str:
    value = raw_event.get(field)
    if not isinstance(value, str) or not value:
        raise SimDslError(
            f"{scenario_id}: event {index} ({raw_event.get('event')}) needs field {field!r}"
        )
    return value


def _asset_exists(
    topology: Topology, key: str, scenario_id: str, index: int, kind: str
) -> None:
    node = topology.nodes.get(key)
    if node is None:
        raise SimDslError(f"{scenario_id}: event {index} {kind}s unknown asset {key!r}")
    if node.node_type != "asset":
        raise SimDslError(
            f"{scenario_id}: event {index} {kind}s {key!r}, which is a service node"
        )


def _scenario_asset_keys(
    events: tuple[SimEvent | _RawWhatIf, ...], topology: Topology
) -> frozenset[str]:
    """Every asset a trace step or delta may legitimately name: base assets
    plus the standbys the scenario's own upgrade events and what-if upgrade
    markers introduce (a post-upgrade SPOF may be a node the base topology
    never had)."""
    keys = {key for key, spec in topology.nodes.items() if spec.node_type == "asset"}
    for event in events:
        if isinstance(event, RedundancyChangeEvent):
            candidate = event.standby_node.get("node_key")
        elif isinstance(event, _RawWhatIf) and isinstance(
            event.intervention, RedundancyChangeEvent
        ):
            candidate = event.intervention.standby_node.get("node_key")
        else:
            continue
        if isinstance(candidate, str):
            keys.add(candidate)
    return frozenset(keys)


def _parse_step(
    scenario_id: str,
    index: int,
    raw_step: object,
    topology: Topology,
    asset_keys: frozenset[str],
) -> ExpectedStep:
    if not isinstance(raw_step, dict):
        raise SimDslError(f"{scenario_id}: expected_trace step {index} is not an object")
    declared_step = raw_step.get("step", index)
    if declared_step != index:
        raise SimDslError(
            f"{scenario_id}: expected_trace step {index} declares step {declared_step}"
        )

    services = topology.service_keys()
    lost = raw_step.get("lost_services", [])
    if not isinstance(lost, list) or not all(isinstance(key, str) for key in lost):
        raise SimDslError(f"{scenario_id}: step {index} lost_services must be a list of keys")
    unknown = set(lost) - services
    if unknown:
        raise SimDslError(
            f"{scenario_id}: step {index} expects losses on non-service keys {sorted(unknown)}"
        )

    tolerated = _parse_service_map(
        scenario_id, index, raw_step.get("service_first_fault_tolerated", {}), services, bool
    )
    spofs = _parse_service_map(
        scenario_id, index, raw_step.get("service_spofs", {}), services, str
    )
    for service_key, spof_keys in spofs.items():
        unknown_spofs = set(spof_keys) - asset_keys
        if unknown_spofs:
            raise SimDslError(
                f"{scenario_id}: step {index} {service_key} names unknown SPOF "
                f"assets {sorted(unknown_spofs)}"
            )
    return ExpectedStep(
        step=index,
        lost_services=frozenset(lost),
        service_first_fault_tolerated={key: bool(value) for key, value in tolerated.items()},
        service_spofs={key: frozenset(value) for key, value in spofs.items()},
    )


def _parse_service_map(
    scenario_id: str,
    index: int,
    raw_map: object,
    services: frozenset[str],
    value_type: type,
) -> dict[str, Any]:
    """Validate a partial per-service assertion map (keys exist; value type)."""
    if not isinstance(raw_map, dict):
        raise SimDslError(f"{scenario_id}: step {index} service map must be an object")
    parsed: dict[str, Any] = {}
    for key, value in raw_map.items():
        if key not in services:
            raise SimDslError(f"{scenario_id}: step {index} names unknown service {key!r}")
        if value_type is bool:
            if not isinstance(value, bool):
                raise SimDslError(f"{scenario_id}: step {index} {key} expects a boolean")
        else:  # str — a list of asset keys for the SPOF assertion
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SimDslError(f"{scenario_id}: step {index} {key} expects a list of keys")
        parsed[key] = value
    return parsed


def check_state_legality(scenario: SimScenario) -> None:
    """Walk the events and reject state-illegal sequences at load time.

    A repair without a matching earlier fault, a second fault of an
    already-faulted asset, or a what-if intervention that is inapplicable
    at its marker's position (unknown or removed target, standby key
    collision with a timeline upgrade) is a scenario-authoring bug —
    surfacing it while loading keeps the failure at review time, not
    mid-run. Markers are validated against a copy of the walked state but
    do not advance it: branches never touch the timeline.
    """
    state = initial_state(scenario.topology)
    for index, event in enumerate(scenario.events, start=1):
        if isinstance(event, WhatIfMarker):
            try:
                apply_intervention(state, event.intervention)
            except SimError as error:
                raise SimDslError(
                    f"{scenario.scenario_id}: what_if {event.label!r} (event {index}) is "
                    f"inapplicable at its position: {error}"
                ) from error
            continue
        try:
            state = apply_event(state, event)
        except Exception as error:
            raise SimDslError(
                f"{scenario.scenario_id}: event {index} ({event!r}) is illegal "
                f"mid-sequence: {error}"
            ) from error
