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

Array order is the execution order; M1 models no wall-clock timing (failover
transfer durations are a WP4 physics question, per the plan).

M3 extension point (deliberately unimplemented): what-if markers arrive as a
new event kind — save the state at the marker, compare branch assessments at
the end. The union validation below is the only code that changes; the trace
schema grows an optional marker field. Unknown kinds are rejected with that
named so a forward-ported file fails loudly instead of silently doing less.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.corpus.common import SEGMENTS
from evals.sim.engine import (
    FaultEvent,
    LoadShiftEvent,
    RedundancyChangeEvent,
    RepairEvent,
    SimDslError,
    SimEvent,
    Topology,
    apply_event,
    initial_state,
)

SIM_SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
FACILITY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")

# The closed M1 event union. M3 appends "what_if" here plus its parser.
EVENT_KINDS = ("fault", "repair", "redundancy_change", "load_shift")


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
class SimScenario:
    """One sequenced scenario: topology, ordered events, adjudicated trace."""

    scenario_id: str
    title: str
    description: str
    market: str
    facility_id: str
    topology_payload: dict[str, object]
    events: tuple[SimEvent, ...]
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
    events = tuple(
        _parse_event(scenario_id, index, raw_event, topology)
        for index, raw_event in enumerate(raw_events)
    )

    expected = raw.get("expected_trace")
    if not isinstance(expected, list):
        raise SimDslError(f"{scenario_id}: 'expected_trace' must be a list")
    if len(expected) != len(events) + 1:
        raise SimDslError(
            f"{scenario_id}: expected_trace has {len(expected)} steps for "
            f"{len(events)} events — expected {len(events) + 1} (baseline + one per event)"
        )
    trace = tuple(
        _parse_step(scenario_id, index, raw_step, topology, _scenario_asset_keys(events, topology))
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
) -> SimEvent:
    if not isinstance(raw_event, dict):
        raise SimDslError(f"{scenario_id}: event {index} is not an object")
    kind = raw_event.get("event")
    if kind not in EVENT_KINDS:
        raise SimDslError(
            f"{scenario_id}: event {index} has unknown kind {kind!r} — supported kinds: "
            f"{', '.join(EVENT_KINDS)} (what-if markers join this union in M3)"
        )
    # Reference checks run here, against the base topology, so a scenario
    # file with a typo'd key fails at load; state-level legality (double
    # fault, repairing an unfaulted asset) is the engine's job at run time.
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
    raise SimDslError(f"{scenario_id}: event {index} kind {kind!r} not parsed")  # pragma: no cover


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


def _scenario_asset_keys(events: tuple[SimEvent, ...], topology: Topology) -> frozenset[str]:
    """Every asset a trace step may legitimately name: base assets plus the
    standbys the scenario's own redundancy-change events introduce (a
    post-upgrade SPOF may be a node the base topology never had)."""
    keys = {key for key, spec in topology.nodes.items() if spec.node_type == "asset"}
    for event in events:
        if not isinstance(event, RedundancyChangeEvent):
            continue
        standby_key = event.standby_node.get("node_key")
        if isinstance(standby_key, str):
            keys.add(standby_key)
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

    A repair without a matching earlier fault, or a second fault of an
    already-faulted asset, is a scenario-authoring bug — surfacing it while
    loading keeps the failure at review time, not mid-run.
    """
    state = initial_state(scenario.topology)
    for index, event in enumerate(scenario.events, start=1):
        try:
            state = apply_event(state, event)
        except Exception as error:
            raise SimDslError(
                f"{scenario.scenario_id}: event {index} ({event!r}) is illegal "
                f"mid-sequence: {error}"
            ) from error
