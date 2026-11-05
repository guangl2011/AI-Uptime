"""Sim engine tests: pure event transitions, cascade closure, materialization.

The engine is the determinism-critical core — these tests pin its semantics
directly: absorption only via live siblings of tolerant groups, cascades
through unprotected chains, repair only for event-faulted assets, and
materialization that keeps dead nodes isolated (so the assessor can name
their services unserved).
"""

from __future__ import annotations

import pytest
from evals.sim.engine import (
    FaultEvent,
    LoadShiftEvent,
    RedundancyChangeEvent,
    RepairEvent,
    SimEventError,
    SimState,
    Topology,
    apply_event,
    dead_nodes,
    initial_state,
    materialize,
)


def _asset(key: str) -> dict[str, object]:
    return {
        "node_key": key,
        "node_type": "asset",
        "asset_type": "compressor",
        "name": key,
        "attributes": {},
        "provenance": {
            "source_kind": "technician_entry",
            "source_system": "test",
            "source_ref": "test",
        },
    }


def _service(key: str) -> dict[str, object]:
    return {
        "node_key": key,
        "node_type": "service",
        "name": key,
        "attributes": {},
        "provenance": {
            "source_kind": "technician_entry",
            "source_system": "test",
            "source_ref": "test",
        },
    }


def _edge(
    from_node: str,
    to_node: str,
    *,
    config: str | None = None,
    group: str | None = None,
) -> dict[str, object]:
    return {
        "from_node": from_node,
        "to_node": to_node,
        "failure_mode": "power_loss",
        "redundancy_config": config,
        "member_group": group,
        "provenance": {
            "source_kind": "technician_entry",
            "source_system": "test",
            "source_ref": "test",
        },
    }


def _redundancy_event(
    group: str,
    config: str,
    standby_key: str,
    consumer: str,
) -> RedundancyChangeEvent:
    return RedundancyChangeEvent(
        member_group=group,
        redundancy_config=config,
        standby_node=_asset(standby_key),
        standby_edge=_edge(standby_key, consumer, config=config, group=group),
    )


def _topology(edges: list[dict[str, object]], *nodes: dict[str, object]) -> Topology:
    return Topology.from_payload(
        {"nodes": list(nodes), "edges": edges}
    )


# ---------------------------------------------------------------------------
# Cascade closure
# ---------------------------------------------------------------------------


def test_fault_cascades_through_unprotected_chain() -> None:
    topology = _topology(
        [
            _edge("SVC-P", "RACK-1", config="N"),
            _edge("COMP-1", "RACK-1", config="N", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-P"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    assert dead_nodes(state) == frozenset({"COMP-1", "RACK-1", "SVC-R"})


def test_live_sibling_absorbs_single_member_fault() -> None:
    topology = _topology(
        [
            _edge("PUMP-1", "CDU-1", config="N+1", group="pumps"),
            _edge("PUMP-2", "CDU-1", config="N+1", group="pumps"),
            _edge("CDU-1", "SVC-C", config="N"),
        ],
        _asset("PUMP-1"),
        _asset("PUMP-2"),
        _asset("CDU-1"),
        _service("SVC-C"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="PUMP-1"))
    assert dead_nodes(state) == frozenset({"PUMP-1"})


def test_dead_group_cannot_absorb() -> None:
    """The dynamic rule: only LIVE siblings count — when the loss that kills
    one pump also kills the other (their own supply), the pair cannot absorb."""
    topology = _topology(
        [
            _edge("CHW", "PUMP-1", config="N"),
            _edge("CHW", "PUMP-2", config="N"),
            _edge("PUMP-1", "CDU-1", config="N+1", group="pumps"),
            _edge("PUMP-2", "CDU-1", config="N+1", group="pumps"),
            _edge("CDU-1", "SVC-C", config="N"),
        ],
        _asset("CHW"),
        _asset("PUMP-1"),
        _asset("PUMP-2"),
        _asset("CDU-1"),
        _service("SVC-C"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="CHW"))
    assert dead_nodes(state) == frozenset(
        {"CHW", "PUMP-1", "PUMP-2", "CDU-1", "SVC-C"}
    )


def test_2n_pair_tolerates_one_source_but_not_two() -> None:
    topology = _topology(
        [
            _edge("UTILITY", "ATS-1", config="2N", group="sources"),
            _edge("GEN", "ATS-1", config="2N", group="sources"),
            _edge("ATS-1", "SVC-P", config="N"),
        ],
        _asset("UTILITY"),
        _asset("GEN"),
        _asset("ATS-1"),
        _service("SVC-P"),
    )
    one_down = apply_event(initial_state(topology), FaultEvent(asset_key="UTILITY"))
    assert dead_nodes(one_down) == frozenset({"UTILITY"})

    both_down = apply_event(one_down, FaultEvent(asset_key="GEN"))
    assert dead_nodes(both_down) == frozenset({"UTILITY", "GEN", "ATS-1", "SVC-P"})


def test_single_member_tolerant_config_does_not_absorb() -> None:
    """A lone member declaring N+1 is still unprotected: the assessor's
    group_sizes >= 2 rule, mirrored by the engine."""
    topology = _topology(
        [
            _edge("COMP-1", "RACK-1", config="N+1", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    assert dead_nodes(state) == frozenset({"COMP-1", "RACK-1", "SVC-R"})


# ---------------------------------------------------------------------------
# Repair semantics
# ---------------------------------------------------------------------------


def test_repair_restores_cascade_through_recomputation() -> None:
    topology = _topology(
        [
            _edge("COMP-1", "RACK-1", config="N", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    assert dead_nodes(state) == frozenset({"COMP-1", "RACK-1", "SVC-R"})

    repaired = apply_event(state, RepairEvent(asset_key="COMP-1"))
    assert dead_nodes(repaired) == frozenset()


def test_cascade_dead_asset_cannot_be_repaired_directly() -> None:
    """RACK-1 died from the cascade, not from an event — repair only undoes
    event faults; restoration flows out of recomputation."""
    topology = _topology(
        [
            _edge("COMP-1", "RACK-1", config="N", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    with pytest.raises(SimEventError, match="not faulted by an event"):
        apply_event(state, RepairEvent(asset_key="RACK-1"))


def test_repair_of_unfaulted_asset_raises() -> None:
    topology = _topology([], _asset("COMP-1"))
    with pytest.raises(SimEventError, match="not faulted by an event"):
        apply_event(initial_state(topology), RepairEvent(asset_key="COMP-1"))


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "match"),
    [
        (FaultEvent(asset_key="NOPE"), "unknown node"),
        (RepairEvent(asset_key="NOPE"), "unknown node"),
    ],
)
def test_events_targeting_unknown_nodes_raise(
    event: FaultEvent | RepairEvent, match: str
) -> None:
    topology = _topology([], _asset("COMP-1"))
    with pytest.raises(SimEventError, match=match):
        apply_event(initial_state(topology), event)


def test_fault_of_service_node_raises() -> None:
    topology = _topology([], _service("SVC-R"))
    with pytest.raises(SimEventError, match="service"):
        apply_event(initial_state(topology), FaultEvent(asset_key="SVC-R"))


def test_double_fault_raises() -> None:
    topology = _topology([], _asset("COMP-1"))
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    with pytest.raises(SimEventError, match="already faulted"):
        apply_event(state, FaultEvent(asset_key="COMP-1"))


# ---------------------------------------------------------------------------
# Redundancy change
# ---------------------------------------------------------------------------


def test_redundancy_change_adds_standby_and_absorbs_faulted_primary() -> None:
    """The failover-ladder centerpiece: N -> N+1 with the primary still down
    restores service before any repair."""
    topology = _topology(
        [
            _edge("COMP-1", "RACK-1", config="N", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    assert dead_nodes(state) == frozenset({"COMP-1", "RACK-1", "SVC-R"})

    upgraded = apply_event(
        state, _redundancy_event("g", "N+1", "COMP-2", "RACK-1")
    )
    # The primary is still faulted; the standby carries the rack.
    assert dead_nodes(upgraded) == frozenset({"COMP-1"})

    group_edges = [edge for edge in upgraded.topology.edges if edge.member_group == "g"]
    assert len(group_edges) == 2
    assert all(edge.redundancy_config == "N+1" for edge in group_edges)


def test_redundancy_change_unknown_group_raises() -> None:
    topology = _topology([], _asset("COMP-1"))
    with pytest.raises(SimEventError, match="unknown member group"):
        apply_event(initial_state(topology), _redundancy_event("g", "N+1", "C2", "RACK-1"))


def test_redundancy_change_unknown_config_raises() -> None:
    topology = _topology(
        [_edge("COMP-1", "RACK-1", config="N", group="g")],
        _asset("COMP-1"),
        _asset("RACK-1"),
    )
    with pytest.raises(SimEventError, match="unknown config"):
        apply_event(
            initial_state(topology), _redundancy_event("g", "3N", "COMP-2", "RACK-1")
        )


def test_redundancy_change_duplicate_standby_raises() -> None:
    topology = _topology(
        [_edge("COMP-1", "RACK-1", config="N", group="g")],
        _asset("COMP-1"),
        _asset("RACK-1"),
    )
    with pytest.raises(SimEventError, match="already exists"):
        apply_event(
            initial_state(topology), _redundancy_event("g", "N+1", "COMP-1", "RACK-1")
        )


def test_redundancy_change_standby_must_target_group_consumer() -> None:
    topology = _topology(
        [_edge("COMP-1", "RACK-1", config="N", group="g")],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _asset("SVC-X"),
    )
    event = RedundancyChangeEvent(
        member_group="g",
        redundancy_config="N+1",
        standby_node=_asset("COMP-2"),
        standby_edge=_edge("COMP-2", "SVC-X", config="N+1", group="g"),
    )
    with pytest.raises(SimEventError, match="must target"):
        apply_event(initial_state(topology), event)


# ---------------------------------------------------------------------------
# Load shift
# ---------------------------------------------------------------------------


def test_load_shift_rewires_unprotected_feed() -> None:
    topology = _topology(
        [
            _edge("RACK-1", "SVC-R", config="N"),
            _edge("RACK-2", "SVC-R", config="N"),
        ],
        _asset("RACK-1"),
        _asset("RACK-2"),
        _service("SVC-R"),
    )
    shifted = apply_event(
        initial_state(topology),
        LoadShiftEvent(consumer="SVC-R", from_node="RACK-1", to_node="RACK-2"),
    )
    from_rack1 = [edge for edge in shifted.topology.edges if edge.from_node == "RACK-1"]
    assert from_rack1 == []
    state = apply_event(shifted, FaultEvent(asset_key="RACK-1"))
    assert dead_nodes(state) == frozenset({"RACK-1"})


def test_load_shift_rejects_group_member_edges() -> None:
    topology = _topology(
        [
            _edge("RACK-1", "SVC-R", config="N+1", group="g"),
            _edge("RACK-2", "SVC-R", config="N+1", group="g"),
        ],
        _asset("RACK-1"),
        _asset("RACK-2"),
        _service("SVC-R"),
    )
    with pytest.raises(SimEventError, match="redundant-group member"):
        apply_event(
            initial_state(topology),
            LoadShiftEvent(consumer="SVC-R", from_node="RACK-1", to_node="RACK-2"),
        )


def test_load_shift_rejects_unknown_nodes_and_missing_edge() -> None:
    topology = _topology(
        [_edge("RACK-1", "SVC-R", config="N")],
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    base = initial_state(topology)
    with pytest.raises(SimEventError, match="unknown consumer"):
        apply_event(
            base, LoadShiftEvent(consumer="SVC-X", from_node="RACK-1", to_node="SVC-R")
        )
    with pytest.raises(SimEventError, match="unknown destination"):
        apply_event(
            base, LoadShiftEvent(consumer="SVC-R", from_node="RACK-1", to_node="RACK-9")
        )
    with pytest.raises(SimEventError, match="exactly one dependency edge"):
        apply_event(
            base, LoadShiftEvent(consumer="SVC-R", from_node="SVC-R", to_node="RACK-1")
        )


# ---------------------------------------------------------------------------
# Purity and determinism
# ---------------------------------------------------------------------------


def test_apply_event_leaves_input_state_untouched() -> None:
    topology = _topology(
        [_edge("COMP-1", "RACK-1", config="N", group="g")],
        _asset("COMP-1"),
        _asset("RACK-1"),
    )
    state = initial_state(topology)
    apply_event(state, FaultEvent(asset_key="COMP-1"))
    assert state.faulted == frozenset()
    assert state.topology is topology


def test_dead_nodes_independent_of_edge_order() -> None:
    edges_a = [
        _edge("COMP-1", "RACK-1", config="N", group="g"),
        _edge("RACK-1", "SVC-R", config="N"),
    ]
    edges_b = list(reversed(edges_a))
    nodes = (_asset("COMP-1"), _asset("RACK-1"), _service("SVC-R"))
    state_a = SimState(topology=_topology(edges_a, *nodes), faulted=frozenset({"COMP-1"}))
    state_b = SimState(topology=_topology(edges_b, *nodes), faulted=frozenset({"COMP-1"}))
    assert dead_nodes(state_a) == dead_nodes(state_b)


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_materialize_keeps_dead_nodes_and_drops_their_edges() -> None:
    topology = _topology(
        [
            _edge("COMP-1", "RACK-1", config="N", group="g"),
            _edge("RACK-1", "SVC-R", config="N"),
        ],
        _asset("COMP-1"),
        _asset("RACK-1"),
        _service("SVC-R"),
    )
    state = apply_event(initial_state(topology), FaultEvent(asset_key="COMP-1"))
    payload = materialize(
        state, dead_nodes(state), author="test", notes="test"
    )
    node_keys = {str(node["node_key"]) for node in payload["nodes"]}
    assert node_keys == {"COMP-1", "RACK-1", "SVC-R"}
    dead = {"COMP-1", "RACK-1", "SVC-R"}
    for raw_edge in payload["edges"]:
        assert str(raw_edge["from_node"]) not in dead
        assert str(raw_edge["to_node"]) not in dead
    assert payload["edges"] == []
