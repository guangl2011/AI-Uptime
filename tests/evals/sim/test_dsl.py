"""Sim DSL tests: scenario validation gates and the shipped corpus.

Mirrors tests/evals/test_scenarios.py's file-integrity style — the shipped
scenario files are hand-adjudicated gates, so the tests validate their
internal consistency and the DSL's rejection paths with small fixtures.
"""

from __future__ import annotations

from typing import Any

import pytest
from evals.sim.dsl import SimDslError, load_sim_scenarios, parse_scenario

SHIPPED_IDS = {
    "sim-cold-chain-compressor-failover-repair",
    "sim-aidc-chiller-cascade-and-standby",
    "sim-hospital-life-safety-ride-through",
}


def test_three_sim_scenarios_ship() -> None:
    """One adjudicated cascade gate per segment, in market order."""
    scenarios = load_sim_scenarios()
    assert {s.scenario_id for s in scenarios} == SHIPPED_IDS
    assert [s.market for s in scenarios] == ["aidc", "cold_chain", "hospital"]


def _minimal_scenario() -> dict[str, Any]:
    provenance = {
        "source_kind": "technician_entry",
        "source_system": "test",
        "source_ref": "test",
    }
    return {
        "scenario_id": "sim-test",
        "title": "Test",
        "description": "Test scenario",
        "market": "cold_chain",
        "facility_id": "sim-test",
        "topology": {
            "nodes": [
                {
                    "node_key": "COMP-1",
                    "node_type": "asset",
                    "asset_type": "compressor",
                    "name": "Compressor 1",
                    "attributes": {},
                    "provenance": provenance,
                },
                {
                    "node_key": "SVC-R",
                    "node_type": "service",
                    "name": "Refrigeration",
                    "attributes": {},
                    "provenance": provenance,
                },
            ],
            "edges": [
                {
                    "from_node": "COMP-1",
                    "to_node": "SVC-R",
                    "failure_mode": "mechanical_failure",
                    "redundancy_config": "N",
                    "provenance": provenance,
                }
            ],
        },
        "events": [
            {"event": "fault", "asset_key": "COMP-1"},
            {"event": "repair", "asset_key": "COMP-1"},
        ],
        "expected_trace": [
            {"step": 0, "lost_services": []},
            {"step": 1, "lost_services": ["SVC-R"]},
            {"step": 2, "lost_services": []},
        ],
        "adjudicated_by": "test",
    }


def test_minimal_scenario_parses() -> None:
    scenario = parse_scenario(_minimal_scenario())
    assert scenario.scenario_id == "sim-test"
    assert scenario.facility_id == "sim-test"
    assert len(scenario.events) == 2
    assert len(scenario.expected_trace) == 3
    assert scenario.expected_trace[1].lost_services == frozenset({"SVC-R"})


def test_unknown_event_kind_is_rejected_with_extension_pointer() -> None:
    raw = _minimal_scenario()
    raw["events"].append({"event": "what_if_marker", "label": "x"})
    with pytest.raises(SimDslError, match="what-if markers join this union in M3"):
        parse_scenario(raw)


def test_trace_length_mismatch_is_rejected() -> None:
    raw = _minimal_scenario()
    del raw["expected_trace"][2]
    with pytest.raises(SimDslError, match="expected_trace has 2 steps"):
        parse_scenario(raw)


def test_unknown_asset_reference_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["events"][0]["asset_key"] = "NOPE"
    with pytest.raises(SimDslError, match="unknown asset"):
        parse_scenario(raw)


def test_unknown_service_reference_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["expected_trace"][1]["lost_services"] = ["SVC-X"]
    with pytest.raises(SimDslError, match="non-service keys"):
        parse_scenario(raw)


def test_step_number_mismatch_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["expected_trace"][2]["step"] = 5
    with pytest.raises(SimDslError, match="declares step 5"):
        parse_scenario(raw)


def test_non_bool_first_fault_tolerated_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["expected_trace"][1]["service_first_fault_tolerated"] = {"SVC-R": "yes"}
    with pytest.raises(SimDslError, match="expects a boolean"):
        parse_scenario(raw)


def test_spof_list_values_must_be_strings() -> None:
    raw = _minimal_scenario()
    raw["expected_trace"][1]["service_spofs"] = {"SVC-R": [42]}
    with pytest.raises(SimDslError, match="expects a list of keys"):
        parse_scenario(raw)


def test_unknown_spof_asset_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["expected_trace"][1]["service_spofs"] = {"SVC-R": ["NOPE"]}
    with pytest.raises(SimDslError, match="unknown SPOF assets"):
        parse_scenario(raw)


def test_spof_may_name_a_scenario_introduced_standby() -> None:
    """A post-upgrade SPOF may be a standby the base topology never had."""
    raw = _minimal_scenario()
    provenance = {
        "source_kind": "technician_entry",
        "source_system": "test",
        "source_ref": "test",
    }
    raw["topology"]["edges"][0]["member_group"] = "g"
    raw["events"].append(
        {
            "event": "redundancy_change",
            "member_group": "g",
            "redundancy_config": "N+1",
            "standby_node": {
                "node_key": "COMP-2",
                "node_type": "asset",
                "asset_type": "compressor",
                "name": "Standby",
                "attributes": {},
                "provenance": provenance,
            },
            "standby_edge": {
                "from_node": "COMP-2",
                "to_node": "SVC-R",
                "failure_mode": "mechanical_failure",
                "redundancy_config": "N+1",
                "member_group": "g",
                "provenance": provenance,
            },
        }
    )
    raw["expected_trace"].append(
        {
            "step": 3,
            "lost_services": [],
            "service_spofs": {"SVC-R": ["COMP-2"]},
        }
    )
    scenario = parse_scenario(raw)
    assert scenario.expected_trace[3].service_spofs["SVC-R"] == frozenset({"COMP-2"})


def test_state_illegal_sequence_is_rejected() -> None:
    raw = _minimal_scenario()
    raw["events"].insert(1, {"event": "fault", "asset_key": "COMP-1"})
    raw["expected_trace"].append({"step": 3, "lost_services": []})
    with pytest.raises(SimDslError, match="already faulted"):
        parse_scenario(raw)


def test_market_must_match_segments() -> None:
    raw = _minimal_scenario()
    raw["market"] = "fab"
    with pytest.raises(SimDslError, match="market"):
        parse_scenario(raw)


def test_facility_id_must_use_sim_namespace() -> None:
    raw = _minimal_scenario()
    raw["facility_id"] = "corpus-cold-chain-00"
    with pytest.raises(SimDslError, match="sim- namespace"):
        parse_scenario(raw)
