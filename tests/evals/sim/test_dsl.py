"""Sim DSL tests: scenario validation gates and the shipped corpus.

Mirrors tests/evals/test_scenarios.py's file-integrity style — the shipped
scenario files are hand-adjudicated gates, so the tests validate their
internal consistency and the DSL's rejection paths with small fixtures.
"""

from __future__ import annotations

from typing import Any

import pytest
from evals.sim.dsl import SimDslError, WhatIfMarker, load_sim_scenarios, parse_scenario
from evals.sim.engine import RemoveAssetEvent

SHIPPED_IDS = {
    # M1 — adjudicated cascade traces, one per segment.
    "sim-cold-chain-compressor-failover-repair",
    "sim-aidc-chiller-cascade-and-standby",
    "sim-hospital-life-safety-ride-through",
    # M3 — adjudicated what-if deltas, one per segment.
    "sim-cold-chain-whatif-condenser-and-rack-upgrade",
    "sim-aidc-whatif-chiller-cascade",
    "sim-hospital-whatif-generator-and-medgas",
}


def test_sim_scenarios_ship_per_segment() -> None:
    """One adjudicated cascade trace (M1) and one adjudicated what-if pair
    (M3) per segment; ids sort into market order."""
    scenarios = load_sim_scenarios()
    assert {s.scenario_id for s in scenarios} == SHIPPED_IDS
    assert [s.market for s in scenarios] == [
        "aidc",
        "aidc",
        "cold_chain",
        "cold_chain",
        "hospital",
        "hospital",
    ]


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


def test_unknown_event_kind_is_rejected_with_supported_list() -> None:
    raw = _minimal_scenario()
    raw["events"].append({"event": "future_kind", "label": "x"})
    with pytest.raises(SimDslError, match="unknown kind 'future_kind'.*supported kinds"):
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


# ---------------------------------------------------------------------------
# What-if markers (M3)
# ---------------------------------------------------------------------------


def _minimal_what_if_event(label: str = "probe") -> dict[str, Any]:
    return {
        "event": "what_if",
        "label": label,
        "intervention": {"type": "remove_asset", "asset_key": "COMP-1"},
        "expected_delta": {
            "lost_services": ["SVC-R"],
            "restored_services": [],
            "service_spofs_added": {},
            "service_spofs_removed": {},
            "service_first_fault_tolerated": {},
            "expect_spof_path_abstention": None,
        },
    }


def test_what_if_marker_parses_with_intervention_and_delta() -> None:
    raw = _minimal_scenario()
    raw["events"].append(_minimal_what_if_event())
    raw["expected_trace"].append({"step": 3, "lost_services": []})
    scenario = parse_scenario(raw)
    markers = [event for event in scenario.events if isinstance(event, WhatIfMarker)]
    assert len(markers) == 1
    marker = markers[0]
    assert marker.label == "probe"
    assert isinstance(marker.intervention, RemoveAssetEvent)
    assert marker.intervention.asset_key == "COMP-1"
    assert marker.expected_delta.lost_services == frozenset({"SVC-R"})
    assert marker.expected_delta.expect_spof_path_abstention is None


def test_what_if_marker_does_not_extend_the_timeline() -> None:
    """Markers occupy a purity trace step but are not timeline events: two
    markers on a two-event scenario yield four events, five trace steps."""
    raw = _minimal_scenario()
    raw["events"].append(_minimal_what_if_event("first"))
    raw["events"].append(_minimal_what_if_event("second"))
    raw["expected_trace"].extend(
        [{"step": 3, "lost_services": []}, {"step": 4, "lost_services": []}]
    )
    scenario = parse_scenario(raw)
    assert len(scenario.events) == 4
    assert len(scenario.expected_trace) == 5


def test_duplicate_what_if_labels_are_rejected() -> None:
    raw = _minimal_scenario()
    raw["events"].append(_minimal_what_if_event())
    raw["events"].append(_minimal_what_if_event())
    with pytest.raises(SimDslError, match="what_if labels must be distinct"):
        parse_scenario(raw)


def test_what_if_delta_naming_an_unknown_service_is_rejected() -> None:
    raw = _minimal_scenario()
    event = _minimal_what_if_event()
    event["expected_delta"]["lost_services"] = ["SVC-X"]
    raw["events"].append(event)
    with pytest.raises(SimDslError, match="non-service keys"):
        parse_scenario(raw)


def test_what_if_delta_naming_an_unknown_asset_is_rejected() -> None:
    raw = _minimal_scenario()
    event = _minimal_what_if_event()
    event["expected_delta"]["service_spofs_removed"] = {"SVC-R": ["NOPE"]}
    raw["events"].append(event)
    with pytest.raises(SimDslError, match="unknown SPOF assets"):
        parse_scenario(raw)


def test_what_if_delta_may_name_a_scenario_introduced_standby() -> None:
    """A delta may reference a standby that only the marker's own upgrade
    introduces (the scenario-wide asset-key set spans every event)."""
    raw = _minimal_scenario()
    raw["topology"]["edges"][0]["member_group"] = "g"
    raw["events"].append(
        {
            "event": "what_if",
            "label": "upgrade",
            "intervention": {
                "type": "upgrade_redundancy",
                "member_group": "g",
                "redundancy_config": "N+1",
                "standby_node": {
                    "node_key": "COMP-2",
                    "node_type": "asset",
                    "asset_type": "compressor",
                    "name": "Compressor 2 (standby)",
                    "attributes": {},
                    "provenance": {
                        "source_kind": "technician_entry",
                        "source_system": "test",
                        "source_ref": "test",
                    },
                },
                "standby_edge": {
                    "from_node": "COMP-2",
                    "to_node": "SVC-R",
                    "failure_mode": "mechanical_failure",
                    "redundancy_config": "N+1",
                    "member_group": "g",
                    "provenance": {
                        "source_kind": "technician_entry",
                        "source_system": "test",
                        "source_ref": "test",
                    },
                },
            },
            "expected_delta": {
                "lost_services": [],
                "restored_services": [],
                "service_spofs_added": {},
                "service_spofs_removed": {"SVC-R": ["COMP-2"]},
                "service_first_fault_tolerated": {"SVC-R": True},
                "expect_spof_path_abstention": False,
            },
        }
    )
    raw["expected_trace"].append({"step": 3, "lost_services": []})
    scenario = parse_scenario(raw)
    markers = [event for event in scenario.events if isinstance(event, WhatIfMarker)]
    assert markers[0].expected_delta.service_spofs_removed == {
        "SVC-R": frozenset({"COMP-2"})
    }


def test_what_if_abstention_pin_must_be_boolean_or_null() -> None:
    raw = _minimal_scenario()
    event = _minimal_what_if_event()
    event["expected_delta"]["expect_spof_path_abstention"] = "maybe"
    raw["events"].append(event)
    raw["expected_trace"].append({"step": 3, "lost_services": []})
    with pytest.raises(
        SimDslError, match="expect_spof_path_abstention must be a boolean"
    ):
        parse_scenario(raw)


def test_what_if_removal_of_a_service_key_is_rejected() -> None:
    raw = _minimal_scenario()
    event = _minimal_what_if_event()
    event["intervention"] = {"type": "remove_asset", "asset_key": "SVC-R"}
    raw["events"].append(event)
    with pytest.raises(SimDslError, match="which is a service node"):
        parse_scenario(raw)
