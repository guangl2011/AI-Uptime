"""Fault-injection scenario tests: file integrity + known-answer runs.

The scenario files are hand-adjudicated gates — these tests validate their
internal consistency (topology references, inject targets, expected keys)
and run each through the real API against its expectations.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest
from evals.scenarios import load_scenarios, run_all_scenarios, run_scenario
from fastapi.testclient import TestClient

from uptime_api.graph.enums import AssetType, FailureMode, RedundancyConfig

N_PLUS_1_SCENARIO = "single-compressor-failure-n-plus-1"


@pytest.fixture()
def scenarios() -> list[dict[str, Any]]:
    return load_scenarios()


def _by_id(scenarios: list[dict[str, Any]], scenario_id: str) -> dict[str, Any]:
    return next(s for s in scenarios if s["scenario_id"] == scenario_id)


def test_four_named_scenarios_ship(scenarios: list[dict[str, Any]]) -> None:
    ids = [str(s["scenario_id"]) for s in scenarios]
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) == {
        "single-compressor-failure-n-plus-1",
        "single-compressor-failure-n",
        "utility-feed-loss-2n",
        "rack-failure-cascade-unserved-rooms",
    }


def test_scenario_files_are_internally_consistent(
    scenarios: list[dict[str, Any]],
) -> None:
    """Topology references resolve; inject targets exist; expectations cover
    declared services; enums are in-taxon; facility ids are API-valid."""
    for scenario in scenarios:
        topology: dict[str, Any] = scenario["topology"]
        nodes = {node["node_key"]: node for node in topology["nodes"]}
        for raw_edge in topology["edges"]:
            assert raw_edge["from_node"] in nodes
            assert raw_edge["to_node"] in nodes
            FailureMode(raw_edge["failure_mode"])
            if raw_edge.get("redundancy_config") is not None:
                RedundancyConfig(raw_edge["redundancy_config"])
        for raw_node in topology["nodes"]:
            if raw_node["node_type"] == "asset":
                AssetType(raw_node["asset_type"])

        inject_key = scenario["inject"]["asset_key"]
        assert inject_key in nodes, scenario["scenario_id"]

        expected: dict[str, Any] = scenario["expected"]
        services = {
            key for key, node in nodes.items() if node["node_type"] == "service"
        }
        assert set(expected["exposed_services"]) <= services
        assert set(expected["service_first_fault_tolerated"]) == services
        assert scenario["market"] == "cold_chain"
        # The runner creates the facility via POST /facilities — the id must
        # satisfy its validation pattern.
        assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,62}", scenario["facility_id"])


def test_scenarios_run_green_through_the_api(client: TestClient) -> None:
    """Every shipped scenario passes its adjudicated expectation (live)."""
    outcomes = run_all_scenarios(client)
    assert len(outcomes) == 4
    for outcome in outcomes:
        assert outcome.passed, (outcome.scenario_id, outcome.failures)


def test_runner_catches_a_tampered_expectation(
    client: TestClient, scenarios: list[dict[str, Any]]
) -> None:
    """The runner is not a rubber stamp: a wrong expectation fails."""
    scenario = copy.deepcopy(_by_id(scenarios, "single-compressor-failure-n"))
    scenario["expected"]["exposed_services"] = []  # wrong on purpose
    outcome = run_scenario(client, scenario)
    assert not outcome.passed
    assert any("exposed services mismatch" in failure for failure in outcome.failures)


def test_runner_catches_a_broken_redundancy_declaration(
    client: TestClient, scenarios: list[dict[str, Any]]
) -> None:
    """If the assessor stopped honoring N+1 absorption, the scenario fails."""
    scenario = copy.deepcopy(_by_id(scenarios, N_PLUS_1_SCENARIO))
    scenario["facility_id"] = "scenario-tampered-nplus1"  # fresh facility id
    for edge in scenario["topology"]["edges"]:
        if edge.get("member_group") == "rack_compressors-1":
            edge["redundancy_config"] = "N"  # degrade the pair to N
    outcome = run_scenario(client, scenario)
    assert not outcome.passed
