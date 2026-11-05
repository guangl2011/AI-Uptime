"""Sim runner tests: loss derivation, trace comparison, and corpus sweeps.

The sweep is exercised on a reduced corpus (one facility per segment) — the
full canonical corpus belongs to `make sim` / the nightly run, not to unit
tests. Sequenced scenarios run against the real API surface.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from evals.sim.dsl import load_sim_scenarios
from evals.sim.runner import (
    SweepCase,
    compare_trace,
    lost_services,
    run_corpus_sweep,
    run_sim_scenario,
)
from fastapi.testclient import TestClient

from uptime_api.app import create_app

# A stable non-canonical seed — these tests are about the runner's mechanics,
# not about pinning canonical-corpus labels (make sim owns that).
REDUCED_COUNTS = {"cold_chain": 1, "aidc": 1, "hospital": 1}


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(f"sqlite:///{tmp_path}/sim-test.db"), raise_server_exceptions=False
    )


def test_lost_services_from_report_findings() -> None:
    report = {
        "findings": [
            {"kind": "unserved_service", "subject": "svc_a"},
            {"kind": "spof", "subject": "COMP-1"},
            {"kind": "unserved_service", "subject": "svc_b"},
        ]
    }
    assert lost_services(report) == frozenset({"svc_a", "svc_b"})


def test_lost_services_ignores_malformed_findings() -> None:
    assert lost_services({"findings": "not-a-list"}) == frozenset()
    assert lost_services({}) == frozenset()


def test_sweep_case_exact_when_labels_agree() -> None:
    case = SweepCase(
        facility_id="f",
        segment="cold_chain",
        asset_key="A",
        expected_exposed=("S1",),
        expected_impacted=("S1",),
        observed_lost=("S1",),
    )
    assert case.ok is True
    assert case.exact is True


def test_sweep_case_violates_when_labels_agree_but_observation_differs() -> None:
    case = SweepCase(
        facility_id="f",
        segment="cold_chain",
        asset_key="A",
        expected_exposed=("S1",),
        expected_impacted=("S1",),
        observed_lost=(),
    )
    assert case.ok is False


def test_sweep_case_within_bounds_passes() -> None:
    case = SweepCase(
        facility_id="f",
        segment="aidc",
        asset_key="CH-1",
        expected_exposed=("S1",),
        expected_impacted=("S1", "S2", "S3"),
        observed_lost=("S1", "S2"),
    )
    assert case.ok is True
    assert case.exact is False


def test_sweep_case_below_exposure_fails() -> None:
    case = SweepCase(
        facility_id="f",
        segment="aidc",
        asset_key="CH-1",
        expected_exposed=("S1", "S2"),
        expected_impacted=("S1", "S2"),
        observed_lost=("S1",),
    )
    assert case.ok is False


def test_sweep_case_above_impact_closure_fails() -> None:
    case = SweepCase(
        facility_id="f",
        segment="aidc",
        asset_key="CH-1",
        expected_exposed=("S1",),
        expected_impacted=("S1", "S2"),
        observed_lost=("S1", "S2", "S3"),
    )
    assert case.ok is False


def test_corpus_sweep_on_reduced_corpus(tmp_path: Path) -> None:
    client = _client(tmp_path)
    result = run_corpus_sweep(
        client, seed=20260101, facilities_per_segment=REDUCED_COUNTS
    )
    assert result.ok(), [
        {
            "facility": case.facility_id,
            "asset": case.asset_key,
            "expected_exposed": case.expected_exposed,
            "expected_impacted": case.expected_impacted,
            "observed": case.observed_lost,
        }
        for case in result.mismatches
    ]
    assert result.cases
    per_segment = result.per_segment()
    assert set(per_segment) == {"cold_chain", "aidc", "hospital"}
    assert all(mismatches == 0 for _, mismatches in per_segment.values())


def test_all_shipped_scenarios_pass_via_real_api(tmp_path: Path) -> None:
    """Every shipped sim scenario — M1 cascades and M3 what-if — passes."""
    client = _client(tmp_path)
    scenarios = load_sim_scenarios()
    assert len(scenarios) == 6
    for scenario in scenarios:
        outcome = run_sim_scenario(client, scenario)
        assert outcome.passed, outcome.failures


def test_compare_trace_reports_stepwise_failures(tmp_path: Path) -> None:
    """Doctor one expectation and confirm the failure names the step and the
    expected-vs-observed sets (the repro detail the CI artifact ships)."""
    scenario = load_sim_scenarios()[0]
    client = _client(tmp_path)
    passing = run_sim_scenario(client, scenario)
    assert passing.passed

    doctored = replace(
        scenario,
        expected_trace=tuple(
            replace(step, lost_services=("SVC-CHW-1",)) for step in scenario.expected_trace
        ),
    )
    outcome = compare_trace(doctored, passing.trace)
    assert not outcome.passed
    assert any("expected ['SVC-CHW-1']" in failure for failure in outcome.failures)
