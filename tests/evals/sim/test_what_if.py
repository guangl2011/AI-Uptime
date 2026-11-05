"""What-if counterfactual tests (Sim M3, Category 5).

Unit coverage for the delta comparison, marker purity, and SPOF-path
verdict helpers, plus end-to-end branch assertions over the shipped AIDC
what-if scenario through the real API surface.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from evals.sim.dsl import SimScenario, WhatIfMarker, load_sim_scenarios
from evals.sim.engine import (
    RedundancyChangeEvent,
    apply_intervention,
    dead_nodes,
    initial_state,
)
from evals.sim.runner import (
    TraceStep,
    _delta_failures,
    _delta_observed,
    _purity_failure,
    _spof_path_verdict,
    run_sim_scenario,
)
from fastapi.testclient import TestClient

from uptime_api.app import create_app

AIDC_WHAT_IF_ID = "sim-aidc-whatif-chiller-cascade"


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(f"sqlite:///{tmp_path}/what-if-test.db"), raise_server_exceptions=False
    )


def _scenario(scenario_id: str) -> SimScenario:
    matches = [s for s in load_sim_scenarios() if s.scenario_id == scenario_id]
    assert matches, f"scenario {scenario_id} is not shipped"
    return matches[0]


def _marker(scenario: SimScenario, label: str) -> WhatIfMarker:
    for event in scenario.events:
        if isinstance(event, WhatIfMarker) and event.label == label:
            return event
    raise AssertionError(f"scenario {scenario.scenario_id} has no marker {label!r}")


def _report(
    spofs: dict[str, list[str]],
    fft: dict[str, bool],
    lost: tuple[str, ...] = (),
) -> dict[str, object]:
    """A minimal assessment report in the runner's extraction shape."""
    return {
        "services": [
            {
                "service_id": svc,
                "exposure": {
                    "unprotected_assets": assets,
                    "first_fault_tolerated": fft[svc],
                },
            }
            for svc, assets in sorted(spofs.items())
        ],
        "findings": [{"kind": "unserved_service", "subject": svc} for svc in lost],
    }


# ---------------------------------------------------------------------------
# Delta observation and adjudication
# ---------------------------------------------------------------------------


def test_delta_observed_reports_lost_restored_spofs_and_fft() -> None:
    baseline = _report({"S1": ["A"], "S2": ["B"]}, {"S1": False, "S2": False}, lost=("S1",))
    counterfactual = _report(
        {"S1": [], "S2": ["B", "C"]}, {"S1": True, "S2": False}, lost=("S2",)
    )
    observed = _delta_observed(baseline, counterfactual)
    assert observed == {
        "lost_services": ["S2"],
        "restored_services": ["S1"],
        "service_spofs_added": {"S2": ["C"]},
        "service_spofs_removed": {"S1": ["A"]},
        "service_first_fault_tolerated": {"S1": {"from": False, "to": True}},
    }


def test_delta_failures_accept_a_matching_adjudication() -> None:
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "upgrade-chillers")
    baseline = _report(
        {"SVC-CHW": ["CH-1"], "SVC-COOLING-0": ["CH-1"]},
        {"SVC-CHW": False, "SVC-COOLING-0": False},
    )
    counterfactual = _report(
        {"SVC-CHW": [], "SVC-COOLING-0": []},
        {"SVC-CHW": True, "SVC-COOLING-0": False},
    )
    failures, checks = _delta_failures(
        scenario.scenario_id,
        marker,
        _delta_observed(baseline, counterfactual),
        counterfactual,
    )
    assert failures == []
    assert checks >= 4  # every declared delta field was adjudicated


def test_delta_failures_name_scenario_label_and_field() -> None:
    """A wrong adjudication fails with the scenario, marker, field, and
    both values — the repro detail the failure output ships."""
    scenario = _scenario(AIDC_WHAT_IF_ID)
    doctored = replace(
        _marker(scenario, "upgrade-chillers"),
        expected_delta=replace(
            _marker(scenario, "upgrade-chillers").expected_delta,
            service_first_fault_tolerated={"SVC-CHW": False},
        ),
    )
    baseline = _report(
        {"SVC-CHW": ["CH-1"], "SVC-COOLING-0": ["CH-1"]},
        {"SVC-CHW": False, "SVC-COOLING-0": False},
    )
    counterfactual = _report(
        {"SVC-CHW": [], "SVC-COOLING-0": []},
        {"SVC-CHW": True, "SVC-COOLING-0": False},
    )
    failures, _ = _delta_failures(
        scenario.scenario_id,
        doctored,
        _delta_observed(baseline, counterfactual),
        counterfactual,
    )
    assert len(failures) == 1
    failure = failures[0]
    assert "sim-aidc-whatif-chiller-cascade" in failure
    assert "'upgrade-chillers'" in failure
    assert "service_first_fault_tolerated[SVC-CHW]" in failure
    assert "declared False" in failure and "observed True" in failure


# ---------------------------------------------------------------------------
# Marker purity
# ---------------------------------------------------------------------------


def test_purity_failure_flags_a_changed_report() -> None:
    report = _report({"S1": ["A"]}, {"S1": False})
    unchanged_step = TraceStep(
        step=0, version_no=1, event="", lost_services=(), report=report
    )
    assert _purity_failure(unchanged_step, report) is None
    changed = _report({"S1": []}, {"S1": False})
    failure = _purity_failure(unchanged_step, changed)
    assert failure is not None
    assert "marker purity" in failure and "step 0 (baseline)" in failure


# ---------------------------------------------------------------------------
# SPOF-path verdicts
# ---------------------------------------------------------------------------


def test_spof_path_verdict_skips_removal_markers() -> None:
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "remove-cdu")
    state = initial_state(scenario.topology)
    verdict, failures = _spof_path_verdict(
        scenario.scenario_id, marker, state, frozenset({"CDU-1"}), {}, {}
    )
    assert verdict is None
    assert failures == []


def test_spof_path_verdict_abstains_below_two_live_members() -> None:
    """With the primary already faulted, a live-member upgrade cannot be
    attributed — the verdict must abstain and say which path and member."""
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "upgrade-chillers-while-dead")
    base_state = replace(initial_state(scenario.topology), faulted=frozenset({"CH-1"}))
    cf_state = apply_intervention(base_state, marker.intervention)
    cf_dead = dead_nodes(cf_state)
    verdict, failures = _spof_path_verdict(
        scenario.scenario_id, marker, cf_state, cf_dead, {}, {}
    )
    assert failures == []
    assert verdict is not None
    assert "chw-chillers-1" in verdict
    assert "CH-1-SB" in verdict
    assert "abstained" in verdict


def test_spof_path_verdict_fails_on_residual_spof() -> None:
    """A baseline SPOF that survives the upgrade with no abstention is the
    silent wrong answer Category 5 exists to catch."""
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "upgrade-chillers")
    cf_state = apply_intervention(initial_state(scenario.topology), marker.intervention)
    baseline_report = _report({"SVC-CHW": ["CH-1"]}, {"SVC-CHW": False})
    residual_report = _report({"SVC-CHW": ["CH-1"]}, {"SVC-CHW": False})
    verdict, failures = _spof_path_verdict(
        scenario.scenario_id,
        marker,
        cf_state,
        dead_nodes(cf_state),
        baseline_report,
        residual_report,
    )
    assert verdict is None
    assert len(failures) == 1
    assert "silent wrong answer" in failures[0]
    assert "CH-1" in failures[0]


def test_spof_path_verdict_clean_when_baseline_spof_removed() -> None:
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "upgrade-chillers")
    cf_state = apply_intervention(initial_state(scenario.topology), marker.intervention)
    baseline_report = _report({"SVC-CHW": ["CH-1"]}, {"SVC-CHW": False})
    clean_report = _report({"SVC-CHW": []}, {"SVC-CHW": True})
    verdict, failures = _spof_path_verdict(
        scenario.scenario_id,
        marker,
        cf_state,
        dead_nodes(cf_state),
        baseline_report,
        clean_report,
    )
    assert verdict is None
    assert failures == []


def test_redundancy_change_event_is_upgrade_payload_shaped() -> None:
    """Marker upgrades reuse the timeline transition type — one redundancy
    semantics implementation backs both paths."""
    scenario = _scenario(AIDC_WHAT_IF_ID)
    marker = _marker(scenario, "upgrade-chillers")
    assert isinstance(marker.intervention, RedundancyChangeEvent)
    assert marker.intervention.redundancy_config == "N+1"


# ---------------------------------------------------------------------------
# End-to-end branches over the real API
# ---------------------------------------------------------------------------


def test_marker_steps_reassess_the_unchanged_state(tmp_path: Path) -> None:
    """A marker step is a purity re-assessment: its observed service loss
    equals the previous timeline step's — markers do not advance the state."""
    scenario = _scenario(AIDC_WHAT_IF_ID)
    outcome = run_sim_scenario(_client(tmp_path), scenario)
    assert outcome.passed, outcome.failures
    marker_steps = [step for step in outcome.trace if step.what_if is not None]
    assert len(marker_steps) == 3
    for step in marker_steps:
        previous = outcome.trace[step.step - 1]
        assert step.lost_services == previous.lost_services


def test_upgrade_branch_records_fft_delta_and_clean_verdict(tmp_path: Path) -> None:
    scenario = _scenario(AIDC_WHAT_IF_ID)
    outcome = run_sim_scenario(_client(tmp_path), scenario)
    assert outcome.passed, outcome.failures
    branch = next(
        step.what_if
        for step in outcome.trace
        if step.what_if is not None and step.what_if.label == "upgrade-chillers"
    )
    assert branch is not None
    assert branch.delta["lost_services"] == []
    assert branch.delta["service_first_fault_tolerated"] == {
        "SVC-CHW": {"from": False, "to": True}
    }
    assert branch.spof_path_abstention is None


def test_degraded_upgrade_branch_records_abstention(tmp_path: Path) -> None:
    scenario = _scenario(AIDC_WHAT_IF_ID)
    outcome = run_sim_scenario(_client(tmp_path), scenario)
    assert outcome.passed, outcome.failures
    branch = next(
        step.what_if
        for step in outcome.trace
        if step.what_if is not None
        and step.what_if.label == "upgrade-chillers-while-dead"
    )
    assert branch is not None
    assert branch.spof_path_abstention is not None
    assert "chw-chillers-1" in branch.spof_path_abstention
