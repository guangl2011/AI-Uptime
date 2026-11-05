"""Sim suite integration tests: the full M1 flow on a reduced corpus.

These run exactly what CI runs — two isolated passes (determinism), gate
evaluation, report artifacts — on one facility per segment so the loop stays
seconds-fast. The canonical-corpus run belongs to `make sim`.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.sim.__main__ import main
from evals.sim.report import SimResults, render_json, render_markdown
from evals.sim.runner import (
    DeterminismResult,
    SweepCase,
    SweepResult,
    evaluate_sim_gates,
    run_sim_suite,
    trace_digest,
)
from fastapi.testclient import TestClient

from uptime_api.app import create_app

REDUCED_COUNTS = {"cold_chain": 1, "aidc": 1, "hospital": 1}


def test_full_suite_green_and_deterministic_on_reduced_corpus(tmp_path: Path) -> None:
    """Two fresh apps, same seed: every gate passes and the digests agree."""
    first = TestClient(create_app(f"sqlite:///{tmp_path}/first.db"))
    second = TestClient(create_app(f"sqlite:///{tmp_path}/second.db"))

    first_outcomes, first_sweep = run_sim_suite(
        first, seed=20260101, facilities_per_segment=REDUCED_COUNTS
    )
    second_outcomes, second_sweep = run_sim_suite(
        second, seed=20260101, facilities_per_segment=REDUCED_COUNTS
    )

    determinism = DeterminismResult(
        first_digest=trace_digest(first_outcomes, first_sweep),
        second_digest=trace_digest(second_outcomes, second_sweep),
    )
    assert determinism.ok, determinism.detail

    gates = evaluate_sim_gates(first_outcomes, first_sweep, determinism)
    failed = [gate for gate in gates if not gate.passed]
    assert not failed, [(gate.name, gate.detail) for gate in failed]
    # Per-segment failover gates exist for all three segments.
    assert {gate.name for gate in gates} >= {
        "cold_chain:failover_semantics",
        "aidc:failover_semantics",
        "hospital:failover_semantics",
        "cascade_traces",
        "determinism",
    }


def test_cli_writes_report_artifacts_and_exits_zero(tmp_path: Path) -> None:
    output_dir = tmp_path / "sim-results"
    exit_code = main(["--facilities=1", "--output-dir", str(output_dir)])
    assert exit_code == 0

    md = (output_dir / "sim-report.md").read_text(encoding="utf-8")
    assert "| `cascade_traces` | PASS |" in md
    assert "| `determinism` | PASS |" in md
    assert "## Failures (repro detail)" not in md

    payload = json.loads((output_dir / "sim-report.json").read_text(encoding="utf-8"))
    assert payload["gates"]
    assert all(gate["passed"] for gate in payload["gates"])
    assert payload["sweep"]["cases"] > 0


def _results_with_sweep_mismatch(seed: int) -> SimResults:
    case = SweepCase(
        facility_id="sim-f",
        segment="aidc",
        asset_key="CH-1",
        expected_exposed=("S1",),
        expected_impacted=("S1",),
        observed_lost=(),
    )
    sweep = SweepResult(
        seed=seed, facilities=1, facilities_by_segment={"aidc": 1}, cases=(case,)
    )
    determinism = DeterminismResult(first_digest="a" * 64, second_digest="a" * 64)
    gates = evaluate_sim_gates((), sweep, determinism)
    return SimResults(
        seed=seed,
        # Deliberately differs from sweep.facilities_by_segment: the report's
        # Facilities column must carry corpus truth, not the request config.
        facilities_per_segment={"aidc": 99},
        scenarios=(),
        sweep=sweep,
        determinism=determinism,
        gates=gates,
    )


def test_report_carries_repro_detail_on_failure() -> None:
    """A gate failure ships its repro case: facility, fault, expected vs observed."""
    results = _results_with_sweep_mismatch(seed=7)
    md = render_markdown(results)
    assert "## Failures (repro detail)" in md
    assert "`sim-f` fault `CH-1`" in md
    assert "aidc:failover_semantics" in md

    payload = render_json(results)
    assert payload["sweep"]["mismatches"][0]["asset_key"] == "CH-1"
    assert payload["sweep"]["facilities_by_segment"] == {"aidc": 1}
    aidc_gate = next(
        gate for gate in payload["gates"] if gate["name"] == "aidc:failover_semantics"
    )
    assert not aidc_gate["passed"]


def test_facilities_column_reports_corpus_truth_not_request_config() -> None:
    """The sweep table's Facilities column counts corpus facilities, never
    the requested per-segment config (which is empty for canonical runs)."""
    results = _results_with_sweep_mismatch(seed=7)
    md = render_markdown(results)
    assert "| aidc | 1 | 1 | 0/1 | FAIL |" in md
