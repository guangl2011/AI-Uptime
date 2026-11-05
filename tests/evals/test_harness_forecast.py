"""Forecast-pass tests: WP2 go/no-go gates and report sections (Phase 5).

Unit tests pin the gate helper's failure modes with constructed metrics;
the integration test runs the real harness on a small corpus and checks
the forecast gates evaluate, pass, and render into both report formats
with the synthetic-only labeling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.harness import (
    FORECAST_BRIER_RELATIVE_GATE,
    CalibrationReport,
    ClassificationMetrics,
    ForecastEval,
    GateResult,
    HarnessConfig,
    HarnessResults,
    _forecast_gates,
    evaluate_gates,
    evaluate_segment_gates,
    render_json,
    render_markdown,
    run_harness,
)
from evals.metrics import ReliabilityBin
from fastapi.testclient import TestClient

from uptime_api.app import create_app
from uptime_api.forecasting.schema import ForecastStatus

# ---------------------------------------------------------------- gate helper


def _classification(**overrides: float | int | tuple[float, float]) -> ClassificationMetrics:
    values: dict[str, float | int | tuple[float, float]] = {
        "cases": 100,
        "positives": 50,
        "auroc": 0.95,
        "auroc_ci": (0.9, 1.0),
        "brier": 0.10,
        "brier_ci": (0.08, 0.12),
        "baseline_brier": 0.25,
        "brier_improvement": 0.15,
        "brier_improvement_ci": (0.10, 0.20),
    }
    values.update(overrides)
    return ClassificationMetrics(**values)  # type: ignore[arg-type]


def _forecast_eval(
    classification: ClassificationMetrics | None = None,
    *,
    cases: int = 100,
    bin_cases: int | None = None,
    abstentions: int = 0,
) -> ForecastEval:
    bins = (
        ReliabilityBin(
            lower=0.0,
            upper=1.0,
            n=bin_cases if bin_cases is not None else cases,
            mean_predicted=0.5,
            observed_rate=0.5,
        ),
    )
    return ForecastEval(
        model="CalibratedForecastModel",
        train_facilities=60,
        eval_facilities=20,
        cases=cases,
        abstentions=abstentions,
        overall=classification if classification is not None else _classification(),
        per_segment={},
        calibration=CalibrationReport(bins=bins, ece=0.02),
        per_segment_abstentions={},
    )


def _by_name(gates: list[GateResult]) -> dict[str, GateResult]:
    return {gate.name: gate for gate in gates}


def test_missing_forecast_pass_is_a_gate_failure_not_a_skip() -> None:
    gates = _forecast_gates(None)
    assert len(gates) == 1
    assert gates[0].name == "forecast_pass_ran"
    assert not gates[0].passed


def test_brier_gate_requires_relative_threshold_and_ci_above_zero() -> None:
    passing = _forecast_gates(_forecast_eval())
    by_name = _by_name(passing)
    assert by_name["forecast_brier_improvement"].passed  # 40% relative, CI > 0

    worse_than_baseline = _forecast_gates(
        _forecast_eval(
            _classification(brier=0.26, brier_improvement=-0.01, brier_improvement_ci=(-0.03, 0.01))
        )
    )
    assert not _by_name(worse_than_baseline)["forecast_brier_improvement"].passed

    ci_touching_zero = _forecast_gates(
        _forecast_eval(_classification(brier_improvement_ci=(-0.01, 0.20)))
    )
    assert not _by_name(ci_touching_zero)["forecast_brier_improvement"].passed


def test_auroc_gate_uses_the_holdout_score() -> None:
    gates = _forecast_gates(_forecast_eval(_classification(auroc=0.5, auroc_ci=(0.4, 0.6))))
    by_name = _by_name(gates)
    assert not by_name["forecast_auroc"].passed
    assert "held-out" in by_name["forecast_auroc"].detail


def test_calibration_gate_covers_cases() -> None:
    assert _by_name(_forecast_gates(_forecast_eval(cases=100)))["forecast_calibration"].passed
    # The reliability table covers only half the holdout cases — a report
    # that cannot account for every scored case must fail the gate.
    partial = _forecast_gates(_forecast_eval(_classification(), cases=200, bin_cases=100))
    assert not _by_name(partial)["forecast_calibration"].passed


def test_gate_threshold_is_the_proposal_fifteen_percent() -> None:
    assert FORECAST_BRIER_RELATIVE_GATE == pytest.approx(0.15)


# ------------------------------------------------------- integration (real run)

FACILITY_COUNTS = {"cold_chain": 6, "aidc": 3, "hospital": 3}


@pytest.fixture(scope="module")
def small_harness_results(tmp_path_factory: pytest.TempPathFactory) -> HarnessResults:
    # File-based DB like tests/conftest.py: ``:memory:`` gives each pooled
    # connection its own empty schema.
    db_path = tmp_path_factory.mktemp("forecast-harness") / "harness.db"
    with TestClient(create_app(f"sqlite:///{db_path}")) as client:
        return run_harness(client, HarnessConfig(facility_counts=FACILITY_COUNTS))


def test_harness_runs_the_forecast_pass_on_the_holdout(
    small_harness_results: HarnessResults,
) -> None:
    forecast = small_harness_results.forecast_eval
    assert forecast is not None
    assert forecast.model == "CalibratedForecastModel"
    assert forecast.train_facilities + forecast.eval_facilities == sum(FACILITY_COUNTS.values())
    assert forecast.cases > 0
    # With only 9 training facilities the no_out_edges cell holds ~20 cases,
    # below MIN_CELL_CASES: the model abstains there instead of guessing
    # (verified against this exact corpus). Abstained cases are excluded
    # from the scored holdout but reported per segment.
    assert forecast.abstentions > 0
    assert sum(forecast.per_segment_abstentions.values()) == forecast.abstentions


def test_harness_forecast_gates_pass_on_the_canonical_path(
    small_harness_results: HarnessResults,
) -> None:
    # Same combined list the runner gates on (evals/__main__.py).
    gates = _by_name(
        evaluate_gates(small_harness_results) + evaluate_segment_gates(small_harness_results)
    )
    for name in ("forecast_auroc", "forecast_brier_improvement", "forecast_calibration"):
        assert name in gates, f"{name} missing from the overall gates"
        assert gates[name].passed, f"{name}: {gates[name].detail}"
    # Per-segment forecast gates apply only to segments with held-out cases:
    # on this 12-facility corpus the 25% holdout is all cold-chain, so the
    # other segments carry no forecast metrics by design.
    forecast = small_harness_results.forecast_eval
    assert forecast is not None
    assert forecast.per_segment, "no segment had held-out forecast cases"
    for segment in forecast.per_segment:
        for name in ("forecast_auroc", "forecast_brier_improvement"):
            gate = gates[f"{segment}:{name}"]
            assert gate.passed, gate.detail
    # Segments with no held-out cases emit no forecast gates at all.
    unscored = set(FACILITY_COUNTS) - set(forecast.per_segment)
    for segment in unscored:
        assert f"{segment}:forecast_auroc" not in gates


def test_forecast_report_sections_render_with_synthetic_only_labeling(
    small_harness_results: HarnessResults,
) -> None:
    payload = render_json(small_harness_results)
    assert isinstance(payload, dict)
    forecast_payload = payload["forecast_eval"]
    assert isinstance(forecast_payload, dict)
    assert forecast_payload["synthetic_only"] is True
    assert forecast_payload["gated"] is True
    assert forecast_payload["model"] == "CalibratedForecastModel"
    assert "calibration" in forecast_payload

    markdown = render_markdown(small_harness_results)
    assert "Phase 5 go/no-go gates apply" in markdown
    assert "Synthetic-only" in markdown
    assert "Forecast reliability table (holdout, synthetic)" in markdown
    assert "nothing executes on them" in markdown


def test_flag_stays_default_off_after_the_forecast_pass(
    small_harness_results: HarnessResults, tmp_path: Path
) -> None:
    # run_harness installs the fitted model only inside the forecast pass;
    # a fresh app resolves to no forecaster (default-off posture).
    del small_harness_results
    with TestClient(create_app(f"sqlite:///{tmp_path / 'default-off.db'}")) as client:
        assert client.app.state.forecaster is None  # type: ignore[attr-defined]


def test_forecast_statuses_serialize_to_their_stated_values() -> None:
    assert ForecastStatus.FORECAST.value == "forecast"
    assert ForecastStatus.ABSTAINED.value == "abstained"
