"""Unit tests for the calibrated forecaster (forecasting/calibrated.py)."""

from __future__ import annotations

import pytest
from snapshot_factories import asset, edge, service, snapshot

from uptime_api.forecasting.baseline import HORIZON_DAYS, PRIORS
from uptime_api.forecasting.calibrated import (
    BOUND_SIGMA,
    MIN_CELL_CASES,
    CalibratedForecastModel,
    flat_prior_fallback,
)
from uptime_api.forecasting.canonical import CANONICAL_TABLE
from uptime_api.forecasting.features import ProtectionStatus
from uptime_api.forecasting.schema import ForecastStatus
from uptime_api.graph.enums import RedundancyConfig

# A realistic cell: 100 cases, 90 positive — a confident but interior posterior.
_REALISTIC_CELL = (100, 90)


def _fitted_model(*cells: tuple[ProtectionStatus, tuple[int, int]]) -> CalibratedForecastModel:
    from uptime_api.forecasting.calibration import CalibrationCell, CalibrationTable

    return CalibratedForecastModel(
        CalibrationTable(
            seed=1,
            train_facilities=10,
            cells={
                status: CalibrationCell(status=status, cases=cases, positives=positives)
                for status, (cases, positives) in cells
            },
        )
    )


def test_calibrated_point_estimate_is_the_posterior_mean() -> None:
    model = _fitted_model(
        (ProtectionStatus.HAS_UNPROTECTED, _REALISTIC_CELL),
        (ProtectionStatus.ALL_PROTECTED, (60, 0)),
        (ProtectionStatus.NO_OUT_EDGES, (40, 0)),
    )
    snap = snapshot([asset("a-1")], [edge("a-1", "svc-1", RedundancyConfig.N, "g")])
    forecasts = model.forecast(snap)
    assert len(forecasts) == 1
    forecast = forecasts[0]
    assert forecast.status is ForecastStatus.FORECAST
    assert forecast.abstention_reason is None
    assert forecast.p_fault == pytest.approx(91 / 102)  # Beta(90+1, 10+1)
    assert forecast.redundancy_class == RedundancyConfig.N.value
    assert forecast.horizon_days == HORIZON_DAYS


def test_bounds_bracket_the_mean_and_stay_in_the_unit_interval() -> None:
    model = _fitted_model(
        (ProtectionStatus.HAS_UNPROTECTED, (200, 200)),
        (ProtectionStatus.ALL_PROTECTED, (200, 0)),
        (ProtectionStatus.NO_OUT_EDGES, (200, 0)),
    )
    snap = snapshot([asset("a-1")], [edge("a-1", "svc-1", RedundancyConfig.N, "g")])
    forecast = model.forecast(snap)[0]
    assert forecast.lower < forecast.p_fault < forecast.upper
    assert 0.0 <= forecast.lower <= forecast.upper <= 1.0
    # ±2-sigma band, not the baseline's cosmetic 0.01/0.99 clip.

    cell = model.table.cell_for(ProtectionStatus.HAS_UNPROTECTED)
    assert cell is not None
    expected_spread = BOUND_SIGMA * cell.sd
    assert forecast.p_fault - forecast.lower == pytest.approx(expected_spread, abs=1e-9)


def test_conflicting_evidence_abstains_with_reason_and_fallback_value() -> None:
    model = _fitted_model(
        (ProtectionStatus.HAS_UNPROTECTED, _REALISTIC_CELL),
        (ProtectionStatus.ALL_PROTECTED, _REALISTIC_CELL),
        (ProtectionStatus.NO_OUT_EDGES, _REALISTIC_CELL),
    )
    # Declares N+1 but its group has one member — self-contradictory snapshot.
    snap = snapshot([asset("a-1")], [edge("a-1", "svc-1", RedundancyConfig.N_PLUS_1, "solo")])
    forecast = model.forecast(snap)[0]
    assert forecast.status is ForecastStatus.ABSTAINED
    assert forecast.abstention_reason is not None
    assert "conflicting evidence" in forecast.abstention_reason
    # The fallback equals the fixed baseline's estimate for the declared class.
    assert forecast.p_fault == pytest.approx(
        PRIORS[RedundancyConfig.N_PLUS_1][0]
        / (PRIORS[RedundancyConfig.N_PLUS_1][0] + PRIORS[RedundancyConfig.N_PLUS_1][1])
    )
    assert forecast.lower == 0.0 and forecast.upper == 1.0
    assert "abstained" in forecast.basis


def test_sparse_cell_abstains_instead_of_guessing() -> None:
    model = _fitted_model(
        (ProtectionStatus.HAS_UNPROTECTED, (5, 5)),  # < MIN_CELL_CASES
        (ProtectionStatus.ALL_PROTECTED, (60, 0)),
        (ProtectionStatus.NO_OUT_EDGES, (40, 0)),
    )
    assert MIN_CELL_CASES > 5
    snap = snapshot([asset("a-1")], [edge("a-1", "svc-1", RedundancyConfig.N, "g")])
    forecast = model.forecast(snap)[0]
    assert forecast.status is ForecastStatus.ABSTAINED
    assert forecast.abstention_reason is not None
    assert "insufficient calibration data" in forecast.abstention_reason
    assert "5 training case(s)" in forecast.abstention_reason
    assert forecast.p_fault == pytest.approx(0.30)  # flat-prior fallback for N


def test_unknown_status_abstains() -> None:
    model = _fitted_model((ProtectionStatus.HAS_UNPROTECTED, _REALISTIC_CELL))
    snap = snapshot([asset("a-1")], [])  # NO_OUT_EDGES has no cell in this table
    forecast = model.forecast(snap)[0]
    assert forecast.status is ForecastStatus.ABSTAINED
    assert "insufficient calibration data" in (forecast.abstention_reason or "")


def test_canonical_table_defaults_and_clean_forecast_has_no_abstention() -> None:
    model = CalibratedForecastModel()
    assert model.table is CANONICAL_TABLE
    snap = snapshot(
        [asset("a-1"), service("svc-1")],
        [edge("a-1", "svc-1", RedundancyConfig.N, "g")],
    )
    forecast = model.forecast(snap)[0]
    assert forecast.status is ForecastStatus.FORECAST
    assert forecast.abstention_reason is None
    assert "synthetic-corpus calibration only" in forecast.basis


def test_forecasts_are_deterministic() -> None:
    model = _fitted_model(
        (ProtectionStatus.HAS_UNPROTECTED, _REALISTIC_CELL),
        (ProtectionStatus.ALL_PROTECTED, (60, 0)),
        (ProtectionStatus.NO_OUT_EDGES, (40, 0)),
    )
    snap = snapshot(
        [asset("b-1"), asset("a-2"), asset("a-3"), service("svc-1")],
        [
            edge("b-1", "svc-1", RedundancyConfig.N, "g"),
            edge("a-2", "svc-1", RedundancyConfig.N2, "pair"),
            edge("a-3", "svc-1", RedundancyConfig.N2, "pair"),
        ],
    )
    first = [f.model_dump(mode="json") for f in model.forecast(snap)]
    second = [f.model_dump(mode="json") for f in model.forecast(snap)]
    assert first == second


def test_flat_prior_fallback_matches_baseline_priors() -> None:
    assert flat_prior_fallback(RedundancyConfig.N.value) == pytest.approx(0.30)
    assert flat_prior_fallback(RedundancyConfig.N_PLUS_1.value) == pytest.approx(0.12)
    assert flat_prior_fallback(RedundancyConfig.N2.value) == pytest.approx(0.05)
