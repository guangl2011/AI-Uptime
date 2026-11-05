"""Unit tests for calibration fitting and the facility-level split."""

from __future__ import annotations

import pytest

from uptime_api.forecasting.calibration import (
    CalibrationCase,
    CalibrationCell,
    fit_calibration,
    train_eval_split,
)
from uptime_api.forecasting.features import ProtectionStatus


def _case(status: ProtectionStatus, label: int, facility: str = "f1") -> CalibrationCase:
    return CalibrationCase(facility_id=facility, status=status, label=label)


def test_fit_laplace_smoothing_keeps_single_outcome_cells_interior() -> None:
    cases = [_case(ProtectionStatus.HAS_UNPROTECTED, 1)]
    table = fit_calibration(cases, seed=1, train_facilities=1)
    cell = table.cell_for(ProtectionStatus.HAS_UNPROTECTED)
    assert cell is not None
    # Beta(1+1, 1): mean 2/3, not the degenerate 1.0 a raw MLE would give.
    assert cell.mean == pytest.approx(2 / 3)


def test_fit_posterior_mean_matches_counts() -> None:
    cases = [
        _case(ProtectionStatus.HAS_UNPROTECTED, 1, "a"),
        _case(ProtectionStatus.HAS_UNPROTECTED, 1, "a"),
        _case(ProtectionStatus.HAS_UNPROTECTED, 0, "b"),
        _case(ProtectionStatus.HAS_UNPROTECTED, 0, "b"),
        _case(ProtectionStatus.HAS_UNPROTECTED, 1, "c"),
    ]
    table = fit_calibration(cases, seed=1, train_facilities=3)
    cell = table.cell_for(ProtectionStatus.HAS_UNPROTECTED)
    assert cell is not None
    assert cell.cases == 5
    assert cell.positives == 3
    assert cell.mean == pytest.approx((3 + 1) / (5 + 2))


def test_fit_sd_shrinks_as_cases_grow() -> None:
    few = fit_calibration([_case(ProtectionStatus.ALL_PROTECTED, 0)], seed=1, train_facilities=1)
    many = fit_calibration(
        [_case(ProtectionStatus.ALL_PROTECTED, 0, f"f{i}") for i in range(100)],
        seed=1,
        train_facilities=100,
    )
    few_cell = few.cell_for(ProtectionStatus.ALL_PROTECTED)
    many_cell = many.cell_for(ProtectionStatus.ALL_PROTECTED)
    assert few_cell is not None and many_cell is not None
    assert many_cell.sd < few_cell.sd


def test_fit_absent_status_has_no_cell() -> None:
    table = fit_calibration([_case(ProtectionStatus.NO_OUT_EDGES, 0)], seed=1, train_facilities=1)
    assert table.cell_for(ProtectionStatus.ALL_PROTECTED) is None


def test_fit_is_deterministic() -> None:
    cases = [
        _case(ProtectionStatus.HAS_UNPROTECTED, 1, "a"),
        _case(ProtectionStatus.ALL_PROTECTED, 0, "b"),
    ]
    first = fit_calibration(cases, seed=7, train_facilities=2)
    second = fit_calibration(cases, seed=7, train_facilities=2)
    for status in ProtectionStatus:
        assert first.cell_for(status) == second.cell_for(status)


def test_split_is_deterministic_and_disjoint() -> None:
    ids = [f"facility-{i:02d}" for i in range(40)]
    train_a, eval_a = train_eval_split(ids, seed=42)
    train_b, eval_b = train_eval_split(ids, seed=42)
    assert train_a == train_b
    assert eval_a == eval_b
    assert not set(train_a) & set(eval_a)
    assert sorted(train_a + eval_a) == sorted(ids)


def test_split_is_facility_level_not_case_level() -> None:
    # Every id appears on exactly one side — no id straddles the split.
    ids = [f"facility-{i:02d}" for i in range(80)]
    train, holdout = train_eval_split(ids, seed=7)
    assert len(train) == 60 and len(holdout) == 20  # canonical 75/25 on 80


def test_split_rejects_degenerate_fractions() -> None:
    with pytest.raises(ValueError, match="eval_fraction"):
        train_eval_split(["a"], seed=1, eval_fraction=0.0)
    with pytest.raises(ValueError, match="eval_fraction"):
        train_eval_split(["a"], seed=1, eval_fraction=1.0)


def test_cell_is_frozen_value() -> None:
    cell = CalibrationCell(status=ProtectionStatus.ALL_PROTECTED, cases=10, positives=2)
    assert cell.mean == pytest.approx(3 / 12)
    assert cell.posterior_alpha == 3.0
    assert cell.posterior_beta == 9.0
