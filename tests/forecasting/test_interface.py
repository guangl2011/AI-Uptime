"""Unit tests for the forecasting flag resolver (forecasting/interface.py)."""

from __future__ import annotations

import pytest

from uptime_api.forecasting import (
    CalibratedForecastModel,
    FlatPriorForecastModel,
    resolve_forecaster,
)


def test_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPTIME_FORECASTING", raising=False)
    assert resolve_forecaster() is None


def test_env_flag_on_resolves_the_calibrated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("UPTIME_FORECASTING", value)
        model = resolve_forecaster()
        assert isinstance(model, CalibratedForecastModel)


def test_env_flag_off_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("", "0", "false", "off", "nonsense"):
        monkeypatch.setenv("UPTIME_FORECASTING", value)
        assert resolve_forecaster() is None


def test_explicit_argument_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UPTIME_FORECASTING", "1")
    assert resolve_forecaster(enabled=False) is None
    monkeypatch.delenv("UPTIME_FORECASTING", raising=False)
    assert isinstance(resolve_forecaster(enabled=True), CalibratedForecastModel)


def test_baseline_model_remains_available_for_the_eval() -> None:
    # The flat-prior model is no longer flag-wired, but it stays the fixed
    # comparison baseline and the abstention fallback.
    assert FlatPriorForecastModel is not None
