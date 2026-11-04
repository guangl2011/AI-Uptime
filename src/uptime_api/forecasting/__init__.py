"""WP2 risk-forecasting groundwork (Phase 4) — feature-flagged, default OFF.

Public surface: the ``ForecastModel`` protocol, the flag resolver, the
baseline estimator, and the ``RiskForecast`` report contract. The module is
pure computation — no I/O, no actions, no write paths of any kind (the
architectural tests in tests/forecasting/ pin this alongside
tests/test_readonly_posture.py).
"""

from __future__ import annotations

from uptime_api.forecasting.baseline import FlatPriorForecastModel
from uptime_api.forecasting.interface import (
    FLAG_ENV_VAR,
    ForecastModel,
    resolve_forecaster,
)
from uptime_api.forecasting.schema import RiskForecast

__all__ = [
    "FLAG_ENV_VAR",
    "FlatPriorForecastModel",
    "ForecastModel",
    "RiskForecast",
    "resolve_forecaster",
]
