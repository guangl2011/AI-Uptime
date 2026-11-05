"""WP2 risk forecasting (Phase 5) — feature-flagged, default OFF.

Public surface: the ``ForecastModel`` protocol, the flag resolver, the
calibrated estimator (fitted Beta posteriors over protection cells with
uncertainty bounds and stated abstention), the flat-prior baseline (the fixed
comparison estimator and abstention fallback), the frozen canonical table,
and the ``RiskForecast`` report contract. The module is pure computation —
no I/O, no actions, no write paths of any kind (pinned by
tests/test_readonly_posture.py and the tests/forecasting/ suite).

Forecasts are informational estimates attached to an assessment; no code
path reads a forecast to make or execute a decision (zero autonomous
control). With the flag off — the default — no forecast is computed and
reports carry none.
"""

from __future__ import annotations

from uptime_api.forecasting.baseline import FlatPriorForecastModel
from uptime_api.forecasting.calibrated import CalibratedForecastModel
from uptime_api.forecasting.canonical import CANONICAL_TABLE
from uptime_api.forecasting.interface import (
    FLAG_ENV_VAR,
    ForecastModel,
    resolve_forecaster,
)
from uptime_api.forecasting.schema import ForecastStatus, RiskForecast

__all__ = [
    "CANONICAL_TABLE",
    "FLAG_ENV_VAR",
    "CalibratedForecastModel",
    "FlatPriorForecastModel",
    "ForecastModel",
    "ForecastStatus",
    "RiskForecast",
    "resolve_forecaster",
]
