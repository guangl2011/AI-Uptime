"""Forecasting module interface (WP2 groundwork, Phase 4) — feature-flagged.

Two things live here: the contract a forecasting estimator implements, and
the flag resolution that decides whether any estimator runs at all.

The flag exists only so the module can be *evaluated* by the harness and
*reviewed* in reports. Forecasts are informational estimates attached to an
assessment; no code path reads a forecast to make or execute a decision
(zero autonomous control). With the flag off — the default — no forecast is
computed and reports carry none.

Implementations must be deterministic and side-effect free: a snapshot in,
forecasts out, no I/O, no clocks, no writes anywhere.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from uptime_api.forecasting.schema import RiskForecast
from uptime_api.graph.schemas import GraphSnapshotOut

# Feature flag: UPTIME_FORECASTING=1 (case-insensitive 1/true/yes/on).
FLAG_ENV_VAR = "UPTIME_FORECASTING"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@runtime_checkable
class ForecastModel(Protocol):
    """A forecasting estimator over one immutable graph snapshot.

    The protocol is deliberately tiny: one pure method. Anything that
    mutates state, reaches a network, or executes an action is not a
    ForecastModel — the architectural test in tests/forecasting/ pins that.
    """

    def forecast(self, snapshot: GraphSnapshotOut) -> list[RiskForecast]:
        """Forecast fault risk for every asset in the snapshot."""
        ...  # pragma: no cover - protocol body


def resolve_forecaster(*, enabled: bool | None = None) -> ForecastModel | None:
    """Resolve the forecasting feature flag to a model, or None when off.

    An explicit ``enabled`` argument wins; otherwise the ``UPTIME_FORECASTING``
    environment variable decides. None means disabled — the default posture:
    the assessor runs without forecasts and reports carry none.
    """
    flag = enabled if enabled is not None else _env_flag()
    if not flag:
        return None
    from uptime_api.forecasting.baseline import FlatPriorForecastModel

    return FlatPriorForecastModel()


def _env_flag() -> bool:
    return os.environ.get(FLAG_ENV_VAR, "").strip().lower() in _TRUTHY
