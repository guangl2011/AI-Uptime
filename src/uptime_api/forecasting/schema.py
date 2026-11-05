"""RiskForecast — the report-contract piece of the WP2 forecasting module.

Kept in its own leaf module so the assessor package and the forecasting
package can both import it without an import cycle (assessor.schemas
re-exports it; forecasting.interface and forecasting.baseline import it
directly).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ForecastStatus(StrEnum):
    """Whether an estimate is a calibrated forecast or a stated abstention.

    Abstention reuses the scanner's pattern: when the data behind an asset is
    sparse or self-contradictory, the forecast says so instead of emitting a
    bare score. An abstained estimate's ``p_fault`` carries the fixed
    flat-prior fallback (never better than the baseline by construction) and
    the bounds widen to the full interval — maximum honesty about what is
    not known.
    """

    FORECAST = "forecast"
    ABSTAINED = "abstained"


class RiskForecast(BaseModel):
    """One WP2 risk forecast with uncertainty bounds (feature-flagged).

    Produced only when the forecasting flag is enabled. Forecasts are
    informational estimates — never actions; nothing in the MVP executes on
    them (zero autonomous control). ``basis`` names the estimator inputs so
    every forecast is as traceable as a score, and every point estimate
    carries uncertainty bounds.
    """

    asset_key: str
    horizon_days: int
    p_fault: float = Field(ge=0.0, le=1.0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    redundancy_class: str | None = None
    basis: str
    status: ForecastStatus = ForecastStatus.FORECAST
    abstention_reason: str | None = None
