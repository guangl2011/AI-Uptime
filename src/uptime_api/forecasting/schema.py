"""RiskForecast — the report-contract piece of the WP2 forecasting module.

Kept in its own leaf module so the assessor package and the forecasting
package can both import it without an import cycle (assessor.schemas
re-exports it; forecasting.interface and forecasting.baseline import it
directly).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskForecast(BaseModel):
    """One WP2 risk forecast with uncertainty bounds (feature-flagged).

    Produced only when the forecasting flag is enabled. Forecasts are
    informational estimates — never actions; nothing in the MVP executes on
    them (zero autonomous control). ``basis`` names the estimator inputs so
    every forecast is as traceable as a score.
    """

    asset_key: str
    horizon_days: int
    p_fault: float = Field(ge=0.0, le=1.0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    redundancy_class: str | None = None
    basis: str
