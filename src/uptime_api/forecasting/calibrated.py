"""Calibrated WP2 forecaster: fitted Beta posteriors over protection cells.

The Phase 5 replacement for the flat-prior placeholder. Where the placeholder
emitted fixed engineering priors (miscalibrated — its Brier score was *worse*
than the flat-prior baseline it must beat), this model reads each asset's
effective protection status from the snapshot (``features.py``) and emits the
cell's fitted Beta posterior: posterior mean as the point estimate, a ±2-sigma
Wald band as the uncertainty bounds, both clipped to [0, 1] only.

Abstention reuses the scanner's pattern — a stated reason, never a bare score
— in two cases:

- **Conflicting evidence**: an edge declares N+1/2N but its group membership
  contradicts the declaration (< 2 members, or no group). The snapshot is
  self-contradictory, so no calibrated probability is asserted; the estimate
  falls back to the flat-prior value for the asset's redundancy class (the
  fixed baseline — abstention can never score better than the baseline) with
  full-width bounds.
- **Sparse calibration data**: the cell behind the asset's status saw fewer
  than ``MIN_CELL_CASES`` training cases (or is absent from the table
  entirely). A posterior over a handful of cases is a guess with a formula
  attached; abstain instead.

Honesty note, carried by every ``basis`` string: the posteriors are fit on
the *synthetic* corpus, whose labels are by-construction — the numbers prove
the calibration machinery, not field performance. The model stays a pure
function of its input snapshot and table: no I/O, no clocks, nothing
executes on its output (informational only, zero autonomous control).
"""

from __future__ import annotations

from uptime_api.forecasting.baseline import HORIZON_DAYS, PRIORS
from uptime_api.forecasting.calibration import CalibrationTable
from uptime_api.forecasting.canonical import CANONICAL_TABLE
from uptime_api.forecasting.features import AssetFeatures, asset_protection_features
from uptime_api.forecasting.schema import ForecastStatus, RiskForecast
from uptime_api.graph.enums import RedundancyConfig
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut

# A cell with fewer training cases than this cannot back a point estimate —
# the forecast abstains (sparse-data reason) instead of guessing.
MIN_CELL_CASES = 30
# ±2-sigma band over the cell's Beta posterior, clipped to [0, 1] only (the
# baseline's tighter cosmetic clip would collapse near-boundary posteriors
# into a degenerate interval that excludes its own mean).
BOUND_SIGMA = 2.0

_SYNTHETIC_ONLY = "synthetic-corpus calibration only — by-construction labels, not field data"


def flat_prior_fallback(redundancy_class: str) -> float:
    """The fixed flat-prior estimate for a redundancy class.

    The abstention fallback: an abstained forecast reports the baseline's
    value, so abstention never scores better than the baseline it defers to.
    """
    alpha, beta = PRIORS[RedundancyConfig(redundancy_class)]
    return alpha / (alpha + beta)


class CalibratedForecastModel:
    """Feature-flagged WP2 model: fitted posteriors, uncertainty bounds,
    stated abstention. Pure snapshot in, forecasts out.

    ``table`` defaults to the frozen canonical fit (``canonical.py``); the
    eval harness passes its own freshly fitted table so the holdout score
    reflects the fitting machinery, not shipped constants.
    """

    def __init__(self, table: CalibrationTable | None = None) -> None:
        self.table = table if table is not None else CANONICAL_TABLE

    def forecast(self, snapshot: GraphSnapshotOut) -> list[RiskForecast]:
        out_edges: dict[str, list[EdgeOut]] = {}
        for edge in snapshot.edges:
            out_edges.setdefault(edge.from_node, []).append(edge)
        return [
            self._forecast_one(feature, out_edges.get(feature.asset_key, []))
            for feature in asset_protection_features(snapshot)
        ]

    def _forecast_one(self, feature: AssetFeatures, edges: list[EdgeOut]) -> RiskForecast:
        p_fault, lower, upper, abstention_reason = self._estimate(feature, edges)
        status = ForecastStatus.ABSTAINED if abstention_reason else ForecastStatus.FORECAST
        return RiskForecast(
            asset_key=feature.asset_key,
            horizon_days=HORIZON_DAYS,
            p_fault=p_fault,
            lower=lower,
            upper=upper,
            redundancy_class=feature.redundancy_class,
            basis=self._basis(feature, abstention_reason),
            status=status,
            abstention_reason=abstention_reason,
        )

    def _estimate(
        self, feature: AssetFeatures, edges: list[EdgeOut]
    ) -> tuple[float, float, float, str | None]:
        """(p_fault, lower, upper, abstention_reason) for one asset."""
        if feature.conflict is not None:
            return (
                flat_prior_fallback(feature.redundancy_class),
                0.0,
                1.0,
                f"conflicting evidence: {feature.conflict}",
            )
        cell = self.table.cell_for(feature.status)
        if cell is None or cell.cases < MIN_CELL_CASES:
            seen = 0 if cell is None else cell.cases
            return (
                flat_prior_fallback(feature.redundancy_class),
                0.0,
                1.0,
                f"insufficient calibration data: status {feature.status.value} "
                f"has {seen} training case(s), need >= {MIN_CELL_CASES}",
            )
        spread = BOUND_SIGMA * cell.sd
        return (
            cell.mean,
            max(0.0, cell.mean - spread),
            min(1.0, cell.mean + spread),
            None,
        )

    def _basis(self, feature: AssetFeatures, abstention_reason: str | None) -> str:
        fit = (
            f"Beta posterior for protection status {feature.status.value} "
            f"(seed {self.table.seed}, {self.table.train_facilities} train facilities)"
        )
        if abstention_reason is not None:
            return f"{fit}; abstained — {abstention_reason}; {_SYNTHETIC_ONLY}"
        return f"{fit}; {_SYNTHETIC_ONLY}"
