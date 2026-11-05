"""WP2 v0 estimator: flat prior per redundancy class with uncertainty bounds.

This is the testing plan's fixed baseline ("flat prior per redundancy class")
formalized as the first ``ForecastModel``. It is deliberately *not* a learned
component: no labels are consumed, nothing is trained, and its calibration is
expected to be mediocre — that is what a baseline is for. WP2's real
estimators (state estimation over evidence windows) drop in behind the same
``ForecastModel`` protocol; the harness evaluates whatever the flag enables
and reports calibration without gating it (locked testing plan: probability-
bearing outputs stay reported-not-gated until labeled data exists).

Class semantics mirror the assessor's propagation rules: an asset's fault is
absorbed exactly when its outgoing supply edges are members of a redundant
group (>= 2 members, N+1/2N), so the class of an asset is the *weakest* of
its outgoing edges' configurations — the same weakest-link notion the
assessor uses for services.
"""

from __future__ import annotations

import math

from uptime_api.forecasting.features import redundancy_class_of
from uptime_api.forecasting.schema import RiskForecast
from uptime_api.graph.enums import RedundancyConfig
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut, NodeOut

# Beta-prior pseudo-counts per redundancy class: (alpha, beta). The means are
# engineering priors for a 90-day horizon (unprotected supply fails more often
# than protected), not fitted values — there is no data to fit yet.
PRIORS: dict[RedundancyConfig, tuple[float, float]] = {
    RedundancyConfig.N: (3.0, 7.0),  # mean 0.30
    RedundancyConfig.N_PLUS_1: (1.2, 8.8),  # mean 0.12
    RedundancyConfig.N2: (0.5, 9.5),  # mean 0.05
}
HORIZON_DAYS = 90
# Wald ±2-sigma band over the Beta prior, clipped into a sane interval —
# honest "wide, not calibrated" bounds without a scipy dependency.
SIGMA_MULTIPLIER = 2.0
BOUND_FLOOR, BOUND_CEILING = 0.01, 0.99


def _group_label(out_edges: list[EdgeOut]) -> str:
    for edge in out_edges:
        if edge.member_group:
            return str(edge.member_group)
    return "no group"


def _bounds(mean: float, alpha: float, beta: float) -> tuple[float, float]:
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    spread = SIGMA_MULTIPLIER * math.sqrt(variance)
    lower = min(BOUND_CEILING, max(BOUND_FLOOR, mean - spread))
    upper = max(BOUND_FLOOR, min(BOUND_CEILING, mean + spread))
    return lower, upper


class FlatPriorForecastModel:
    """Deterministic per-asset fault-prior forecasts; pure snapshot in."""

    def forecast(self, snapshot: GraphSnapshotOut) -> list[RiskForecast]:
        out_edges: dict[str, list[EdgeOut]] = {}
        for edge in snapshot.edges:
            out_edges.setdefault(edge.from_node, []).append(edge)

        forecasts: list[RiskForecast] = []
        for node in sorted(snapshot.nodes, key=_node_sort_key):
            if node.node_type.value != "asset":
                continue
            edges = out_edges.get(node.node_key, [])
            asset_class = redundancy_class_of(edges)
            alpha, beta = PRIORS[asset_class]
            mean = alpha / (alpha + beta)
            lower, upper = _bounds(mean, alpha, beta)
            forecasts.append(
                RiskForecast(
                    asset_key=node.node_key,
                    horizon_days=HORIZON_DAYS,
                    p_fault=mean,
                    lower=lower,
                    upper=upper,
                    redundancy_class=asset_class.value,
                    basis=(
                        f"flat Beta prior for redundancy class {asset_class.value} "
                        f"(group {_group_label(edges)}); WP2 v0 baseline — "
                        "no learned component"
                    ),
                )
            )
        return forecasts


def _node_sort_key(node: NodeOut) -> str:
    return node.node_key
