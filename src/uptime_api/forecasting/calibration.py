"""Calibration fitting for the WP2 forecaster: labeled cases in, table out.

The Phase 4 placeholder emitted fixed engineering priors (0.30/0.12/0.05 per
redundancy class) that are miscalibrated against even the synthetic corpus —
its Brier score was *worse* than the flat-prior baseline. This module
replaces guessed priors with fitted Beta posteriors over protection-status
cells:

- Cases are (protection status, label) pairs — one per asset per training
  facility. The label is the by-construction answer the corpus adjudicates
  from construction intent (does this single fault interrupt >= 1 service?).
- Each cell's posterior is Beta(positives + 1, negatives + 1) — a Beta(1, 1)
  Laplace prior keeps every cell's mean strictly inside (0, 1) and prices
  the possibility that the training distribution understates the truth.
- The fit is a facility-level holdout discipline: the harness fits on the
  training facilities' cases and scores on facilities the fit never saw, so
  the eval measures out-of-sample calibration, not memory.

Everything here is pure and deterministic: no I/O, no clocks, no product
state. ``train_eval_split`` is the shared split rule — the frozen canonical
constants (``canonical.py``) cite it, and the harness re-derives the same
split from the same seed at eval time.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from uptime_api.forecasting.features import ProtectionStatus

# Beta(1, 1) Laplace prior: one pseudo-positive and one pseudo-negative per
# cell, so a cell that saw only one outcome still admits the other.
LAPLACE_ALPHA = 1.0
LAPLACE_BETA = 1.0

# Canonical holdout: 75% of facilities fit the table, 25% score it.
CANONICAL_EVAL_FRACTION = 0.25


@dataclass(frozen=True)
class CalibrationCase:
    """One labeled training case: an asset's protection status and its
    by-construction single-fault label (1 = interrupts a service)."""

    facility_id: str
    status: ProtectionStatus
    label: int


@dataclass(frozen=True)
class CalibrationCell:
    """One protection-status cell's training evidence and Beta posterior."""

    status: ProtectionStatus
    cases: int
    positives: int

    @property
    def posterior_alpha(self) -> float:
        return self.positives + LAPLACE_ALPHA

    @property
    def posterior_beta(self) -> float:
        return self.cases - self.positives + LAPLACE_BETA

    @property
    def mean(self) -> float:
        """Posterior mean — the calibrated point estimate for the cell."""
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)

    @property
    def sd(self) -> float:
        """Posterior standard deviation — the uncertainty behind the bounds.

        ``math.sqrt`` rather than ``** 0.5``: this mypy/typeshed combination
        types ``float.__pow__`` as Any, and an untyped exponent would leak
        Any into the uncertainty bounds.
        """
        alpha = float(self.posterior_alpha)
        beta = float(self.posterior_beta)
        total = alpha + beta
        variance = (alpha * beta) / (total * total * (total + 1.0))
        return math.sqrt(variance)


@dataclass(frozen=True)
class CalibrationTable:
    """Fitted calibration over protection-status cells.

    ``seed`` and ``train_facilities`` carry the fit's provenance so a table
    can always be tied back to the corpus and split that produced it.
    """

    seed: int
    train_facilities: int
    cells: dict[ProtectionStatus, CalibrationCell]

    def cell_for(self, status: ProtectionStatus) -> CalibrationCell | None:
        return self.cells.get(status)


def fit_calibration(
    cases: Sequence[CalibrationCase], *, seed: int, train_facilities: int
) -> CalibrationTable:
    """Fit per-cell Beta posteriors from labeled cases (pure, deterministic)."""
    positives: dict[ProtectionStatus, int] = {}
    totals: dict[ProtectionStatus, int] = {}
    for case in cases:
        totals[case.status] = totals.get(case.status, 0) + 1
        if case.label:
            positives[case.status] = positives.get(case.status, 0) + 1
    cells = {
        status: CalibrationCell(
            status=status, cases=totals[status], positives=positives.get(status, 0)
        )
        for status in sorted(totals, key=lambda s: s.value)
    }
    return CalibrationTable(seed=seed, train_facilities=train_facilities, cells=cells)


def train_eval_split(
    facility_ids: Sequence[str],
    *,
    seed: int,
    eval_fraction: float = CANONICAL_EVAL_FRACTION,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Facility-level holdout split: (train ids, eval ids).

    Facilities — not cases — are split, so no facility's labels ever appear
    on both sides (a case-level split would leak the fit into the score).
    Deterministic: sorted ids shuffled by ``seed``; the tail
    ``eval_fraction`` lands in eval. The two sets are disjoint and their
    union is exactly the input.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    ordered = sorted(facility_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(len(ordered) * (1.0 - eval_fraction))
    return tuple(ordered[:cut]), tuple(ordered[cut:])
