"""Pure evaluation metrics — no I/O, no product imports.

Each function is a named, testable transformation from paired score/label
sequences to the testing plan's metrics (art_XaCiEWwo): AUROC on labeled
faults, Brier score and reliability-bin calibration over the score-as-
probability mapping, rank agreement with adjudicated orderings, and set
precision/recall. Bootstrap confidence intervals round out every headline
number — a point estimate without a spread is not reported (sparse-label
honesty).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

# Calibration bins over the [0, 1] predicted-probability range: five equal
# bins, the last inclusive of 1.0 so p = 1.0 never falls off the table.
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

PairedStat = Callable[[Sequence[float], Sequence[float]], float]


@dataclass(frozen=True)
class PrecisionRecall:
    """Precision/recall over id sets, with the counts behind them."""

    true_positives: int
    predicted_total: int
    actual_total: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted_total if self.predicted_total else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.actual_total if self.actual_total else 0.0


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-table row: predicted probability vs observed rate."""

    lower: float
    upper: float
    n: int
    mean_predicted: float | None
    observed_rate: float | None


def precision_recall(predicted: Collection[str], actual: Collection[str]) -> PrecisionRecall:
    """Precision/recall of a predicted id set against a ground-truth id set."""
    predicted_set, actual_set = set(predicted), set(actual)
    return PrecisionRecall(
        true_positives=len(predicted_set & actual_set),
        predicted_total=len(predicted_set),
        actual_total=len(actual_set),
    )


def rank_with_ties(values: Sequence[float]) -> list[float]:
    """1-based ascending average ranks: smallest value = rank 1 (scipy-style);
    equal values share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; raises on zero variance (correlation undefined)."""
    if len(xs) != len(ys):
        raise ValueError("paired sequences must have equal length")
    n = len(xs)
    if n < 2:
        raise ValueError("correlation needs at least two paired points")
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov: float = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x: float = sum((x - mean_x) ** 2 for x in xs)
    var_y: float = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        raise ValueError("correlation undefined for a constant sequence")
    return cov / math.sqrt(var_x * var_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average-rank transforms)."""
    return pearson(rank_with_ties(xs), rank_with_ties(ys))


def auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """AUROC of scores against binary labels (Mann-Whitney via average ranks).

    Labels are 1 (positive: the injected fault interrupts at least one
    service) or 0 (negative: redundancy absorbs it); ints or floats.
    Ties earn half credit.
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("auc needs non-empty paired scores and labels")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary 0/1")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc undefined with a single label class")
    ranks = rank_with_ties(list(scores))
    rank_sum_pos = sum(r for r, label in zip(ranks, labels, strict=True) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    """Mean (p - y)^2 over labeled cases (binary 0/1 labels)."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("brier needs non-empty paired probabilities and labels")
    return sum((p - label) ** 2 for p, label in zip(probabilities, labels, strict=True)) / len(
        probabilities
    )


def brier_flat_baseline(labels: Sequence[float]) -> float:
    """Brier score of the flat-prior baseline: predict the positive rate.

    The fixed baseline the testing plan scores against (the model must beat
    it for the WP2 go/no-go target; the MVP reports the delta without a
    threshold).
    """
    if not labels:
        raise ValueError("baseline needs labels")
    prior = sum(labels) / len(labels)
    return brier([prior] * len(labels), labels)


def relative_brier_improvement(model_brier: float, baseline_brier: float) -> float | None:
    """Fraction of the baseline Brier the model removes: (base - model) / base.

    The Phase 5 WP2 go/no-go gate is *relative* (>= 15% better than the
    flat-prior baseline). None when the baseline is zero — an all-one-class
    label set, where a relative claim is meaningless (the caller gates that
    as a failure rather than dividing by zero).
    """
    if baseline_brier <= 0.0:
        return None
    return (baseline_brier - model_brier) / baseline_brier


def _bin_index(p: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        if (lower <= p <= upper) if last else (lower <= p < upper):
            return i
    raise ValueError(f"probability {p} outside calibration edges {edges}")


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[int],
    edges: tuple[float, ...] = CALIBRATION_EDGES,
) -> list[ReliabilityBin]:
    """Reliability table: per bin, count, mean predicted p, observed rate."""
    if len(probabilities) != len(labels):
        raise ValueError("paired probabilities and labels required")
    counts = [0] * (len(edges) - 1)
    predicted_sums = [0.0] * (len(edges) - 1)
    positives = [0] * (len(edges) - 1)
    for p, label in zip(probabilities, labels, strict=True):
        index = _bin_index(p, edges)
        counts[index] += 1
        predicted_sums[index] += p
        positives[index] += int(label)
    table: list[ReliabilityBin] = []
    for i in range(len(edges) - 1):
        count = counts[i]
        table.append(
            ReliabilityBin(
                lower=edges[i],
                upper=edges[i + 1],
                n=count,
                mean_predicted=predicted_sums[i] / count if count else None,
                observed_rate=positives[i] / count if count else None,
            )
        )
    return table


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    edges: tuple[float, ...] = CALIBRATION_EDGES,
) -> float:
    """ECE: bin-share-weighted mean absolute predicted-vs-observed gap."""
    table = reliability_bins(probabilities, labels, edges)
    total = sum(bucket.n for bucket in table)
    if total == 0:
        raise ValueError("calibration error needs cases")
    return sum(
        (bucket.n / total) * abs(bucket.mean_predicted - bucket.observed_rate)
        for bucket in table
        if bucket.n > 0 and bucket.mean_predicted is not None and bucket.observed_rate is not None
    )


def bootstrap_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    stat: PairedStat,
    rng: random.Random,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` over paired (xs, ys) resamples.

    Pairs are resampled jointly so any statistic of the pairing survives.
    Deterministic given the rng.
    """
    if len(xs) != len(ys) or not xs:
        raise ValueError("bootstrap needs non-empty paired sequences")
    draws: list[float] = []
    n = len(xs)
    for _ in range(n_boot):
        index = [rng.randrange(n) for _ in range(n)]
        draws.append(stat([xs[i] for i in index], [ys[i] for i in index]))
    draws.sort()
    lo_index = int((alpha / 2) * n_boot)
    hi_index = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return draws[lo_index], draws[hi_index]
