"""Unit tests for the pure evaluation metrics (evals/metrics.py)."""

from __future__ import annotations

import random

import pytest
from evals.metrics import (
    auc,
    bootstrap_ci,
    brier,
    brier_flat_baseline,
    expected_calibration_error,
    precision_recall,
    rank_with_ties,
    relative_brier_improvement,
    reliability_bins,
    spearman,
)


def test_auc_perfect_separation() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert auc(scores, labels) == 1.0


def test_auc_inverted_scores_scores_half_credit_on_ties() -> None:
    # A single tied pair straddling classes earns exactly half credit.
    assert auc([0.5, 0.5], [1, 0]) == 0.5


def test_auc_inverted_separation() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    assert auc(scores, labels) == 0.0


def test_auc_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="single label class"):
        auc([0.9, 0.8], [1, 1])


def test_auc_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        auc([0.9, 0.1], [1, 2])


def test_rank_with_ties_uses_average_ranks() -> None:
    # Ascending convention (smallest value = rank 1, scipy rankdata-style);
    # two tied values at positions 1-2 share rank 1.5, the max sits last.
    assert rank_with_ties([30.0, 10.0, 10.0, 90.0]) == [3.0, 1.5, 1.5, 4.0]


def test_brier_known_value() -> None:
    # Mean of (0.8-1)^2, (0.4-0)^2 = (0.04 + 0.16) / 2.
    assert brier([0.8, 0.4], [1, 0]) == pytest.approx(0.10)


def test_brier_flat_baseline_equals_predicting_the_prior() -> None:
    labels = [1, 0, 1, 1, 0, 1]
    prior = sum(labels) / len(labels)
    assert brier_flat_baseline(labels) == pytest.approx(brier([prior] * len(labels), labels))


def test_brier_rejects_empty_and_mismatched() -> None:
    with pytest.raises(ValueError):
        brier([], [])
    with pytest.raises(ValueError):
        brier([0.5], [1, 0])


def test_reliability_bins_counts_and_edges() -> None:
    probabilities = [0.1, 0.15, 0.5, 0.95, 1.0]
    labels = [0, 1, 0, 1, 1]
    table = reliability_bins(probabilities, labels)
    assert [b.n for b in table] == [2, 0, 1, 0, 2]
    assert table[0].lower == 0.0 and table[0].upper == 0.2
    # p = 1.0 lands in the last (inclusive) bin, not off the table.
    assert table[-1].observed_rate == 1.0
    assert table[0].mean_predicted == pytest.approx(0.125)
    assert table[0].observed_rate == pytest.approx(0.5)


def test_expected_calibration_error_is_share_weighted() -> None:
    # One case in each of two extreme bins, both labeled 0: bin gaps are
    # |0.0 - 0.1| = 0.1 and |0.0 - 0.9| = 0.9, equally shared.
    probabilities = [0.1, 0.9]
    labels = [0, 0]
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_and_covers_point_estimate() -> None:
    xs = [0.9, 0.8, 0.75, 0.3, 0.2, 0.1]
    ys = [1, 1, 1, 0, 0, 0]

    def stat(pxs: list[float], pys: list[float]) -> float:
        # Bootstrap resamples can be single-class by chance; the stat must
        # stay total — score degenerate samples as random guessing.
        if len(set(pys)) < 2:
            return 0.5
        return auc(pxs, [int(y) for y in pys])

    first = bootstrap_ci(xs, ys, stat, random.Random(7), n_boot=200)
    second = bootstrap_ci(xs, ys, stat, random.Random(7), n_boot=200)
    assert first == second  # same seed, same interval
    low, high = first
    assert 0.0 <= low <= high <= 1.0
    assert low <= auc(xs, ys) <= high


def test_spearman_tracks_monotone_and_inverted() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_handles_heavy_ties() -> None:
    # Perfect agreement within the tied structure: ranks equal in both.
    assert spearman([5.0, 5.0, 1.0, 1.0], [90.0, 90.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_precision_recall_counts() -> None:
    stats = precision_recall({"a", "b", "c"}, {"b", "c", "d"})
    assert stats.true_positives == 2
    assert stats.precision == pytest.approx(2 / 3)
    assert stats.recall == pytest.approx(2 / 3)


def test_relative_brier_improvement_positive_change() -> None:
    # (0.25 - 0.10) / 0.25 = 0.60: a 60% relative improvement.
    assert relative_brier_improvement(0.10, 0.25) == pytest.approx(0.60)


def test_relative_brier_improvement_negative_change() -> None:
    # Worse than baseline: (0.25 - 0.26) / 0.25 = -0.04.
    assert relative_brier_improvement(0.26, 0.25) == pytest.approx(-0.04)


def test_relative_brier_improvement_zero_change() -> None:
    assert relative_brier_improvement(0.25, 0.25) == pytest.approx(0.0)


def test_relative_brier_improvement_returns_none_for_zero_baseline() -> None:
    # A zero baseline makes any relative claim meaningless (division by zero
    # or negative denominators) — the function reports None and the harness
    # gates that as a failure, not a number.
    assert relative_brier_improvement(0.10, 0.0) is None
