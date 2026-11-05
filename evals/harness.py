"""Full evaluation harness: corpus in, gated report out.

Runs the testing plan's (art_XaCiEWwo) evaluation loop end to end over the
real HTTP surface: generate the synthetic corpus, load every facility the
way an operator does (inventory CSV upload, then the confirmed topology
commit, then an assessment), and score the assessor against the corpus's
by-construction labels:

- topology recovery — what the inventory parser + operator commit built
  vs the generator's intended nodes (assets, attributes, provenance);
- exposure correctness — per asset, assessed ``exposed_services`` vs the
  adjudicated set (risk-propagation correctness against known answers);
- SPOF detection — set precision/recall of predicted single points of
  failure (which assets interrupt at least one service);
- criticality ranking — Spearman agreement between the assessor's
  criticality scores and the adjudicated impact ordering;
- fault classification — AUROC (go/no-go >= 0.80) and Brier score over
  the score-as-probability mapping, with bootstrap CIs and the flat-prior
  baseline delta;
- calibration — reliability bins and expected calibration error;
- no-leakage — every replay fold checked against both split invariants;
- scenarios — the named fault-injection known-answer gates.

Phase 4 added per-market-segment reporting: every metric is computed per
segment (cold chain / AIDC / hospitals) as well as overall, and the same
thresholds gate each segment, so a regression in one segment is visible and
blocking instead of averaged away.

Phase 5 turns the forecasting pass into the WP2 go/no-go evaluation. The
calibrated forecaster (``CalibratedForecastModel``) is *fit* on the training
facilities' labeled cases — a facility-level holdout via
``train_eval_split``, so no scored case was ever seen by the fit — and its
held-out ``p_fault`` estimates are scored against the same by-construction
labels. The proposal's go/no-go gates now apply to that held-out score and
gate the build: AUROC >= 0.80, and a Brier score at least 15% better than
the flat-prior baseline with the bootstrap improvement interval excluding
zero. Calibration (ECE + reliability bins) is reported alongside. Abstained
forecasts carry the flat-prior fallback value, so abstention can never score
better than the baseline it defers to. The gates are honestly winnable on
the synthetic corpus (the placeholder's fixed priors were miscalibrated —
its Brier was *worse* than baseline) and honestly labeled: every forecast
number is synthetic-only, proving the calibration machinery, not field
performance.

Gates are threshold checks evaluated in ``evaluate_gates`` (overall) and
``evaluate_segment_gates`` (per segment); the runner (``python -m evals``)
fails nonzero if any fails, and every number in the report is labeled
synthetic — the corpus proves the machinery, not field performance.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from evals.corpus import SEGMENTS, Corpus, FacilityCorpus, generate_corpus
from evals.metrics import (
    ReliabilityBin,
    auc,
    bootstrap_ci,
    brier,
    brier_flat_baseline,
    expected_calibration_error,
    relative_brier_improvement,
    reliability_bins,
    spearman,
)
from evals.scenarios import ScenarioOutcome, run_all_scenarios
from evals.splits import LeakageCheck, build_folds, check_no_leakage
from uptime_api.forecasting.calibrated import CalibratedForecastModel
from uptime_api.forecasting.calibration import (
    CANONICAL_EVAL_FRACTION,
    CalibrationCase,
    fit_calibration,
    train_eval_split,
)
from uptime_api.forecasting.features import asset_protection_features
from uptime_api.forecasting.schema import ForecastStatus
from uptime_api.graph.schemas import GraphSnapshotOut

# Gate thresholds. AUROC >= 0.80 is the proposal's go/no-go target applied
# to labeled synthetic faults; the rest are the harness's own regression
# gates on a deterministic assessor (an accuracy below 1.0 is a bug, not
# noise — the corpus is exact by construction).
AUROC_GATE = 0.80
TOPOLOGY_RECOVERY_GATE = 1.0
EXPOSURE_GATE = 1.0
SPOF_RECALL_GATE = 1.0
SPOF_FPR_GATE = 0.02
RANKING_TAU_GATE = 0.60
BRIER_GATE = 0.10
BOOTSTRAP_SAMPLES = 400
# Phase 5 WP2 go/no-go: the calibrated forecaster must beat the flat-prior
# baseline by >= 15% relative Brier, established with the paired bootstrap
# improvement interval excluding zero, on the held-out facility split.
FORECAST_BRIER_RELATIVE_GATE = 0.15

# Replay feature window for leakage-checked folds (see evals.splits).
FEATURE_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class HarnessConfig:
    """Harness parameters (defaults = the canonical deterministic corpus)."""

    seed: int = 20261104
    # Per-segment facility counts (None = canonical: 40 cold chain,
    # 20 AIDC, 20 hospitals). ``--facilities`` applies one count to all.
    facility_counts: dict[str, int] | None = None
    bootstrap_samples: int = BOOTSTRAP_SAMPLES


@dataclass(frozen=True)
class TopologyRecovery:
    """Inventory-parser + operator-commit fidelity vs generator intent."""

    expected_nodes: int
    recovered_nodes: int
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    attribute_mismatches: tuple[str, ...]
    unresolved_rows: int
    unresolved_reasons: dict[str, int]

    @property
    def recovery_rate(self) -> float:
        return self.recovered_nodes / self.expected_nodes if self.expected_nodes else 1.0


@dataclass(frozen=True)
class ExposureCorrectness:
    """Risk-propagation correctness: per-asset exposure sets vs labels."""

    cases: int
    correct: int
    mismatches: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.cases if self.cases else 1.0


@dataclass(frozen=True)
class SpofQuality:
    """SPOF detection pooled across the corpus (set precision/recall)."""

    true_positives: int
    predicted_total: int
    actual_total: int
    total_cases: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted_total if self.predicted_total else 1.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.actual_total if self.actual_total else 1.0

    @property
    def false_positive_rate(self) -> float:
        false_positives = self.predicted_total - self.true_positives
        actual_negatives = self.total_cases - self.actual_total
        return false_positives / actual_negatives if actual_negatives else 0.0


@dataclass(frozen=True)
class RankingQuality:
    """Criticality ranking agreement (Spearman vs adjudicated impact)."""

    mean_tau: float
    per_facility: tuple[float, ...]


@dataclass(frozen=True)
class ClassificationMetrics:
    """AUROC + Brier over labeled single-fault cases, with CIs and baseline."""

    cases: int
    positives: int
    auroc: float
    auroc_ci: tuple[float, float]
    brier: float
    brier_ci: tuple[float, float]
    baseline_brier: float
    brier_improvement: float
    brier_improvement_ci: tuple[float, float]

    @property
    def relative_improvement(self) -> float | None:
        """Fraction of the baseline Brier removed; None if baseline is 0."""
        return relative_brier_improvement(self.brier, self.baseline_brier)


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability table and expected calibration error (synthetic)."""

    bins: tuple[ReliabilityBin, ...]
    ece: float


@dataclass(frozen=True)
class MetricBundle:
    """The six scored sections, computed over one set of labeled cases."""

    topology: TopologyRecovery
    exposure: ExposureCorrectness
    spof: SpofQuality
    ranking: RankingQuality
    classification: ClassificationMetrics
    calibration: CalibrationReport


@dataclass(frozen=True)
class SegmentMetrics:
    """One market segment's scored results (Phase 4 per-segment reporting)."""

    segment: str
    facilities: int
    fault_cases: int
    metrics: MetricBundle


@dataclass(frozen=True)
class ForecastEval:
    """Scored WP2 forecasting pass — Phase 5 go/no-go gates apply.

    The calibrated forecaster is fit on the training facilities' labeled
    cases and scored on the held-out facilities the fit never saw. The
    proposal's go/no-go gates (AUROC >= 0.80; Brier >= 15% better than the
    flat-prior baseline, with the bootstrap improvement CI excluding zero)
    apply to this held-out score. Synthetic-only: labels are
    by-construction, so these numbers prove the calibration machinery, not
    field performance.
    """

    model: str
    train_facilities: int
    eval_facilities: int
    cases: int
    abstentions: int
    overall: ClassificationMetrics
    per_segment: dict[str, ClassificationMetrics]
    calibration: CalibrationReport
    per_segment_abstentions: dict[str, int]


@dataclass(frozen=True)
class HarnessResults:
    """Everything the report renders, in one bundle."""

    config: HarnessConfig
    facilities: int
    fault_cases: int
    shape_histogram: dict[str, int]
    topology: TopologyRecovery
    exposure: ExposureCorrectness
    spof: SpofQuality
    ranking: RankingQuality
    classification: ClassificationMetrics
    calibration: CalibrationReport
    leakage: LeakageCheck
    scenarios: tuple[ScenarioOutcome, ...]
    per_segment: dict[str, SegmentMetrics] = field(default_factory=dict)
    forecast_eval: ForecastEval | None = None


@dataclass
class _ScoreCollector:
    """Accumulates labeled cases and section counters across facilities."""

    labels: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    expected_nodes: int = 0
    recovered_nodes: int = 0
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    attribute_mismatches: list[str] = field(default_factory=list)
    unresolved_rows: int = 0
    unresolved_reasons: dict[str, int] = field(default_factory=dict)
    exposure_cases: int = 0
    exposure_correct: int = 0
    exposure_mismatches: list[str] = field(default_factory=list)
    tp: int = 0
    predicted: int = 0
    actual: int = 0
    taus: list[float] = field(default_factory=list)


def _merge_collectors(collectors: Sequence[_ScoreCollector]) -> _ScoreCollector:
    """Pool per-segment collectors into the overall collector (pure)."""
    merged = _ScoreCollector()
    for collector in collectors:
        merged.labels.extend(collector.labels)
        merged.probabilities.extend(collector.probabilities)
        merged.expected_nodes += collector.expected_nodes
        merged.recovered_nodes += collector.recovered_nodes
        merged.missing.extend(collector.missing)
        merged.extra.extend(collector.extra)
        merged.attribute_mismatches.extend(collector.attribute_mismatches)
        merged.unresolved_rows += collector.unresolved_rows
        for reason, count in collector.unresolved_reasons.items():
            merged.unresolved_reasons[reason] = merged.unresolved_reasons.get(reason, 0) + count
        merged.exposure_cases += collector.exposure_cases
        merged.exposure_correct += collector.exposure_correct
        merged.exposure_mismatches.extend(collector.exposure_mismatches)
        merged.tp += collector.tp
        merged.predicted += collector.predicted
        merged.actual += collector.actual
        merged.taus.extend(collector.taus)
    return merged


def load_facility(client: TestClient, facility: FacilityCorpus) -> dict[str, object]:
    """Load one corpus facility through the real onboarding surface.

    Inventory CSV upload first (the parser path — version 1), then the
    technician-confirmed topology commit (version 2), then the assessment
    pinned to that version. Returns the AssessmentReport payload.
    """
    created = client.post(
        "/facilities",
        json={
            "id": facility.facility_id,
            "name": facility.name,
            "segment": facility.segment,
        },
    )
    assert created.status_code == 201, created.text
    uploaded = client.post(
        f"/facilities/{facility.facility_id}/inventory",
        files={"file": ("inventory.csv", facility.inventory_csv.encode("utf-8"), "text/csv")},
    )
    assert uploaded.status_code == 201, uploaded.text
    committed = client.post(
        f"/facilities/{facility.facility_id}/graph/versions",
        json=facility.graph,
    )
    assert committed.status_code == 201, committed.text
    version_no: int = committed.json()["version_no"]
    run = client.post(
        f"/facilities/{facility.facility_id}/assessments",
        json={"author": "eval_harness", "version_no": version_no},
    )
    assert run.status_code == 201, run.text
    report: dict[str, object] = run.json()
    return report


def check_topology_recovery(
    client: TestClient, facility: FacilityCorpus, collector: _ScoreCollector
) -> None:
    """Compare the inventory-imported version-1 nodes with generator intent."""
    snapshot = client.get(f"/facilities/{facility.facility_id}/graph?version=1")
    assert snapshot.status_code == 200, snapshot.text
    nodes = snapshot.json()["nodes"]
    # Inventory imports are namespaced INV-<asset_id> so parser-created nodes
    # can never collide with technician-entered keys.
    recovered = {str(node["node_key"]): node for node in nodes if node["node_type"] == "asset"}
    intended = {f"INV-{row.asset_id}": row for row in facility.clean_rows}

    collector.expected_nodes += len(intended)
    collector.recovered_nodes += len(set(recovered) & set(intended))
    collector.missing.extend(sorted(set(intended) - set(recovered)))
    collector.extra.extend(sorted(set(recovered) - set(intended)))
    for key in sorted(set(recovered) & set(intended)):
        row = intended[key]
        node = recovered[key]
        if node["name"] != row.name or node["location"] != row.location:
            collector.attribute_mismatches.append(f"{facility.facility_id}:{key} name/location")
        elif node["attributes"] != row.attributes:
            collector.attribute_mismatches.append(f"{facility.facility_id}:{key} attributes")


def collect_case_scores(
    report: dict[str, object], facility: FacilityCorpus, collector: _ScoreCollector
) -> None:
    """Accumulate labeled classification cases and section counters.

    Labels come from the corpus construction; scores from the assessor.
    A case is one asset's single-fault question: does this fault interrupt
    at least one service (1) or is it absorbed by redundancy (0)?
    """
    raw_assets = report["assets"]
    assert isinstance(raw_assets, list)
    scores: dict[str, int] = {}
    for raw_asset in raw_assets:
        assert isinstance(raw_asset, dict)
        key = str(raw_asset["node_key"])
        criticality = raw_asset["criticality"]
        assert isinstance(criticality, dict)
        scores[key] = int(criticality["score"])
        exposed = set(raw_asset["exposed_services"])
        label = 1 if facility.expected_exposed[key] else 0
        collector.labels.append(label)
        collector.probabilities.append(scores[key] / 100.0)
        collector.exposure_cases += 1
        if exposed == set(facility.expected_exposed[key]):
            collector.exposure_correct += 1
        else:
            collector.exposure_mismatches.append(
                f"{facility.facility_id}:{key} expected "
                f"{sorted(facility.expected_exposed[key])}, got {sorted(exposed)}"
            )
        if exposed:
            collector.predicted += 1
        if facility.expected_exposed[key]:
            collector.actual += 1
            if exposed:
                collector.tp += 1
    # Ranking agreement: Spearman between the adjudicated priority ordering
    # and the score. Priority is exposure-first (the product surfaces
    # unprotected damage first), with total reach as the tiebreak:
    # 1000 * exposed_count + impacted_count. This is construction-derived —
    # it never reads the assessor's score — so a propagation bug that moves
    # exposure between services still fails the metric.
    keys = sorted(facility.expected_impacted)
    expected_order = [
        float(1000 * len(facility.expected_exposed[key]) + len(facility.expected_impacted[key]))
        for key in keys
    ]
    actual_order = [float(scores[key]) for key in keys]
    collector.taus.append(spearman(expected_order, actual_order))


def _classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    seed: int,
    bootstrap_samples: int,
) -> ClassificationMetrics:
    """AUROC + Brier with bootstrap CIs and the flat-prior baseline delta."""
    rng = random.Random(seed)
    auroc_ci = bootstrap_ci(probabilities, labels, auc, rng, bootstrap_samples)
    brier_value = brier(probabilities, labels)
    brier_ci = bootstrap_ci(probabilities, labels, brier, rng, bootstrap_samples)
    baseline = brier_flat_baseline(labels)
    improvement = baseline - brier_value

    def _delta(ps: Sequence[float], ys: Sequence[float]) -> float:
        return brier_flat_baseline(ys) - brier(ps, ys)

    improvement_ci = bootstrap_ci(probabilities, labels, _delta, rng, bootstrap_samples)
    return ClassificationMetrics(
        cases=len(labels),
        positives=sum(labels),
        auroc=auc(probabilities, labels),
        auroc_ci=auroc_ci,
        brier=brier_value,
        brier_ci=brier_ci,
        baseline_brier=baseline,
        brier_improvement=improvement,
        brier_improvement_ci=improvement_ci,
    )


def _finalize_bundle(collector: _ScoreCollector, *, seed: int, samples: int) -> MetricBundle:
    """Scored sections for one collector (overall or single-segment)."""
    classification = _classification_metrics(
        collector.labels,
        collector.probabilities,
        seed=seed,
        bootstrap_samples=samples,
    )
    return MetricBundle(
        topology=TopologyRecovery(
            expected_nodes=collector.expected_nodes,
            recovered_nodes=collector.recovered_nodes,
            missing=tuple(collector.missing),
            extra=tuple(collector.extra),
            attribute_mismatches=tuple(collector.attribute_mismatches),
            unresolved_rows=collector.unresolved_rows,
            unresolved_reasons=dict(collector.unresolved_reasons),
        ),
        exposure=ExposureCorrectness(
            cases=collector.exposure_cases,
            correct=collector.exposure_correct,
            mismatches=tuple(collector.exposure_mismatches),
        ),
        spof=SpofQuality(
            true_positives=collector.tp,
            predicted_total=collector.predicted,
            actual_total=collector.actual,
            total_cases=collector.exposure_cases,
        ),
        ranking=RankingQuality(
            mean_tau=sum(collector.taus) / len(collector.taus) if collector.taus else 0.0,
            per_facility=tuple(collector.taus),
        ),
        classification=classification,
        calibration=CalibrationReport(
            bins=tuple(reliability_bins(collector.probabilities, collector.labels)),
            ece=expected_calibration_error(collector.probabilities, collector.labels),
        ),
    )


def _get_snapshot(client: TestClient, facility_id: str, version: int) -> dict[str, object]:
    """Fetch one immutable graph version (the fit's read-only input)."""
    response = client.get(f"/facilities/{facility_id}/graph?version={version}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def run_forecast_pass(
    client: TestClient,
    corpus: Corpus,
    versions: dict[str, int],
    *,
    seed: int,
    bootstrap_samples: int,
) -> ForecastEval:
    """Fit the calibrated forecaster on a facility-level holdout and score it.

    Training cases come from the *training* facilities' snapshots (read-only
    graph GETs) paired with by-construction labels — no assessment runs
    during the fit. The fitted model is then installed on the app for one
    extra assessment per *held-out* facility, and every emitted ``p_fault``
    is scored against the same construction labels (AUROC, Brier vs the
    flat-prior baseline with a paired-bootstrap improvement CI, calibration).
    Abstained estimates carry the flat-prior fallback value, so abstention
    can never score better than the baseline it defers to. Restores the
    previous forecaster on the way out.
    """
    train_ids, eval_ids = train_eval_split(
        [facility.facility_id for facility in corpus.facilities],
        seed=seed,
        eval_fraction=CANONICAL_EVAL_FRACTION,
    )
    facilities_by_id = {facility.facility_id: facility for facility in corpus.facilities}
    # httpx types TestClient.app as the raw ASGI callable; it is the FastAPI
    # instance, and the forecaster slot is where the flag resolution landed.
    app = cast(FastAPI, client.app)
    previous = app.state.forecaster

    # Fit: read-only snapshot pulls paired with construction labels.
    train_cases: list[CalibrationCase] = []
    for facility_id in train_ids:
        facility = facilities_by_id[facility_id]
        snapshot = GraphSnapshotOut.model_validate(
            _get_snapshot(client, facility_id, versions[facility_id])
        )
        for feature in asset_protection_features(snapshot):
            train_cases.append(
                CalibrationCase(
                    facility_id=facility_id,
                    status=feature.status,
                    label=1 if facility.expected_exposed[feature.asset_key] else 0,
                )
            )
    table = fit_calibration(train_cases, seed=seed, train_facilities=len(train_ids))

    # Score: held-out facilities only, through the real assessment surface.
    segment_cases: dict[str, list[tuple[int, float]]] = {segment: [] for segment in SEGMENTS}
    segment_abstentions: dict[str, int] = {segment: 0 for segment in SEGMENTS}
    app.state.forecaster = CalibratedForecastModel(table)
    try:
        for facility_id in eval_ids:
            facility = facilities_by_id[facility_id]
            run = client.post(
                f"/facilities/{facility_id}/assessments",
                json={
                    "author": "eval_forecast_pass",
                    "version_no": versions[facility_id],
                },
            )
            assert run.status_code == 201, run.text
            report = run.json()
            forecasts = {str(item["asset_key"]): item for item in report["forecasts"]}
            cases = segment_cases[facility.segment]
            for asset_key, expected in facility.expected_exposed.items():
                item = forecasts.get(asset_key)
                assert item is not None, f"{facility_id}:{asset_key} has no forecast"
                label = 1 if expected else 0
                cases.append((label, float(item["p_fault"])))
                if item.get("status") == ForecastStatus.ABSTAINED.value:
                    segment_abstentions[facility.segment] += 1
    finally:
        app.state.forecaster = previous

    per_segment: dict[str, ClassificationMetrics] = {}
    for segment in SEGMENTS:
        cases = segment_cases[segment]
        if not cases:
            continue  # hand-shrunken corpora may leave a segment unscored
        per_segment[segment] = _classification_metrics(
            [label for label, _ in cases],
            [probability for _, probability in cases],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
    overall_labels = [label for segment in SEGMENTS for label, _ in segment_cases[segment]]
    overall_probabilities = [
        probability for segment in SEGMENTS for _, probability in segment_cases[segment]
    ]
    overall = _classification_metrics(
        overall_labels,
        overall_probabilities,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return ForecastEval(
        model=CalibratedForecastModel.__name__,
        train_facilities=len(train_ids),
        eval_facilities=len(eval_ids),
        cases=len(overall_labels),
        abstentions=sum(segment_abstentions.values()),
        overall=overall,
        per_segment=per_segment,
        calibration=CalibrationReport(
            bins=tuple(reliability_bins(overall_probabilities, overall_labels)),
            ece=expected_calibration_error(overall_probabilities, overall_labels),
        ),
        per_segment_abstentions=segment_abstentions,
    )


def run_harness(client: TestClient, config: HarnessConfig | None = None) -> HarnessResults:
    """Run the full evaluation loop and return every section's results."""
    cfg = config or HarnessConfig()
    corpus = generate_corpus(cfg.seed, cfg.facility_counts)

    # One collector per segment; the overall collector is their merge.
    segment_collectors = {segment: _ScoreCollector() for segment in SEGMENTS}
    versions: dict[str, int] = {}
    for facility in corpus.facilities:
        report = load_facility(client, facility)
        graph_version = report["graph_version"]
        assert isinstance(graph_version, int)
        versions[facility.facility_id] = graph_version
        collector = segment_collectors[facility.segment]
        check_topology_recovery(client, facility, collector)
        collect_case_scores(report, facility, collector)

    overall = _merge_collectors(list(segment_collectors.values()))
    per_segment = {
        segment: SegmentMetrics(
            segment=segment,
            facilities=len([f for f in corpus.facilities if f.segment == segment]),
            fault_cases=len(collector.labels),
            metrics=_finalize_bundle(collector, seed=cfg.seed, samples=cfg.bootstrap_samples),
        )
        for segment, collector in segment_collectors.items()
    }
    bundle = _finalize_bundle(overall, seed=cfg.seed, samples=cfg.bootstrap_samples)

    leakage = check_no_leakage(
        [event for facility in corpus.facilities for event in facility.events],
        [
            fold
            for facility in corpus.facilities
            for fold in build_folds(facility.replay_cases, FEATURE_WINDOW)
        ],
    )
    scenarios = tuple(run_all_scenarios(client))
    forecast_eval = run_forecast_pass(
        client,
        corpus,
        versions,
        seed=cfg.seed,
        bootstrap_samples=cfg.bootstrap_samples,
    )

    return HarnessResults(
        config=cfg,
        facilities=len(corpus.facilities),
        fault_cases=len(overall.labels),
        shape_histogram=corpus.shape_histogram,
        topology=bundle.topology,
        exposure=bundle.exposure,
        spof=bundle.spof,
        ranking=bundle.ranking,
        classification=bundle.classification,
        calibration=bundle.calibration,
        leakage=leakage,
        scenarios=scenarios,
        per_segment=per_segment,
        forecast_eval=forecast_eval,
    )


@dataclass(frozen=True)
class GateResult:
    """One threshold check against the results bundle."""

    name: str
    passed: bool
    detail: str


def _forecast_gates(forecast: ForecastEval | None) -> list[GateResult]:
    """WP2 go/no-go gates on the held-out forecast score (Phase 5).

    A missing forecast pass is a gate *failure*, not a skip — the Phase 5
    gates are merge-blocking, so silence would be a false pass. The Brier
    gate needs both the relative-improvement threshold and the paired
    bootstrap improvement interval excluding zero.
    """
    if forecast is None:
        return [
            GateResult(
                "forecast_pass_ran",
                False,
                "forecast pass did not run — the WP2 go/no-go gates cannot be evaluated",
            )
        ]
    relative = forecast.overall.relative_improvement
    improvement_ci = forecast.overall.brier_improvement_ci
    relative_text = f"{relative:+.1%}" if relative is not None else "n/a"
    return [
        GateResult(
            "forecast_auroc",
            forecast.overall.auroc >= AUROC_GATE,
            f"holdout AUROC {forecast.overall.auroc:.3f} "
            f"[{forecast.overall.auroc_ci[0]:.3f}, {forecast.overall.auroc_ci[1]:.3f}] "
            f"over {forecast.cases} cases from {forecast.eval_facilities} held-out "
            f"facilities (fit on {forecast.train_facilities}; gate {AUROC_GATE})",
        ),
        GateResult(
            "forecast_brier_improvement",
            relative is not None
            and relative >= FORECAST_BRIER_RELATIVE_GATE
            and improvement_ci[0] > 0.0,
            f"holdout Brier {forecast.overall.brier:.4f} vs flat-prior baseline "
            f"{forecast.overall.baseline_brier:.4f} — {relative_text} relative "
            f"(gate >= {FORECAST_BRIER_RELATIVE_GATE:.0%}), improvement CI "
            f"[{improvement_ci[0]:+.4f}, {improvement_ci[1]:+.4f}]; "
            f"{forecast.abstentions} abstention(s) scored at the flat-prior fallback",
        ),
        GateResult(
            "forecast_calibration",
            sum(bucket.n for bucket in forecast.calibration.bins) == forecast.cases,
            f"reliability table covers "
            f"{sum(bucket.n for bucket in forecast.calibration.bins)}/{forecast.cases} "
            f"holdout cases, ECE {forecast.calibration.ece:.3f} (reported, not gated); "
            "synthetic-only — by-construction labels, proves the machinery, not "
            "field performance",
        ),
    ]


def evaluate_gates(results: HarnessResults) -> list[GateResult]:
    """Every overall gate, in report order. The runner fails nonzero if any fails."""
    gates = [
        GateResult(
            "topology_recovery",
            results.topology.recovery_rate >= TOPOLOGY_RECOVERY_GATE
            and not results.topology.missing
            and not results.topology.extra
            and not results.topology.attribute_mismatches,
            f"recovery {results.topology.recovered_nodes}/{results.topology.expected_nodes}, "
            f"{len(results.topology.missing)} missing, {len(results.topology.extra)} extra, "
            f"{len(results.topology.attribute_mismatches)} attribute mismatches",
        ),
        GateResult(
            "exposure_correctness",
            results.exposure.accuracy >= EXPOSURE_GATE,
            f"{results.exposure.correct}/{results.exposure.cases} exposure sets exact",
        ),
        GateResult(
            "spof_detection",
            results.spof.recall >= SPOF_RECALL_GATE
            and results.spof.false_positive_rate <= SPOF_FPR_GATE,
            f"recall {results.spof.recall:.3f}, FPR {results.spof.false_positive_rate:.3f}",
        ),
        GateResult(
            "ranking_agreement",
            results.ranking.mean_tau >= RANKING_TAU_GATE,
            f"mean Spearman {results.ranking.mean_tau:.3f} (gate {RANKING_TAU_GATE})",
        ),
        GateResult(
            "auroc",
            results.classification.auroc >= AUROC_GATE,
            f"AUROC {results.classification.auroc:.3f} "
            f"[{results.classification.auroc_ci[0]:.3f}, {results.classification.auroc_ci[1]:.3f}] "
            f"(gate {AUROC_GATE})",
        ),
        GateResult(
            "brier",
            results.classification.brier <= BRIER_GATE,
            f"Brier {results.classification.brier:.3f} "
            f"[{results.classification.brier_ci[0]:.3f}, {results.classification.brier_ci[1]:.3f}] "
            f"vs baseline {results.classification.baseline_brier:.3f} "
            f"(improvement {results.classification.brier_improvement:+.3f})",
        ),
        GateResult(
            "calibration_reporting",
            sum(bucket.n for bucket in results.calibration.bins) == results.classification.cases,
            f"reliability table covers {sum(bucket.n for bucket in results.calibration.bins)}"
            f"/{results.classification.cases} cases, ECE {results.calibration.ece:.3f} "
            "(reported, not gated: calibrated probabilities are a WP2 deliverable — "
            "the MVP asserts topological exposure, not calibrated risk)",
        ),
        *_forecast_gates(results.forecast_eval),
        GateResult(
            "no_leakage",
            results.leakage.ok,
            f"{results.leakage.cases} folds checked, "
            f"{len(results.leakage.violations)} violations, "
            f"{results.leakage.post_repair_events_excluded} post-repair events excluded",
        ),
        GateResult(
            "scenarios",
            all(outcome.passed for outcome in results.scenarios),
            f"{sum(outcome.passed for outcome in results.scenarios)}/"
            f"{len(results.scenarios)} scenarios pass",
        ),
    ]
    return gates


def evaluate_segment_gates(results: HarnessResults) -> list[GateResult]:
    """The same thresholds applied per market segment (Phase 4).

    Names are prefixed ``<segment>:<gate>`` so a regression in one segment
    is visible and blocking instead of averaged away by the overall pool.
    """
    gates: list[GateResult] = []
    for segment in SEGMENTS:
        segment_metrics = results.per_segment.get(segment)
        if segment_metrics is None:
            continue
        metrics = segment_metrics.metrics
        prefix = f"{segment}:"
        gates.extend(
            [
                GateResult(
                    f"{prefix}topology_recovery",
                    metrics.topology.recovery_rate >= TOPOLOGY_RECOVERY_GATE
                    and not metrics.topology.missing
                    and not metrics.topology.extra
                    and not metrics.topology.attribute_mismatches,
                    f"recovery {metrics.topology.recovered_nodes}/"
                    f"{metrics.topology.expected_nodes}, "
                    f"{len(metrics.topology.missing)} missing, "
                    f"{len(metrics.topology.extra)} extra, "
                    f"{len(metrics.topology.attribute_mismatches)} attribute mismatches",
                ),
                GateResult(
                    f"{prefix}exposure_correctness",
                    metrics.exposure.accuracy >= EXPOSURE_GATE,
                    f"{metrics.exposure.correct}/{metrics.exposure.cases} exposure sets exact",
                ),
                GateResult(
                    f"{prefix}spof_detection",
                    metrics.spof.recall >= SPOF_RECALL_GATE
                    and metrics.spof.false_positive_rate <= SPOF_FPR_GATE,
                    f"recall {metrics.spof.recall:.3f}, FPR {metrics.spof.false_positive_rate:.3f}",
                ),
                GateResult(
                    f"{prefix}ranking_agreement",
                    metrics.ranking.mean_tau >= RANKING_TAU_GATE,
                    f"mean Spearman {metrics.ranking.mean_tau:.3f} (gate {RANKING_TAU_GATE})",
                ),
                GateResult(
                    f"{prefix}auroc",
                    metrics.classification.auroc >= AUROC_GATE,
                    f"AUROC {metrics.classification.auroc:.3f} "
                    f"[{metrics.classification.auroc_ci[0]:.3f}, "
                    f"{metrics.classification.auroc_ci[1]:.3f}] (gate {AUROC_GATE})",
                ),
                GateResult(
                    f"{prefix}brier",
                    metrics.classification.brier <= BRIER_GATE,
                    f"Brier {metrics.classification.brier:.3f} "
                    f"[{metrics.classification.brier_ci[0]:.3f}, "
                    f"{metrics.classification.brier_ci[1]:.3f}] vs baseline "
                    f"{metrics.classification.baseline_brier:.3f}",
                ),
            ]
        )
        forecast_metrics = (
            results.forecast_eval.per_segment.get(segment)
            if results.forecast_eval is not None
            else None
        )
        if forecast_metrics is None:
            # A missing forecast pass already fails the overall gates loudly;
            # per-segment forecast gates only apply to scored segments.
            continue
        relative = forecast_metrics.relative_improvement
        relative_text = f"{relative:+.1%}" if relative is not None else "n/a"
        gates.extend(
            [
                GateResult(
                    f"{prefix}forecast_auroc",
                    forecast_metrics.auroc >= AUROC_GATE,
                    f"holdout AUROC {forecast_metrics.auroc:.3f} "
                    f"[{forecast_metrics.auroc_ci[0]:.3f}, "
                    f"{forecast_metrics.auroc_ci[1]:.3f}] (gate {AUROC_GATE})",
                ),
                GateResult(
                    f"{prefix}forecast_brier_improvement",
                    relative is not None
                    and relative >= FORECAST_BRIER_RELATIVE_GATE
                    and forecast_metrics.brier_improvement_ci[0] > 0.0,
                    f"holdout Brier {forecast_metrics.brier:.4f} vs baseline "
                    f"{forecast_metrics.baseline_brier:.4f} — {relative_text} "
                    f"relative (gate >= {FORECAST_BRIER_RELATIVE_GATE:.0%}), "
                    f"improvement CI [{forecast_metrics.brier_improvement_ci[0]:+.4f}, "
                    f"{forecast_metrics.brier_improvement_ci[1]:+.4f}]",
                ),
            ]
        )
    return gates


def _classification_payload(metrics: ClassificationMetrics) -> dict[str, object]:
    return {
        "cases": metrics.cases,
        "positives": metrics.positives,
        "auroc": metrics.auroc,
        "auroc_ci": list(metrics.auroc_ci),
        "brier": metrics.brier,
        "brier_ci": list(metrics.brier_ci),
        "baseline_brier": metrics.baseline_brier,
        "brier_improvement": metrics.brier_improvement,
        "brier_improvement_ci": list(metrics.brier_improvement_ci),
        "brier_relative_improvement": metrics.relative_improvement,
    }


def _segment_payload(segment: SegmentMetrics) -> dict[str, object]:
    metrics = segment.metrics
    return {
        "segment": segment.segment,
        "facilities": segment.facilities,
        "fault_cases": segment.fault_cases,
        "topology_recovery": {
            "expected_nodes": metrics.topology.expected_nodes,
            "recovered_nodes": metrics.topology.recovered_nodes,
            "recovery_rate": metrics.topology.recovery_rate,
            "missing": list(metrics.topology.missing),
            "extra": list(metrics.topology.extra),
            "attribute_mismatches": list(metrics.topology.attribute_mismatches),
        },
        "exposure_correctness": {
            "cases": metrics.exposure.cases,
            "correct": metrics.exposure.correct,
            "accuracy": metrics.exposure.accuracy,
            "mismatches": list(metrics.exposure.mismatches),
        },
        "spof_detection": {
            "true_positives": metrics.spof.true_positives,
            "predicted_total": metrics.spof.predicted_total,
            "actual_total": metrics.spof.actual_total,
            "precision": metrics.spof.precision,
            "recall": metrics.spof.recall,
            "false_positive_rate": metrics.spof.false_positive_rate,
        },
        "ranking_agreement": {
            "mean_spearman": metrics.ranking.mean_tau,
        },
        "classification": _classification_payload(metrics.classification),
        "calibration": {
            "ece": metrics.calibration.ece,
        },
    }


def render_json(results: HarnessResults) -> dict[str, object]:
    """Machine-readable report payload (workflow artifact / diffing)."""
    return {
        "seed": results.config.seed,
        "facilities": results.facilities,
        "fault_cases": results.fault_cases,
        "shape_histogram": results.shape_histogram,
        "topology_recovery": {
            "expected_nodes": results.topology.expected_nodes,
            "recovered_nodes": results.topology.recovered_nodes,
            "recovery_rate": results.topology.recovery_rate,
            "missing": list(results.topology.missing),
            "extra": list(results.topology.extra),
            "attribute_mismatches": list(results.topology.attribute_mismatches),
        },
        "exposure_correctness": {
            "cases": results.exposure.cases,
            "correct": results.exposure.correct,
            "accuracy": results.exposure.accuracy,
            "mismatches": list(results.exposure.mismatches),
        },
        "spof_detection": {
            "true_positives": results.spof.true_positives,
            "predicted_total": results.spof.predicted_total,
            "actual_total": results.spof.actual_total,
            "precision": results.spof.precision,
            "recall": results.spof.recall,
            "false_positive_rate": results.spof.false_positive_rate,
        },
        "ranking_agreement": {
            "mean_spearman": results.ranking.mean_tau,
        },
        "classification": _classification_payload(results.classification),
        "calibration": {
            "ece": results.calibration.ece,
            "bins": [
                {
                    "lower": bucket.lower,
                    "upper": bucket.upper,
                    "n": bucket.n,
                    "mean_predicted": bucket.mean_predicted,
                    "observed_rate": bucket.observed_rate,
                }
                for bucket in results.calibration.bins
            ],
        },
        "per_segment": {
            segment: _segment_payload(segment_metrics)
            for segment, segment_metrics in results.per_segment.items()
        },
        "forecast_eval": None
        if results.forecast_eval is None
        else {
            "model": results.forecast_eval.model,
            "train_facilities": results.forecast_eval.train_facilities,
            "eval_facilities": results.forecast_eval.eval_facilities,
            "cases": results.forecast_eval.cases,
            "abstentions": results.forecast_eval.abstentions,
            "synthetic_only": True,
            "gated": True,
            "overall": _classification_payload(results.forecast_eval.overall),
            "per_segment": {
                segment: _classification_payload(metrics)
                for segment, metrics in results.forecast_eval.per_segment.items()
            },
            "per_segment_abstentions": results.forecast_eval.per_segment_abstentions,
            "calibration": {
                "ece": results.forecast_eval.calibration.ece,
                "bins": [
                    {
                        "lower": bucket.lower,
                        "upper": bucket.upper,
                        "n": bucket.n,
                        "mean_predicted": bucket.mean_predicted,
                        "observed_rate": bucket.observed_rate,
                    }
                    for bucket in results.forecast_eval.calibration.bins
                ],
            },
        },
        "no_leakage": {
            "ok": results.leakage.ok,
            "cases": results.leakage.cases,
            "feature_events_used": results.leakage.feature_events_used,
            "post_repair_events_excluded": results.leakage.post_repair_events_excluded,
            "violations": list(results.leakage.violations),
        },
        "scenarios": [
            {
                "scenario_id": outcome.scenario_id,
                "passed": outcome.passed,
                "checks": outcome.checks,
                "failures": list(outcome.failures),
            }
            for outcome in results.scenarios
        ],
    }


def render_markdown(results: HarnessResults) -> str:
    """The human-readable report (also the CI artifact's markdown half)."""
    gates = evaluate_gates(results) + evaluate_segment_gates(results)
    segment_counts = {
        segment: segment_metrics.facilities
        for segment, segment_metrics in results.per_segment.items()
    }
    lines = [
        "# AI-Uptime Evaluation Report",
        "",
        f"- Corpus: {results.facilities} synthetic facilities across "
        f"{len(results.per_segment)} market segments (cold chain → AIDC → hospitals; "
        f"per segment: {segment_counts}; seed {results.config.seed}), "
        f"{results.fault_cases} labeled single-fault cases",
        f"- Shape histogram: {results.shape_histogram}",
        "",
        "> **All labels in this report are synthetic and by-construction.** The corpus",
        "> proves the assessment machinery, not field performance: no real facility",
        "> history exists yet (the beachhead market is sparse-label by nature). These",
        "> numbers make no claim about real-world facilities — that gate opens with the",
        "> first pilot's replay (WP4). Synthetic-vs-real breakdown: 100% synthetic.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Topology recovery | {results.topology.recovered_nodes}/"
        f"{results.topology.expected_nodes} nodes, "
        f"{len(results.topology.attribute_mismatches)} attribute mismatches |",
        f"| Exposure correctness | {results.exposure.correct}/{results.exposure.cases} "
        f"({results.exposure.accuracy:.1%}) |",
        f"| SPOF detection | precision {results.spof.precision:.3f}, "
        f"recall {results.spof.recall:.3f}, FPR {results.spof.false_positive_rate:.3f} |",
        f"| Ranking agreement | mean Spearman {results.ranking.mean_tau:.3f} |",
        f"| AUROC | {results.classification.auroc:.3f} "
        f"[{results.classification.auroc_ci[0]:.3f}, {results.classification.auroc_ci[1]:.3f}] |",
        f"| Brier | {results.classification.brier:.3f} "
        f"[{results.classification.brier_ci[0]:.3f}, {results.classification.brier_ci[1]:.3f}] "
        f"(flat-prior baseline {results.classification.baseline_brier:.3f}, "
        f"improvement {results.classification.brier_improvement:+.3f} "
        f"[{results.classification.brier_improvement_ci[0]:.3f}, "
        f"{results.classification.brier_improvement_ci[1]:.3f}]) |",
        f"| Calibration (ECE) | {results.calibration.ece:.3f} |",
        f"| No-leakage folds | {results.leakage.cases} checked, "
        f"{len(results.leakage.violations)} violations, "
        f"{results.leakage.post_repair_events_excluded} post-repair events excluded |",
        f"| Scenarios | {sum(o.passed for o in results.scenarios)}/{len(results.scenarios)} pass |",
        "",
        "## Per-segment results (market sequence: cold chain → AIDC → hospitals)",
        "",
        "| Segment | Facilities | Cases | Topology | Exposure | SPOF R/FPR "
        "| Spearman | AUROC | Brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for segment, segment_metrics in results.per_segment.items():
        metrics = segment_metrics.metrics
        lines.append(
            f"| {segment} | {segment_metrics.facilities} | {segment_metrics.fault_cases} "
            f"| {metrics.topology.recovered_nodes}/{metrics.topology.expected_nodes} "
            f"| {metrics.exposure.correct}/{metrics.exposure.cases} "
            f"| {metrics.spof.recall:.3f}/{metrics.spof.false_positive_rate:.3f} "
            f"| {metrics.ranking.mean_tau:.3f} "
            f"| {metrics.classification.auroc:.3f} "
            f"| {metrics.classification.brier:.3f} |"
        )
    if results.forecast_eval is not None:
        forecast = results.forecast_eval
        relative = forecast.overall.relative_improvement
        relative_text = f"{relative:+.1%}" if relative is not None else "n/a"
        lines.extend(
            [
                "",
                "## WP2 forecasting pass (calibrated; Phase 5 go/no-go gates apply)",
                "",
                f"Model `{forecast.model}` — fitted Beta posteriors over protection "
                f"cells, fit on {forecast.train_facilities} facilities' labeled cases "
                f"and scored on the {forecast.eval_facilities} held-out facilities "
                f"the fit never saw ({forecast.cases} cases; {forecast.abstentions} "
                "abstention(s) scored at the flat-prior fallback).",
                "Synthetic-only: labels are by-construction — these numbers prove",
                "the calibration machinery, not field performance. Forecasts are",
                "informational estimates; nothing executes on them.",
                "",
                "| Segment | AUROC | Brier | Baseline | Rel. improvement | Abstentions |",
                "|---|---|---|---|---|---|",
            ]
        )
        for segment, forecast_metrics in forecast.per_segment.items():
            seg_relative = forecast_metrics.relative_improvement
            seg_relative_text = f"{seg_relative:+.1%}" if seg_relative is not None else "n/a"
            lines.append(
                f"| {segment} | {forecast_metrics.auroc:.3f} | {forecast_metrics.brier:.3f} "
                f"| {forecast_metrics.baseline_brier:.3f} | {seg_relative_text} "
                f"| {forecast.per_segment_abstentions.get(segment, 0)} |"
            )
        lines.append(
            f"| **overall** | {forecast.overall.auroc:.3f} "
            f"| {forecast.overall.brier:.3f} | {forecast.overall.baseline_brier:.3f} "
            f"| {relative_text} | {forecast.abstentions} |"
        )
        lines.extend(
            [
                "",
                f"Holdout calibration: ECE {forecast.calibration.ece:.3f} "
                "(forecast reliability table below).",
            ]
        )
    lines.extend(
        [
            "",
            "## Calibration reliability table (synthetic)",
            "",
            "| Bin | n | Mean predicted | Observed rate |",
            "|---|---|---|---|",
        ]
    )
    for bucket in results.calibration.bins:
        lines.append(
            f"| [{bucket.lower:.1f}, {bucket.upper:.1f}] | {bucket.n} "
            f"| {bucket.mean_predicted if bucket.mean_predicted is not None else '—'} "
            f"| {bucket.observed_rate if bucket.observed_rate is not None else '—'} |"
        )
    if results.forecast_eval is not None:
        lines.extend(
            [
                "",
                "### Forecast reliability table (holdout, synthetic)",
                "",
                "| Bin | n | Mean predicted | Observed rate |",
                "|---|---|---|---|",
            ]
        )
        for bucket in results.forecast_eval.calibration.bins:
            lines.append(
                f"| [{bucket.lower:.1f}, {bucket.upper:.1f}] | {bucket.n} "
                f"| {bucket.mean_predicted if bucket.mean_predicted is not None else '—'} "
                f"| {bucket.observed_rate if bucket.observed_rate is not None else '—'} |"
            )
    lines.extend(["", "## Gates", "", "| Gate | Status | Detail |", "|---|---|---|"])
    for gate in gates:
        lines.append(f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |")
    if any(not gate.passed for gate in gates):
        lines.extend(["", "## Gate failures", ""])
        for gate in gates:
            if not gate.passed:
                lines.append(f"- **{gate.name}**: {gate.detail}")
        if results.exposure.mismatches:
            lines.extend(["", "### Exposure mismatches (first 20)", ""])
            lines.extend(f"- {m}" for m in results.exposure.mismatches[:20])
        if results.topology.missing or results.topology.extra:
            lines.extend(["", "### Topology differences (first 20)", ""])
            lines.extend(f"- missing: {m}" for m in results.topology.missing[:20])
            lines.extend(f"- extra: {m}" for m in results.topology.extra[:20])
    lines.extend(
        [
            "",
            "## Scenario outcomes",
            "",
            "| Scenario | Checks | Result |",
            "|---|---|---|",
        ]
    )
    for outcome in results.scenarios:
        detail = "pass" if outcome.passed else "; ".join(outcome.failures)
        lines.append(f"| {outcome.scenario_id} | {outcome.checks} | {detail} |")
    lines.append("")
    return "\n".join(lines)
