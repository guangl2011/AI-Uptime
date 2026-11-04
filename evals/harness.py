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

Gates are threshold checks evaluated in ``evaluate_gates``; the runner
(``python -m evals``) fails nonzero if any gate fails, and every number in
the report is labeled synthetic — the corpus proves the machinery, not
field performance.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from fastapi.testclient import TestClient

from evals.corpus import FacilityCorpus, generate_corpus
from evals.metrics import (
    ReliabilityBin,
    auc,
    bootstrap_ci,
    brier,
    brier_flat_baseline,
    expected_calibration_error,
    reliability_bins,
    spearman,
)
from evals.scenarios import ScenarioOutcome, run_all_scenarios
from evals.splits import LeakageCheck, build_folds, check_no_leakage

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

# Replay feature window for leakage-checked folds (see evals.splits).
FEATURE_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class HarnessConfig:
    """Harness parameters (defaults = the canonical deterministic corpus)."""

    seed: int = 20261104
    n_facilities: int = 40
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


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability table and expected calibration error (synthetic)."""

    bins: tuple[ReliabilityBin, ...]
    ece: float


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


def load_facility(client: TestClient, facility: FacilityCorpus) -> dict[str, object]:
    """Load one corpus facility through the real onboarding surface.

    Inventory CSV upload first (the parser path — version 1), then the
    technician-confirmed topology commit (version 2), then the assessment
    pinned to that version. Returns the AssessmentReport payload.
    """
    created = client.post(
        "/facilities",
        json={"id": facility.facility_id, "name": facility.name, "segment": "cold_chain"},
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
    recovered = {
        str(node["node_key"]): node for node in nodes if node["node_type"] == "asset"
    }
    intended = {f"INV-{row.asset_id}": row for row in facility.clean_rows}

    collector.expected_nodes += len(intended)
    collector.recovered_nodes += len(set(recovered) & set(intended))
    collector.missing.extend(sorted(set(intended) - set(recovered)))
    collector.extra.extend(sorted(set(recovered) - set(intended)))
    for key in sorted(set(recovered) & set(intended)):
        row = intended[key]
        node = recovered[key]
        if node["name"] != row.name or node["location"] != row.location:
            collector.attribute_mismatches.append(
                f"{facility.facility_id}:{key} name/location"
            )
        elif node["attributes"] != row.attributes:
            collector.attribute_mismatches.append(
                f"{facility.facility_id}:{key} attributes"
            )


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
    # Ranking agreement: Spearman between adjudicated impact size and score.
    keys = sorted(facility.expected_impacted)
    expected_order = [float(len(facility.expected_impacted[key])) for key in keys]
    actual_order = [float(scores[key]) for key in keys]
    collector.taus.append(spearman(expected_order, actual_order))


def run_harness(client: TestClient, config: HarnessConfig | None = None) -> HarnessResults:
    """Run the full evaluation loop and return every section's results."""
    cfg = config or HarnessConfig()
    corpus = generate_corpus(cfg.seed, cfg.n_facilities)
    collector = _ScoreCollector()

    for facility in corpus.facilities:
        report = load_facility(client, facility)
        check_topology_recovery(client, facility, collector)
        collect_case_scores(report, facility, collector)

    rng = random.Random(cfg.seed)
    auroc_ci = bootstrap_ci(
        collector.probabilities, collector.labels, auc, rng, cfg.bootstrap_samples
    )
    brier_value = brier(collector.probabilities, collector.labels)
    brier_ci = bootstrap_ci(
        collector.probabilities,
        collector.labels,
        brier,
        rng,
        cfg.bootstrap_samples,
    )
    baseline = brier_flat_baseline(collector.labels)
    improvement = baseline - brier_value

    def _delta(ps: Sequence[float], ys: Sequence[float]) -> float:
        return brier_flat_baseline(ys) - brier(ps, ys)

    improvement_ci = bootstrap_ci(
        collector.probabilities, collector.labels, _delta, rng, cfg.bootstrap_samples
    )

    leakage = check_no_leakage(
        [event for facility in corpus.facilities for event in facility.events],
        [
            fold
            for facility in corpus.facilities
            for fold in build_folds(facility.replay_cases, FEATURE_WINDOW)
        ],
    )
    scenarios = tuple(run_all_scenarios(client))

    return HarnessResults(
        config=cfg,
        facilities=len(corpus.facilities),
        fault_cases=len(collector.labels),
        shape_histogram=corpus.shape_histogram,
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
        classification=ClassificationMetrics(
            cases=len(collector.labels),
            positives=sum(collector.labels),
            auroc=auc(collector.probabilities, collector.labels),
            auroc_ci=auroc_ci,
            brier=brier_value,
            brier_ci=brier_ci,
            baseline_brier=baseline,
            brier_improvement=improvement,
            brier_improvement_ci=improvement_ci,
        ),
        calibration=CalibrationReport(
            bins=tuple(reliability_bins(collector.probabilities, collector.labels)),
            ece=expected_calibration_error(collector.probabilities, collector.labels),
        ),
        leakage=leakage,
        scenarios=scenarios,
    )


@dataclass(frozen=True)
class GateResult:
    """One threshold check against the results bundle."""

    name: str
    passed: bool
    detail: str


def evaluate_gates(results: HarnessResults) -> list[GateResult]:
    """Every gate, in report order. The runner fails nonzero if any fails."""
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
        "classification": {
            "cases": results.classification.cases,
            "positives": results.classification.positives,
            "auroc": results.classification.auroc,
            "auroc_ci": list(results.classification.auroc_ci),
            "brier": results.classification.brier,
            "brier_ci": list(results.classification.brier_ci),
            "baseline_brier": results.classification.baseline_brier,
            "brier_improvement": results.classification.brier_improvement,
            "brier_improvement_ci": list(results.classification.brier_improvement_ci),
        },
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
    gates = evaluate_gates(results)
    lines = [
        "# AI-Uptime Evaluation Report",
        "",
        f"- Corpus: {results.facilities} synthetic cold-chain facilities "
        f"(seed {results.config.seed}), {results.fault_cases} labeled single-fault cases",
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
        "## Calibration reliability table (synthetic)",
        "",
        "| Bin | n | Mean predicted | Observed rate |",
        "|---|---|---|---|",
    ]
    for bucket in results.calibration.bins:
        lines.append(
            f"| [{bucket.lower:.1f}, {bucket.upper:.1f}] | {bucket.n} "
            f"| {bucket.mean_predicted if bucket.mean_predicted is not None else '—'} "
            f"| {bucket.observed_rate if bucket.observed_rate is not None else '—'} |"
        )
    lines.extend(["", "## Gates", "", "| Gate | Status | Detail |", "|---|---|---|"])
    for gate in gates:
        lines.append(
            f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |"
        )
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
