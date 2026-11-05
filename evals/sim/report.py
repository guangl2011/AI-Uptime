"""Sim report rendering: markdown and JSON artifacts, mirroring the eval report.

Same discipline as the harness report (harness.render_markdown): a gate
table up front, segment-level results, and — on any gate failure — the
repro detail (failing scenario/step or sweep case, expected vs observed)
inline so a failure ships with its repro case, per the plan's CI rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.sim.runner import (
    DeterminismResult,
    GateResult,
    SimScenarioOutcome,
    SweepResult,
    WhatIfOutcome,
)


@dataclass(frozen=True)
class SimResults:
    """Everything one ``python -m evals.sim`` run produced."""

    seed: int
    facilities_per_segment: dict[str, int]
    scenarios: tuple[SimScenarioOutcome, ...]
    sweep: SweepResult
    determinism: DeterminismResult
    gates: tuple[GateResult, ...]


def render_json(results: SimResults) -> dict[str, object]:
    """The full structured run, including every trace step (CI artifact)."""
    return {
        "seed": results.seed,
        "facilities_per_segment": results.facilities_per_segment,
        "gates": [
            {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
            for gate in results.gates
        ],
        "scenarios": [
            {
                "scenario_id": outcome.scenario_id,
                "market": outcome.market,
                "passed": outcome.passed,
                "checks": outcome.checks,
                "failures": list(outcome.failures),
                "what_ifs": [
                    {
                        "label": wi.label,
                        "intervention": wi.intervention,
                        "intervention_kind": wi.intervention_kind,
                        "passed": wi.passed,
                        "checks": wi.checks,
                        "delta_failures": list(wi.delta_failures),
                        "spof_path_failures": list(wi.spof_path_failures),
                        "purity_failures": list(wi.purity_failures),
                        "abstention": wi.abstention,
                    }
                    for wi in outcome.what_ifs
                ],
                "trace": [step.to_json() for step in outcome.trace],
            }
            for outcome in results.scenarios
        ],
        "sweep": {
            "facilities": results.sweep.facilities,
            "facilities_by_segment": results.sweep.facilities_by_segment,
            "cases": len(results.sweep.cases),
            "per_segment": {
                segment: {"cases": cases, "mismatches": mismatches}
                for segment, (cases, mismatches) in results.sweep.per_segment().items()
            },
            "mismatches": [
                {
                    "facility_id": case.facility_id,
                    "segment": case.segment,
                    "asset_key": case.asset_key,
                    "expected_exposed": list(case.expected_exposed),
                    "expected_impacted": list(case.expected_impacted),
                    "observed_lost": list(case.observed_lost),
                }
                for case in results.sweep.mismatches
            ],
        },
        "determinism": {
            "ok": results.determinism.ok,
            "first_digest": results.determinism.first_digest,
            "second_digest": results.determinism.second_digest,
        },
    }


def render_markdown(results: SimResults) -> str:
    """The human-readable report (PR comment / workflow artifact)."""
    sweep_total = len(results.sweep.cases)
    scenario_checks = sum(outcome.checks for outcome in results.scenarios)
    what_ifs = tuple(wi for outcome in results.scenarios for wi in outcome.what_ifs)
    what_if_checks = sum(wi.checks for wi in what_ifs)
    abstentions = [wi for wi in what_ifs if wi.abstention is not None]
    lines = [
        "# Simulation report (Sim M1–M3)",
        "",
        f"Seed: {results.seed} · Facilities: {results.sweep.facilities} · "
        f"Sequenced scenarios: {len(results.scenarios)} ({scenario_checks} stepwise checks) · "
        f"What-if markers: {len(what_ifs)} ({what_if_checks} counterfactual checks) · "
        f"Sweep cases: {sweep_total}",
        "",
        "Every sequenced scenario is adjudicated by construction intent; the sweep "
        "faults every asset of every facility and asserts the loss the construction "
        "labels pin (equal to the what-if label where unambiguous, within the "
        "exposure/impact bounds where a protecting group sits downstream). Zero "
        "tolerance: a violation is a bug, not a tolerance.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for gate in results.gates:
        marker = "PASS" if gate.passed else "FAIL"
        lines.append(f"| `{gate.name}` | {marker} | {gate.detail} |")

    lines += [
        "",
        "## Sequenced scenarios (cascade dynamics, per-segment coverage)",
        "",
        "| Scenario | Segment | Events | Checks | Result |",
        "|---|---|---|---|---|",
    ]
    for outcome in results.scenarios:
        marker = "PASS" if outcome.passed else "FAIL"
        lines.append(
            f"| `{outcome.scenario_id}` | {outcome.market} | {len(outcome.trace) - 1} | "
            f"{outcome.checks} | {marker} |"
        )

    lines += [
        "",
        "## What-if counterfactuals (Category 5, per-segment coverage)",
        "",
        "| Scenario | Segment | Marker | Intervention | Delta | SPOF-path | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for outcome in results.scenarios:
        for wi in outcome.what_ifs:
            delta_cell = "PASS" if not (wi.delta_failures or wi.purity_failures) else "FAIL"
            if wi.spof_path_failures:
                spof_cell = "FAIL (residual SPOF)"
            elif wi.abstention is not None:
                spof_cell = "abstained"
            else:
                spof_cell = "PASS"
            result = "PASS" if wi.passed else "FAIL"
            abstention_note = f" — {wi.abstention}" if wi.abstention is not None else ""
            lines.append(
                f"| `{outcome.scenario_id}` | {outcome.market} | `{wi.label}` | "
                f"{wi.intervention} | {delta_cell} | {spof_cell} | {result} |"
            )
            if wi.abstention is not None:
                lines.append(
                    f"| | | | | | | abstention reason:{abstention_note} |"
                )

    lines += [
        "",
        "## Corpus failover sweep (failover semantics, per segment)",
        "",
        "| Segment | Facilities | Fault cases | Agreements | Result |",
        "|---|---|---|---|---|",
    ]
    per_segment = results.sweep.per_segment()
    facilities_by_segment = results.sweep.facilities_by_segment
    for segment, (cases, mismatches) in per_segment.items():
        marker = "PASS" if mismatches == 0 else "FAIL"
        facilities = facilities_by_segment.get(segment, 0)
        lines.append(
            f"| {segment} | {facilities} | {cases} | {cases - mismatches}/{cases} | {marker} |"
        )

    if what_ifs:
        by_segment: dict[str, list[WhatIfOutcome]] = {}
        for wi in what_ifs:
            by_segment.setdefault(wi.market, []).append(wi)
        coverage = ", ".join(
            f"{segment} {sum(1 for wi in marker_list if wi.passed)}/{len(marker_list)}"
            for segment, marker_list in sorted(by_segment.items())
        )
        abstention_note = (
            f" Abstentions: {len(abstentions)} (stated reasons above)."
            if abstentions
            else " No abstention paths fired."
        )
        lines += [
            "",
            f"Category 5 coverage: {len(what_ifs)} markers across "
            f"{len(by_segment)} segments ({coverage}); deltas and SPOF-path "
            "verdicts above are 100%-or-fail gates against the adjudicated "
            f"expectations.{abstention_note}",
        ]

    lines += ["", "## Determinism", ""]
    if results.determinism.ok:
        lines.append(f"PASS — {results.determinism.detail} (canonical full-trace SHA-256).")
    else:
        lines.append(f"FAIL — {results.determinism.detail}.")

    failure_lines = _failure_detail(results)
    if failure_lines:
        lines += ["", "## Failures (repro detail)", ""] + failure_lines
    return "\n".join(lines) + "\n"


def _failure_detail(results: SimResults) -> list[str]:
    """The repro payload for every failing check: expected vs observed."""
    lines: list[str] = []
    for outcome in results.scenarios:
        if not outcome.passed:
            lines.append(f"### `{outcome.scenario_id}` (seed {results.seed})")
            lines.append("")
            lines.extend(f"- {failure}" for failure in outcome.failures)
            lines.append("")
        for wi in outcome.what_ifs:
            if wi.passed:
                continue
            lines.append(
                f"### `{outcome.scenario_id}` what_if `{wi.label}` "
                f"({wi.intervention}, seed {results.seed})"
            )
            lines.append("")
            lines.extend(f"- {failure}" for failure in wi.failures)
            lines.append("")
    if results.sweep.mismatches:
        lines.append(f"### corpus sweep (seed {results.seed})")
        lines.append("")
        for case in results.sweep.mismatches:
            lines.append(
                f"- `{case.facility_id}` fault `{case.asset_key}`: expected exposed "
                f"{list(case.expected_exposed)}, full-impact "
                f"{list(case.expected_impacted)}, observed {list(case.observed_lost)}"
            )
        lines.append("")
    if not results.determinism.ok:
        lines.append(f"### determinism (seed {results.seed})")
        lines.append("")
        lines.append(f"- {results.determinism.detail}")
        lines.append("")
    return lines
