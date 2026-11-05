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
    lines = [
        "# Simulation report (Sim M1)",
        "",
        f"Seed: {results.seed} · Facilities: {results.sweep.facilities} · "
        f"Sequenced scenarios: {len(results.scenarios)} ({scenario_checks} stepwise checks) · "
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
        if outcome.passed:
            continue
        lines.append(f"### `{outcome.scenario_id}` (seed {results.seed})")
        lines.append("")
        lines.extend(f"- {failure}" for failure in outcome.failures)
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
