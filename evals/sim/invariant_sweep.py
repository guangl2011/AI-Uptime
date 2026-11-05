"""``python -m evals.sim.invariant_sweep`` — the Sim M2 invariant sweep.

Runs the plan's Category 3-4 invariants (``evals.sim.invariants``) over
generated seeds: one facility per segment per seed, engine-level checks on
every seed, plus the API-level recovery-state check on a deterministic
subsample (throwaway SQLite apps, same posture as ``make sim``). The plan's
gate is zero violations across >= 1,000 generated seeds; the full sweep runs
nightly (eval-nightly.yml), while push CI runs the Hypothesis property
seeds inside pytest so a push stays under a few minutes.

Writes ``invariant-sweep.{md,json}`` next to the sim reports (never
committed — ``eval-results/`` is gitignored) and exits nonzero on any
violation, with the failing seed and minimal repro in both the report and
the console output. Nothing here touches a facility system or anything
external — the simulator drives the assessment engine only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from evals.corpus import SEGMENTS, FacilityCorpus, generate_segment_corpus
from evals.sim.invariants import (
    INVARIANT_NAMES,
    InvariantResult,
    InvariantViolation,
    check_monotonicity,
    check_recovery_state,
    check_recovery_via_api,
    check_repair_recovery,
    check_replay_folds,
    check_replay_recovery,
    check_trace_determinism,
    check_whatif_counterfactual,
)
from uptime_api.app import create_app

DEFAULT_OUTPUT_DIR = Path("sim-results")
# (facility, seed) -> InvariantResult; injectable so failure-path tests can
# feed a known violation through the real aggregation and report path.
InvariantCheck = Callable[[FacilityCorpus, int], InvariantResult]
ENGINE_CHECKS: tuple[InvariantCheck, ...] = (
    check_monotonicity,
    check_repair_recovery,
    check_trace_determinism,
    check_replay_folds,
    check_replay_recovery,
    check_recovery_state,
    check_whatif_counterfactual,
)


@dataclass(frozen=True)
class SweepTotals:
    """Aggregated sweep results, one row per invariant plus segment counts."""

    seeds: int
    start_seed: int
    api_every: int
    facilities: int
    api_facilities: int
    runtime_seconds: float
    checks: dict[str, int]
    violations: dict[str, tuple[InvariantViolation, ...]]

    def gates(self) -> list[tuple[str, bool, str]]:
        """One zero-tolerance gate per invariant (a silent zero-check run fails)."""
        rows: list[tuple[str, bool, str]] = []
        for name in INVARIANT_NAMES:
            count = len(self.violations.get(name, ()))
            checks = self.checks.get(name, 0)
            passed = count == 0 and checks > 0
            detail = f"{checks} property checks across {self.seeds} seeds: {count} violations"
            rows.append((name, passed, detail))
        return rows

    @property
    def all_violations(self) -> tuple[InvariantViolation, ...]:
        return tuple(violation for bucket in self.violations.values() for violation in bucket)

    def violations_by_segment(self) -> dict[str, int]:
        counts = {segment: 0 for segment in SEGMENTS}
        for violation in self.all_violations:
            counts[violation.segment] = counts.get(violation.segment, 0) + 1
        return counts


def run_sweep(
    seeds: range,
    api_every: int,
    checks: Sequence[InvariantCheck] = ENGINE_CHECKS,
) -> SweepTotals:
    """Run every engine-level invariant on every seed; API recovery on a subsample."""
    started = time.monotonic()
    check_counts: dict[str, int] = {name: 0 for name in INVARIANT_NAMES}
    violations: dict[str, tuple[InvariantViolation, ...]] = {name: () for name in INVARIANT_NAMES}
    facilities = 0
    api_facilities = 0

    for seed in seeds:
        for segment in SEGMENTS:
            facility = generate_segment_corpus(segment, seed, 1)[0]
            facilities += 1
            for check in checks:
                result = check(facility, seed)
                check_counts[result.invariant] += result.checks
                violations[result.invariant] += result.violations

        if api_every > 0 and (seed - seeds.start) % api_every == 0:
            with tempfile.TemporaryDirectory(prefix="aiuptime-invariants-") as tmp:
                app = create_app(f"sqlite:///{Path(tmp) / 'sweep.db'}")
                with TestClient(app) as client:
                    for segment in SEGMENTS:
                        facility = generate_segment_corpus(segment, seed, 1)[0]
                        result = check_recovery_via_api(client, facility, seed)
                        api_facilities += 1
                        check_counts[result.invariant] += result.checks
                        violations[result.invariant] += result.violations

    return SweepTotals(
        seeds=len(seeds),
        start_seed=seeds.start,
        api_every=api_every,
        facilities=facilities,
        api_facilities=api_facilities,
        runtime_seconds=time.monotonic() - started,
        checks=check_counts,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Report artifacts (same convention as the sim report)
# ---------------------------------------------------------------------------


def seed_count_note(seed_count: int) -> str:
    """The plan's gate statement, so the artifact carries its own evidence."""
    if seed_count >= 1_000:
        return f"{seed_count} generated seeds — meets the plan's >= 1,000-seed gate"
    return f"{seed_count} generated seeds — below the plan's 1,000-seed gate (quick/local run)"


def render_json(totals: SweepTotals, *, seed_count: int) -> dict[str, object]:
    """The structured sweep run, including every violation's repro (CI artifact)."""
    return {
        "seeds": totals.seeds,
        "start_seed": totals.start_seed,
        "api_every": totals.api_every,
        "facilities": totals.facilities,
        "api_facilities": totals.api_facilities,
        "runtime_seconds": round(totals.runtime_seconds, 2),
        "gates": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in totals.gates()
        ],
        "checks_by_invariant": dict(sorted(totals.checks.items())),
        "violations_by_segment": totals.violations_by_segment(),
        "violations": [
            {
                "invariant": violation.invariant,
                "seed": violation.seed,
                "segment": violation.segment,
                "facility_id": violation.facility_id,
                "detail": violation.detail,
                "repro": violation.repro,
            }
            for violation in totals.all_violations
        ],
        "generated_at": datetime.now(UTC).isoformat(),
        "seed_count_note": seed_count_note(seed_count),
    }


def render_markdown(totals: SweepTotals) -> str:
    """The human-readable sweep report (workflow artifact)."""
    lines = [
        "# Invariant sweep report (Sim M2)",
        "",
        f"Seeds: {totals.seeds} (start {totals.start_seed}) · Facilities: "
        f"{totals.facilities} · API-level recovery facilities: {totals.api_facilities} "
        f"(every {totals.api_every}th seed) · Runtime: {totals.runtime_seconds:.1f}s",
        "",
        seed_count_note(totals.seeds),
        "",
        "Category 3-4 invariants stated as mathematical properties over generated "
        "seeds — never as expected outputs. Zero tolerance: any violation is a bug "
        "in the engine or the corpus, and fails CI with the failing seed and a "
        "minimal repro.",
        "",
        "## Gates",
        "",
        "| Invariant | Result | Detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in totals.gates():
        marker = "PASS" if passed else "FAIL"
        lines.append(f"| `{name}` | {marker} | {detail} |")

    lines += ["", "## Violations by segment", ""]
    for segment, count in totals.violations_by_segment().items():
        lines.append(f"- {segment}: {count}")

    if totals.all_violations:
        lines += ["", "## Violations (repro detail)", ""]
        lines.extend(f"- {violation.render()}" for violation in totals.all_violations)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    checks: Sequence[InvariantCheck] = ENGINE_CHECKS,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.sim.invariant_sweep",
        description="Run the Sim M2 invariant sweep (Categories 3-4 over generated seeds).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1_000,
        help="number of generated seeds (default: 1,000 — the plan's gate)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="first seed (default: 1)",
    )
    parser.add_argument(
        "--api-every",
        type=int,
        default=25,
        help="run the API-level recovery check every Nth seed (0 disables; default: 25)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where invariant-sweep.{md,json} land (default: sim-results/)",
    )
    args = parser.parse_args(argv)
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.api_every < 0:
        parser.error("--api-every must be zero or positive")

    totals = run_sweep(range(args.start_seed, args.start_seed + args.seeds), args.api_every, checks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "invariant-sweep.md"
    json_path = args.output_dir / "invariant-sweep.json"
    md_path.write_text(render_markdown(totals), encoding="utf-8")
    json_path.write_text(
        json.dumps(render_json(totals, seed_count=args.seeds), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Invariant sweep report: {md_path} (+ {json_path.name})")
    print(
        f"Seeds: {totals.seeds} · Facilities: {totals.facilities} "
        f"(API-level: {totals.api_facilities}) · Runtime: {totals.runtime_seconds:.1f}s"
    )
    for name, passed, detail in totals.gates():
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name}: {detail}")
    for violation in totals.all_violations:
        print(f"  VIOLATION {violation.render()}")
    if not all(passed for _, passed, _ in totals.gates()):
        print("Invariant gates failed — see the report artifacts for the repro detail.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
