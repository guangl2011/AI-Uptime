"""``python -m evals.sim`` — run the sim gates and write report artifacts.

Creates two throwaway SQLite app instances (never touches real data), runs
the identical sim pass against each — same seed, same scenarios, same
corpus — for the determinism gate, then writes ``sim-report.md`` +
``sim-report.json`` into the output directory, prints the gate table, and
exits nonzero if any gate fails. CI uploads the two artifacts per run (the
report carries the failing seed, trace step, and diff on a gate failure, so
a repro case ships with the failure); nothing is committed back into the
repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from evals.corpus import CORPUS_SEED, SEGMENTS
from evals.sim.report import SimResults, render_json, render_markdown
from evals.sim.runner import (
    DeterminismResult,
    evaluate_sim_gates,
    run_sim_suite,
    trace_digest,
)
from uptime_api.app import create_app

DEFAULT_OUTPUT_DIR = Path("sim-results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.sim",
        description="Run the AI-Uptime simulation gates (Sim M1).",
    )
    parser.add_argument(
        "--seed", type=int, default=CORPUS_SEED, help="corpus seed (default: canonical)"
    )
    parser.add_argument(
        "--facilities",
        type=int,
        default=None,
        help="facility count applied to every segment (default: canonical counts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where sim-report.{md,json} land (default: sim-results/)",
    )
    args = parser.parse_args(argv)

    facilities_per_segment = (
        None
        if args.facilities is None
        else {segment: args.facilities for segment in SEGMENTS}
    )

    with tempfile.TemporaryDirectory(prefix="aiuptime-sim-") as tmp:
        # Two fresh apps, same seed, no shared state: the determinism gate
        # compares the two full traces.
        first_app = create_app(f"sqlite:///{Path(tmp) / 'sim-first.db'}")
        with TestClient(first_app) as first_client:
            first_outcomes, first_sweep = run_sim_suite(
                first_client, seed=args.seed, facilities_per_segment=facilities_per_segment
            )
        second_app = create_app(f"sqlite:///{Path(tmp) / 'sim-second.db'}")
        with TestClient(second_app) as second_client:
            second_outcomes, second_sweep = run_sim_suite(
                second_client, seed=args.seed, facilities_per_segment=facilities_per_segment
            )

    determinism = DeterminismResult(
        first_digest=trace_digest(first_outcomes, first_sweep),
        second_digest=trace_digest(second_outcomes, second_sweep),
    )
    gates = evaluate_sim_gates(first_outcomes, first_sweep, determinism)
    results = SimResults(
        seed=args.seed,
        facilities_per_segment=facilities_per_segment or {},
        scenarios=first_outcomes,
        sweep=first_sweep,
        determinism=determinism,
        gates=gates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "sim-report.md"
    json_path = args.output_dir / "sim-report.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_payload = render_json(results)
    json_payload["generated_at"] = datetime.now(UTC).isoformat()
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Sim report: {md_path} (+ {json_path.name})")
    print(f"Corpus: {first_sweep.facilities} facilities, {len(first_sweep.cases)} fault cases")
    for gate in gates:
        marker = "PASS" if gate.passed else "FAIL"
        print(f"  [{marker}] {gate.name}: {gate.detail}")
    if not all(gate.passed for gate in gates):
        print("Sim gates failed — see the report artifacts for the repro detail.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
