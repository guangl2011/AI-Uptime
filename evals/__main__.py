"""``python -m evals`` — run the full harness and write report artifacts.

Creates a throwaway SQLite app instance (never touches real data), runs the
harness against it, writes ``eval-report.md`` + ``eval-report.json`` into the
output directory, prints the gate table, and exits nonzero if any gate
fails. CI (nightly) uploads the two artifacts per run; nothing is committed
back into the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from evals.harness import (
    HarnessConfig,
    evaluate_gates,
    evaluate_segment_gates,
    render_json,
    render_markdown,
    run_harness,
)
from uptime_api.app import create_app

DEFAULT_OUTPUT_DIR = Path("eval-results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run the AI-Uptime evaluation harness (synthetic corpus).",
    )
    parser.add_argument("--seed", type=int, default=None, help="corpus seed (default: canonical)")
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
        help="where eval-report.{md,json} land (default: eval-results/)",
    )
    args = parser.parse_args(argv)

    config = HarnessConfig(
        seed=args.seed if args.seed is not None else HarnessConfig.seed,
        facility_counts=None
        if args.facilities is None
        else {segment: args.facilities for segment in ("cold_chain", "aidc", "hospital")},
    )

    with tempfile.TemporaryDirectory(prefix="aiuptime-eval-") as tmp:
        app = create_app(f"sqlite:///{Path(tmp) / 'eval.db'}")
        with TestClient(app) as client:
            results = run_harness(client, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "eval-report.md"
    json_path = args.output_dir / "eval-report.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_payload = render_json(results)
    json_payload["generated_at"] = datetime.now(UTC).isoformat()
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")

    gates = evaluate_gates(results) + evaluate_segment_gates(results)
    print(f"Eval report: {md_path} (+ {json_path.name})")
    print(f"Corpus: {results.facilities} facilities, {results.fault_cases} labeled fault cases")
    for gate in gates:
        marker = "PASS" if gate.passed else "FAIL"
        print(f"  [{marker}] {gate.name}: {gate.detail}")
    if not all(gate.passed for gate in gates):
        print("Gates failed — see the report's Gate failures section.", file=sys.stderr)
        return 1
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
