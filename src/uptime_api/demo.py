"""`make demo` runbook — Phase 2.

Exercises the onboarding scanner the way a customer would: the seed
cold-chain facility is created and committed through the real HTTP surface,
the graph is read back with provenance intact, and the run ends with a scored
assessment — ranked findings, per-service risk, and the seed facility's known
single points of failure asserted. Nothing here touches a facility system —
the only writes are to AI-Uptime's own database.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from uptime_api.app import create_app

SEED_PATH = Path(__file__).resolve().parents[2] / "seed" / "example_facility.json"

# Known SPOFs of the seed facility, derived from its topology: the single
# refrigeration rack, its lone condenser, and the single-path power chain
# (ATS -> UPS -> PDU) downstream of the 2N sources. The N+1 compressor pair
# and the 2N utility/generator sources must NOT appear.
KNOWN_SPOFS = {"ATS-1", "COND-1", "PDU-1", "RACK-1", "UPS-1"}
KNOWN_REDUNDANT = {"GEN-1", "UTILITY-1", "COMP-1", "COMP-2"}


def main() -> None:
    """Load the seed facility, assess it, and verify the known SPOFs."""
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    facility_id = seed["facility_id"]

    with tempfile.TemporaryDirectory() as tmp:
        with TestClient(create_app(f"sqlite:///{tmp}/demo.db")) as client:
            created = client.post(
                "/facilities",
                json={"id": facility_id, "name": seed["name"], "segment": seed["segment"]},
            )
            assert created.status_code == 201, created.text
            print(f"[demo] facility created: {facility_id} ({seed['segment']})")

            committed = client.post(
                f"/facilities/{facility_id}/graph/versions",
                json={
                    "author": seed["author"],
                    "notes": seed["notes"],
                    "nodes": seed["nodes"],
                    "edges": seed["edges"],
                },
            )
            assert committed.status_code == 201, committed.text
            print(f"[demo] graph version {committed.json()['version_no']} committed")

            snapshot = client.get(f"/facilities/{facility_id}/graph").json()
            untraced = [
                element
                for element in snapshot["nodes"] + snapshot["edges"]
                if not element["provenance"]["source_ref"]
            ]
            assert not untraced, f"untraceable elements: {untraced}"
            assets = sum(1 for n in snapshot["nodes"] if n["node_type"] == "asset")
            print(
                f"[demo] read back v{snapshot['version_no']}: {assets} assets, "
                f"{len(snapshot['nodes']) - assets} services, "
                f"{len(snapshot['edges'])} edges — provenance intact on every element"
            )

            # The scanner wedge: one pinned, reproducible assessment.
            run = client.post(f"/facilities/{facility_id}/assessments", json={})
            assert run.status_code == 201, run.text
            report = run.json()
            print(
                f"[demo] assessment run on graph v{report['graph_version']} — "
                f"{len(report['services'])} services, {len(report['assets'])} assets scored"
            )

            spof_subjects = {
                f["subject"] for f in report["findings"]
                if f["kind"] == "single_point_of_failure"
            }
            assert KNOWN_SPOFS <= spof_subjects, (
                f"missing known SPOFs: {KNOWN_SPOFS - spof_subjects}"
            )
            assert not (spof_subjects & KNOWN_REDUNDANT), (
                f"redundant members flagged as SPOFs: {spof_subjects & KNOWN_REDUNDANT}"
            )
            for finding in report["findings"]:
                assert finding["evidence"], f"finding without evidence: {finding['title']}"
            print("[demo] ranked findings (most services affected first):")
            for finding in report["findings"]:
                print(
                    f"  - {finding['kind']}: {finding['title']} — "
                    f"{len(finding['evidence'])} citation(s)"
                )

            fetch = client.get(f"/facilities/{facility_id}/assessments/1")
            assert fetch.status_code == 200, fetch.text
            assert fetch.json() == report, "stored report differs from the run response"
            print("[demo] stored report retrieved byte-identical — reproducibility holds")

    print("[demo] Phase 2 onboarding scanner complete — scored report round-tripped")


if __name__ == "__main__":
    main()
