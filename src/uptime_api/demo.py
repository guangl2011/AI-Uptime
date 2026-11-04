"""`make demo` runbook — Phase 1.

Exercises the core domain the way a customer onboarding would: the seed
cold-chain facility is created and committed through the real HTTP surface,
read back, and every node and edge is checked for provenance. Nothing here
touches a facility system — the only writes are to AI-Uptime's own database.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from uptime_api.app import create_app

SEED_PATH = Path(__file__).resolve().parents[2] / "seed" / "example_facility.json"


def main() -> None:
    """Load the seed facility through the API and verify provenance end to end."""
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

    print("[demo] Phase 1 core domain complete — seed facility round-tripped through the API")


if __name__ == "__main__":
    main()
