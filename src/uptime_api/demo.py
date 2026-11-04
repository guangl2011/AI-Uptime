"""`make demo` runbook — Phase 0 stub.

Exercises the three things later phases will extend: the seed corpus loads,
the backend boots and serves its liveness probe, and the frontend builds
(Makefile runs that step). Every stage prints what it did; nothing here
touches a facility system — integrations are read-only by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from uptime_api.app import create_app

SEED_PATH = Path(__file__).resolve().parents[2] / "seed" / "example_facility.json"


def main() -> None:
    """Run the Phase 0 demo: load the seed facility, boot the API, check liveness."""
    facility = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    print(f"[demo] seed facility loaded: {facility['facility_id']} ({facility['segment']})")
    print(f"[demo] assets in stub inventory: {len(facility['assets'])}")

    with TestClient(create_app("sqlite:///:memory:")) as client:
        response = client.get("/healthz")
        assert response.status_code == 200, response.text
        print(f"[demo] backend booted, {response.url.path} -> {response.json()}")

    print("[demo] frontend build runs as the third `make demo` step (see Makefile)")
    print("[demo] Phase 0 stub complete — Phase 1 replaces the stub inventory with the graph model")


if __name__ == "__main__":
    main()
