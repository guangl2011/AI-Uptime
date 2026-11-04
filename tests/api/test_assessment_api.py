"""Assessment API: run/list/read, 404s, and version-pinned reproducibility."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

FACILITY_ID = "assess_api_facility"


def _create_facility_with_graph(client: TestClient) -> None:
    assert client.post(
        "/facilities",
        json={"id": FACILITY_ID, "name": "Assessment API Facility"},
    ).status_code == 201
    committed = client.post(
        f"/facilities/{FACILITY_ID}/graph/versions",
        json={
            "author": "test",
            "notes": None,
            "nodes": [
                {
                    "node_key": "SRC-1",
                    "node_type": "asset",
                    "asset_type": "utility_feed",
                    "name": "Source 1",
                    "attributes": {},
                    "provenance": {
                        "source_kind": "inventory_import",
                        "source_system": "test",
                        "source_ref": "inv.csv#row=2",
                    },
                },
                {
                    "node_key": "SVC-1",
                    "node_type": "service",
                    "name": "Service 1",
                    "attributes": {},
                    "provenance": {
                        "source_kind": "technician_entry",
                        "source_system": "test",
                        "source_ref": "walkthrough",
                    },
                },
            ],
            "edges": [
                {
                    "from_node": "SRC-1",
                    "to_node": "SVC-1",
                    "provenance": {
                        "source_kind": "technician_entry",
                        "source_system": "test",
                        "source_ref": "walkthrough",
                    },
                }
            ],
        },
    )
    assert committed.status_code == 201, committed.text


def test_run_and_read_assessment_round_trips(client: TestClient) -> None:
    _create_facility_with_graph(client)

    run = client.post(f"/facilities/{FACILITY_ID}/assessments", json={"author": "tester"})
    assert run.status_code == 201, run.text
    report: dict[str, Any] = run.json()
    assert report["facility_id"] == FACILITY_ID
    assert report["graph_version"] == 1
    assert report["findings"], "connected N chain must produce findings"

    fetched = client.get(f"/facilities/{FACILITY_ID}/assessments/1")
    assert fetched.status_code == 200
    assert fetched.json() == report


def test_assessment_list_reports_history(client: TestClient) -> None:
    _create_facility_with_graph(client)
    assert client.post(f"/facilities/{FACILITY_ID}/assessments", json={}).status_code == 201
    assert client.post(f"/facilities/{FACILITY_ID}/assessments", json={}).status_code == 201

    listed = client.get(f"/facilities/{FACILITY_ID}/assessments")
    assert listed.status_code == 200
    summaries = listed.json()
    assert [s["id"] for s in summaries] == [1, 2]
    assert all(s["graph_version_no"] == 1 for s in summaries)


def test_assessment_is_pinned_to_its_graph_version(client: TestClient) -> None:
    _create_facility_with_graph(client)
    first = client.post(f"/facilities/{FACILITY_ID}/assessments", json={}).json()
    assert first["graph_version"] == 1

    # The twin moves on — the stored report must not.
    new_node = {
        "node_key": "SVC-2",
        "node_type": "service",
        "name": "Service 2",
        "attributes": {},
        "provenance": {
            "source_kind": "technician_entry",
            "source_system": "test",
            "source_ref": "walkthrough",
        },
    }
    latest = client.get(f"/facilities/{FACILITY_ID}/graph").json()
    client.post(
        f"/facilities/{FACILITY_ID}/graph/versions",
        json={
            "author": "test",
            "notes": "topology edit",
            "nodes": latest["nodes"] + [new_node],
            "edges": latest["edges"],
        },
    )

    assert client.get(f"/facilities/{FACILITY_ID}/assessments/1").json() == first


def test_repeat_assessment_of_pinned_version_is_reproducible(client: TestClient) -> None:
    _create_facility_with_graph(client)
    client.post(f"/facilities/{FACILITY_ID}/assessments", json={})
    client.post(f"/facilities/{FACILITY_ID}/assessments", json={})

    a = client.get(f"/facilities/{FACILITY_ID}/assessments/1").json()
    b = client.get(f"/facilities/{FACILITY_ID}/assessments/2").json()
    assert a == b


def test_assessment_can_pin_a_specific_version(client: TestClient) -> None:
    _create_facility_with_graph(client)
    latest = client.get(f"/facilities/{FACILITY_ID}/graph").json()
    client.post(
        f"/facilities/{FACILITY_ID}/graph/versions",
        json={
            "author": "test",
            "notes": "second version",
            "nodes": latest["nodes"],
            "edges": latest["edges"],
        },
    )

    run = client.post(
        f"/facilities/{FACILITY_ID}/assessments", json={"version_no": 1}
    ).json()
    assert run["graph_version"] == 1


def test_unknown_facility_or_version_returns_404(client: TestClient) -> None:
    assert client.post("/facilities/ghost/assessments", json={}).status_code == 404
    assert client.get("/facilities/ghost/assessments").status_code == 404
    assert client.get("/facilities/ghost/assessments/1").status_code == 404


def test_unknown_assessment_id_returns_404(client: TestClient) -> None:
    _create_facility_with_graph(client)
    missing = client.get(f"/facilities/{FACILITY_ID}/assessments/99")
    assert missing.status_code == 404


def test_report_findings_all_carry_evidence_citations(client: TestClient) -> None:
    _create_facility_with_graph(client)
    report = client.post(f"/facilities/{FACILITY_ID}/assessments", json={}).json()

    assert report["findings"], "test graph must produce findings"
    for finding in report["findings"]:
        assert finding["evidence"], f"finding without evidence: {finding['title']}"
        for citation in finding["evidence"]:
            assert citation["record_id"] > 0
            assert citation["claim"]


def test_assessment_payload_is_valid_json_round_trip(client: TestClient) -> None:
    _create_facility_with_graph(client)
    raw = client.post(f"/facilities/{FACILITY_ID}/assessments", json={}).text

    assert json.loads(raw)["facility_id"] == FACILITY_ID
