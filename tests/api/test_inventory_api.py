"""API tests: equipment-inventory upload — CSV/JSON import, flagged ambiguity."""

from __future__ import annotations

from fastapi.testclient import TestClient

CSV_OK = (
    b"asset_id,asset_type,name,location\n"
    b"COMP-1,compressor,Compressor 1,machine room\n"
    b"CR-1,cold_room,Cold Room 1,freezer wing\n"
)

CSV_MIXED = CSV_OK + b"WEIRD-1,quantum_heater,Odd unit,basement\n"


def _create_facility(client: TestClient, facility_id: str = "f1") -> None:
    response = client.post(
        "/facilities",
        json={"id": facility_id, "name": "Import Target", "segment": "cold_chain"},
    )
    assert response.status_code == 201, response.text


def test_csv_import_creates_version_with_provenance(client: TestClient) -> None:
    _create_facility(client)
    import_response = client.post(
        "/facilities/f1/inventory",
        files={"file": ("inventory.csv", CSV_OK, "text/csv")},
    )
    assert import_response.status_code == 201, import_response.text
    body = import_response.json()
    assert body["version_no"] == 1
    assert body["nodes_created"] == 2
    assert body["unresolved"] == []

    snapshot = client.get("/facilities/f1/graph").json()
    assert snapshot["version_no"] == 1
    assert len(snapshot["nodes"]) == 2 and snapshot["edges"] == []
    for node in snapshot["nodes"]:
        assert node["provenance"]["source_kind"] == "inventory_import"
        assert "#row=" in node["provenance"]["source_ref"]


def test_json_import_works_too(client: TestClient) -> None:
    _create_facility(client)
    payload = b'{"assets": [{"asset_id": "GEN-1", "asset_type": "generator"}]}'
    import_response = client.post(
        "/facilities/f1/inventory",
        files={"file": ("inventory.json", payload, "application/json")},
    )
    assert import_response.status_code == 201, import_response.text
    assert import_response.json()["nodes_created"] == 1


def test_ambiguous_rows_flagged_not_imported(client: TestClient) -> None:
    _create_facility(client)
    import_response = client.post(
        "/facilities/f1/inventory",
        files={"file": ("inventory.csv", CSV_MIXED, "text/csv")},
    )
    assert import_response.status_code == 201, import_response.text
    body = import_response.json()
    assert body["nodes_created"] == 2  # the good rows
    assert len(body["unresolved"]) == 1
    assert body["unresolved"][0]["reason"] == "unknown_asset_type"

    # The flagged row must NOT exist in the graph.
    snapshot = client.get("/facilities/f1/graph").json()
    assert {n["node_key"] for n in snapshot["nodes"]} == {"INV-COMP-1", "INV-CR-1"}


def test_second_import_conflicts_rather_than_overwrites(client: TestClient) -> None:
    _create_facility(client)
    for _ in range(2):
        response = client.post(
            "/facilities/f1/inventory",
            files={"file": ("inventory.csv", CSV_OK, "text/csv")},
        )
    assert response.status_code == 409
    assert "already has a graph version" in response.json()["detail"]


def test_unsupported_format_and_missing_facility(client: TestClient) -> None:
    _create_facility(client)
    bad_format = client.post(
        "/facilities/f1/inventory",
        files={"file": ("inventory.xlsx", b"binary-ish", "application/octet-stream")},
    )
    assert bad_format.status_code == 422

    missing = client.post(
        "/facilities/ghost/inventory",
        files={"file": ("inventory.csv", CSV_OK, "text/csv")},
    )
    assert missing.status_code == 404
