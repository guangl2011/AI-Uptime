"""API tests: the graph twin surface — commit, read back, provenance intact."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _provenance(ref: str, kind: str = "technician_entry") -> dict[str, str]:
    return {"source_kind": kind, "source_system": "test", "source_ref": ref}


def _node(key: str, **overrides: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "node_key": key,
        "node_type": "asset",
        "asset_type": "compressor",
        "name": f"node {key}",
        "attributes": {},
        "provenance": _provenance(f"src#{key}"),
    }
    node.update(overrides)
    return node


def _make_facility(client: TestClient, facility_id: str = "f1") -> None:
    response = client.post(
        "/facilities",
        json={"id": facility_id, "name": "Test Facility", "segment": "cold_chain"},
    )
    assert response.status_code == 201, response.text


def test_facility_create_list_get_and_conflict(client: TestClient) -> None:
    _make_facility(client)
    _make_facility(client, "f2")

    listed = client.get("/facilities")
    assert listed.status_code == 200
    assert {f["id"] for f in listed.json()} == {"f1", "f2"}
    assert all(f["segment"] == "cold_chain" for f in listed.json())

    one = client.get("/facilities/f1")
    assert one.status_code == 200
    assert one.json()["name"] == "Test Facility"

    assert client.post(
        "/facilities", json={"id": "f1", "name": "Dup", "segment": "cold_chain"}
    ).status_code == 409
    assert client.get("/facilities/ghost").status_code == 404


def test_commit_read_round_trip_preserves_provenance(client: TestClient) -> None:
    _make_facility(client)
    commit = client.post(
        "/facilities/f1/graph/versions",
        json={
            "author": "tech@example.com",
            "notes": "initial topology",
            "nodes": [_node("A"), _node("B", node_type="service", asset_type=None)],
            "edges": [
                {
                    "from_node": "A",
                    "to_node": "B",
                    "failure_mode": "mechanical_failure",
                    "redundancy_config": "N",
                    "provenance": _provenance("edge-A-B"),
                }
            ],
        },
    )
    assert commit.status_code == 201, commit.text
    body = commit.json()
    assert body["version_no"] == 1
    assert body["node_count"] == 2 and body["edge_count"] == 1

    snapshot = client.get("/facilities/f1/graph").json()
    assert snapshot["version_no"] == 1
    assert snapshot["author"] == "tech@example.com"
    by_key = {n["node_key"]: n for n in snapshot["nodes"]}
    assert by_key["A"]["provenance"]["source_ref"] == "src#A"
    assert by_key["B"]["asset_type"] is None
    edge = snapshot["edges"][0]
    assert (edge["from_node"], edge["to_node"]) == ("A", "B")
    assert edge["provenance"]["source_ref"] == "edge-A-B"
    assert edge["redundancy_config"] == "N"


def test_pinned_version_reads_are_immutable(client: TestClient) -> None:
    _make_facility(client)
    for keys in (["A"], ["A", "B"]):
        commit = client.post(
            "/facilities/f1/graph/versions",
            json={"author": "test", "nodes": [_node(k) for k in keys], "edges": []},
        )
        assert commit.status_code == 201, commit.text

    latest = client.get("/facilities/f1/graph").json()
    v1 = client.get("/facilities/f1/graph", params={"version": 1}).json()
    assert latest["version_no"] == 2
    assert {n["node_key"] for n in latest["nodes"]} == {"A", "B"}
    assert {n["node_key"] for n in v1["nodes"]} == {"A"}
    assert [v["version_no"] for v in client.get("/facilities/f1/graph/versions").json()] == [1, 2]


def test_graph_404s(client: TestClient) -> None:
    assert client.get("/facilities/ghost/graph").status_code == 404
    _make_facility(client)
    assert client.get("/facilities/f1/graph").status_code == 404  # no versions yet
    commit = client.post(
        "/facilities/f1/graph/versions",
        json={"author": "test", "nodes": [_node("A")], "edges": []},
    )
    assert commit.status_code == 201
    assert client.get("/facilities/f1/graph", params={"version": 9}).status_code == 404


def test_invalid_topologies_rejected_with_422(client: TestClient) -> None:
    _make_facility(client)
    dangling = client.post(
        "/facilities/f1/graph/versions",
        json={"author": "test", "nodes": [_node("A")], "edges": [_edge_ghost()]},
    )
    assert dangling.status_code == 422

    duplicate = client.post(
        "/facilities/f1/graph/versions",
        json={"author": "test", "nodes": [_node("A"), _node("A")], "edges": []},
    )
    assert duplicate.status_code == 422


def _edge_ghost() -> dict[str, Any]:
    return {
        "from_node": "A",
        "to_node": "GHOST",
        "provenance": _provenance("bad"),
    }
