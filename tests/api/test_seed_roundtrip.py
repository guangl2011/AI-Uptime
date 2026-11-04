"""Phase 1 gate: the seed cold-chain facility round-trips through the API.

The committed seed file is loaded via the real HTTP surface (facility create,
graph version commit), read back, and every node and edge must carry
resolvable provenance. This is the acceptance check from the agent-team plan:
"a cold-chain example facility loads via seed script and round-trips through
the API with provenance intact."
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

ALLOWED_SOURCE_KINDS = {
    "inventory_import",
    "technician_entry",
    "oem_doc",
    "alarm",
    "work_order",
    "sensor",
}


def test_seed_round_trips_with_provenance_intact(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    load_seed_via_api(client, seed_facility)

    snapshot = client.get(f"/facilities/{seed_facility['facility_id']}/graph").json()
    assert snapshot["version_no"] == 1

    # Provenance intact on EVERY node and edge — the Phase 1 gate.
    for node in snapshot["nodes"]:
        provenance = node["provenance"]
        assert provenance["id"] > 0
        assert provenance["source_kind"] in ALLOWED_SOURCE_KINDS
        assert provenance["source_system"]
        assert provenance["source_ref"]
    for edge in snapshot["edges"]:
        provenance = edge["provenance"]
        assert provenance["source_kind"] in ALLOWED_SOURCE_KINDS
        assert provenance["source_system"]
        assert provenance["source_ref"]

    # Round-trip fidelity: nothing renamed, dropped, or invented.
    seed_keys = {n["node_key"] for n in seed_facility["nodes"]}
    assert {n["node_key"] for n in snapshot["nodes"]} == seed_keys
    assert len(snapshot["edges"]) == len(seed_facility["edges"])


def test_seed_facility_shape_matches_phase1_contract(
    seed_facility: dict[str, Any],
) -> None:
    assets = [n for n in seed_facility["nodes"] if n["node_type"] == "asset"]
    asset_types = {n["asset_type"] for n in assets}

    assert 5 <= len(assets) <= 15
    # One refrigeration rack plus a power chain, per the Phase 1 brief.
    assert "refrigeration_rack" in asset_types
    assert {"ups", "ats", "generator"} <= asset_types
    assert "compressor" in asset_types
    assert any(n["node_type"] == "service" for n in seed_facility["nodes"])

    # Redundancy configs are present and canonical (N / N+1 / 2N).
    configs = {e.get("redundancy_config") for e in seed_facility["edges"]}
    assert {"N", "N+1", "2N"} <= configs
    groups = {e["member_group"] for e in seed_facility["edges"] if e.get("member_group")}
    assert "rack_compressors" in groups and "power_sources" in groups

    # Every element carries provenance before it ever reaches the API.
    for element in [*seed_facility["nodes"], *seed_facility["edges"]]:
        provenance = element["provenance"]
        assert provenance["source_kind"] in ALLOWED_SOURCE_KINDS
        assert provenance["source_ref"]
