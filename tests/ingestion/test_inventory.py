"""Inventory parser tests: determinism, provenance refs, explicit abstention."""

from __future__ import annotations

import pytest

from uptime_api.ingestion.inventory import (
    InventoryFormatError,
    parse_inventory,
)

CSV_OK = (
    b"asset_id,asset_type,name,location,rated_kw\n"
    b"COMP-1,compressor,Compressor 1,machine room,22\n"
    b"COMP-2,compressor,Compressor 2,machine room,22\n"
    b"CR-1,cold_room,Cold Room 1,freezer wing,\n"
)

JSON_OK = (
    b'{"assets": ['
    b'{"asset_id": "PUMP-1", "asset_type": "compressor", "location": "plant"},'
    b'{"asset_id": "PUMP-2", "asset_type": "compressor", "name": "Pump 2"}'
    b"]}"
)


def test_csv_rows_become_nodes_with_row_provenance() -> None:
    parsed = parse_inventory(CSV_OK, "inventory.csv")
    assert [n.node_key for n in parsed.nodes] == ["INV-COMP-1", "INV-COMP-2", "INV-CR-1"]
    assert parsed.unresolved == []
    first = parsed.nodes[0]
    assert first.asset_type == "compressor"
    assert first.location == "machine room"
    assert first.provenance.source_kind == "inventory_import"
    assert first.provenance.source_ref == "inventory.csv#row=2"
    # Non-schema columns land in attributes; blanks are dropped.
    assert first.attributes == {"rated_kw": "22"}
    assert parsed.nodes[2].attributes == {}


def test_json_assets_wrapper_becomes_nodes() -> None:
    parsed = parse_inventory(JSON_OK, "inventory.json")
    assert [n.node_key for n in parsed.nodes] == ["INV-PUMP-1", "INV-PUMP-2"]
    assert parsed.nodes[1].name == "Pump 2"
    assert parsed.nodes[0].provenance.source_ref == "inventory.json#row=2"


def test_unknown_asset_type_is_flagged_not_guessed() -> None:
    content = b"asset_id,asset_type\nX-1,quantum_heater\n"
    parsed = parse_inventory(content, "inventory.csv")
    assert parsed.nodes == []
    assert len(parsed.unresolved) == 1
    flagged = parsed.unresolved[0]
    assert flagged.reason == "unknown_asset_type"
    assert "quantum_heater" in flagged.detail
    assert flagged.row == 2


def test_missing_asset_id_is_flagged() -> None:
    content = b"asset_id,asset_type\n,compressor\n"
    parsed = parse_inventory(content, "inventory.csv")
    assert parsed.nodes == []
    assert parsed.unresolved[0].reason == "missing_asset_id"


def test_duplicate_asset_ids_are_flagged_first_wins() -> None:
    content = (
        b"asset_id,asset_type\n"
        b"COMP-1,compressor\n"
        b"COMP-1,compressor\n"
    )
    parsed = parse_inventory(content, "inventory.csv")
    assert [n.node_key for n in parsed.nodes] == ["INV-COMP-1"]
    assert parsed.unresolved[0].reason == "duplicate_asset_id"
    assert parsed.unresolved[0].row == 3


def test_missing_required_csv_columns_raise() -> None:
    with pytest.raises(InventoryFormatError, match="asset_type"):
        parse_inventory(b"asset_id,name\nA-1,Alpha\n", "inventory.csv")


def test_unsupported_extension_raises() -> None:
    with pytest.raises(InventoryFormatError, match="xlsx"):
        parse_inventory(b"whatever", "inventory.xlsx")


def test_parsing_is_deterministic() -> None:
    """Same bytes in, byte-identical node payloads out (no hidden state)."""
    first = parse_inventory(CSV_OK, "inventory.csv")
    second = parse_inventory(CSV_OK, "inventory.csv")
    assert [n.model_dump() for n in first.nodes] == [n.model_dump() for n in second.nodes]
