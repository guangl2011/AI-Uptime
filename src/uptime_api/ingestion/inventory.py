"""Equipment-inventory import: CSV or JSON, deterministic, provenance per row.

A row either becomes a well-formed asset node or is returned in ``unresolved``
with a concrete reason — ambiguous rows are flagged, never guessed into
existence (explicit-abstention practice). The parser holds no state and no
network surface; the caller decides what to persist.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

from pydantic import ValidationError

from uptime_api.graph.enums import AssetType, NodeType, SourceKind, UnresolvedReason
from uptime_api.graph.schemas import NodeIn, ProvenanceIn

DEFAULT_SOURCE_SYSTEM = "inventory_upload"
REQUIRED_CSV_COLUMNS = ("asset_id", "asset_type")


class InventoryFormatError(ValueError):
    """The uploaded file is not a parseable inventory (bad headers / bad JSON)."""


@dataclass(frozen=True)
class UnresolvedRow:
    """An inventory row that could not become a node, and why."""

    row: int
    reason: UnresolvedReason
    detail: str


@dataclass(frozen=True)
class ParsedInventory:
    """Parse result: well-formed nodes plus the rows flagged along the way."""

    nodes: list[NodeIn]
    unresolved: list[UnresolvedRow]


def parse_inventory(content: bytes, filename: str) -> ParsedInventory:
    """Parse an inventory upload into nodes + unresolved rows.

    Format comes from the filename extension (``.csv`` or ``.json``) —
    content sniffing is nondeterministic, so we don't do it.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        return _parse_csv(content.decode("utf-8-sig"), filename)
    if name.endswith(".json"):
        return _parse_json(content.decode("utf-8-sig"), filename)
    raise InventoryFormatError(f"unsupported inventory format: {filename!r} (.csv or .json)")


def _parse_csv(text: str, filename: str) -> ParsedInventory:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise InventoryFormatError("empty CSV inventory")
    headers = [field.strip() for field in reader.fieldnames]
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in headers]
    if missing:
        raise InventoryFormatError(f"CSV inventory missing required columns: {missing}")

    nodes: list[NodeIn] = []
    unresolved: list[UnresolvedRow] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(reader, start=2):  # row 1 is the header
        row = {key.strip(): (value or "").strip() for key, value in raw.items()}
        row_ref = _row_ref(filename, line_no)
        asset_id = row.get("asset_id", "")
        if not asset_id:
            unresolved.append(
                UnresolvedRow(line_no, UnresolvedReason.MISSING_ASSET_ID, "blank asset_id")
            )
            continue
        if asset_id in seen_ids:
            unresolved.append(
                UnresolvedRow(
                    line_no, UnresolvedReason.DUPLICATE_ASSET_ID, f"asset_id {asset_id!r} repeats"
                )
            )
            continue
        try:
            asset_type = AssetType(row.get("asset_type", ""))
        except ValueError:
            unresolved.append(
                UnresolvedRow(
                    line_no,
                    UnresolvedReason.UNKNOWN_ASSET_TYPE,
                    f"asset_type {row.get('asset_type', '')!r} is not in the taxonomy",
                )
            )
            continue
        seen_ids.add(asset_id)
        attributes = {
            key: value
            for key, value in row.items()
            if key not in ("asset_id", "asset_type", "name", "location") and value != ""
        }
        nodes.append(
            NodeIn(
                node_key=_unique_key(asset_id),
                node_type=NodeType.ASSET,
                asset_type=asset_type,
                name=row.get("name") or asset_id,
                location=row.get("location") or None,
                attributes=attributes,
                provenance=ProvenanceIn(
                    source_kind=SourceKind.INVENTORY_IMPORT,
                    source_system=DEFAULT_SOURCE_SYSTEM,
                    source_ref=row_ref,
                    payload={"asset_id": asset_id},
                ),
            )
        )
    return ParsedInventory(nodes=nodes, unresolved=unresolved)


def _parse_json(text: str, filename: str) -> ParsedInventory:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise InventoryFormatError(f"invalid JSON inventory: {error}") from error
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict) and isinstance(document.get("assets"), list):
        rows = document["assets"]
    else:
        raise InventoryFormatError('JSON inventory must be a list or {"assets": [...]}')

    nodes: list[NodeIn] = []
    unresolved: list[UnresolvedRow] = []
    for index, entry in enumerate(rows):
        line_no = index + 2
        if not isinstance(entry, dict):
            unresolved.append(
                UnresolvedRow(line_no, UnresolvedReason.MISSING_ASSET_ID, "row is not an object")
            )
            continue
        try:
            parsed = _json_row_to_node(entry, filename, line_no)
        except (ValueError, ValidationError) as error:
            unresolved.append(
                UnresolvedRow(line_no, UnresolvedReason.UNKNOWN_ASSET_TYPE, str(error)[:300])
            )
            continue
        if parsed is None:
            unresolved.append(
                UnresolvedRow(line_no, UnresolvedReason.MISSING_ASSET_ID, "blank asset_id")
            )
            continue
        nodes.append(parsed)
    return _drop_duplicate_keys(nodes, unresolved)


def _json_row_to_node(entry: dict[str, object], filename: str, line_no: int) -> NodeIn | None:
    asset_id = str(entry.get("asset_id", "") or "").strip()
    if not asset_id:
        return None
    asset_type = AssetType(str(entry.get("asset_type", "") or ""))
    known = {"asset_id", "asset_type", "name", "location"}
    attributes = {key: value for key, value in entry.items() if key not in known}
    return NodeIn(
        node_key=_unique_key(asset_id),
        node_type=NodeType.ASSET,
        asset_type=asset_type,
        name=str(entry.get("name") or asset_id),
        location=str(entry["location"]) if entry.get("location") else None,
        attributes=attributes,
        provenance=ProvenanceIn(
            source_kind=SourceKind.INVENTORY_IMPORT,
            source_system=DEFAULT_SOURCE_SYSTEM,
            source_ref=_row_ref(filename, line_no),
            payload={"asset_id": asset_id},
        ),
    )


def _drop_duplicate_keys(
    nodes: list[NodeIn], unresolved: list[UnresolvedRow]
) -> ParsedInventory:
    """Keep the first node per key; flag later repeats (deterministic)."""
    seen: set[str] = set()
    unique: list[NodeIn] = []
    for node in nodes:
        if node.node_key in seen:
            unresolved.append(
                UnresolvedRow(
                    0, UnresolvedReason.DUPLICATE_ASSET_ID, f"asset_id {node.node_key!r} repeats"
                )
            )
            continue
        seen.add(node.node_key)
        unique.append(node)
    return ParsedInventory(nodes=unique, unresolved=unresolved)


def _unique_key(asset_id: str) -> str:
    """Inventory ids become graph node keys; prefix keeps namespaces separate."""
    return f"INV-{asset_id}"


def _row_ref(filename: str, line_no: int) -> str:
    # A per-row reference the review UI can later resolve back to the upload.
    # Purely a function of the input so parsing stays deterministic.
    return f"{filename}#row={line_no}"
