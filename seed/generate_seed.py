"""Generate the Phase 1 seed: a cold-chain example facility with provenance.

Produces ``seed/example_facility.json`` — facility metadata plus a full graph
snapshot (nodes + edges, each with provenance) in the exact shape the API's
``POST /facilities/{fid}/graph/versions`` accepts, so the seed round-trips
through the real HTTP surface. 11 assets (one refrigeration rack + the power
chain) and 3 services.

Assets trace to an inventory import; services and topology edges trace to a
technician walkthrough; the rack's capacity note traces to an OEM doc —
three source kinds, all resolvable.

Usage: python seed/generate_seed.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FACILITY_ID = "cold_chain_example_01"
INVENTORY_FILE = "inventory_2026_11.csv"
WALKTHROUGH_REF = "technician_walkthrough_2026-11-01"
OEM_REF = "rack_oem_datasheet_revC#p=4"


def _node(
    key: str,
    name: str,
    asset_type: str | None,
    location: str | None,
    ref: str,
    kind: str = "inventory_import",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one node payload; services have no asset_type by contract."""
    node: dict[str, Any] = {
        "node_key": key,
        "name": name,
        "node_type": "service" if asset_type is None else "asset",
        "attributes": attributes or {},
        "provenance": {
            "source_kind": kind,
            "source_system": "seed_corpus",
            "source_ref": ref,
        },
    }
    if asset_type is not None:
        node["asset_type"] = asset_type
    if location is not None:
        node["location"] = location
    return node


def _edge(
    from_key: str,
    to_key: str,
    failure_mode: str,
    ref: str,
    redundancy: str | None = None,
    member_group: str | None = None,
) -> dict[str, Any]:
    """Build one dependency edge with technician provenance."""
    edge: dict[str, Any] = {
        "from_node": from_key,
        "to_node": to_key,
        "failure_mode": failure_mode,
        "provenance": {
            "source_kind": "technician_entry",
            "source_system": "seed_corpus",
            "source_ref": ref,
        },
    }
    if redundancy is not None:
        edge["redundancy_config"] = redundancy
    if member_group is not None:
        edge["member_group"] = member_group
    return edge


def _inventory_row(row_no: int) -> str:
    return f"{INVENTORY_FILE}#row={row_no}"


def build_seed() -> dict[str, Any]:
    """The deterministic seed facility payload."""
    nodes = [
        # Power chain (assets, inventory rows 2-6)
        _node("UTILITY-1", "Utility feed", "utility_feed", "site entrance", _inventory_row(2)),
        _node(
            "GEN-1",
            "Standby generator",
            "generator",
            "electrical room",
            _inventory_row(3),
            attributes={"rated_kw": 750, "fuel": "diesel"},
        ),
        _node("ATS-1", "Automatic transfer switch", "ats", "electrical room", _inventory_row(4)),
        _node(
            "UPS-1",
            "UPS module 1",
            "ups",
            "electrical room",
            _inventory_row(5),
            attributes={"rated_kva": 100, "battery_minutes": 15},
        ),
        _node("PDU-1", "Main PDU", "pdu", "electrical room", _inventory_row(6)),
        # Refrigeration rack train (assets, inventory rows 7-10)
        # Rack capacity verified against the OEM datasheet — OEM-doc provenance.
        _node(
            "RACK-1",
            "Compressor rack 1",
            "refrigeration_rack",
            "machine room",
            OEM_REF,
            kind="oem_doc",
            attributes={"capacity_kw": 180, "refrigerant": "R-404A"},
        ),
        _node(
            "COMP-1",
            "Compressor 1",
            "compressor",
            "machine room",
            _inventory_row(8),
            attributes={"rated_kw": 22},
        ),
        _node(
            "COMP-2",
            "Compressor 2",
            "compressor",
            "machine room",
            _inventory_row(9),
            attributes={"rated_kw": 22},
        ),
        _node(
            "COND-1",
            "Rooftop condenser",
            "condenser",
            "roof",
            _inventory_row(10),
            attributes={"rated_kw": 200},
        ),
        # Cold rooms (assets, inventory rows 11-12)
        _node(
            "CR-1",
            "Cold Room 1 (freezer)",
            "cold_room",
            "freezer wing",
            _inventory_row(11),
            attributes={"setpoint_c": -18, "volume_m3": 850},
        ),
        _node(
            "CR-2",
            "Cold Room 2 (cooler)",
            "cold_room",
            "cooler wing",
            _inventory_row(12),
            attributes={"setpoint_c": 2, "volume_m3": 420},
        ),
        # Services (technician-defined, not in any inventory)
        _node(
            "SVC-POWER",
            "Facility power distribution",
            None,
            None,
            WALKTHROUGH_REF,
            kind="technician_entry",
        ),
        _node(
            "SVC-REFRIG-CR1",
            "Cold Room 1 refrigeration",
            None,
            None,
            WALKTHROUGH_REF,
            kind="technician_entry",
        ),
        _node(
            "SVC-REFRIG-CR2",
            "Cold Room 2 refrigeration",
            None,
            None,
            WALKTHROUGH_REF,
            kind="technician_entry",
        ),
    ]

    edges = [
        # Power chain: two sources into the ATS (2N), then downstream N feeds.
        _edge(
            "UTILITY-1", "ATS-1", "power_loss", WALKTHROUGH_REF, "2N", "power_sources"
        ),
        _edge("GEN-1", "ATS-1", "power_loss", WALKTHROUGH_REF, "2N", "power_sources"),
        _edge("ATS-1", "UPS-1", "power_loss", WALKTHROUGH_REF, "N"),
        _edge("UPS-1", "PDU-1", "power_loss", WALKTHROUGH_REF, "N"),
        _edge("PDU-1", "SVC-POWER", "power_loss", WALKTHROUGH_REF, "N"),
        _edge("SVC-POWER", "RACK-1", "power_loss", WALKTHROUGH_REF, "N"),
        _edge("SVC-POWER", "CR-1", "power_loss", WALKTHROUGH_REF, "N"),
        _edge("SVC-POWER", "CR-2", "power_loss", WALKTHROUGH_REF, "N"),
        # Refrigeration: N+1 compressor pair, one condenser, rack serves both rooms.
        _edge(
            "COMP-1", "RACK-1", "mechanical_failure", WALKTHROUGH_REF, "N+1", "rack_compressors"
        ),
        _edge(
            "COMP-2", "RACK-1", "mechanical_failure", WALKTHROUGH_REF, "N+1", "rack_compressors"
        ),
        _edge("COND-1", "RACK-1", "refrigerant_loss", WALKTHROUGH_REF, "N"),
        _edge("RACK-1", "SVC-REFRIG-CR1", "mechanical_failure", WALKTHROUGH_REF, "N"),
        _edge("RACK-1", "SVC-REFRIG-CR2", "mechanical_failure", WALKTHROUGH_REF, "N"),
        _edge("SVC-REFRIG-CR1", "CR-1", "refrigerant_loss", WALKTHROUGH_REF, "N"),
        _edge("SVC-REFRIG-CR2", "CR-2", "refrigerant_loss", WALKTHROUGH_REF, "N"),
    ]

    return {
        "facility_id": FACILITY_ID,
        "name": "Example Cold Storage",
        "segment": "cold_chain",
        "author": "seed_corpus",
        "notes": "Phase 1 seed: rack (N+1 compressors) + power chain (2N sources).",
        "nodes": nodes,
        "edges": edges,
    }


def write_seed(out_path: Path) -> Path:
    """Write the seed facility JSON deterministically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_seed(), indent=2, sort_keys=True) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("seed/example_facility.json"))
    args = parser.parse_args()
    written = write_seed(args.out)
    seed = build_seed()
    assets = sum(1 for node in seed["nodes"] if node["node_type"] == "asset")
    print(
        f"[seed] wrote {written} ({assets} assets, "
        f"{len(seed['nodes']) - assets} services, {len(seed['edges'])} edges)"
    )


if __name__ == "__main__":
    main()
