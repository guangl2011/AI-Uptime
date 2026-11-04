"""Synthetic cold-chain facilities: the beachhead corpus (Phase 2/3, unchanged).

Moved verbatim from the pre-Phase-4 ``evals/corpus.py`` — the RNG stream is
part of the frozen Phase 3 contract (same seed -> identical cold-chain
facilities, labels, and events), so this module adds only the ``segment``
tag and the generalized event/corrupt-row hooks. Phase 4 extends the corpus
with AIDC and hospital segments; it does not disturb this one.
"""

from __future__ import annotations

import random

from evals.corpus.common import (
    SOURCE_REF_INVENTORY,
    SOURCE_REF_TOPOLOGY,
    CorruptRow,
    FacilityCorpus,
    InventoryRow,
    _edge,
    _inventory_csv,
    _node,
    _prov,
    generate_events,
)

REPLAY_PREFIXES = ("RACK", "COMP", "COND")
SEGMENT = "cold_chain"


def generate_cold_chain(seed: int, index: int) -> FacilityCorpus:
    """One cold-chain facility: shape drawn from the seed, labels from construction."""
    rng = random.Random(seed * 1_000_003 + index)
    facility_id = f"eval_coldchain_{index:03d}"
    walk_ref = _prov("technician_entry", SOURCE_REF_TOPOLOGY.format(facility=facility_id))

    n_rooms = rng.randint(2, 5)
    n_racks = rng.randint(1, 2)
    sources_2n = rng.random() < 0.7
    compressor_counts = [rng.randint(1, 3) for _ in range(n_racks)]
    shape = (
        f"src-{'2N' if sources_2n else 'N'}/racks-{n_racks}/rooms-{n_rooms}/"
        f"comps-{'.'.join(str(count) for count in compressor_counts)}"
    )

    # Room r lands on rack r % n_racks — balanced by construction.
    rack_rooms: list[list[int]] = [[] for _ in range(n_racks)]
    for room in range(n_rooms):
        rack_rooms[room % n_racks].append(room)

    all_services = [f"SVC-REFRIG-{room}" for room in range(n_rooms)] + ["SVC-POWER"]

    nodes: list[dict[str, object]] = []
    clean_rows: list[InventoryRow] = []

    def add_asset(
        key: str, name: str, asset_type: str, location: str, attributes: dict[str, str]
    ) -> None:
        row_no = 2 + len(clean_rows)  # header is row 1
        prov = _prov(
            "inventory_import",
            SOURCE_REF_INVENTORY.format(facility=facility_id, row=row_no),
        )
        nodes.append(
            _node(
                key,
                name,
                asset_type,
                location,
                prov,
                dict(attributes),
            )
        )
        clean_rows.append(
            InventoryRow(
                asset_id=key,
                asset_type=asset_type,
                name=name,
                location=location,
                attributes=dict(attributes),
            )
        )

    # Power chain.
    add_asset("UTILITY-1", "Utility feed", "utility_feed", "site entrance", {})
    if sources_2n:
        add_asset("GEN-1", "Standby generator", "generator", "electrical room", {"fuel": "diesel"})
    add_asset("ATS-1", "Automatic transfer switch", "ats", "electrical room", {})
    add_asset("UPS-1", "UPS module 1", "ups", "electrical room", {"battery_minutes": "15"})
    add_asset("PDU-1", "Main PDU", "pdu", "electrical room", {})
    # Refrigeration train and cold rooms.
    for rack in range(n_racks):
        add_asset(
            f"RACK-{rack + 1}",
            f"Compressor rack {rack + 1}",
            "refrigeration_rack",
            "machine room",
            {"refrigerant": "R-404A", "capacity_kw": "180"},
        )
        for comp in range(compressor_counts[rack]):
            add_asset(
                f"COMP-{rack + 1}-{comp + 1}",
                f"Compressor {rack + 1}-{comp + 1}",
                "compressor",
                "machine room",
                {"rated_kw": "22"},
            )
        add_asset(f"COND-{rack + 1}", f"Condenser {rack + 1}", "condenser", "roof", {})
    for room in range(n_rooms):
        setpoint = "-18" if room % 2 == 0 else "2"
        add_asset(
            f"CR-{room}",
            f"Cold Room {room}",
            "cold_room",
            "freezer wing" if setpoint == "-18" else "cooler wing",
            {"setpoint_c": setpoint, "volume_m3": "850" if setpoint == "-18" else "420"},
        )
    for room in range(n_rooms):
        nodes.append(
            _node(f"SVC-REFRIG-{room}", f"Cold Room {room} refrigeration", None, None, walk_ref)
        )
    nodes.append(_node("SVC-POWER", "Facility power distribution", None, None, walk_ref))

    edges: list[dict[str, object]] = []
    if sources_2n:
        edges.append(_edge("UTILITY-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
        edges.append(_edge("GEN-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
    else:
        edges.append(_edge("UTILITY-1", "ATS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("ATS-1", "UPS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("UPS-1", "PDU-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("PDU-1", "SVC-POWER", "power_loss", walk_ref, "N"))
    for rack in range(n_racks):
        rack_key = f"RACK-{rack + 1}"
        group = f"rack_compressors-{rack + 1}"
        config = "N+1" if compressor_counts[rack] >= 2 else "N"
        for comp in range(compressor_counts[rack]):
            edges.append(
                _edge(
                    f"COMP-{rack + 1}-{comp + 1}",
                    rack_key,
                    "mechanical_failure",
                    walk_ref,
                    config,
                    group,
                )
            )
        edges.append(_edge(f"COND-{rack + 1}", rack_key, "refrigerant_loss", walk_ref, "N"))
        edges.append(_edge("SVC-POWER", rack_key, "power_loss", walk_ref, "N"))
        for room in rack_rooms[rack]:
            edges.append(
                _edge(rack_key, f"SVC-REFRIG-{room}", "mechanical_failure", walk_ref, "N")
            )
            edges.append(
                _edge(f"SVC-REFRIG-{room}", f"CR-{room}", "refrigerant_loss", walk_ref, "N")
            )
            edges.append(_edge("SVC-POWER", f"CR-{room}", "power_loss", walk_ref, "N"))

    # Adjudicated labels, straight from the construction (NOT the assessor).
    expected_exposed: dict[str, frozenset[str]] = {}
    expected_impacted: dict[str, frozenset[str]] = {}
    expected_exposed["UTILITY-1"] = frozenset() if sources_2n else frozenset(all_services)
    expected_impacted["UTILITY-1"] = frozenset(all_services)
    if sources_2n:
        expected_exposed["GEN-1"] = frozenset()
        expected_impacted["GEN-1"] = frozenset(all_services)
    for key in ("ATS-1", "UPS-1", "PDU-1"):
        expected_exposed[key] = frozenset(all_services)
        expected_impacted[key] = frozenset(all_services)
    for rack in range(n_racks):
        rack_services = frozenset(f"SVC-REFRIG-{room}" for room in rack_rooms[rack])
        expected_exposed[f"RACK-{rack + 1}"] = rack_services
        expected_impacted[f"RACK-{rack + 1}"] = rack_services
        for comp in range(compressor_counts[rack]):
            comp_key = f"COMP-{rack + 1}-{comp + 1}"
            # A compressor alone in its group is an unprotected supplier;
            # a partner (N+1, >= 2 members) absorbs its fault.
            expected_exposed[comp_key] = (
                frozenset() if compressor_counts[rack] >= 2 else rack_services
            )
            expected_impacted[comp_key] = rack_services
        expected_exposed[f"COND-{rack + 1}"] = rack_services
        expected_impacted[f"COND-{rack + 1}"] = rack_services
    for room in range(n_rooms):
        expected_exposed[f"CR-{room}"] = frozenset()
        expected_impacted[f"CR-{room}"] = frozenset()

    asset_keys = tuple(
        str(node["node_key"]) for node in nodes if node["node_type"] == "asset"
    )

    corrupt = [
        CorruptRow(2 + len(clean_rows), "blank_asset_id", "missing_asset_id"),
        CorruptRow(3 + len(clean_rows), "unknown_asset_type", "unknown_asset_type"),
        CorruptRow(4 + len(clean_rows), "duplicate_asset_id", "duplicate_asset_id"),
    ]

    events, replay_cases = generate_events(facility_id, expected_exposed, REPLAY_PREFIXES)

    graph: dict[str, object] = {
        "author": "eval_corpus",
        "notes": f"Synthetic cold-chain facility {index} (shape {shape})",
        "nodes": nodes,
        "edges": edges,
    }
    return FacilityCorpus(
        facility_id=facility_id,
        name=f"Eval Cold Storage {index}",
        segment=SEGMENT,
        shape=shape,
        graph=graph,
        inventory_csv=_inventory_csv(tuple(clean_rows), clean_rows[0].asset_id, "cold_room"),
        clean_rows=tuple(clean_rows),
        corrupt_rows=tuple(corrupt),
        assets=asset_keys,
        expected_exposed=expected_exposed,
        expected_impacted=expected_impacted,
        events=events,
        replay_cases=replay_cases,
    )
