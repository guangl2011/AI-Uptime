"""Synthetic AIDC facilities: CDU/chiller/UPS/generator chains (Phase 4).

The DOE proposal's flagship domain: utility + standby generator (2N source
pair in most shapes), ATS -> UPS -> PDU power chain, and per-row cooling
trains — chillers (N or N+1) feeding a chilled-water service, pumps
(N or N+1) feeding a CDU, then CRAH, rack cooling, and IT power.

Construction grammar (every shape):
- SVC-POWER feeds chillers and racks; chilled water flows chiller -> pump ->
  CDU -> CRAH -> row cooling; racks serve IT per row.
- A chiller pair (N+1, group ``chillers-<row>``) absorbs any one member's
  fault; a lone chiller is an unprotected supplier of its row's water loop.
- Dual pumps (N+1, group ``chw_pumps-<row>``) absorb a pump fault; a lone
  pump is an unprotected supplier of CDU/CRAH cooling.
- The 2N source pair (group ``power_sources``) absorbs a utility OR generator
  fault; without it, the lone utility feed is an unprotected supplier of
  everything downstream of the ATS.

Because redundancy absorbs faults mid-tree, the exposed set for upstream
assets is NOT every service: a row whose chillers are N+1 keeps its
chilled-water service protected, so an upstream (utility/ATS/UPS/PDU) fault
exposes only the services no group shields — power itself, the row IT
services, and (where a lone chiller faces dual pumps) the chilled-water
service while row cooling stays absorbed at the pumps. Labels below encode
exactly that; the exposure gate pins the assessor to it.
"""

from __future__ import annotations

import random

from evals.corpus.common import (
    AIDC_SEED_OFFSET,
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

REPLAY_PREFIXES = ("CDU", "CHILLER", "PUMP", "CRAH")
SEGMENT = "aidc"


def generate_aidc(seed: int, index: int) -> FacilityCorpus:
    """One AIDC facility: shape drawn from the seed, labels from construction."""
    rng = random.Random(seed * 1_000_003 + AIDC_SEED_OFFSET + index)
    facility_id = f"eval_aidc_{index:03d}"
    walk_ref = _prov("technician_entry", SOURCE_REF_TOPOLOGY.format(facility=facility_id))

    n_rows = rng.randint(2, 3)
    n_chillers = rng.randint(1, 3)
    sources_2n = rng.random() < 0.7
    dual_pumps = rng.random() < 0.6
    shape = (
        f"aidc/rows-{n_rows}/chillers-{n_chillers}"
        f"/src-{'2N' if sources_2n else 'N'}/pumps-{'n+1' if dual_pumps else 'n'}"
    )

    rows = range(1, n_rows + 1)
    row_services = {
        row: frozenset({f"SVC-CHW-{row}", f"SVC-COOLING-{row}", f"SVC-IT-{row}"})
        for row in rows
    }
    all_services = frozenset({"SVC-POWER"}) | frozenset().union(*row_services.values())

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
        nodes.append(_node(key, name, asset_type, location, prov, dict(attributes)))
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
    # Cooling plant and data-hall rows.
    for row in rows:
        for chiller in range(1, n_chillers + 1):
            add_asset(
                f"CHILLER-{row}-{chiller}",
                f"Chiller {row}-{chiller}",
                "chiller",
                "chilled water plant",
                {"capacity_kw": "600", "refrigerant": "R-134a"},
            )
        pump_count = 2 if dual_pumps else 1
        for pump in range(1, pump_count + 1):
            add_asset(
                f"PUMP-{row}-{pump}",
                f"CHW pump {row}-{pump}",
                "pump",
                "chilled water plant",
                {"rated_kw": "11"},
            )
        add_asset(
            f"CDU-{row}", f"Cooling distribution unit {row}", "cdu", f"data hall row {row}",
            {"rated_kw": "40"},
        )
        add_asset(
            f"CRAH-{row}", f"CRAH unit {row}", "crah", f"data hall row {row}",
            {"capacity_kw": "45"},
        )
        add_asset(
            f"RACK-{row}", f"IT rack row {row}", "it_rack", f"data hall row {row}",
            {"rated_kw": "30"},
        )
    nodes.append(_node("SVC-POWER", "Facility power distribution", None, None, walk_ref))
    for row in rows:
        nodes.append(
            _node(f"SVC-CHW-{row}", f"Row {row} chilled water", None, None, walk_ref)
        )
        nodes.append(
            _node(f"SVC-COOLING-{row}", f"Row {row} rack cooling", None, None, walk_ref)
        )
        nodes.append(_node(f"SVC-IT-{row}", f"Row {row} IT load", None, None, walk_ref))

    edges: list[dict[str, object]] = []
    if sources_2n:
        edges.append(_edge("UTILITY-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
        edges.append(_edge("GEN-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
    else:
        edges.append(_edge("UTILITY-1", "ATS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("ATS-1", "UPS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("UPS-1", "PDU-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("PDU-1", "SVC-POWER", "power_loss", walk_ref, "N"))
    for row in rows:
        chiller_group = f"chillers-{row}"
        chiller_config = "N+1" if n_chillers >= 2 else "N"
        for chiller in range(1, n_chillers + 1):
            edges.append(
                _edge("SVC-POWER", f"CHILLER-{row}-{chiller}", "power_loss", walk_ref, "N")
            )
            edges.append(
                _edge(
                    f"CHILLER-{row}-{chiller}",
                    f"SVC-CHW-{row}",
                    "mechanical_failure",
                    walk_ref,
                    chiller_config if n_chillers >= 2 else None,
                    chiller_group if n_chillers >= 2 else None,
                )
            )
        for pump in range(1, (2 if dual_pumps else 1) + 1):
            edges.append(_edge(f"SVC-CHW-{row}", f"PUMP-{row}-{pump}", "power_loss", walk_ref, "N"))
            edges.append(
                _edge(
                    f"PUMP-{row}-{pump}",
                    f"CDU-{row}",
                    "mechanical_failure",
                    walk_ref,
                    "N+1" if dual_pumps else "N",
                    f"chw_pumps-{row}" if dual_pumps else None,
                )
            )
        edges.append(_edge(f"CDU-{row}", f"CRAH-{row}", "mechanical_failure", walk_ref, "N"))
        edges.append(
            _edge(f"CRAH-{row}", f"SVC-COOLING-{row}", "mechanical_failure", walk_ref, "N")
        )
        edges.append(
            _edge(f"SVC-COOLING-{row}", f"RACK-{row}", "mechanical_failure", walk_ref, "N")
        )
        edges.append(_edge("SVC-POWER", f"RACK-{row}", "power_loss", walk_ref, "N"))
        edges.append(_edge(f"RACK-{row}", f"SVC-IT-{row}", "power_loss", walk_ref, "N"))

    # Adjudicated labels, straight from the construction (NOT the assessor).
    expected_exposed: dict[str, frozenset[str]] = {}
    expected_impacted: dict[str, frozenset[str]] = {}
    # What an upstream (pre-chiller) fault exposes: power itself, every row IT
    # service (racks are power-fed directly), plus whatever a lone chiller's
    # failure would expose that no downstream group shields.
    upstream_exposed = {"SVC-POWER"} | {f"SVC-IT-{row}" for row in rows}
    for row in rows:
        if n_chillers == 1:
            upstream_exposed.add(f"SVC-CHW-{row}")
            if not dual_pumps:
                upstream_exposed.add(f"SVC-COOLING-{row}")
    upstream = frozenset(upstream_exposed)

    expected_exposed["UTILITY-1"] = frozenset() if sources_2n else upstream
    expected_impacted["UTILITY-1"] = all_services
    if sources_2n:
        expected_exposed["GEN-1"] = frozenset()
        expected_impacted["GEN-1"] = all_services
    for key in ("ATS-1", "UPS-1", "PDU-1"):
        expected_exposed[key] = upstream
        expected_impacted[key] = all_services
    for row in rows:
        for chiller in range(1, n_chillers + 1):
            key = f"CHILLER-{row}-{chiller}"
            # N+1 partner absorbs; a lone chiller exposes its water loop —
            # and, without standby pumps, the row cooling and IT beneath it.
            if n_chillers >= 2:
                expected_exposed[key] = frozenset()
            elif dual_pumps:
                expected_exposed[key] = frozenset({f"SVC-CHW-{row}"})
            else:
                expected_exposed[key] = row_services[row]
            expected_impacted[key] = row_services[row]
        pump_count = 2 if dual_pumps else 1
        for pump in range(1, pump_count + 1):
            key = f"PUMP-{row}-{pump}"
            cooling_and_it = frozenset({f"SVC-COOLING-{row}", f"SVC-IT-{row}"})
            expected_exposed[key] = frozenset() if dual_pumps else cooling_and_it
            expected_impacted[key] = cooling_and_it
        for key in (f"CDU-{row}", f"CRAH-{row}"):
            cooling_and_it = frozenset({f"SVC-COOLING-{row}", f"SVC-IT-{row}"})
            expected_exposed[key] = cooling_and_it
            expected_impacted[key] = cooling_and_it
        expected_exposed[f"RACK-{row}"] = frozenset({f"SVC-IT-{row}"})
        expected_impacted[f"RACK-{row}"] = frozenset({f"SVC-IT-{row}"})

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
        "notes": f"Synthetic AIDC facility {index} (shape {shape})",
        "nodes": nodes,
        "edges": edges,
    }
    return FacilityCorpus(
        facility_id=facility_id,
        name=f"Eval Data Hall {index}",
        segment=SEGMENT,
        shape=shape,
        graph=graph,
        inventory_csv=_inventory_csv(tuple(clean_rows), clean_rows[0].asset_id, "it_rack"),
        clean_rows=tuple(clean_rows),
        corrupt_rows=tuple(corrupt),
        assets=asset_keys,
        expected_exposed=expected_exposed,
        expected_impacted=expected_impacted,
        events=events,
        replay_cases=replay_cases,
    )
