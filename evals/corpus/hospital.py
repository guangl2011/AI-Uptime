"""Synthetic hospital facilities: life-safety branching and isolation (Phase 4).

Construction grammar (every shape):
- Two source branches. The **life-safety branch**: the 2N source pair
  (utility + generator, group ``power_sources``) feeds the ATS, which feeds
  the life-safety panel — marked ``load_class=life_safety`` — which supplies
  the life-safety services (branch power, OR ventilation, medical gas). The
  **general branch**: the utility feed (normal power only — the general path
  does not reach the emergency bus) feeds UPS -> PDU -> the general
  distribution panel (``load_class=general``), which supplies imaging and
  general HVAC.
- Redundant pairs where hospital practice puts them: standby OR air
  handlers, duplex med-gas manifolds, and (in most shapes) paired general
  air handlers — all N+1 groups absorbing any one member's fault.
- Isolation invariant: a life-safety service must draw **only** through the
  life-safety branch. The ``mixed`` shape wires the (duplex) med-gas
  manifolds from the general distribution panel instead — a red-line
  violation the assessor must report (``life_safety_isolation`` finding),
  not absorb.

Labels encode the two-branch topology under single-fault semantics: the
OR air handlers and med-gas manifolds are N+1 pairs, so a single fault
anywhere upstream of them is absorbed and exposes nothing — med-gas and
OR air never appear in any exposed set, in either shape. The generator's
fault is absorbed by the 2N pair; the utility feed's fault exposes only
the general branch's unprotected tail (imaging, plus HVAC when its air
handlers are unpaired); the isolation violation is visible in the feeder
wiring and asserted on the assessor's finding, not in exposure labels.
"""

from __future__ import annotations

import random

from evals.corpus.common import (
    HOSPITAL_SEED_OFFSET,
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

REPLAY_PREFIXES = ("LS-PANEL", "OR-AH", "MEDGAS", "IMG", "AH")
SEGMENT = "hospital"


def generate_hospital(seed: int, index: int) -> FacilityCorpus:
    """One hospital facility: shape drawn from the seed, labels from construction."""
    rng = random.Random(seed * 1_000_003 + HOSPITAL_SEED_OFFSET + index)
    facility_id = f"eval_hospital_{index:03d}"
    walk_ref = _prov("technician_entry", SOURCE_REF_TOPOLOGY.format(facility=facility_id))

    mixed_medgas = rng.random() < 0.3
    paired_general_air = rng.random() < 0.4
    shape = f"hospital/ls-branch-{'mixed' if mixed_medgas else 'clean'}"
    shape += f"/general-air-{'n+1' if paired_general_air else 'n'}"

    ls_services = frozenset({"SVC-LS-POWER", "SVC-OR-AIR", "SVC-MEDGAS"})
    general_services = frozenset({"SVC-IMAGING", "SVC-GENERAL-HVAC"})
    all_services = ls_services | general_services
    # In the mixed shape the med-gas service is fed by the general branch.
    ls_services_served = ls_services - frozenset({"SVC-MEDGAS"}) if mixed_medgas else ls_services
    general_services_served = general_services | (
        frozenset({"SVC-MEDGAS"}) if mixed_medgas else frozenset()
    )

    nodes: list[dict[str, object]] = []
    clean_rows: list[InventoryRow] = []

    def add_asset(
        key: str,
        name: str,
        asset_type: str,
        location: str,
        attributes: dict[str, str],
        load_class: str | None = None,
        life_safety: bool | None = None,
    ) -> None:
        row_no = 2 + len(clean_rows)  # header is row 1
        prov = _prov(
            "inventory_import",
            SOURCE_REF_INVENTORY.format(facility=facility_id, row=row_no),
        )
        node_attrs: dict[str, object] = dict(attributes)
        if load_class is not None:
            node_attrs["load_class"] = load_class
        if life_safety is not None:
            node_attrs["life_safety"] = life_safety
        nodes.append(_node(key, name, asset_type, location, prov, node_attrs))
        # Inventory CSVs carry nameplate attributes only; branch classes are
        # technician-entered topology annotations (graph commit), not columns.
        clean_rows.append(
            InventoryRow(
                asset_id=key,
                asset_type=asset_type,
                name=name,
                location=location,
                attributes=dict(attributes),
            )
        )

    # Source pair and the two branch panels.
    add_asset("UTILITY-1", "Utility feed", "utility_feed", "site entrance", {})
    add_asset("GEN-1", "Standby generator", "generator", "electrical room", {"fuel": "diesel"})
    add_asset("ATS-1", "Automatic transfer switch", "ats", "electrical room", {})
    add_asset(
        "LS-PANEL",
        "Life-safety distribution panel",
        "distribution_panel",
        "life-safety branch",
        {},
        load_class="life_safety",
    )
    add_asset("UPS-1", "UPS module 1", "ups", "electrical room", {"battery_minutes": "15"})
    add_asset("PDU-1", "General PDU", "pdu", "electrical room", {})
    add_asset(
        "GENERAL-PANEL",
        "General distribution panel",
        "distribution_panel",
        "general branch",
        {},
        load_class="general",
    )
    # Life-safety branch equipment.
    add_asset(
        "OR-AH-1",
        "OR air handler 1 (duty)",
        "air_handler",
        "OR suite",
        {},
        load_class="life_safety",
    )
    add_asset(
        "OR-AH-2",
        "OR air handler 2 (standby)",
        "air_handler",
        "OR suite",
        {},
        load_class="life_safety",
    )
    add_asset(
        "MEDGAS-1",
        "Med-gas manifold 1 (duty)",
        "med_gas_manifold",
        "med-gas room",
        {},
        load_class="life_safety",
    )
    add_asset(
        "MEDGAS-2",
        "Med-gas manifold 2 (standby)",
        "med_gas_manifold",
        "med-gas room",
        {},
        load_class="life_safety",
    )
    # General branch equipment.
    add_asset("IMG-1", "Imaging unit 1", "medical_imaging", "imaging suite", {})
    add_asset(
        "AH-2",
        "General air handler 2",
        "air_handler",
        "general wings",
        {},
        load_class="general",
    )
    if paired_general_air:
        add_asset(
            "AH-3",
            "General air handler 3 (standby)",
            "air_handler",
            "general wings",
            {},
            load_class="general",
        )
    nodes.append(
        _node(
            "SVC-LS-POWER", "Life-safety branch power", None, None, walk_ref,
            {"life_safety": True},
        )
    )
    nodes.append(
        _node("SVC-OR-AIR", "OR suite ventilation", None, None, walk_ref, {"life_safety": True})
    )
    nodes.append(
        _node("SVC-MEDGAS", "Medical gas delivery", None, None, walk_ref, {"life_safety": True})
    )
    nodes.append(_node("SVC-IMAGING", "Imaging suite power", None, None, walk_ref))
    nodes.append(_node("SVC-GENERAL-HVAC", "General HVAC", None, None, walk_ref))

    edges: list[dict[str, object]] = []
    edges.append(_edge("UTILITY-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
    edges.append(_edge("GEN-1", "ATS-1", "power_loss", walk_ref, "2N", "power_sources"))
    edges.append(_edge("ATS-1", "LS-PANEL", "power_loss", walk_ref, "N"))
    edges.append(_edge("LS-PANEL", "SVC-LS-POWER", "power_loss", walk_ref, "N"))
    edges.append(_edge("LS-PANEL", "OR-AH-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("LS-PANEL", "OR-AH-2", "power_loss", walk_ref, "N"))
    edges.append(
        _edge("OR-AH-1", "SVC-OR-AIR", "mechanical_failure", walk_ref, "N+1", "or_air_handlers")
    )
    edges.append(
        _edge("OR-AH-2", "SVC-OR-AIR", "mechanical_failure", walk_ref, "N+1", "or_air_handlers")
    )
    # The isolation pivot: med-gas fed from the life-safety panel (clean) or
    # from the general panel (mixed — the red-line violation).
    medgas_feeder = "GENERAL-PANEL" if mixed_medgas else "LS-PANEL"
    edges.append(_edge(medgas_feeder, "MEDGAS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge(medgas_feeder, "MEDGAS-2", "power_loss", walk_ref, "N"))
    edges.append(
        _edge("MEDGAS-1", "SVC-MEDGAS", "mechanical_failure", walk_ref, "N+1", "med_gas_manifolds")
    )
    edges.append(
        _edge("MEDGAS-2", "SVC-MEDGAS", "mechanical_failure", walk_ref, "N+1", "med_gas_manifolds")
    )
    # General branch: normal power only.
    edges.append(_edge("UTILITY-1", "UPS-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("UPS-1", "PDU-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("PDU-1", "GENERAL-PANEL", "power_loss", walk_ref, "N"))
    edges.append(_edge("GENERAL-PANEL", "IMG-1", "power_loss", walk_ref, "N"))
    edges.append(_edge("IMG-1", "SVC-IMAGING", "power_loss", walk_ref, "N"))
    edges.append(_edge("GENERAL-PANEL", "AH-2", "power_loss", walk_ref, "N"))
    edges.append(
        _edge(
            "AH-2",
            "SVC-GENERAL-HVAC",
            "mechanical_failure",
            walk_ref,
            "N+1" if paired_general_air else "N",
            "general_air_handlers" if paired_general_air else None,
        )
    )
    if paired_general_air:
        edges.append(_edge("GENERAL-PANEL", "AH-3", "power_loss", walk_ref, "N"))
        edges.append(
            _edge(
                "AH-3",
                "SVC-GENERAL-HVAC",
                "mechanical_failure",
                walk_ref,
                "N+1",
                "general_air_handlers",
            )
        )

    # Adjudicated labels, straight from the construction (NOT the assessor).
    # Single-fault semantics: the OR air handlers and the med-gas manifolds
    # are N+1 pairs, so a single fault anywhere upstream of them (utility,
    # ATS, either panel) is absorbed at those groups. Exposed sets therefore
    # name only services reached without passing an absorbing pair; the
    # isolation violation itself is asserted by the assessor finding test.
    general_exposed = frozenset(
        {"SVC-IMAGING"} | (set() if paired_general_air else {"SVC-GENERAL-HVAC"})
    )
    expected_exposed: dict[str, frozenset[str]] = {}
    expected_impacted: dict[str, frozenset[str]] = {}
    expected_exposed["UTILITY-1"] = general_exposed
    expected_impacted["UTILITY-1"] = all_services
    expected_exposed["GEN-1"] = frozenset()  # 2N pair absorbs; life-safety branch rides through
    expected_impacted["GEN-1"] = ls_services_served
    for key in ("ATS-1", "LS-PANEL"):
        expected_exposed[key] = frozenset({"SVC-LS-POWER"})  # OR-AIR absorbed by the N+1 handlers
        expected_impacted[key] = ls_services_served
    for key in ("UPS-1", "PDU-1", "GENERAL-PANEL"):
        expected_exposed[key] = general_exposed  # MEDGAS absorbed by the duplex manifold
        expected_impacted[key] = general_services_served
    for key in ("OR-AH-1", "OR-AH-2"):
        expected_exposed[key] = frozenset()  # N+1 partner absorbs
        expected_impacted[key] = frozenset({"SVC-OR-AIR"})
    for key in ("MEDGAS-1", "MEDGAS-2"):
        expected_exposed[key] = frozenset()  # N+1 partner absorbs
        expected_impacted[key] = frozenset({"SVC-MEDGAS"})
    expected_exposed["IMG-1"] = frozenset({"SVC-IMAGING"})
    expected_impacted["IMG-1"] = frozenset({"SVC-IMAGING"})
    ah_exposed = frozenset() if paired_general_air else frozenset({"SVC-GENERAL-HVAC"})
    for key in ("AH-2", "AH-3"):
        if key == "AH-3" and not paired_general_air:
            continue
        expected_exposed[key] = ah_exposed
        expected_impacted[key] = frozenset({"SVC-GENERAL-HVAC"})

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
        "notes": f"Synthetic hospital facility {index} (shape {shape})",
        "nodes": nodes,
        "edges": edges,
    }
    return FacilityCorpus(
        facility_id=facility_id,
        name=f"Eval Hospital {index}",
        segment=SEGMENT,
        shape=shape,
        graph=graph,
        inventory_csv=_inventory_csv(tuple(clean_rows), clean_rows[0].asset_id, "air_handler"),
        clean_rows=tuple(clean_rows),
        corrupt_rows=tuple(corrupt),
        assets=asset_keys,
        expected_exposed=expected_exposed,
        expected_impacted=expected_impacted,
        events=events,
        replay_cases=replay_cases,
    )
