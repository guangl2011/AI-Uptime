"""Synthetic cold-chain corpus with adjudicated, by-construction labels.

Generates plausible cold-chain facilities (cold rooms, refrigeration racks
with N/N+1 compressor groups, condensers, and the power chain) exactly the
way the onboarding scanner meets a real site: an operator-style inventory
CSV upload, then a technician-confirmed topology commit. Alongside each
facility the generator records the ground truth implied by its *construction*
— which single faults must interrupt which services, which are absorbed by
redundancy — plus a synthetic monitoring event log with known fault/repair
times for the leakage-split machinery.

Adjudication discipline: labels come from the construction intent (``this
compressor is alone in its group, so its failure must expose the rooms its
rack serves``), never from running the assessor. The harness then checks the
assessor against those labels — two independent derivations of the same
physics. A disagreement is a bug in one of them; the eval's job is to make
that visible, not to reconcile it silently.

Everything is seeded and deterministic: the same seed produces a
byte-identical corpus, so evaluations are reproducible and comparable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# Canonical corpus seed — the deterministic corpus `make eval` and CI gate on.
CORPUS_SEED = 20261104
DEFAULT_FACILITY_COUNT = 40

SOURCE_SYSTEM = "eval_corpus"
SOURCE_REF_TOPOLOGY = "{facility}/technician_walkthrough"
SOURCE_REF_INVENTORY = "{facility}/inventory.csv#row={row}"

# Replay episodes (see evals.splits): the fault lands in the monitoring
# window and the repair follows one day later.
REPLAY_BASE = datetime(2026, 9, 1, tzinfo=UTC)
REPLAY_FAULT_OFFSETS = (timedelta(days=10), timedelta(days=20))
REPLAY_REPAIR_LAG = timedelta(days=1)

# Attribute columns the operator CSV carries; a row fills the ones that
# apply to its asset type and leaves the rest blank.
INVENTORY_COLUMNS = (
    "asset_id",
    "asset_type",
    "name",
    "location",
    "setpoint_c",
    "volume_m3",
    "rated_kw",
    "rated_kva",
    "battery_minutes",
    "capacity_kw",
    "refrigerant",
    "fuel",
)


@dataclass(frozen=True)
class InventoryRow:
    """One clean inventory row (ground truth for topology-recovery eval)."""

    asset_id: str
    asset_type: str
    name: str
    location: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class CorruptRow:
    """A deliberately unresolvable inventory row the parser must flag."""

    row_no: int  # 1-based CSV line number (row 1 is the header)
    kind: str
    expected_reason: str


@dataclass(frozen=True)
class CorpusEvent:
    """One synthetic monitoring event in a facility's event log."""

    event_id: str
    facility_id: str
    asset_key: str
    occurred_at: datetime
    kind: str  # sensor | alarm | work_order | repair


@dataclass(frozen=True)
class ReplayCase:
    """A known fault/repair episode: the seed of one leakage-checked fold."""

    case_id: str
    facility_id: str
    asset_key: str
    interrupted_service: str  # by construction: the service the fault takes out
    fault_time: datetime
    repair_time: datetime


@dataclass(frozen=True)
class FacilityCorpus:
    """One generated facility: inputs, topology, labels, and event log."""

    facility_id: str
    name: str
    shape: str
    graph: dict[str, object]  # GraphVersionIn-compatible topology payload
    inventory_csv: str
    clean_rows: tuple[InventoryRow, ...]
    corrupt_rows: tuple[CorruptRow, ...]
    assets: tuple[str, ...]
    expected_exposed: dict[str, frozenset[str]]
    expected_impacted: dict[str, frozenset[str]]
    events: tuple[CorpusEvent, ...]
    replay_cases: tuple[ReplayCase, ...]


@dataclass(frozen=True)
class Corpus:
    """The generated corpus: facilities plus the seed that produced them."""

    seed: int
    facilities: tuple[FacilityCorpus, ...]

    @property
    def fault_case_count(self) -> int:
        """Labeled single-fault cases: one per asset per facility."""
        return sum(len(facility.assets) for facility in self.facilities)

    @property
    def shape_histogram(self) -> dict[str, int]:
        shapes: dict[str, int] = {}
        for facility in self.facilities:
            shapes[facility.shape] = shapes.get(facility.shape, 0) + 1
        return dict(sorted(shapes.items()))


def _prov(source_kind: str, ref: str) -> dict[str, str]:
    return {"source_kind": source_kind, "source_system": SOURCE_SYSTEM, "source_ref": ref}


def _node(
    key: str,
    name: str,
    asset_type: str | None,
    location: str | None,
    provenance: dict[str, str],
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    node: dict[str, object] = {
        "node_key": key,
        "name": name,
        "node_type": "service" if asset_type is None else "asset",
        "attributes": attributes or {},
        "provenance": provenance,
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
    provenance: dict[str, str],
    redundancy: str | None = None,
    member_group: str | None = None,
) -> dict[str, object]:
    edge: dict[str, object] = {
        "from_node": from_key,
        "to_node": to_key,
        "failure_mode": failure_mode,
        "provenance": provenance,
    }
    if redundancy is not None:
        edge["redundancy_config"] = redundancy
    if member_group is not None:
        edge["member_group"] = member_group
    return edge


def _inventory_csv(clean_rows: tuple[InventoryRow, ...], first_asset_id: str) -> str:
    """Render the operator-style CSV: clean rows, then three corrupt rows.

    Line numbers are deterministic (no embedded commas or quotes), so the
    corrupt row numbers in the corpus match the parser's reported rows
    exactly. The corrupt rows exercise all three abstention reasons: blank
    asset_id, unknown asset_type, duplicate asset_id.
    """
    lines = [",".join(INVENTORY_COLUMNS)]
    for row in clean_rows:
        values = [
            row.asset_id,
            row.asset_type,
            row.name,
            row.location,
            *[
                row.attributes.get(column, "")
                for column in INVENTORY_COLUMNS[4:]
            ],
        ]
        lines.append(",".join(values))
    lines.append(",cold_room,Unlabeled row,backroom,,,,,,,,")  # blank asset_id
    lines.append("BOGUS-1,hyperencabulator,Mystery unit,backroom,,,,,,,,")  # unknown type
    lines.append(f"{first_asset_id},cold_room,Duplicate id,freezer wing,,,,,,,,")  # duplicate
    return "\n".join(lines) + "\n"


def _generate_facility(seed: int, index: int) -> FacilityCorpus:
    """One facility: shape drawn from the seed, labels from construction."""
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

    events, replay_cases = _generate_events(facility_id, expected_exposed)

    graph: dict[str, object] = {
        "author": "eval_corpus",
        "notes": f"Synthetic cold-chain facility {index} (shape {shape})",
        "nodes": nodes,
        "edges": edges,
    }
    return FacilityCorpus(
        facility_id=facility_id,
        name=f"Eval Cold Storage {index}",
        shape=shape,
        graph=graph,
        inventory_csv=_inventory_csv(tuple(clean_rows), clean_rows[0].asset_id),
        clean_rows=tuple(clean_rows),
        corrupt_rows=tuple(corrupt),
        assets=asset_keys,
        expected_exposed=expected_exposed,
        expected_impacted=expected_impacted,
        events=events,
        replay_cases=replay_cases,
    )


def _generate_events(
    facility_id: str,
    expected_exposed: dict[str, frozenset[str]],
) -> tuple[tuple[CorpusEvent, ...], tuple[ReplayCase, ...]]:
    """Monitoring event log with two replay episodes (one rack-scoped fault,
    one power-chain fault), including post-repair telemetry that must never
    leak into a pre-repair feature window."""
    spof_assets = sorted(key for key, exposed in expected_exposed.items() if exposed)
    rack_scoped = [key for key in spof_assets if key.startswith(("RACK", "COMP", "COND"))]
    replay_assets = [*rack_scoped[:1], "ATS-1"]

    events: list[CorpusEvent] = []
    cases: list[ReplayCase] = []
    for slot, asset_key in enumerate(replay_assets):
        fault_time = REPLAY_BASE + REPLAY_FAULT_OFFSETS[slot] + timedelta(hours=slot * 3)
        repair_time = fault_time + REPLAY_REPAIR_LAG
        case_id = f"{facility_id}:{asset_key}"

        # Timestamps first (data), then construction — no loop closure.
        stamps: list[tuple[str, datetime]] = [
            ("sensor", fault_time - timedelta(hours=8 - hour)) for hour in range(8)
        ]
        stamps += [
            ("alarm", fault_time - timedelta(hours=6)),
            ("alarm", fault_time - timedelta(hours=2)),
            ("work_order", fault_time),
            # Label window: the interruption and its repair.
            ("repair", repair_time),
        ]
        # Post-repair telemetry — must stay out of every feature window.
        stamps += [
            ("sensor", repair_time + timedelta(hours=hour + 1)) for hour in range(8)
        ]
        for seq, (kind, at) in enumerate(stamps, start=1):
            events.append(
                CorpusEvent(
                    event_id=f"{case_id}#{seq}",
                    facility_id=facility_id,
                    asset_key=asset_key,
                    occurred_at=at,
                    kind=kind,
                )
            )
        interrupted = sorted(expected_exposed[asset_key])[0]
        cases.append(
            ReplayCase(
                case_id=case_id,
                facility_id=facility_id,
                asset_key=asset_key,
                interrupted_service=interrupted,
                fault_time=fault_time,
                repair_time=repair_time,
            )
        )
    return tuple(events), tuple(cases)


def generate_corpus(seed: int = CORPUS_SEED, n_facilities: int = DEFAULT_FACILITY_COUNT) -> Corpus:
    """Deterministic synthetic cold-chain corpus with adjudicated labels."""
    return Corpus(
        seed=seed,
        facilities=tuple(_generate_facility(seed, index) for index in range(n_facilities)),
    )
