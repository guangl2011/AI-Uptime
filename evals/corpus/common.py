"""Shared corpus machinery: record types, constants, and builder helpers.

The segment generators (cold_chain, aidc, hospital) produce facilities the
way the onboarding scanner meets a real site — an operator-style inventory
CSV upload, then a technician-confirmed topology commit — and record the
ground truth implied by construction: which single faults must interrupt
which services, which are absorbed by redundancy, plus a synthetic event log
with known fault/repair times for the leakage-split machinery.

Adjudication discipline (binding on every segment): labels come from the
construction intent (``this compressor is alone in its group, so its failure
must expose the rooms its rack serves``), never from running the assessor.
The harness then checks the assessor against those labels — two independent
derivations of the same physics. A disagreement is a bug in one of them; the
eval's job is to make that visible, not to reconcile it silently.
"""

from __future__ import annotations

import random  # noqa: F401  (re-exported for segment generators' discipline)
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# Canonical corpus seed — the deterministic corpus `make eval` and CI gate on.
CORPUS_SEED = 20261104

# Segment registry: market sequence order (locked: cold chain -> AIDC ->
# hospitals) and the default facility count per segment.
SEGMENTS: tuple[str, ...] = ("cold_chain", "aidc", "hospital")
SEGMENT_FACILITY_COUNTS: dict[str, int] = {
    "cold_chain": 40,
    "aidc": 20,
    "hospital": 20,
}
# Per-segment RNG namespaces: one base offset per segment keeps generators
# independent — regenerating or resizing one segment never shifts another's
# random stream.
AIDC_SEED_OFFSET = 10_000_000
HOSPITAL_SEED_OFFSET = 20_000_000

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
    segment: str  # cold_chain | aidc | hospital (market-sequence order)
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

    def by_segment(self, segment: str) -> tuple[FacilityCorpus, ...]:
        """The facilities of one market segment, in generation order."""
        return tuple(f for f in self.facilities if f.segment == segment)


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


def _inventory_csv(
    clean_rows: tuple[InventoryRow, ...], first_asset_id: str, corrupt_asset_type: str
) -> str:
    """Render the operator-style CSV: clean rows, then three corrupt rows.

    Line numbers are deterministic (no embedded commas or quotes), so the
    corrupt row numbers in the corpus match the parser's reported rows
    exactly. The corrupt rows exercise all three abstention reasons: blank
    asset_id, unknown asset_type, duplicate asset_id. The corrupt rows'
    asset type is the segment's representative type — the rows are
    unresolvable either way, but the CSV should read like the segment's site.
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
    lines.append(f",{corrupt_asset_type},Unlabeled row,backroom,,,,,,,,")  # blank asset_id
    lines.append("BOGUS-1,hyperencabulator,Mystery unit,backroom,,,,,,,,")  # unknown type
    lines.append(
        f"{first_asset_id},{corrupt_asset_type},Duplicate id,backroom,,,,,,,,"
    )  # duplicate
    return "\n".join(lines) + "\n"


def generate_events(
    facility_id: str,
    expected_exposed: dict[str, frozenset[str]],
    replay_prefixes: tuple[str, ...],
) -> tuple[tuple[CorpusEvent, ...], tuple[ReplayCase, ...]]:
    """Monitoring event log with two replay episodes: one branch-scoped fault
    (first asset whose key starts with a segment prefix) and one power-chain
    fault (ATS-1, an SPOF in every segment shape), including post-repair
    telemetry that must never leak into a pre-repair feature window."""
    spof_assets = sorted(key for key, exposed in expected_exposed.items() if exposed)
    branch_scoped = [
        key for key in spof_assets if key.startswith(replay_prefixes)
    ]
    replay_assets = [*branch_scoped[:1], "ATS-1"]

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
