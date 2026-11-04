"""Synthetic multi-segment corpus with adjudicated, by-construction labels.

Market-sequence order (locked: cold chain -> AIDC -> hospitals):
``cold_chain`` — the Phase 2/3 beachhead corpus (40 facilities, RNG stream
frozen); ``aidc`` — CDU/chiller/UPS/generator chains (Phase 4); ``hospital``
— life-safety branching and isolation logic (Phase 4). Each segment is its
own RNG namespace: resizing or regenerating one never shifts another.

Every facility ships the way the onboarding scanner meets a real site — an
operator-style inventory CSV upload, then a technician-confirmed topology
commit — plus ground truth implied by construction: which single faults
must interrupt which services, which are absorbed by redundancy, and a
synthetic monitoring event log with known fault/repair times for the
leakage-split machinery. Labels come from construction intent, never from
running the assessor (see common.py's adjudication discipline).
"""

from __future__ import annotations

from evals.corpus.aidc import generate_aidc
from evals.corpus.cold_chain import generate_cold_chain
from evals.corpus.common import (
    AIDC_SEED_OFFSET,
    CORPUS_SEED,
    HOSPITAL_SEED_OFFSET,
    INVENTORY_COLUMNS,
    SEGMENT_FACILITY_COUNTS,
    SEGMENTS,
    Corpus,
    CorpusEvent,
    CorruptRow,
    FacilityCorpus,
    InventoryRow,
    ReplayCase,
)
from evals.corpus.hospital import generate_hospital

_SEGMENT_GENERATORS = {
    "cold_chain": generate_cold_chain,
    "aidc": generate_aidc,
    "hospital": generate_hospital,
}

__all__ = [
    "AIDC_SEED_OFFSET",
    "CORPUS_SEED",
    "Corpus",
    "CorpusEvent",
    "CorruptRow",
    "FacilityCorpus",
    "HOSPITAL_SEED_OFFSET",
    "INVENTORY_COLUMNS",
    "InventoryRow",
    "ReplayCase",
    "SEGMENTS",
    "SEGMENT_FACILITY_COUNTS",
    "generate_corpus",
    "generate_segment_corpus",
]


def generate_segment_corpus(
    segment: str, seed: int = CORPUS_SEED, n_facilities: int | None = None
) -> tuple[FacilityCorpus, ...]:
    """One segment's facilities, in deterministic generation order."""
    if segment not in _SEGMENT_GENERATORS:
        raise ValueError(f"unknown segment {segment!r} (known: {', '.join(SEGMENTS)})")
    count = SEGMENT_FACILITY_COUNTS[segment] if n_facilities is None else n_facilities
    generator = _SEGMENT_GENERATORS[segment]
    return tuple(generator(seed, index) for index in range(count))


def generate_corpus(
    seed: int = CORPUS_SEED,
    facilities_per_segment: dict[str, int] | None = None,
) -> Corpus:
    """The full deterministic corpus: every segment in market-sequence order.

    ``facilities_per_segment`` overrides per-segment counts (default: the
    canonical counts); a segment missing from the mapping is skipped, so
    callers can evaluate a single segment.
    """
    counts = facilities_per_segment or SEGMENT_FACILITY_COUNTS
    facilities: list[FacilityCorpus] = []
    for segment in SEGMENTS:
        if counts.get(segment, 0) > 0:
            facilities.extend(generate_segment_corpus(segment, seed, counts[segment]))
    return Corpus(seed=seed, facilities=tuple(facilities))
