"""Corpus generator tests: determinism, label integrity, parser round-trip.

The corpus is the ground truth the harness scores against — these tests
check the labels agree with the construction (not with the assessor) and
that the operator-style inputs round-trip through the real parser.
"""

from __future__ import annotations

from datetime import timedelta

from evals.corpus import (
    CORPUS_SEED,
    DEFAULT_FACILITY_COUNT,
    INVENTORY_COLUMNS,
    FacilityCorpus,
    generate_corpus,
)
from evals.harness import FEATURE_WINDOW
from evals.splits import build_folds, check_no_leakage

from uptime_api.ingestion.inventory import parse_inventory


def test_corpus_is_deterministic() -> None:
    """Same seed -> identical corpus (graphs, CSVs, labels, events)."""
    first = generate_corpus(CORPUS_SEED, 5)
    second = generate_corpus(CORPUS_SEED, 5)
    assert first.facilities == second.facilities


def test_corpus_shape_diversity() -> None:
    """40 facilities produce a meaningful mix of shapes, both power configs."""
    corpus = generate_corpus()
    assert len(corpus.facilities) == DEFAULT_FACILITY_COUNT
    assert len(corpus.shape_histogram) >= 10  # varied rooms/racks/compressors
    shapes = list(corpus.shape_histogram)
    assert any("src-2N" in shape for shape in shapes), "both power configs must occur"
    assert any("src-N" in shape for shape in shapes), "both power configs must occur"


def _assert_labels_match_construction(facility: FacilityCorpus) -> None:
    """N+1 pairs absorb their members' faults; lone members do not.

    Read straight from the committed edges — the same evidence a reviewer
    would use — so a label bug cannot hide behind the generator.
    """
    members_by_group: dict[tuple[str, str], list[str]] = {}
    for raw_edge in facility.graph["edges"]:
        assert isinstance(raw_edge, dict)
        if raw_edge.get("redundancy_config") == "N+1":
            group = (str(raw_edge["to_node"]), str(raw_edge["member_group"]))
            members_by_group.setdefault(group, []).append(str(raw_edge["from_node"]))
    for members in members_by_group.values():
        if len(members) < 2:
            continue
        for member in members:
            assert not facility.expected_exposed[member], (
                f"{facility.facility_id}:{member} is in an N+1 pair but labeled exposed"
            )


def test_labels_agree_with_construction_across_corpus() -> None:
    """Every facility in the canonical corpus: labels match its edges."""
    for facility in generate_corpus().facilities:
        _assert_labels_match_construction(facility)


def test_seed_change_keeps_labels_construction_consistent() -> None:
    """A different seed changes shapes, but labels still follow construction."""
    for facility in generate_corpus(CORPUS_SEED + 1, DEFAULT_FACILITY_COUNT).facilities:
        _assert_labels_match_construction(facility)


def test_fault_case_count_matches_assets() -> None:
    corpus = generate_corpus(CORPUS_SEED, 7)
    assert corpus.fault_case_count == sum(len(f.assets) for f in corpus.facilities)


def test_inventory_csv_line_numbers_match_corrupt_rows() -> None:
    """The corrupt rows sit at exactly the line numbers the corpus claims."""
    for facility in generate_corpus(CORPUS_SEED, 6).facilities:
        lines = facility.inventory_csv.splitlines()
        assert lines[0].split(",") == list(INVENTORY_COLUMNS)
        assert len(lines) == 1 + len(facility.clean_rows) + len(facility.corrupt_rows)
        for corrupt in facility.corrupt_rows:
            row = lines[corrupt.row_no - 1]
            if corrupt.expected_reason == "missing_asset_id":
                assert row.startswith(",")
            elif corrupt.expected_reason == "unknown_asset_type":
                assert row.startswith("BOGUS-1,")
            else:
                asset_id = row.split(",")[0]
                assert any(r.asset_id == asset_id for r in facility.clean_rows)


def test_inventory_csv_round_trips_through_parser() -> None:
    """Clean rows become INV-prefixed nodes; corrupt rows flag with the
    exact reasons the corpus records — through the real parser."""
    for facility in generate_corpus(CORPUS_SEED, 6).facilities:
        parsed = parse_inventory(facility.inventory_csv.encode("utf-8"), "inventory.csv")
        assert {node.node_key for node in parsed.nodes} == {
            f"INV-{row.asset_id}" for row in facility.clean_rows
        }
        reasons = [(u.row, u.reason.value) for u in parsed.unresolved]
        expected = [(c.row_no, c.expected_reason) for c in facility.corrupt_rows]
        assert reasons == expected, facility.facility_id


def test_replay_cases_are_self_consistent() -> None:
    """Every replay episode: repair follows the fault, folds are leak-free."""
    corpus = generate_corpus()
    for facility in corpus.facilities:
        assert facility.replay_cases, "every facility ships two replay episodes"
        folds = build_folds(facility.replay_cases, FEATURE_WINDOW)
        check = check_no_leakage(facility.events, folds)
        assert check.ok, check.violations
        for case in facility.replay_cases:
            assert case.repair_time - case.fault_time == timedelta(days=1)
            # The interrupted service must be one the fault actually exposes.
            assert case.interrupted_service in facility.expected_exposed[case.asset_key]
