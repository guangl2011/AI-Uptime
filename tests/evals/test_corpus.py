"""Corpus generator tests: determinism, label integrity, parser round-trip.

The corpus is the ground truth the harness scores against — these tests
check the labels agree with the construction (not with the assessor) and
that the operator-style inputs round-trip through the real parser. The
Phase 4 corpus spans three market segments (cold chain -> AIDC -> hospitals);
the construction-integrity checks run across all of them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import cast

from evals.corpus import (
    CORPUS_SEED,
    INVENTORY_COLUMNS,
    SEGMENT_FACILITY_COUNTS,
    SEGMENTS,
    FacilityCorpus,
    generate_corpus,
    generate_segment_corpus,
)
from evals.harness import FEATURE_WINDOW
from evals.splits import build_folds, check_no_leakage

from uptime_api.ingestion.inventory import parse_inventory

NodeList = list[dict[str, object]]
EdgeList = list[dict[str, object]]


def _graph_nodes_and_edges(facility: FacilityCorpus) -> tuple[NodeList, EdgeList]:
    return (
        cast(NodeList, facility.graph["nodes"]),
        cast(EdgeList, facility.graph["edges"]),
    )


def _asset_load_classes(facility: FacilityCorpus) -> dict[str, object]:
    nodes, _ = _graph_nodes_and_edges(facility)
    return {
        str(node["node_key"]): (node["attributes"] or {}).get("load_class")
        for node in nodes
        if node["node_type"] == "asset"
    }


def test_corpus_is_deterministic() -> None:
    """Same seed -> identical corpus (graphs, CSVs, labels, events)."""
    counts = {"cold_chain": 5, "aidc": 5, "hospital": 5}
    first = generate_corpus(CORPUS_SEED, counts)
    second = generate_corpus(CORPUS_SEED, counts)
    assert first.facilities == second.facilities


def test_segment_generation_is_independent_and_deterministic() -> None:
    """Per-segment generation is stable, and resizing one segment never
    shifts another's random stream (independent RNG namespaces)."""
    five_aidc = generate_segment_corpus("aidc", CORPUS_SEED, 5)
    again_five_aidc = generate_segment_corpus("aidc", CORPUS_SEED, 5)
    assert five_aidc == again_five_aidc
    six_aidc = generate_segment_corpus("aidc", CORPUS_SEED, 6)
    assert [f.facility_id for f in five_aidc] == [f.facility_id for f in six_aidc[:5]]
    assert five_aidc[0].graph == six_aidc[0].graph


def test_canonical_corpus_spans_all_segments_in_market_order() -> None:
    """The canonical corpus: every segment present, in market-sequence order,
    with the locked per-segment facility counts."""
    corpus = generate_corpus()
    order = [facility.segment for facility in corpus.facilities]
    assert order == sorted(order, key=SEGMENTS.index)
    for segment in SEGMENTS:
        facilities = corpus.by_segment(segment)
        assert len(facilities) == SEGMENT_FACILITY_COUNTS[segment], segment
        assert all(f.facility_id.startswith(f"eval_{segment[:4]}") for f in facilities)


def test_corpus_shape_diversity_per_segment() -> None:
    """Each segment produces a meaningful shape mix covering both power
    configurations — the corpus must exercise N, N+1, and 2N everywhere."""
    for segment in SEGMENTS:
        shapes = {facility.shape for facility in generate_segment_corpus(segment)}
        assert len(shapes) >= 3, segment
        if segment in ("cold_chain", "aidc"):
            assert any("src-2N" in shape for shape in shapes), segment
            assert any("src-N" in shape for shape in shapes), segment
        if segment == "hospital":
            assert any("clean" in shape for shape in shapes), segment
            assert any("mixed" in shape for shape in shapes), segment


def _assert_labels_match_construction(facility: FacilityCorpus) -> None:
    """N+1 pairs absorb their members' faults; lone members do not.

    Read straight from the committed edges — the same evidence a reviewer
    would use — so a label bug cannot hide behind the generator.
    """
    _, edges = _graph_nodes_and_edges(facility)
    members_by_group: dict[tuple[str, str], list[str]] = {}
    for raw_edge in edges:
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
    for facility in generate_corpus(CORPUS_SEED + 1).facilities:
        _assert_labels_match_construction(facility)


def test_aidc_construction_grammar() -> None:
    """AIDC facilities: CDU/chiller/UPS/generator chains with per-row cooling
    trains; the services the spec's canonical AIDC chains name."""
    aidc_types = {"cdu", "crah", "it_rack", "chiller", "pump"}
    for facility in generate_segment_corpus("aidc", CORPUS_SEED, 20):
        asset_types = {str(row.asset_type) for row in facility.clean_rows}
        assert asset_types >= aidc_types, facility.facility_id
        nodes, edges = _graph_nodes_and_edges(facility)
        services = {
            str(node["node_key"]) for node in nodes if node["node_type"] == "service"
        }
        assert "SVC-POWER" in services
        assert any(key.startswith("SVC-IT-") for key in services)
        assert any(key.startswith("SVC-CHW-") for key in services)
        # One source edge per shape kind: the 2N pair or the lone utility.
        source_edges = [edge for edge in edges if edge["to_node"] == "ATS-1"]
        assert len(source_edges) == (2 if "src-2N" in facility.shape else 1)
        if len(source_edges) == 2:
            assert all(edge["redundancy_config"] == "2N" for edge in source_edges)


def test_hospital_isolation_construction() -> None:
    """Hospital facilities: life-safety services marked; mixed shapes wire
    med-gas from the general panel (the red-line violation the assessor must
    flag), clean shapes feed it from the life-safety panel."""
    for facility in generate_segment_corpus("hospital", CORPUS_SEED, 20):
        load_class = _asset_load_classes(facility)
        _, edges = _graph_nodes_and_edges(facility)
        medgas_feeders = [
            str(edge["from_node"])
            for edge in edges
            if str(edge["to_node"]).startswith("MEDGAS-")
        ]
        if "mixed" in facility.shape:
            assert set(medgas_feeders) == {"GENERAL-PANEL"}, facility.facility_id
            assert load_class["GENERAL-PANEL"] == "general"
        else:
            assert set(medgas_feeders) == {"LS-PANEL"}, facility.facility_id
        # Single-fault semantics: the duplex med-gas manifold is an N+1 pair,
        # so NO single fault anywhere exposes med-gas — the violation lives in
        # the feeder wiring above and the assessor's isolation finding, not in
        # the exposure labels. OR air behaves the same way; branch power is
        # fed by a plain N edge, so panel faults expose it in both shapes.
        assert all(
            "SVC-MEDGAS" not in exposed
            for exposed in facility.expected_exposed.values()
        )
        assert all(
            "SVC-OR-AIR" not in exposed
            for exposed in facility.expected_exposed.values()
        )
        assert facility.expected_exposed["ATS-1"] == frozenset({"SVC-LS-POWER"})
        assert facility.expected_impacted["ATS-1"] >= {"SVC-LS-POWER", "SVC-OR-AIR"}
        nodes, _ = _graph_nodes_and_edges(facility)
        ls_services = {
            str(node["node_key"])
            for node in nodes
            if node["node_type"] == "service"
            and (node["attributes"] or {}).get("life_safety")
        }
        assert {"SVC-LS-POWER", "SVC-OR-AIR", "SVC-MEDGAS"} <= ls_services


def test_fault_case_count_matches_assets() -> None:
    corpus = generate_corpus(CORPUS_SEED, {"cold_chain": 7})
    assert corpus.fault_case_count == sum(len(f.assets) for f in corpus.facilities)


def test_inventory_csv_line_numbers_match_corrupt_rows() -> None:
    """The corrupt rows sit at exactly the line numbers the corpus claims."""
    counts = {"cold_chain": 3, "aidc": 3, "hospital": 3}
    for facility in generate_corpus(CORPUS_SEED, counts).facilities:
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
    counts = {"cold_chain": 3, "aidc": 3, "hospital": 3}
    for facility in generate_corpus(CORPUS_SEED, counts).facilities:
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
