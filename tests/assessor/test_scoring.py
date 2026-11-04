"""Criticality scoring: determinism and the N / N+1 / 2N redundancy math."""

from __future__ import annotations

from typing import Any

from uptime_api.assessor.enums import RiskLevel
from uptime_api.assessor.scoring import (
    EXPOSED_SERVICE_WEIGHT,
    SERVICE_BASE_SCORE,
    UNPROTECTED_SUPPLIER_WEIGHT,
    build_report,
    risk_level_for,
)

# -- Determinism --------------------------------------------------------------


def test_analysis_is_deterministic(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "FAN-1", "SVC-1")
    first = build_report(b.snapshot(nodes, edges))
    second = build_report(b.snapshot(nodes, edges))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_same_snapshot_scored_twice_is_byte_identical(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    snap = b.snapshot(nodes, edges)

    assert build_report(snap).model_dump_json() == build_report(snap).model_dump_json()


# -- Redundancy math (spec gate: N fails on first fault, N+1 tolerates one,
#    2N tolerates full-path loss) --------------------------------------------


def test_n_config_fails_on_first_fault(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    report = build_report(b.snapshot(nodes, edges))

    asset_scores = {a.node_key: a for a in report.assets}
    assert asset_scores["SRC-1"].exposed_services == ["SVC-1"]


def test_n_plus_1_tolerates_one_fault(b: Any) -> None:
    nodes = [
        b.asset("SRC-A", b.asset_type.GENERATOR),
        b.asset("SRC-B", b.asset_type.GENERATOR),
        b.service("SVC-1"),
    ]
    edges = [
        b.edge("SRC-A", "SVC-1", b.redundancy.N_PLUS_1, "feeds"),
        b.edge("SRC-B", "SVC-1", b.redundancy.N_PLUS_1, "feeds"),
    ]
    report = build_report(b.snapshot(nodes, edges))

    asset_scores = {a.node_key: a for a in report.assets}
    for source in ("SRC-A", "SRC-B"):
        assert asset_scores[source].exposed_services == []
        assert asset_scores[source].tolerated_services == ["SVC-1"]

    service_assessment = report.services[0]
    assert service_assessment.exposure.first_fault_tolerated is True
    assert service_assessment.exposure.concurrent_faults_tolerated == 1


def test_2n_tolerates_full_path_loss(b: Any) -> None:
    # Two complete supply paths into one service; losing either is absorbed.
    nodes = [
        b.asset("PATH-A", b.asset_type.GENERATOR),
        b.asset("PATH-B", b.asset_type.GENERATOR),
        b.service("SVC-1"),
    ]
    edges = [
        b.edge("PATH-A", "SVC-1", b.redundancy.N2, "paths"),
        b.edge("PATH-B", "SVC-1", b.redundancy.N2, "paths"),
    ]
    report = build_report(b.snapshot(nodes, edges))

    assert all(a.exposed_services == [] for a in report.assets)
    assert report.services[0].exposure.first_fault_tolerated is True


def test_declared_redundancy_with_one_member_absorbs_nothing(b: Any) -> None:
    # N+1 declared but a single member: redundancy exists on paper only.
    nodes, edges = b.chain(
        "SRC-1", "SVC-1", redundancy=b.redundancy.N_PLUS_1, member_group="feeds"
    )
    report = build_report(b.snapshot(nodes, edges))

    assert report.assets[0].exposed_services == ["SVC-1"]
    assert report.services[0].exposure.first_fault_tolerated is False


# -- Scoring model ------------------------------------------------------------


def test_asset_score_counts_exposed_services_only(b: Any) -> None:
    # Calibration discipline: a fault the redundancy fully absorbs interrupts
    # nothing, so it must score zero — absorbed impact is reported in
    # tolerated_services, never scored as if it were exposure.
    nodes = [
        b.asset("PROTECTED", b.asset_type.GENERATOR),
        b.asset("SPARE", b.asset_type.GENERATOR),
        b.service("SVC-EXPOSED"),
        b.service("SVC-TOLERATED"),
    ]
    edges = [
        b.edge("PROTECTED", "SVC-EXPOSED"),  # unprotected single feed
        b.edge("SPARE", "SVC-TOLERATED", b.redundancy.N_PLUS_1, "feeds"),
        b.edge("PROTECTED", "SVC-TOLERATED", b.redundancy.N_PLUS_1, "feeds"),
    ]
    report = build_report(b.snapshot(nodes, edges))

    scores = {a.node_key: a.criticality.score for a in report.assets}
    # PROTECTED: one exposed service, the other feed absorbed.
    assert scores["PROTECTED"] == EXPOSED_SERVICE_WEIGHT
    assert scores["PROTECTED"] > scores["SPARE"]
    # SPARE: only absorbed impact — zero, but still visible in tolerated.
    assert scores["SPARE"] == 0
    spare = next(a for a in report.assets if a.node_key == "SPARE")
    assert spare.tolerated_services == ["SVC-TOLERATED"]
    assert any("absorbed" in driver for driver in spare.criticality.drivers)


def test_asset_score_is_capped_at_100(b: Any) -> None:
    # One asset exposing many services would exceed 100 raw.
    nodes: list[Any] = [b.asset("HUB")] + [b.service(f"SVC-{i}") for i in range(4)]
    edges = [b.edge("HUB", f"SVC-{i}") for i in range(4)]
    report = build_report(b.snapshot(nodes, edges))

    assert report.assets[0].criticality.score == 100
    assert report.assets[0].criticality.risk_level is RiskLevel.HIGH


def test_service_score_counts_unprotected_suppliers(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "MID-1", "SVC-1")
    report = build_report(b.snapshot(nodes, edges))

    # Both SRC-1 and MID-1 interrupt SVC-1: 30 + 15 * 2.
    score = report.services[0].criticality.score
    assert score == SERVICE_BASE_SCORE + 2 * UNPROTECTED_SUPPLIER_WEIGHT


def test_risk_bands() -> None:
    assert risk_level_for(70) is RiskLevel.HIGH
    assert risk_level_for(30) is RiskLevel.ELEVATED
    assert risk_level_for(29) is RiskLevel.LOW


def test_report_sections_sorted_deterministically(b: Any) -> None:
    nodes, edges = b.chain("B-SRC", "A-SVC-1")
    nodes.append(b.asset("A-SRC"))
    edges.append(b.edge("A-SRC", "A-SVC-1"))
    report = build_report(b.snapshot(nodes, edges))

    assert [a.node_key for a in report.assets] == ["A-SRC", "B-SRC"]
    assert [s.service_id for s in report.services] == ["A-SVC-1"]
