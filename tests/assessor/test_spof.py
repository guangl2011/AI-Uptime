"""SPOF detection and finding rules: R1 spof, R2 missing redundancy, R3 unserved."""

from __future__ import annotations

from typing import Any

from uptime_api.assessor.enums import FindingKind
from uptime_api.assessor.scoring import build_report

# -- R1: single points of failure ---------------------------------------------


def test_single_unprotected_supplier_is_a_spof(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    report = build_report(b.snapshot(nodes, edges))

    spofs = [f for f in report.findings if f.kind is FindingKind.SINGLE_POINT_OF_FAILURE]
    assert [f.subject for f in spofs] == ["SRC-1"]
    assert spofs[0].affected_services == ["SVC-1"]


def test_redundant_suppliers_are_not_spofs(b: Any) -> None:
    nodes = [
        b.asset("SRC-A", b.asset_type.GENERATOR),
        b.asset("SRC-B", b.asset_type.GENERATOR),
        b.service("SVC-1"),
    ]
    edges = [
        b.edge("SRC-A", "SVC-1", b.redundancy.N2, "paths"),
        b.edge("SRC-B", "SVC-1", b.redundancy.N2, "paths"),
    ]
    report = build_report(b.snapshot(nodes, edges))

    assert not any(f.kind is FindingKind.SINGLE_POINT_OF_FAILURE for f in report.findings)


def test_spof_exposes_every_service_downstream(b: Any) -> None:
    # One rack feeding two services: a single rack failure hits both.
    nodes, edges = b.chain("RACK-1", "SVC-CR1")
    nodes.append(b.service("SVC-CR2"))
    edges.append(b.edge("RACK-1", "SVC-CR2"))
    report = build_report(b.snapshot(nodes, edges))

    rack_finding = next(
        f for f in report.findings if f.subject == "RACK-1"
    )
    assert rack_finding.kind is FindingKind.SINGLE_POINT_OF_FAILURE
    assert rack_finding.affected_services == ["SVC-CR1", "SVC-CR2"]


# -- R2: missing redundancy -----------------------------------------------------


def test_service_without_redundant_supply_gets_missing_redundancy_finding(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    report = build_report(b.snapshot(nodes, edges))

    missing = [f for f in report.findings if f.kind is FindingKind.MISSING_REDUNDANCY]
    assert [f.subject for f in missing] == ["SVC-1"]
    assert "SRC-1" in missing[0].detail


def test_service_with_redundant_supply_gets_no_missing_redundancy_finding(b: Any) -> None:
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

    assert not any(f.kind is FindingKind.MISSING_REDUNDANCY for f in report.findings)


# -- R3: unserved services ------------------------------------------------------


def test_unserved_service_is_flagged(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    nodes.append(b.service("SVC-ORPHAN"))
    report = build_report(b.snapshot(nodes, edges))

    unserved = [f for f in report.findings if f.kind is FindingKind.UNSERVED_SERVICE]
    assert [f.subject for f in unserved] == ["SVC-ORPHAN"]
    assert "service SVC-ORPHAN has no incoming supply edges" in report.data_gaps


def test_orphan_asset_lands_in_data_gaps_not_findings(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    nodes.append(b.asset("STRANDED", b.asset_type.CONDENSER))
    report = build_report(b.snapshot(nodes, edges))

    assert not any(f.subject == "STRANDED" for f in report.findings)
    assert any("STRANDED" in gap for gap in report.data_gaps)


# -- Ranking --------------------------------------------------------------------


def test_findings_ranked_most_services_affected_first(b: Any) -> None:
    # HUB exposes two services; SRC-1 exposes one. HUB must rank first.
    nodes, edges = b.chain("SRC-1", "SVC-1")
    nodes.extend([b.asset("HUB"), b.service("SVC-2")])
    edges.extend([b.edge("HUB", "SVC-1"), b.edge("HUB", "SVC-2")])
    report = build_report(b.snapshot(nodes, edges))

    spofs = [f for f in report.findings if f.kind is FindingKind.SINGLE_POINT_OF_FAILURE]
    assert [f.subject for f in spofs] == ["HUB", "SRC-1"]


def test_every_finding_carries_evidence(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    report = build_report(b.snapshot(nodes, edges))

    assert report.findings, "test graph should produce findings"
    for finding in report.findings:
        assert finding.evidence, f"finding without evidence: {finding.title}"


def test_findings_order_is_deterministic(b: Any) -> None:
    nodes, edges = b.chain("SRC-1", "SVC-1")
    nodes.append(b.service("SVC-2"))
    edges.append(b.edge("SRC-1", "SVC-2"))

    first = build_report(b.snapshot(nodes, edges)).model_dump(mode="json")
    second = build_report(b.snapshot(nodes, edges)).model_dump(mode="json")
    assert [f["subject"] for f in first["findings"]] == [
        f["subject"] for f in second["findings"]
    ]
