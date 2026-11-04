"""Phase 2 gate: the seed facility's known SPOFs are detected and ranked.

The seed cold-chain facility has, by construction: a single refrigeration
rack serving two cold rooms (with an N+1 compressor pair and a lone
condenser behind it), and a power chain whose 2N sources terminate in a
single-path ATS -> UPS -> PDU feed. The assessment must find exactly the
unprotected assets as SPOFs, rank the power chain highest (it affects all
three services), and carry evidence on every finding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

KNOWN_SPOFS = {"ATS-1", "COND-1", "PDU-1", "RACK-1", "UPS-1"}
KNOWN_REDUNDANT = {"COMP-1", "COMP-2", "GEN-1", "UTILITY-1"}


def _assess_seed(
    client: TestClient,
    seed: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> dict[str, Any]:
    load_seed_via_api(client, seed)
    run = client.post(f"/facilities/{seed['facility_id']}/assessments", json={})
    assert run.status_code == 201, run.text
    return run.json()


def test_seed_known_spofs_are_detected(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    report = _assess_seed(client, seed_facility, load_seed_via_api)

    spofs = {
        f["subject"] for f in report["findings"]
        if f["kind"] == "single_point_of_failure"
    }
    assert KNOWN_SPOFS <= spofs, f"missing known SPOFs: {KNOWN_SPOFS - spofs}"
    assert not spofs & KNOWN_REDUNDANT, (
        f"redundant members flagged as SPOFs: {spofs & KNOWN_REDUNDANT}"
    )


def test_power_chain_spofs_rank_above_rack_spofs(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    report = _assess_seed(client, seed_facility, load_seed_via_api)

    spof_subjects = [
        f["subject"] for f in report["findings"]
        if f["kind"] == "single_point_of_failure"
    ]
    # Power chain hits all three services (power + both refrigeration rooms);
    # the rack and condenser only hit the two refrigeration services.
    top3 = set(spof_subjects[:3])
    assert top3 == {"ATS-1", "PDU-1", "UPS-1"}
    assert spof_subjects.index("RACK-1") > 2
    assert spof_subjects.index("COND-1") > 2


def test_seed_services_score_and_carry_evidence(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    report = _assess_seed(client, seed_facility, load_seed_via_api)

    services = {s["service_id"]: s for s in report["services"]}
    assert set(services) == {"SVC-POWER", "SVC-REFRIG-CR1", "SVC-REFRIG-CR2"}

    # Refrigeration depends on the power chain, so it accumulates more
    # unprotected suppliers and scores at least as high as power itself.
    assert (
        services["SVC-REFRIG-CR1"]["criticality"]["score"]
        >= services["SVC-POWER"]["criticality"]["score"]
    )
    for service in services.values():
        assert service["evidence"], f"service without evidence: {service['service_id']}"
        assert service["criticality"]["drivers"]
        assert service["redundancy"] == "N"  # all seed services sit behind N feeds

    # Spof lists per service are populated for the refrigeration services.
    for service_id in ("SVC-REFRIG-CR1", "SVC-REFRIG-CR2"):
        spof_ids = {e["asset_id"] for e in services[service_id]["single_points_of_failure"]}
        assert {"ATS-1", "COND-1", "PDU-1", "RACK-1", "UPS-1"} <= spof_ids


def test_seed_asset_scores_match_redundancy_structure(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    report = _assess_seed(client, seed_facility, load_seed_via_api)

    scores = {a["node_key"]: a["criticality"] for a in report["assets"]}
    tolerated = {a["node_key"]: a["tolerated_services"] for a in report["assets"]}
    # Power chain mid-section exposes all three services -> capped at 100.
    for key in ("ATS-1", "PDU-1", "UPS-1"):
        assert scores[key]["score"] == 100
    # Rack and its condenser expose the two refrigeration services -> capped.
    assert scores["RACK-1"]["score"] == 100
    assert scores["COND-1"]["score"] == 100
    # Redundant members expose nothing; absorbed impact shows up only in
    # tolerated_services, never in the score (calibration discipline).
    for key in ("GEN-1", "UTILITY-1", "COMP-1", "COMP-2"):
        assert scores[key]["score"] == 0, key
        assert tolerated[key], key


def test_seed_report_carries_no_orphans_and_honest_data_gaps(
    client: TestClient,
    seed_facility: dict[str, Any],
    load_seed_via_api: Callable[[TestClient, dict[str, Any]], None],
) -> None:
    report = _assess_seed(client, seed_facility, load_seed_via_api)

    # Every seed asset reaches at least one service.
    assert not any("does not reach any service" in gap for gap in report["data_gaps"])
    # Topology-only honesty: the window is unset and the gap says so.
    assert report["evidence_window"] is None
    assert any("topology-only" in gap for gap in report["data_gaps"])
    # Missing-redundancy findings exist for all three services.
    missing = [f for f in report["findings"] if f["kind"] == "missing_redundancy"]
    assert {f["subject"] for f in missing} == {
        "SVC-POWER",
        "SVC-REFRIG-CR1",
        "SVC-REFRIG-CR2",
    }
