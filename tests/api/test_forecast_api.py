"""Assessment API forecast surface: flag-off absence, flag-on shape, abstention.

The unit suites in tests/forecasting/ cover the models; these pin what the
HTTP surface actually serializes and stores — including the abstention
fields, which must survive the persistence round-trip intact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uptime_api.app import create_app

FACILITY_ID = "forecast_api_facility"

_PROVENANCE = {
    "source_kind": "technician_entry",
    "source_system": "test",
    "source_ref": "walkthrough",
}


def _tmp_db() -> str:
    return f"sqlite:///{Path(tempfile.mkdtemp()) / 'forecast.db'}"


def _graph_payload(edge: dict[str, object]) -> dict[str, object]:
    return {
        "author": "test",
        "notes": None,
        "nodes": [
            {
                "node_key": "SRC-1",
                "node_type": "asset",
                "asset_type": "utility_feed",
                "name": "Source 1",
                "attributes": {},
                "provenance": {
                    "source_kind": "inventory_import",
                    "source_system": "test",
                    "source_ref": "inv.csv#row=2",
                },
            },
            {
                "node_key": "SVC-1",
                "node_type": "service",
                "name": "Service 1",
                "attributes": {},
                "provenance": _PROVENANCE,
            },
        ],
        "edges": [edge],
    }


def _create_facility(client: TestClient, edge: dict[str, object]) -> None:
    assert (
        client.post(
            "/facilities", json={"id": FACILITY_ID, "name": "Forecast API Facility"}
        ).status_code
        == 201
    )
    committed = client.post(f"/facilities/{FACILITY_ID}/graph/versions", json=_graph_payload(edge))
    assert committed.status_code == 201, committed.text


def _run_assessment(client: TestClient) -> dict[str, object]:
    run = client.post(f"/facilities/{FACILITY_ID}/assessments", json={"author": "tester"})
    assert run.status_code == 201, run.text
    report: dict[str, object] = run.json()
    return report


def test_no_forecasts_when_the_flag_is_default_off(client: TestClient) -> None:
    _create_facility(client, {"from_node": "SRC-1", "to_node": "SVC-1", "provenance": _PROVENANCE})
    report = _run_assessment(client)
    assert report["forecasts"] == []


def test_flag_on_emits_calibrated_forecasts_with_bounds_and_basis() -> None:
    with TestClient(create_app(_tmp_db(), enable_forecasting=True)) as client:
        _create_facility(
            client, {"from_node": "SRC-1", "to_node": "SVC-1", "provenance": _PROVENANCE}
        )
        report = _run_assessment(client)
    forecasts = report["forecasts"]
    assert isinstance(forecasts, list) and forecasts, "flag on must emit forecasts"
    forecast = forecasts[0]
    assert isinstance(forecast, dict)
    assert forecast["asset_key"] == "SRC-1"
    assert forecast["status"] == "forecast"
    assert forecast["abstention_reason"] is None
    assert "synthetic-corpus calibration only" in forecast["basis"]
    assert forecast["redundancy_class"] == "N"
    assert forecast["horizon_days"] == 90
    p, lower, upper = forecast["p_fault"], forecast["lower"], forecast["upper"]
    assert 0.0 <= lower <= p <= upper <= 1.0


def test_conflicting_topology_abstains_with_reason_and_fallback() -> None:
    with TestClient(create_app(_tmp_db(), enable_forecasting=True)) as client:
        _create_facility(
            client,
            {
                "from_node": "SRC-1",
                "to_node": "SVC-1",
                "redundancy_config": "N+1",
                "member_group": "solo-group",
                "provenance": _PROVENANCE,
            },
        )
        report = _run_assessment(client)
    forecast = report["forecasts"][0]
    assert forecast["status"] == "abstained"
    assert "conflicting evidence" in forecast["abstention_reason"]
    # The abstained value is the flat-prior fallback for the declared class
    # (0.12), with full-width bounds — never a confident guess.
    assert forecast["p_fault"] == pytest.approx(0.12)
    assert forecast["lower"] == 0.0 and forecast["upper"] == 1.0
    assert "abstained" in forecast["basis"]


def test_stored_assessment_round_trips_the_forecast_fields() -> None:
    with TestClient(create_app(_tmp_db(), enable_forecasting=True)) as client:
        _create_facility(
            client, {"from_node": "SRC-1", "to_node": "SVC-1", "provenance": _PROVENANCE}
        )
        report = _run_assessment(client)
        fetched = client.get(f"/facilities/{FACILITY_ID}/assessments/1")
    assert fetched.status_code == 200
    assert fetched.json() == report
