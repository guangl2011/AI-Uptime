"""Named fault-injection scenarios: hand-adjudicated known-answer tests.

Each scenario is a committed, reviewable JSON file under ``evals/scenarios/``
— a small cold-chain topology, one injected single fault (the asset key), and
the adjudicated expectation (which services the fault interrupts, which
supply paths absorb it). The runner commits the topology through the real
API, runs the assessor, and compares. Unlike the corpus metrics (which
measure agreement statistically across hundreds of generated cases), a
scenario is a named regression gate a reviewer can read: if the redundancy
semantics ever regress — an N+1 group stops absorbing, a 2N source pair
stops carrying — the failing scenario names exactly which physics broke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass(frozen=True)
class ScenarioOutcome:
    """One scenario's known-answer check: pass/fail with the mismatches."""

    scenario_id: str
    passed: bool
    checks: int
    failures: tuple[str, ...]


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[dict[str, Any]]:
    """All scenario payloads, sorted by id (deterministic run order)."""
    scenarios = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    return sorted(scenarios, key=lambda scenario: scenario["scenario_id"])


def _load_topology(client: TestClient, scenario: dict[str, Any]) -> None:
    """Create the scenario's facility and commit its topology (real HTTP)."""
    created = client.post(
        "/facilities",
        json={
            "id": scenario["facility_id"],
            "name": scenario["title"],
            "segment": scenario["market"],
        },
    )
    assert created.status_code == 201, created.text
    committed = client.post(
        f"/facilities/{scenario['facility_id']}/graph/versions",
        json=scenario["topology"],
    )
    assert committed.status_code == 201, committed.text


def run_scenario(client: TestClient, scenario: dict[str, Any]) -> ScenarioOutcome:
    """Run one fault-injection scenario against its adjudicated expectation."""
    scenario_id = scenario["scenario_id"]
    failures: list[str] = []
    checks = 0

    _load_topology(client, scenario)
    run = client.post(f"/facilities/{scenario['facility_id']}/assessments", json={})
    assert run.status_code == 201, run.text
    report: dict[str, Any] = run.json()

    inject_key: str = scenario["inject"]["asset_key"]
    injected = next(
        (asset for asset in report["assets"] if asset["node_key"] == inject_key), None
    )
    checks += 1
    if injected is None:
        return ScenarioOutcome(
            scenario_id=scenario_id,
            passed=False,
            checks=checks,
            failures=(f"injected asset {inject_key} missing from the assessment",),
        )

    expected_exposed: list[str] = sorted(scenario["expected"]["exposed_services"])
    checks += 1
    if sorted(injected["exposed_services"]) != expected_exposed:
        failures.append(
            f"exposed services mismatch: expected {expected_exposed}, "
            f"got {sorted(injected['exposed_services'])}"
        )

    services_by_id = {service["service_id"]: service for service in report["services"]}
    for service_id, expected_tolerated in sorted(
        scenario["expected"]["service_first_fault_tolerated"].items()
    ):
        checks += 1
        service = services_by_id.get(service_id)
        if service is None:
            failures.append(f"service {service_id} missing from the assessment")
            continue
        actual = service["exposure"]["first_fault_tolerated"]
        if actual != expected_tolerated:
            failures.append(
                f"service {service_id} first-fault tolerance: expected "
                f"{expected_tolerated}, got {actual}"
            )

    # Optional finding assertions: a scenario may pin that the assessor
    # raises a specific finding (e.g. the hospital's life-safety isolation
    # violation) — kind plus subject, matched against the report's findings.
    for expected_finding in scenario["expected"].get("findings", []):
        checks += 1
        present = any(
            finding["kind"] == expected_finding["kind"]
            and finding["subject"] == expected_finding["subject"]
            for finding in report["findings"]
        )
        if not present:
            failures.append(
                f"expected finding {expected_finding['kind']} on "
                f"{expected_finding['subject']} missing from the assessment"
            )

    checks += 1
    if not injected["evidence"]:
        failures.append(f"injected asset {inject_key} carries no evidence citations")

    return ScenarioOutcome(
        scenario_id=scenario_id, passed=not failures, checks=checks, failures=tuple(failures)
    )


def run_all_scenarios(client: TestClient) -> list[ScenarioOutcome]:
    """Every shipped scenario, in deterministic order."""
    return [run_scenario(client, scenario) for scenario in load_scenarios()]
