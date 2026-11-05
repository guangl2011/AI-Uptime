"""Sim runner: drives the real assessment API and evaluates the M1 gates.

Everything runs through the same TestClient the eval harness uses — the
simulator drives the assessment engine only, per the plan's governance
constraints. Each trace step commits a new immutable graph version and pins
an assessment to it, so every recorded state is reproducible in the API's
own terms (version_no), not just in memory.

Gates (art_non0bSFL categories 1-2 + determinism; zero tolerance):

- ``<segment>:failover_semantics`` — the corpus sweep: fault every asset of
  every facility, assert the service loss the construction-derived labels
  pin: equal to ``expected_exposed`` where the construction is unambiguous,
  within ``[expected_exposed, expected_impacted]`` where a protecting group
  sits downstream of the fault (a cascade may destroy a group's supply —
  the divergence the what-if label cannot see), per segment, zero tolerance.
- ``cascade_traces`` — the adjudicated sequenced scenarios, asserted
  stepwise (lost-service sets, degraded tolerance, SPOF churn).
- ``determinism`` — the same seed, run twice, produces an identical full
  trace (canonical-JSON digest of every step's report).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from fastapi.testclient import TestClient

from evals.corpus import generate_corpus
from evals.harness import load_facility
from evals.sim.dsl import SimScenario, load_sim_scenarios
from evals.sim.engine import (
    SimState,
    Topology,
    apply_event,
    dead_nodes,
    describe_event,
    initial_state,
    materialize,
)

SIM_AUTHOR = "eval_sim"
UNSERVED_KIND = "unserved_service"


@dataclass(frozen=True)
class TraceStep:
    """One recorded assessment state: version, event, observed service loss."""

    step: int
    version_no: int
    event: str  # "" for the baseline step
    lost_services: tuple[str, ...]
    report: dict[str, object]

    def to_json(self) -> dict[str, object]:
        """The full step as canonical-serializable JSON (determinism input)."""
        return {
            "step": self.step,
            "version_no": self.version_no,
            "event": self.event,
            "lost_services": list(self.lost_services),
            "report": self.report,
        }


@dataclass(frozen=True)
class SimScenarioOutcome:
    """The run of one sequenced scenario against its adjudicated trace."""

    scenario_id: str
    market: str
    passed: bool
    checks: int
    failures: tuple[str, ...]
    trace: tuple[TraceStep, ...]


def lost_services(report: dict[str, object]) -> frozenset[str]:
    """Services the report declares unserved — the observed service loss.

    Derived from the report's own findings (not from simulator internals):
    materialization leaves dead services isolated, and the assessor names
    each one with an unserved_service finding, so the derivation holds for
    any dead service the engine produces.
    """
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return frozenset()
    return frozenset(
        finding["subject"]
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("kind") == UNSERVED_KIND
        and isinstance(finding.get("subject"), str)
    )


def _commit_and_assess(
    client: TestClient, facility_id: str, payload: dict[str, object]
) -> tuple[int, dict[str, object]]:
    """Commit one snapshot and pin an assessment to it (the real API path)."""
    committed = client.post(f"/facilities/{facility_id}/graph/versions", json=payload)
    if committed.status_code != 201:
        raise AssertionError(
            f"sim snapshot commit failed ({committed.status_code}): {committed.text[:2000]}"
        )
    version_no = committed.json()["version_no"]
    run = client.post(
        f"/facilities/{facility_id}/assessments",
        json={"author": SIM_AUTHOR, "version_no": version_no},
    )
    if run.status_code != 201:
        raise AssertionError(f"sim assessment failed ({run.status_code}): {run.text[:2000]}")
    return version_no, run.json()


def _start_scenario(client: TestClient, scenario: SimScenario) -> None:
    """Create the scenario's facility and commit its base topology."""
    created = client.post(
        "/facilities",
        json={"id": scenario.facility_id, "name": scenario.title, "segment": scenario.market},
    )
    if created.status_code != 201:
        raise AssertionError(f"facility create failed: {created.text[:2000]}")
    _commit_and_assess(client, scenario.facility_id, dict(scenario.topology_payload))


def _services_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    """Report services indexed by id (report payloads are JSON dicts)."""
    services = report.get("services", [])
    if not isinstance(services, list):
        return {}
    by_id: dict[str, dict[str, object]] = {}
    for service in services:
        if isinstance(service, dict) and isinstance(service.get("service_id"), str):
            by_id[service["service_id"]] = service
    return by_id


def _exposure_field(service: dict[str, object], key: str) -> object:
    exposure = service.get("exposure")
    if not isinstance(exposure, dict):
        return None
    return exposure.get(key)


def compare_trace(scenario: SimScenario, trace: tuple[TraceStep, ...]) -> SimScenarioOutcome:
    """Assert the observed trace against the adjudicated expectations."""
    failures: list[str] = []
    checks = 0
    for step, expected in enumerate(scenario.expected_trace):
        observed = trace[step]
        checks += 1
        if observed.lost_services != tuple(sorted(expected.lost_services)):
            failures.append(
                f"step {step} ({observed.event or 'baseline'}): lost services expected "
                f"{sorted(expected.lost_services)}, observed {list(observed.lost_services)}"
            )
        report_service = _services_by_id(observed.report)
        for service_id, expected_fft in expected.service_first_fault_tolerated.items():
            service = report_service.get(service_id)
            if service is None:
                failures.append(f"step {step}: expected service {service_id} missing from report")
                continue
            checks += 1
            observed_fft = _exposure_field(service, "first_fault_tolerated")
            if observed_fft is not expected_fft:
                failures.append(
                    f"step {step} ({observed.event or 'baseline'}): {service_id} "
                    f"first_fault_tolerated expected {expected_fft}, observed {observed_fft}"
                )
        for service_id, expected_spofs in expected.service_spofs.items():
            service = report_service.get(service_id)
            if service is None:
                failures.append(f"step {step}: expected service {service_id} missing from report")
                continue
            checks += 1
            observed_spofs = _exposure_field(service, "unprotected_assets")
            if observed_spofs != sorted(expected_spofs):
                failures.append(
                    f"step {step} ({observed.event or 'baseline'}): {service_id} SPOFs expected "
                    f"{sorted(expected_spofs)}, observed {observed_spofs}"
                )
    return SimScenarioOutcome(
        scenario_id=scenario.scenario_id,
        market=scenario.market,
        passed=not failures,
        checks=checks,
        failures=tuple(failures),
        trace=trace,
    )


def run_sim_scenario(client: TestClient, scenario: SimScenario) -> SimScenarioOutcome:
    """Execute one sequenced scenario and compare it to its adjudicated trace."""
    _start_scenario(client, scenario)
    trace: list[TraceStep] = []

    state = initial_state(scenario.topology)
    dead = dead_nodes(state)
    payload = materialize(
        state, dead, author=SIM_AUTHOR, notes=f"sim {scenario.scenario_id} step 0: baseline"
    )
    version_no, report = _commit_and_assess(client, scenario.facility_id, payload)
    trace.append(
        TraceStep(
            step=0,
            version_no=version_no,
            event="",
            lost_services=tuple(sorted(lost_services(report))),
            report=report,
        )
    )

    for index, event in enumerate(scenario.events, start=1):
        state = apply_event(state, event)
        dead = dead_nodes(state)
        payload = materialize(
            state,
            dead,
            author=SIM_AUTHOR,
            notes=f"sim {scenario.scenario_id} step {index}: {describe_event(event)}",
        )
        version_no, report = _commit_and_assess(client, scenario.facility_id, payload)
        trace.append(
            TraceStep(
                step=index,
                version_no=version_no,
                event=describe_event(event),
                lost_services=tuple(sorted(lost_services(report))),
                report=report,
            )
        )

    return compare_trace(scenario, tuple(trace))


# ---------------------------------------------------------------------------
# Corpus sweep (Category 1: failover semantics, per segment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCase:
    """One single-fault sweep case against its construction-derived labels.

    The corpus carries two by-construction labels per asset: the single-fault
    what-if exposure (``expected_exposed`` — propagation stops at redundant
    groups of the intact graph) and the full-impact closure
    (``expected_impacted`` — the service set lost when the asset's function
    is completely gone). A stateful cascade sits between them: it can exceed
    the what-if label when the fault's secondary losses destroy a protecting
    group's own supply (a lone chiller's death kills both CHW pumps), but it
    never exceeds the full-impact closure. Where the two labels agree, the
    construction is unambiguous and the cascade must match exactly; where
    they differ, the cascade must lie within the bounds.
    """

    facility_id: str
    segment: str
    asset_key: str
    expected_exposed: tuple[str, ...]
    expected_impacted: tuple[str, ...]
    observed_lost: tuple[str, ...]

    @property
    def ok(self) -> bool:
        if self.expected_exposed == self.expected_impacted:
            return self.observed_lost == self.expected_exposed
        exposed = frozenset(self.expected_exposed)
        observed = frozenset(self.observed_lost)
        return exposed <= observed and observed <= frozenset(self.expected_impacted)

    @property
    def exact(self) -> bool:
        """The cascade equals the what-if label outright (no bound in play)."""
        return self.observed_lost == self.expected_exposed


@dataclass(frozen=True)
class SweepResult:
    """Aggregated corpus sweep over every facility's every asset."""

    seed: int
    facilities: int
    facilities_by_segment: dict[str, int]
    cases: tuple[SweepCase, ...]

    @property
    def mismatches(self) -> tuple[SweepCase, ...]:
        return tuple(case for case in self.cases if not case.ok)

    def per_segment(self) -> dict[str, tuple[int, int]]:
        """segment -> (cases, mismatches)."""
        totals: dict[str, tuple[int, int]] = {}
        for case in self.cases:
            cases, mismatches = totals.get(case.segment, (0, 0))
            totals[case.segment] = (cases + 1, mismatches + (0 if case.ok else 1))
        return dict(sorted(totals.items()))

    def ok(self) -> bool:
        return not self.mismatches


def run_corpus_sweep(
    client: TestClient,
    *,
    seed: int,
    facilities_per_segment: dict[str, int] | None = None,
) -> SweepResult:
    """Fault every asset of every corpus facility; assert the labeled loss.

    Reuses the eval harness's loader and the corpus's construction-derived
    labels (``expected_exposed``, ``expected_impacted``). Each case faults
    one asset, materializes the cascade, commits, and re-assesses — so the
    corpus adjudication gates the simulator's stateful semantics, not just
    the one-shot what-if the static eval already covers.
    """
    corpus = generate_corpus(seed, facilities_per_segment)
    cases: list[SweepCase] = []
    for facility in corpus.facilities:
        load_facility(client, facility)
        topology = Topology.from_payload(facility.graph)
        for asset_key in facility.assets:
            state = SimState(topology=topology, faulted=frozenset({asset_key}))
            dead = dead_nodes(state)
            payload = materialize(
                state,
                dead,
                author=SIM_AUTHOR,
                notes=f"sim sweep fault {asset_key} (facility {facility.facility_id})",
            )
            _, report = _commit_and_assess(client, facility.facility_id, payload)
            observed = tuple(sorted(lost_services(report)))
            expected_exposed = tuple(sorted(facility.expected_exposed[asset_key]))
            expected_impacted = tuple(sorted(facility.expected_impacted[asset_key]))
            cases.append(
                SweepCase(
                    facility_id=facility.facility_id,
                    segment=facility.segment,
                    asset_key=asset_key,
                    expected_exposed=expected_exposed,
                    expected_impacted=expected_impacted,
                    observed_lost=observed,
                )
            )
    counts: dict[str, int] = {}
    for facility in corpus.facilities:
        counts[facility.segment] = counts.get(facility.segment, 0) + 1
    return SweepResult(
        seed=seed,
        facilities=len(corpus.facilities),
        facilities_by_segment=dict(sorted(counts.items())),
        cases=tuple(cases),
    )


# ---------------------------------------------------------------------------
# Determinism gate and aggregate gates
# ---------------------------------------------------------------------------


def run_sim_suite(
    client: TestClient,
    *,
    seed: int,
    facilities_per_segment: dict[str, int] | None = None,
) -> tuple[tuple[SimScenarioOutcome, ...], SweepResult]:
    """One full sim pass: every sequenced scenario, then the corpus sweep.

    Called twice by the CLI (two throwaway clients, same seed) for the
    determinism gate; called once by tests on reduced corpora.
    """
    outcomes = tuple(
        run_sim_scenario(client, scenario) for scenario in load_sim_scenarios()
    )
    sweep = run_corpus_sweep(client, seed=seed, facilities_per_segment=facilities_per_segment)
    return outcomes, sweep


@dataclass(frozen=True)
class DeterminismResult:
    """Two same-seed runs must produce an identical full-trace digest."""

    first_digest: str
    second_digest: str

    @property
    def ok(self) -> bool:
        return self.first_digest == self.second_digest

    @property
    def detail(self) -> str:
        if self.ok:
            return f"identical full-trace digest {self.first_digest[:16]}"
        return f"digests differ: {self.first_digest[:16]} vs {self.second_digest[:16]}"


@dataclass(frozen=True)
class GateResult:
    """One named gate and its pass/fail with a human detail line."""

    name: str
    passed: bool
    detail: str


def trace_digest(outcomes: tuple[SimScenarioOutcome, ...], sweep: SweepResult) -> str:
    """Canonical digest of the full observed trace (scenarios + sweep)."""
    payload = {
        "scenarios": [
            {
                "scenario_id": outcome.scenario_id,
                "steps": [step.to_json() for step in outcome.trace],
            }
            for outcome in outcomes
        ],
        "sweep": [
            {
                "facility_id": case.facility_id,
                "asset_key": case.asset_key,
                "observed_lost": list(case.observed_lost),
            }
            for case in sweep.cases
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluate_sim_gates(
    outcomes: tuple[SimScenarioOutcome, ...], sweep: SweepResult, determinism: DeterminismResult
) -> tuple[GateResult, ...]:
    """The M1 gates, each with its evidence line."""
    gates: list[GateResult] = []

    passed = sum(1 for outcome in outcomes if outcome.passed)
    detail = f"{passed}/{len(outcomes)} adjudicated traces match stepwise"
    if passed != len(outcomes):
        first_failure = next(outcome for outcome in outcomes if not outcome.passed)
        detail += f"; first failure {first_failure.scenario_id}: {first_failure.failures[0]}"
    gates.append(GateResult(name="cascade_traces", passed=passed == len(outcomes), detail=detail))

    for segment, (cases, mismatches) in sweep.per_segment().items():
        exact = sum(1 for case in sweep.cases if case.segment == segment and case.exact)
        gates.append(
            GateResult(
                name=f"{segment}:failover_semantics",
                passed=mismatches == 0,
                detail=f"{cases} single-fault cases: {exact} equal the what-if label, "
                f"{cases - exact} within construction bounds (exposed <= observed <= "
                f"impacted), {mismatches} violations",
            )
        )

    gates.append(GateResult(name="determinism", passed=determinism.ok, detail=determinism.detail))
    return tuple(gates)
