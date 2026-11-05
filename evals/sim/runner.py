"""Sim runner: drives the real assessment API and evaluates the M1–M3 gates.

Everything runs through the same TestClient the eval harness uses — the
simulator drives the assessment engine only, per the plan's governance
constraints. Each trace step commits a new immutable graph version and pins
an assessment to it, so every recorded state is reproducible in the API's
own terms (version_no), not just in memory.

Gates (art_non0bSFL categories 1-2 + 5, plus determinism; zero tolerance):

- ``<segment>:failover_semantics`` — the corpus sweep: fault every asset of
  every facility, assert the service loss the construction-derived labels
  pin: equal to ``expected_exposed`` where the construction is unambiguous,
  within ``[expected_exposed, expected_impacted]`` where a protecting group
  sits downstream of the fault (a cascade may destroy a group's supply —
  the divergence the what-if label cannot see), per segment, zero tolerance.
- ``cascade_traces`` — the adjudicated sequenced scenarios, asserted
  stepwise (lost-service sets, degraded tolerance, SPOF churn).
- ``whatif_deltas`` — Category 5: each marker's assessment delta (baseline
  vs counterfactual) matches the adjudicated delta exactly; the marker's
  timeline step re-assesses the unchanged state (marker purity).
- ``whatif_spof_path`` — Category 5: an upgrade marker removes every SPOF
  finding on the upgraded path in the counterfactual, or the evaluator
  abstains with a stated reason. Abstention is accepted; a residual SPOF
  is a silent wrong answer and fails.
- ``determinism`` — the same seed, run twice, produces an identical full
  trace (canonical-JSON digest of every step's report, branches included).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from fastapi.testclient import TestClient

from evals.corpus import generate_corpus
from evals.harness import load_facility
from evals.sim.dsl import SimScenario, WhatIfMarker, load_sim_scenarios
from evals.sim.engine import (
    RedundancyChangeEvent,
    SimState,
    Topology,
    apply_event,
    apply_intervention,
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
    what_if: WhatIfBranch | None = None  # set on what-if marker steps (M3)

    def to_json(self) -> dict[str, object]:
        """The full step as canonical-serializable JSON (determinism input).

        The ``what_if`` key is omitted (not null) when absent, so steps
        from marker-free scenarios serialize byte-identically to their
        pre-M3 form — the M1 digest stays a stable fingerprint.
        """
        payload = {
            "step": self.step,
            "version_no": self.version_no,
            "event": self.event,
            "lost_services": list(self.lost_services),
            "report": self.report,
        }
        if self.what_if is not None:
            payload["what_if"] = self.what_if.to_json()
        return payload


@dataclass(frozen=True)
class WhatIfBranch:
    """The counterfactual side of one marker: intervention, assessment, delta.

    Recorded in the trace step (and therefore the determinism digest), so a
    what-if run replays byte-for-byte like any other step.
    """

    label: str
    intervention: str
    cf_version_no: int
    cf_report: dict[str, object]
    delta: dict[str, object]
    spof_path_abstention: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "intervention": self.intervention,
            "cf_version_no": self.cf_version_no,
            "cf_report": self.cf_report,
            "delta": self.delta,
            "spof_path_abstention": self.spof_path_abstention,
        }


@dataclass(frozen=True)
class WhatIfOutcome:
    """The verdict on one what-if marker, split by the Category 5 gates that
    consume it (``failures`` is the union for the repro section)."""

    scenario_id: str
    market: str
    label: str
    intervention: str
    intervention_kind: str  # "remove_asset" | "upgrade_redundancy"
    checks: int
    delta_failures: tuple[str, ...]
    spof_path_failures: tuple[str, ...]
    purity_failures: tuple[str, ...]
    abstention: str | None

    @property
    def failures(self) -> tuple[str, ...]:
        return self.delta_failures + self.spof_path_failures + self.purity_failures

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def delta_ok(self) -> bool:
        """The delta-adjudication side (what the ``whatif_deltas`` gate owns)."""
        return not self.delta_failures and not self.purity_failures


@dataclass(frozen=True)
class SimScenarioOutcome:
    """The run of one sequenced scenario against its adjudicated trace."""

    scenario_id: str
    market: str
    passed: bool
    checks: int
    failures: tuple[str, ...]
    trace: tuple[TraceStep, ...]
    what_ifs: tuple[WhatIfOutcome, ...] = ()


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


def compare_trace(
    scenario: SimScenario,
    trace: tuple[TraceStep, ...],
    what_ifs: tuple[WhatIfOutcome, ...] = (),
) -> SimScenarioOutcome:
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
        what_ifs=what_ifs,
    )


# ---------------------------------------------------------------------------
# What-if markers (M3, Category 5): branches, deltas, SPOF-path verdicts
# ---------------------------------------------------------------------------


def _service_spofs(report: dict[str, object]) -> dict[str, frozenset[str]]:
    """Per-service SPOF sets from the report's exposure blocks."""
    spofs: dict[str, frozenset[str]] = {}
    for service_id, service in _services_by_id(report).items():
        assets = _exposure_field(service, "unprotected_assets")
        spofs[service_id] = frozenset(assets) if isinstance(assets, list) else frozenset()
    return spofs


def _service_fft(report: dict[str, object]) -> dict[str, object]:
    """Per-service first-fault tolerance from the report's exposure blocks."""
    fft: dict[str, object] = {}
    for service_id, service in _services_by_id(report).items():
        fft[service_id] = _exposure_field(service, "first_fault_tolerated")
    return fft


def _delta_observed(
    baseline_report: dict[str, object], cf_report: dict[str, object]
) -> dict[str, object]:
    """The assessment delta between the baseline and counterfactual reports.

    Both sides come from the assessor's own output (exposure blocks and
    unserved_service findings) — the simulator never substitutes its own
    closure into the adjudication. Sets are sorted for canonical JSON.
    """
    base_lost = lost_services(baseline_report)
    cf_lost = lost_services(cf_report)
    base_spofs = _service_spofs(baseline_report)
    cf_spofs = _service_spofs(cf_report)
    base_fft = _service_fft(baseline_report)
    cf_fft = _service_fft(cf_report)
    spofs_added: dict[str, list[str]] = {}
    spofs_removed: dict[str, list[str]] = {}
    fft_changed: dict[str, dict[str, object]] = {}
    for service_id in sorted(set(base_spofs) | set(cf_spofs)):
        added = cf_spofs.get(service_id, frozenset()) - base_spofs.get(service_id, frozenset())
        removed = base_spofs.get(service_id, frozenset()) - cf_spofs.get(service_id, frozenset())
        if added:
            spofs_added[service_id] = sorted(added)
        if removed:
            spofs_removed[service_id] = sorted(removed)
        if base_fft.get(service_id) != cf_fft.get(service_id):
            fft_changed[service_id] = {
                "from": base_fft.get(service_id),
                "to": cf_fft.get(service_id),
            }
    return {
        "lost_services": sorted(cf_lost - base_lost),
        "restored_services": sorted(base_lost - cf_lost),
        "service_spofs_added": spofs_added,
        "service_spofs_removed": spofs_removed,
        "service_first_fault_tolerated": fft_changed,
    }


def _delta_failures(
    scenario_id: str,
    marker: WhatIfMarker,
    observed: dict[str, object],
    cf_report: dict[str, object],
) -> tuple[list[str], int]:
    """Compare the observed delta to the adjudicated one; every declared
    comparison is one check. ``service_first_fault_tolerated`` declares the
    counterfactual's absolute value (the state the graph should be in), so
    it is compared against the counterfactual report, not the changed-only
    delta map."""
    expected = marker.expected_delta
    failures: list[str] = []
    checks = 0

    def _expect(field: str, declared: object, observed_value: object) -> None:
        nonlocal checks
        checks += 1
        if declared != observed_value:
            failures.append(
                f"{scenario_id} what_if {marker.label!r}: {field} declared "
                f"{declared}, observed {observed_value}"
            )

    _expect(
        "lost_services",
        sorted(expected.lost_services),
        observed["lost_services"],
    )
    _expect(
        "restored_services",
        sorted(expected.restored_services),
        observed["restored_services"],
    )
    added = observed["service_spofs_added"]
    removed = observed["service_spofs_removed"]
    for service_id, declared in sorted(expected.service_spofs_added.items()):
        _expect(
            f"service_spofs_added[{service_id}]",
            sorted(declared),
            added.get(service_id, []) if isinstance(added, dict) else [],
        )
    for service_id, declared in sorted(expected.service_spofs_removed.items()):
        _expect(
            f"service_spofs_removed[{service_id}]",
            sorted(declared),
            removed.get(service_id, []) if isinstance(removed, dict) else [],
        )
    cf_fft = _service_fft(cf_report)
    for service_id, expected_fft in sorted(expected.service_first_fault_tolerated.items()):
        _expect(
            f"service_first_fault_tolerated[{service_id}]",
            expected_fft,
            cf_fft.get(service_id),
        )
    return failures, checks


def _strip_volatile(value: object) -> object:
    """Drop per-commit fields (provenance record ids, version numbers) so
    report comparison asks about content, not about which commit produced it."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in ("record_id", "graph_version")
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _purity_failure(previous: TraceStep, marker_report: dict[str, object]) -> str | None:
    """Marker purity: re-assessing the unchanged state must reproduce the
    previous step's report (the marker recorded no timeline event)."""
    if _strip_volatile(previous.report) != _strip_volatile(marker_report):
        return (
            "marker purity: re-assessing the unchanged state changed the report vs "
            f"step {previous.step} ({previous.event or 'baseline'}) — the marker "
            "advanced the timeline"
        )
    return None


def _spof_path_verdict(
    scenario_id: str,
    marker: WhatIfMarker,
    cf_state: SimState,
    cf_dead: frozenset[str],
    baseline_report: dict[str, object],
    cf_report: dict[str, object],
) -> tuple[str | None, list[str]]:
    """Category 5's structural verdict for an upgrade marker (removals: none).

    Every live member of the upgraded group that was a SPOF of one of its
    services at baseline must have lost that finding in the counterfactual
    (an N → N+1 upgrade must remove the SPOF finding on that path). When the
    counterfactual snapshot cannot support the verdict — fewer than two live
    members — the evaluator abstains with a stated reason, which the plan
    accepts; a residual SPOF is a silent wrong answer and fails.
    """
    if not isinstance(marker.intervention, RedundancyChangeEvent):
        return None, []
    group = marker.intervention.member_group
    members = {edge.from_node for edge in cf_state.topology.edges if edge.member_group == group}
    live = sorted(members - cf_dead)
    if len(live) < 2:
        reason = (
            f"upgrade of {group}: {len(live)} live member(s) in the counterfactual "
            f"snapshot ({', '.join(live) or 'none'}) — single-fault tolerance for the "
            "path is structurally undecided at this member count, so the SPOF-path "
            "verdict is abstained rather than asserted"
        )
        return reason, []
    base_spofs = _service_spofs(baseline_report)
    cf_spofs = _service_spofs(cf_report)
    failures: list[str] = []
    for service_id in sorted(base_spofs):
        baseline_group_spofs = base_spofs[service_id] & members
        if not baseline_group_spofs:
            continue
        residual = sorted(baseline_group_spofs & cf_spofs.get(service_id, frozenset()))
        for asset_key in residual:
            failures.append(
                f"{scenario_id} what_if {marker.label!r}: upgrade of {group} left "
                f"{asset_key} a single point of failure of {service_id} in the "
                "counterfactual — a silent wrong answer (an N → N+1 upgrade must "
                "remove the SPOF finding on that path or abstain with a reason)"
            )
    return None, failures


def _run_what_if_branch(
    client: TestClient,
    scenario: SimScenario,
    marker: WhatIfMarker,
    index: int,
    state: SimState,
    baseline_report: dict[str, object],
) -> tuple[WhatIfBranch, WhatIfOutcome]:
    """Run one marker's counterfactual branch through the real API.

    The branch starts from the marker's saved state (a copy — the timeline
    continues from ``state`` untouched), applies the intervention, commits
    the counterfactual snapshot, and re-assesses. Returns the branch record
    (trace + determinism digest) and its verdict (Category 5 gates).
    """
    cf_state = apply_intervention(state, marker.intervention)
    cf_dead = dead_nodes(cf_state)
    payload = materialize(
        cf_state,
        cf_dead,
        author=SIM_AUTHOR,
        notes=(
            f"sim {scenario.scenario_id} step {index}: what_if {marker.label} "
            f"counterfactual ({marker.described})"
        ),
    )
    cf_version_no, cf_report = _commit_and_assess(client, scenario.facility_id, payload)
    delta = _delta_observed(baseline_report, cf_report)
    delta_failures, checks = _delta_failures(scenario.scenario_id, marker, delta, cf_report)
    abstention, spof_failures = _spof_path_verdict(
        scenario.scenario_id, marker, cf_state, cf_dead, baseline_report, cf_report
    )
    checks += 1  # the SPOF-path verdict itself
    declared_abstention = marker.expected_delta.expect_spof_path_abstention
    if declared_abstention is True and abstention is None:
        delta_failures.append(
            f"{scenario.scenario_id} what_if {marker.label!r}: adjudication says the "
            "SPOF-path verdict must be abstained, but none fired"
        )
    elif declared_abstention is False and abstention is not None:
        delta_failures.append(
            f"{scenario.scenario_id} what_if {marker.label!r}: SPOF-path verdict "
            f"abstained unexpectedly: {abstention}"
        )
    branch = WhatIfBranch(
        label=marker.label,
        intervention=marker.described,
        cf_version_no=cf_version_no,
        cf_report=cf_report,
        delta=delta,
        spof_path_abstention=abstention,
    )
    outcome = WhatIfOutcome(
        scenario_id=scenario.scenario_id,
        market=scenario.market,
        label=marker.label,
        intervention=marker.described,
        intervention_kind=(
            "upgrade_redundancy"
            if isinstance(marker.intervention, RedundancyChangeEvent)
            else "remove_asset"
        ),
        checks=checks,
        delta_failures=tuple(delta_failures),
        spof_path_failures=tuple(spof_failures),
        purity_failures=(),
        abstention=abstention,
    )
    return branch, outcome


def run_sim_scenario(client: TestClient, scenario: SimScenario) -> SimScenarioOutcome:
    """Execute one sequenced scenario and compare it to its adjudicated trace."""
    _start_scenario(client, scenario)
    trace: list[TraceStep] = []
    what_ifs: list[WhatIfOutcome] = []

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
        if isinstance(event, WhatIfMarker):
            # The marker re-assesses the unchanged state through the real
            # API: the timeline record doubles as the branch's baseline, and
            # the re-assessment doubles as the purity check — the report
            # must equal the previous step's, or the marker advanced the
            # timeline. Branches never touch the timeline.
            dead = dead_nodes(state)
            payload = materialize(
                state,
                dead,
                author=SIM_AUTHOR,
                notes=(
                    f"sim {scenario.scenario_id} step {index}: what_if "
                    f"{event.label} (state unchanged)"
                ),
            )
            version_no, report = _commit_and_assess(client, scenario.facility_id, payload)
            purity_failures: list[str] = []
            if trace:
                purity = _purity_failure(trace[-1], report)
                if purity is not None:
                    purity_failures.append(purity)
            branch, outcome = _run_what_if_branch(
                client, scenario, event, index, state, report
            )
            if purity_failures:
                outcome = replace(
                    outcome,
                    purity_failures=tuple(purity_failures),
                    checks=outcome.checks + 1,
                )
            trace.append(
                TraceStep(
                    step=index,
                    version_no=version_no,
                    event=f"what_if {event.label}",
                    lost_services=tuple(sorted(lost_services(report))),
                    report=report,
                    what_if=branch,
                )
            )
            what_ifs.append(outcome)
            continue
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

    return compare_trace(scenario, tuple(trace), tuple(what_ifs))


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
class ForecastDeterminismResult:
    """Sim M4 (Category 6 (e)): two same-seed passes must produce an
    identical forecast-across-dynamics trace digest."""

    first_digest: str
    second_digest: str

    @property
    def ok(self) -> bool:
        return self.first_digest == self.second_digest

    @property
    def detail(self) -> str:
        if self.ok:
            return f"identical forecast-trace digest {self.first_digest[:16]}"
        return (
            f"forecast digests differ: {self.first_digest[:16]} "
            f"vs {self.second_digest[:16]}"
        )


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
    outcomes: tuple[SimScenarioOutcome, ...],
    sweep: SweepResult,
    determinism: DeterminismResult,
    forecast_determinism: ForecastDeterminismResult | None = None,
) -> tuple[GateResult, ...]:
    """The M1–M3 gates, each with its evidence line.

    ``forecast_determinism`` (Sim M4, Category 6 (e)) folds into the
    determinism row when provided: one determinism gate covering both the
    M1–M3 assessment trace and the forecast-across-dynamics trace.
    """
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

    det_detail = determinism.detail
    det_passed = determinism.ok
    if forecast_determinism is not None:
        det_passed = det_passed and forecast_determinism.ok
        det_detail += f"; forecast-under-dynamics {forecast_determinism.detail}"
    gates.append(GateResult(name="determinism", passed=det_passed, detail=det_detail))

    what_ifs = tuple(wi for outcome in outcomes for wi in outcome.what_ifs)
    if not what_ifs:
        empty_detail = (
            "no what-if markers loaded — Category 5 coverage is empty "
            "(check the scenario directory)"
        )
        gates.append(GateResult(name="whatif_deltas", passed=True, detail=empty_detail))
        gates.append(GateResult(name="whatif_spof_path", passed=True, detail=empty_detail))
        return tuple(gates)

    delta_passed = sum(1 for wi in what_ifs if wi.delta_ok)
    by_segment: dict[str, list[WhatIfOutcome]] = {}
    for wi in what_ifs:
        by_segment.setdefault(wi.market, []).append(wi)
    segment_detail = ", ".join(
        f"{segment} {sum(1 for wi in marker_list if wi.delta_ok)}/{len(marker_list)}"
        for segment, marker_list in sorted(by_segment.items())
    )
    detail = (
        f"{delta_passed}/{len(what_ifs)} what-if deltas match the adjudication "
        f"({segment_detail})"
    )
    if delta_passed != len(what_ifs):
        first = next(wi for wi in what_ifs if not wi.delta_ok)
        first_detail = first.delta_failures or first.purity_failures
        detail += f"; first failure {first.scenario_id}/{first.label}: {first_detail[0]}"
    gates.append(
        GateResult(
            name="whatif_deltas",
            passed=delta_passed == len(what_ifs),
            detail=detail,
        )
    )

    upgrades = [wi for wi in what_ifs if wi.intervention_kind == "upgrade_redundancy"]
    abstentions = [wi for wi in what_ifs if wi.abstention is not None]
    spof_violations = sum(1 for wi in what_ifs if wi.spof_path_failures)
    spof_detail = (
        f"{len(upgrades)} upgrade marker(s): {len(upgrades) - spof_violations} clean, "
        f"{len(abstentions)} abstained with a stated reason, {spof_violations} "
        "residual-SPOF violations"
    )
    if abstentions:
        first_abstention = abstentions[0]
        spof_detail += (
            f"; first abstention {first_abstention.scenario_id}/"
            f"{first_abstention.label}: {first_abstention.abstention}"
        )
    gates.append(
        GateResult(
            name="whatif_spof_path",
            passed=spof_violations == 0,
            detail=spof_detail,
        )
    )
    return tuple(gates)
