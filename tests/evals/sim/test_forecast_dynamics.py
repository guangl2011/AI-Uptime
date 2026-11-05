"""Sim M4 tests: forecast-under-dynamics gates (plan Category 6).

Two layers, mirroring the M1–M3 test split:

- Unit tests prove each gate is non-vacuous: they tamper one recorded row of
  a real reduced-corpus pass (a decrease, a cleared abstention, a narrower
  interval, a drifted report row, a drifted intact reference) and assert the
  corresponding check fires with the seed and repro attached — plus the
  sanctioned behaviors it must accept (a widened interval, a conflict
  registration, bounds narrowed by added redundancy).
- Integration tests run the real pass on throwaway SQLite apps (one facility
  per segment): every per-segment gate green, the fault → conflict-abstention
  → repair-restores behavior through the actual API, same-seed forecast-trace
  determinism, and the nightly sweep CLI on one seed.

What is never asserted here: accuracy. Synthetic labels are a deterministic
function of the same protection facts the forecast features read, so accuracy
under dynamics would be tautological (the plan's degenerate-separation
caveat). Gates assert structural behavior only.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from evals.corpus import FacilityCorpus, generate_corpus, generate_segment_corpus
from evals.harness import load_facility
from evals.sim.engine import (
    TOLERANT_CONFIGS,
    FaultEvent,
    LoadShiftEvent,
    RedundancyChangeEvent,
    RepairEvent,
    SimState,
    Topology,
    apply_event,
    initial_state,
    materialize,
)
from evals.sim.forecast_dynamics import (
    BOUNDS,
    SIGN,
    DynamicsStep,
    ForecastDynamicsResult,
    StableFeed,
    StablePathScope,
    _fetch_snapshot,
    _stable_path_feeds,
    check_abstention_identity,
    check_bounds_monotone,
    check_degradation_sign,
    check_repair_restores,
    evaluate_forecast_gates,
    forecast_trace_digest,
    is_fault_step,
    run_forecast_dynamics,
    run_forecast_sweep,
)
from evals.sim.report import SimResults, render_json, render_markdown
from evals.sim.runner import (
    SIM_AUTHOR,
    DeterminismResult,
    ForecastDeterminismResult,
    SweepResult,
    _commit_and_assess,
)
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from uptime_api.app import create_app
from uptime_api.forecasting.calibrated import CalibratedForecastModel
from uptime_api.forecasting.schema import ForecastStatus, RiskForecast

REDUCED_COUNTS = {"cold_chain": 1, "aidc": 1, "hospital": 1}
SEED = 20261104


def with_row(row: RiskForecast, **updates: object) -> RiskForecast:
    """A copied forecast row with the given fields updated (pydantic copy)."""
    return row.model_copy(update=updates)

# Stable Hypothesis draw range — any seed here is a legal corpus seed.
SEEDS = st.integers(min_value=1, max_value=10_000)


def reduced_pass(
    tmp_path: Path, seed: int = SEED, name: str = "pass"
) -> ForecastDynamicsResult:
    """One real forecast-under-dynamics pass on one facility per segment.

    Each pass gets its own SQLite file: corpus facility ids must not collide
    across passes on the same app database.
    """
    app = create_app(f"sqlite:///{tmp_path}/dynamics-{seed}-{name}.db")
    with TestClient(app) as client:
        return run_forecast_dynamics(
            client, seed=seed, facilities_per_segment=REDUCED_COUNTS
        )


def corpus_facility(trace_segment_facility: tuple[str, str]) -> FacilityCorpus:
    """The corpus facility a recorded trace came from (same seed → same corpus)."""
    segment, facility_id = trace_segment_facility
    corpus = generate_corpus(SEED, REDUCED_COUNTS)
    for facility in corpus.facilities:
        if facility.segment == segment and facility.facility_id == facility_id:
            return facility
    raise AssertionError(f"{facility_id} not in the seed-{SEED} corpus")


# ---------------------------------------------------------------------------
# Non-vacuity: each gate catches the defect it exists for
# ---------------------------------------------------------------------------


def find_feed(
    result: ForecastDynamicsResult, predicate: object
) -> tuple[FacilityCorpus, DynamicsStep, DynamicsStep, StablePathScope, StableFeed]:
    """The first fault-step stable feed matching ``predicate(row)``, across all traces."""
    for trace in result.traces:
        facility = corpus_facility((trace.segment, trace.facility_id))
        for pre, post in zip(trace.steps, trace.steps[1:], strict=False):
            if not post.event.lower().startswith("fault"):
                continue
            scope = _stable_path_feeds(pre, post)
            for feed in scope.pairs:
                row = pre.forecasts.get(feed.asset_key)
                if row is not None and row.status is ForecastStatus.FORECAST and predicate(row):  # type: ignore[arg-type]
                    return facility, pre, post, scope, feed
    raise AssertionError("no matching stable feed found in the reduced pass")


def test_sign_gate_catches_a_tampered_decrease(tmp_path: Path) -> None:
    """Lowering a stable feed's post-fault forecast must fire the sign gate."""
    result = reduced_pass(tmp_path)
    facility, pre, post, scope, feed = find_feed(result, lambda row: row.p_fault > 0.01)
    pre_row = pre.forecasts[feed.asset_key]
    tampered_post = replace(
        post,
        forecasts={
            **post.forecasts,
            feed.asset_key: with_row(pre_row, p_fault=pre_row.p_fault - 0.01),
        },
    )
    check = check_degradation_sign(facility, SEED, pre, tampered_post, scope, ())
    assert check.violations, "the sign gate missed a tampered decrease"
    assert "DECREASED" in check.violations[0].render()
    assert f"seed={SEED}" in check.violations[0].render()


def test_sign_gate_catches_a_cleared_abstention(tmp_path: Path) -> None:
    """An abstention must never clear across a fault; clearing fires the gate."""
    result = reduced_pass(tmp_path)
    facility, pre, post, scope, feed = find_feed(result, lambda row: True)
    tampered_pre = replace(
        pre,
        forecasts={
            **pre.forecasts,
            feed.asset_key: RiskForecast(
                asset_key=feed.asset_key,
                horizon_days=30,
                status=ForecastStatus.ABSTAINED,
                abstention_reason="sparse data: no calibration support",
                p_fault=1 / 3,
                lower=0.0,
                upper=1.0,
                basis="synthetic test row",
            ),
        },
    )
    check = check_degradation_sign(facility, SEED, tampered_pre, post, scope, ())
    assert check.violations, "the sign gate missed a cleared abstention"
    assert "cleared" in check.violations[0].render()


def test_bounds_gate_catches_a_shrinking_interval(tmp_path: Path) -> None:
    """A fault must never silently tighten a stable feed's interval."""
    result = reduced_pass(tmp_path)
    facility, pre, post, scope, feed = find_feed(
        result, lambda row: row.upper > row.lower
    )
    pre_row = pre.forecasts[feed.asset_key]
    span = pre_row.upper - pre_row.lower
    tampered_post = replace(
        post,
        forecasts={
            **post.forecasts,
            feed.asset_key: with_row(
                pre_row,
                lower=pre_row.lower + span * 0.25,
                upper=pre_row.upper - span * 0.25,
            ),
        },
    )
    check = check_bounds_monotone(facility, SEED, pre, tampered_post, scope, ())
    assert check.violations, "the bounds gate missed a shrinking interval"
    assert "SHRANK" in check.violations[0].render()


def test_bounds_gate_accepts_a_widened_interval(tmp_path: Path) -> None:
    """Information loss widening the bounds is the sanctioned direction."""
    result = reduced_pass(tmp_path)
    facility, pre, post, scope, feed = find_feed(result, lambda row: True)
    pre_row = pre.forecasts[feed.asset_key]
    tampered_post = replace(
        post,
        forecasts={
            **post.forecasts,
            feed.asset_key: with_row(
                pre_row, lower=max(0.0, pre_row.lower - 0.1), upper=1.0
            ),
        },
    )
    check = check_bounds_monotone(facility, SEED, pre, tampered_post, scope, ())
    assert not check.violations, [violation.render() for violation in check.violations]


def test_upgrade_steps_are_out_of_scope_for_sign_and_bounds(
    tmp_path: Path,
) -> None:
    """A redundancy upgrade ADDS protection (information): tightening bounds
    or lowering a forecast there is legitimate, so the sign/bounds gates key
    on fault steps only — and real passes do run upgrades."""
    result = reduced_pass(tmp_path)
    assert any(
        step.event.startswith("redundancy_change")
        for trace in result.traces
        for step in trace.steps
    ), "the reduced pass contains no redundancy upgrade — scope test vacuous"
    assert is_fault_step(FaultEvent(asset_key="GEN-1"))
    assert not is_fault_step(
        RedundancyChangeEvent(
            member_group="CHILLERS",
            redundancy_config="N+1",
            standby_node={},
            standby_edge={},
        )
    )
    assert not is_fault_step(RepairEvent(asset_key="GEN-1"))
    assert not is_fault_step(
        LoadShiftEvent(consumer="service", from_node="a", to_node="b")
    )


def test_abstention_identity_gate_catches_a_drifted_report_row(
    tmp_path: Path,
) -> None:
    """The report's rows must equal the static path's, exactly, at every state."""
    result = reduced_pass(tmp_path)
    facility, _, _, _, feed = find_feed(result, lambda row: True)
    trace = result.traces[0]
    step = trace.steps[-1]
    asset_key = feed.asset_key if feed.asset_key in step.forecasts else sorted(step.forecasts)[0]
    tampered = dict(step.forecasts)
    tampered[asset_key] = with_row(
        tampered[asset_key],
        p_fault=min(1.0, tampered[asset_key].p_fault + 0.05),
    )
    tampered_step = replace(step, forecasts=tampered)
    model = CalibratedForecastModel()
    check = check_abstention_identity(facility, SEED, tampered_step, model, ())
    assert check.violations, "the identity gate missed a drifted report row"
    assert asset_key in check.violations[0].render()


def test_repair_restores_gate_catches_a_drifted_reference(tmp_path: Path) -> None:
    """Post-repair rows must equal the fresh intact reference, exactly."""
    result = reduced_pass(tmp_path)
    facility, _, _, _, feed = find_feed(result, lambda row: True)
    trace = next(
        t for t in result.traces if t.steps[-1].event.startswith("intact reference")
    )
    repaired, intact = trace.steps[-2], trace.steps[-1]
    asset_key = (
        feed.asset_key if feed.asset_key in intact.forecasts else sorted(intact.forecasts)[0]
    )
    tampered_reference = replace(
        intact,
        forecasts={
            **intact.forecasts,
            asset_key: with_row(
                intact.forecasts[asset_key],
                p_fault=min(1.0, intact.forecasts[asset_key].p_fault + 0.05),
            ),
        },
    )
    check = check_repair_restores(facility, SEED, repaired, tampered_reference, ())
    assert check.violations, "the restore gate missed a drifted reference"
    assert asset_key in check.violations[0].render()


# ---------------------------------------------------------------------------
# Integration: the canonical behavior through the real API
# ---------------------------------------------------------------------------


def _pair_group_facility() -> tuple[FacilityCorpus, str, str]:
    """The first aidc corpus facility with a live tolerant pair of ASSET members."""
    for seed in range(1, 25):
        facility = generate_segment_corpus("aidc", seed, 1)[0]
        topology = Topology.from_payload(facility.graph)
        asset_keys = {
            node_key
            for node_key, node in topology.nodes.items()
            if node.node_type == "asset"
        }
        groups: dict[tuple[str, str], list[str]] = {}
        for edge in topology.edges:
            if (
                edge.redundancy_config in TOLERANT_CONFIGS
                and edge.member_group is not None
                and edge.from_node in asset_keys
            ):
                groups.setdefault((edge.to_node, edge.member_group), []).append(
                    edge.from_node
                )
        for members in groups.values():
            live = sorted(set(members))
            if len(live) >= 2:
                return facility, live[0], live[1]
    raise AssertionError("no tolerant asset pair group found in aidc seeds 1..24")


def test_conflict_registration_and_repair_restoration_via_api(tmp_path: Path) -> None:
    """The canonical M4 behavior through the real API surface.

    Faulting one member of a live tolerant group must flip the surviving
    feed to a conflict abstention (widest interval); the sign gate must
    accept exactly that registration and reject an unregistered one; and
    repairing the member must restore the baseline forecast rows exactly
    (frozen, drift-pinned table).
    """
    facility, member_a, member_b = _pair_group_facility()
    app = create_app(f"sqlite:///{tmp_path}/behavior.db")
    with TestClient(app) as client:
        # The canonical M4 surface: forecasts ride on the app-level model
        # (product default stays off; this throwaway app opts in explicitly).
        model = CalibratedForecastModel()
        client.app.state.forecaster = model
        state = initial_state(Topology.from_payload(facility.graph))
        load_facility(client, facility)

        def step_of(
            live_state: SimState, dead: frozenset[str], step: int, notes: str
        ) -> DynamicsStep:
            payload = materialize(live_state, dead, author=SIM_AUTHOR, notes=notes)
            version_no, report = _commit_and_assess(
                client, facility.facility_id, payload
            )
            return DynamicsStep(
                step=step,
                event=notes,
                version_no=version_no,
                snapshot=_fetch_snapshot(client, facility.facility_id, version_no),
                forecasts={
                    RiskForecast.model_validate(item).asset_key: RiskForecast.model_validate(
                        item
                    )
                    for item in report["forecasts"]
                },
                dead=dead,
            )

        baseline = step_of(state, frozenset(), 0, "intact baseline (test)")
        assert (
            baseline.forecasts[member_b].status is ForecastStatus.FORECAST
        ), "test precondition: the surviving feed starts non-abstained"

        faulted_state = apply_event(
            initial_state(Topology.from_payload(facility.graph)),
            FaultEvent(asset_key=member_a),
        )
        faulted = step_of(
            faulted_state,
            frozenset({member_a}),
            1,
            f"fault {member_a} (test)",
        )
        post_row = faulted.forecasts[member_b]
        assert post_row.status is ForecastStatus.ABSTAINED
        assert "conflicting" in post_row.abstention_reason.lower()
        assert post_row.lower == 0.0 and post_row.upper == 1.0

        scope = _stable_path_feeds(baseline, faulted)
        clean = check_degradation_sign(facility, SEED, baseline, faulted, scope, ())
        assert not clean.violations, [
            violation.render() for violation in clean.violations
        ]

        unregistered = replace(
            faulted,
            forecasts={
                **faulted.forecasts,
                member_b: with_row(
                    post_row,
                    status=ForecastStatus.FORECAST,
                    abstention_reason=None,
                    p_fault=0.5,
                    lower=0.3,
                    upper=0.7,
                ),
            },
        )
        broken = check_degradation_sign(
            facility, SEED, baseline, unregistered, scope, ()
        )
        assert broken.violations, "an unregistered redundancy loss slipped through"
        assert "not registered" in broken.violations[0].render()

        repaired = step_of(state, frozenset(), 2, "repair restored (test)")
        assert repaired.forecasts == baseline.forecasts
        restored = check_repair_restores(facility, SEED, repaired, baseline, ())
        assert not restored.violations, [
            violation.render() for violation in restored.violations
        ]
        identity = check_abstention_identity(facility, SEED, repaired, model, ())
        assert not identity.violations, [
            violation.render() for violation in identity.violations
        ]


def test_full_pass_gates_are_green_on_reduced_corpus(tmp_path: Path) -> None:
    result = reduced_pass(tmp_path)
    assert result.facilities == 3
    assert result.states_assessed > 0
    for segment in ("aidc", "cold_chain", "hospital"):
        assert result.facilities_by_segment[segment] == 1
    gates = evaluate_forecast_gates(result)
    failed = [gate for gate in gates if not gate.passed]
    assert not failed, [(gate.name, gate.detail) for gate in failed]
    names = {gate.name for gate in gates}
    assert names >= {
        f"{segment}:{name}"
        for segment in REDUCED_COUNTS
        for name in (
            SIGN,
            "forecast_repair_restores",
            "forecast_abstention_identity",
            BOUNDS,
        )
    }
    # No gate is vacuous on any segment: identity holds at every state,
    # restore runs once per facility, and sign/bounds cover fault steps.
    for segment in REDUCED_COUNTS:
        assert result.checks_for(SIGN, segment) > 0
        assert result.checks_for("forecast_abstention_identity", segment) > 0
        assert result.checks_for(BOUNDS, segment) > 0
        assert result.checks_for("forecast_repair_restores", segment) >= 1


def test_digest_stable_across_same_seed_passes(tmp_path: Path) -> None:
    first = reduced_pass(tmp_path, seed=SEED, name="first")
    second = reduced_pass(tmp_path, seed=SEED, name="second")
    assert forecast_trace_digest(first.traces) == forecast_trace_digest(second.traces)


def test_digest_tracks_trace_content(tmp_path: Path) -> None:
    """A different seed regenerates a different corpus — a different digest."""
    first = reduced_pass(tmp_path, seed=SEED, name="first")
    second = reduced_pass(tmp_path, seed=SEED + 1, name="second")
    assert forecast_trace_digest(first.traces) != forecast_trace_digest(second.traces)


# ---------------------------------------------------------------------------
# Property: determinism of the digest across same-seed reduced passes
# ---------------------------------------------------------------------------


@given(SEEDS)
@settings(max_examples=3, deadline=None)
def test_property_same_seed_digest_is_stable(seed: int) -> None:
    with tempfile.TemporaryDirectory(prefix="aiuptime-m4-prop-") as tmp:
        tmp_path = Path(tmp)
        first = reduced_pass(tmp_path, seed=seed, name="first")
        second = reduced_pass(tmp_path, seed=seed, name="second")
    assert forecast_trace_digest(first.traces) == forecast_trace_digest(second.traces)


# ---------------------------------------------------------------------------
# Gate evaluation failure paths, sweep CLI, and report rendering
# ---------------------------------------------------------------------------


def test_gate_evaluation_fails_with_repro_on_a_violation(tmp_path: Path) -> None:
    result = reduced_pass(tmp_path)
    trace = result.traces[0]
    step = trace.steps[-1]
    asset_key = sorted(step.forecasts)[0]
    tampered_step = replace(
        step,
        forecasts={
            **step.forecasts,
            asset_key: with_row(
                step.forecasts[asset_key],
                p_fault=min(1.0, step.forecasts[asset_key].p_fault + 0.05),
            ),
        },
    )
    tampered_trace = replace(
        trace,
        steps=(*trace.steps[:-1], tampered_step),
        violations=(
            *trace.violations,
            *check_abstention_identity(
                corpus_facility((trace.segment, trace.facility_id)),
                result.seed,
                tampered_step,
                CalibratedForecastModel(),
                (),
            ).violations,
        ),
    )
    broken = replace(result, traces=(tampered_trace, *result.traces[1:]))
    gates = evaluate_forecast_gates(broken)
    identity = [
        gate
        for gate in gates
        if gate.name == f"{trace.segment}:forecast_abstention_identity"
    ]
    assert identity and not identity[0].passed
    assert asset_key in identity[0].detail


def test_gate_evaluation_fails_when_a_property_is_vacuous(tmp_path: Path) -> None:
    result = reduced_pass(tmp_path)
    hollow = ForecastDynamicsResult(
        seed=result.seed,
        facilities=0,
        facilities_by_segment={segment: 0 for segment in REDUCED_COUNTS},
        states_assessed=0,
        runtime_seconds=0.0,
        traces=(),
    )
    gates = evaluate_forecast_gates(hollow)
    assert gates and not any(gate.passed for gate in gates)
    assert all(gate.detail.startswith("0 checks") for gate in gates)


def test_sweep_cli_smoke_on_one_seed(tmp_path: Path) -> None:
    exit_code = run_forecast_sweep(
        seed_count=1,
        start_seed=7,
        facilities_per_segment=1,
        output_dir=tmp_path,
    )
    assert exit_code == 0
    md = (tmp_path / "forecast-dynamics.md").read_text(encoding="utf-8")
    assert "Forecast-under-dynamics sweep (Sim M4, Category 6)" in md
    assert "never accuracy" in md
    assert "| `cold_chain:forecast_degradation_sign` | PASS |" in md
    payload_md = (tmp_path / "forecast-dynamics.json").read_text(encoding="utf-8")
    assert '"per_property"' in payload_md


def test_report_renders_the_m4_section_and_json_payload(tmp_path: Path) -> None:
    result = reduced_pass(tmp_path)
    gates = evaluate_forecast_gates(result)
    digest = forecast_trace_digest(result.traces)
    results = SimResults(
        seed=result.seed,
        facilities_per_segment=dict(result.facilities_by_segment),
        scenarios=(),
        sweep=SweepResult(
            seed=result.seed,
            facilities=3,
            facilities_by_segment=dict(result.facilities_by_segment),
            cases=(),
        ),
        determinism=DeterminismResult(first_digest="a", second_digest="a"),
        gates=gates,
        forecast_dynamics=result,
        forecast_determinism=ForecastDeterminismResult(
            first_digest=digest, second_digest=digest
        ),
    )
    md = render_markdown(results)
    assert "## Forecast under dynamics (Sim M4, Category 6, per-segment)" in md
    assert "accuracy under dynamics (AUROC/Brier) would be tautological" in md
    assert "| cold_chain |" in md
    assert "Category 6 (e) determinism: identical forecast-trace digest" in md
    assert "## Failures (repro detail)" not in md
    payload = render_json(results)
    forecast_json = payload["forecast_dynamics"]
    assert forecast_json["states_assessed"] == result.states_assessed
    assert forecast_json["per_property"][SIGN]["cold_chain"]["checks"] > 0
    assert payload["determinism"]["forecast_dynamics"]["ok"] is True


def test_report_renders_not_run_when_forecast_absent() -> None:
    results = SimResults(
        seed=SEED,
        facilities_per_segment={},
        scenarios=(),
        sweep=SweepResult(
            seed=SEED, facilities=0, facilities_by_segment={}, cases=()
        ),
        determinism=DeterminismResult(first_digest="a", second_digest="a"),
        gates=(),
    )
    md = render_markdown(results)
    assert "NOT RUN" in md
    assert "its gates are absent" in md
    payload = render_json(results)
    assert "forecast_dynamics" not in payload
