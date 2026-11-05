"""Hypothesis property tests for the Sim M2 invariants (plan Categories 3-4).

Properties pinned here, stated as mathematical properties over generated
seeds — never as expected outputs:

1. Redundancy monotonicity: adding redundancy (a standby member via the
   N → N+1 → 2N ladder) never increases exposure — for the intact state and
   every single-asset fault, the upgraded dead set is a subset of the
   original's, and the standby itself never dies.
2. Repair recovery: a repair never re-exposes a dead service, and repairing
   every event fault restores the intact graph exactly.
3. Trace determinism: the same seed regenerates the facility identically
   and replays to an identical materialized trace.
4. Replay folds: the existing ``evals.splits`` discipline holds on corpus
   replay cases — no post-fault event in the feature window, fault and
   repair inside the label window.
5. Non-clairvoyance: pre-repair trace steps are identical with and without
   the future repair in the replay input.
6. Recovery state: the constructed interruption is visible pre-repair and
   the post-repair state equals the intact baseline — engine level, and
   through the real assessment API on a subsample of examples (throwaway
   SQLite apps; a fresh app per example keeps facilities from colliding).

Each check returns structured violations carrying the failing seed and a
minimal repro; a property fails with that render, so a repro ships with the
failure. The plan's 1,000-seed gate belongs to the nightly sweep
(``python -m evals.sim.invariant_sweep``); push CI runs these property
seeds so a push stays under a few minutes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evals.corpus import SEGMENTS, FacilityCorpus, generate_segment_corpus
from evals.sim.invariant_sweep import main as sweep_main
from evals.sim.invariants import (
    InvariantResult,
    InvariantViolation,
    check_monotonicity,
    check_recovery_state,
    check_recovery_via_api,
    check_repair_recovery,
    check_replay_folds,
    check_replay_recovery,
    check_trace_determinism,
)
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from uptime_api.app import create_app

# Stable Hypothesis draw range — any seed here is a legal corpus seed.
SEEDS = st.integers(min_value=1, max_value=10_000)


def facility_for(segment: str, seed: int) -> FacilityCorpus:
    """One generated facility per (segment, seed) — the property test's subject."""
    return generate_segment_corpus(segment, seed, 1)[0]


def assert_clean(invariant: str, result: InvariantResult) -> None:
    """Fail with the violation render (seed + repro) if the check found any."""
    assert not result.violations, "\n".join(
        [f"{invariant}: {len(result.violations)} violation(s)"]
        + [violation.render() for violation in result.violations]
    )


# 1. Redundancy monotonicity --------------------------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_added_redundancy_never_increases_exposure(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("redundancy_monotonicity", check_monotonicity(facility, seed))


# 2. Repair recovery -----------------------------------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_repair_never_reexposes_and_restores(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("repair_recovery", check_repair_recovery(facility, seed))


# 3. Trace determinism ----------------------------------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_same_seed_replays_identically(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("trace_determinism", check_trace_determinism(facility, seed))


# 4. Replay folds (existing splits discipline) ----------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_replay_folds_keep_the_splits_discipline(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("replay_no_leakage", check_replay_folds(facility, seed))


# 5. Non-clairvoyance ------------------------------------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_pre_repair_states_ignore_the_future_repair(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("replay_no_clairvoyance", check_replay_recovery(facility, seed))


# 6. Recovery state: engine level, then the real API ------------------


@given(SEEDS)
@settings(max_examples=40, deadline=None)
def test_property_post_repair_equals_intact_baseline(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        assert_clean("recovery_state", check_recovery_state(facility, seed))


@given(SEEDS)
@settings(max_examples=6, deadline=None)  # each example spins up a throwaway app
def test_property_post_repair_report_matches_baseline_via_api(seed: int) -> None:
    for segment in SEGMENTS:
        facility = facility_for(segment, seed)
        with tempfile.TemporaryDirectory(prefix="aiuptime-m2-prop-") as tmp:
            app = create_app(f"sqlite:///{Path(tmp) / 'prop.db'}")
            with TestClient(app, raise_server_exceptions=False) as client:
                assert_clean("recovery_state", check_recovery_via_api(client, facility, seed))


# 7. Sweep CLI mechanics (not a property: the tool that owns the 1,000 gate) --


def test_sweep_cli_runs_clean_on_a_single_seed(tmp_path: Path) -> None:
    exit_code = sweep_main(["--seeds", "1", "--api-every", "1", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    report = (tmp_path / "invariant-sweep.md").read_text(encoding="utf-8")
    assert "PASS" in report
    assert "below the plan's 1,000-seed gate" in report


def test_sweep_cli_failure_carries_seed_and_repro(tmp_path: Path) -> None:
    failing = InvariantViolation(
        invariant="trace_determinism",
        seed=7,
        segment="cold_chain",
        facility_id="cold_chain-1",
        detail="deliberate: injected for the failure-path test",
        repro="generate_segment_corpus('cold_chain', seed=7, n_facilities=1)[-1]",
    )

    def fake_check(facility: FacilityCorpus, seed: int) -> InvariantResult:
        return InvariantResult("trace_determinism", 3, (failing,))

    exit_code = sweep_main(
        ["--seeds", "1", "--api-every", "0", "--output-dir", str(tmp_path)],
        checks=(fake_check,),
    )
    assert exit_code == 1
    report = (tmp_path / "invariant-sweep.md").read_text(encoding="utf-8")
    assert "FAIL" in report
    assert "seed=7" in report
    assert "generate_segment_corpus" in report
