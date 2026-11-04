"""No-leakage split tests: the window discipline, enforced in code.

The testing plan's leakage row is a dedicated test: no event timestamped at
or after a repair may appear in a pre-repair feature window, across seeds.
These tests build folds over the corpus event logs and over hand-made
adversarial cases that would leak if the utilities were naive.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from evals.corpus import CORPUS_SEED, CorpusEvent, ReplayCase, generate_corpus
from evals.harness import FEATURE_WINDOW
from evals.splits import (
    FoldCase,
    LeakageError,
    assert_no_leakage,
    build_folds,
    check_no_leakage,
    feature_events,
    label_window_events,
)

FAULT = datetime(2026, 9, 10, 6, 0, tzinfo=UTC)
REPAIR = FAULT + timedelta(days=1)


def _event(event_id: str, at: datetime, asset_key: str = "RACK-1") -> CorpusEvent:
    return CorpusEvent(
        event_id=event_id,
        facility_id="f1",
        asset_key=asset_key,
        occurred_at=at,
        kind="sensor",
    )


def _case(feature_start_offset: timedelta = timedelta(days=7)) -> FoldCase:
    replay = ReplayCase(
        case_id="f1:RACK-1",
        facility_id="f1",
        asset_key="RACK-1",
        interrupted_service="SVC-REFRIG-0",
        fault_time=FAULT,
        repair_time=REPAIR,
    )
    return FoldCase.from_replay(replay, feature_start_offset)


def test_fold_windows_end_at_the_fault() -> None:
    case = _case()
    assert case.feature_start == FAULT - FEATURE_WINDOW
    assert case.fault_time == FAULT
    assert case.repair_time == REPAIR


def test_fold_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        _case(feature_start_offset=timedelta(0))
    replay = ReplayCase(
        case_id="bad",
        facility_id="f1",
        asset_key="RACK-1",
        interrupted_service="SVC-REFRIG-0",
        fault_time=REPAIR,
        repair_time=FAULT,  # repair before fault
    )
    with pytest.raises(ValueError, match="repair must follow"):
        FoldCase.from_replay(replay, FEATURE_WINDOW)


def test_feature_events_exclude_fault_time_and_later() -> None:
    case = _case()
    events = [
        _event("e1", case.feature_start),  # boundary: inside window (inclusive)
        _event("e2", case.fault_time - timedelta(minutes=1)),  # last allowed
        _event("e3", case.fault_time),  # the fault itself: excluded
        _event("e4", REPAIR + timedelta(days=1)),  # post-repair: excluded
    ]
    features = feature_events(events, case)
    assert [event.event_id for event in features] == ["e1", "e2"]


def test_feature_events_filter_by_asset() -> None:
    case = _case()
    other_asset = _event("other", case.fault_time - timedelta(hours=1), asset_key="UPS-1")
    assert feature_events([other_asset], case) == []


def test_feature_events_raise_on_repair_inside_window() -> None:
    """A repair sitting inside the feature window cannot be scored — raise.

    Unreachable for valid folds (repair always follows the fault), which is
    the point: the guard exists for malformed direct-built folds (e.g. a
corrupt clock putting the repair before the fault) so they fail loudly
    instead of silently scoring leaked features.
    """
    case = FoldCase(
        case_id="f1:RACK-1",
        facility_id="f1",
        asset_key="RACK-1",
        feature_start=FAULT - FEATURE_WINDOW,
        fault_time=FAULT,
        repair_time=FAULT - timedelta(days=3),  # malformed: repair before fault
    )
    events = [_event("leak", FAULT - timedelta(days=2))]  # in window, after "repair"
    with pytest.raises(LeakageError, match="post-repair event"):
        feature_events(events, case)


def test_label_window_contains_fault_through_repair() -> None:
    case = _case()
    events = [
        _event("pre", case.fault_time - timedelta(seconds=1)),
        _event("fault", case.fault_time),
        _event("mid", case.fault_time + timedelta(hours=12)),
        _event("repair", case.repair_time),  # boundary: inside (inclusive)
        _event("after", case.repair_time + timedelta(seconds=1)),
    ]
    assert [event.event_id for event in label_window_events(events, case)] == [
        "fault",
        "mid",
        "repair",
    ]


def test_check_no_leakage_counts_post_repair_exclusions() -> None:
    case = _case()
    events = [
        _event("pre1", FAULT - timedelta(hours=3)),
        _event("pre2", FAULT - timedelta(hours=1)),
        _event("post1", REPAIR + timedelta(hours=1)),
        _event("post2", REPAIR + timedelta(hours=2)),
    ]
    check = check_no_leakage(events, [case])
    assert check.ok
    assert check.cases == 1
    assert check.feature_events_used == 2
    assert check.post_repair_events_excluded == 2
    assert check.violations == ()


def test_check_no_leakage_catches_malformed_folds() -> None:
    """A fold whose repair sits inside its feature window is a violation,
    not silently scored — the aggregated check reports it."""
    case = FoldCase(
        case_id="f1:RACK-1",
        facility_id="f1",
        asset_key="RACK-1",
        feature_start=FAULT - FEATURE_WINDOW,
        fault_time=FAULT,
        repair_time=FAULT - timedelta(days=3),  # malformed: repair before fault
    )
    events = [_event("leak", FAULT - timedelta(days=2))]
    check = check_no_leakage(events, [case])
    assert not check.ok
    assert any("post-repair event" in violation for violation in check.violations)


def test_assert_no_leakage_raises_on_violations() -> None:
    case = FoldCase(
        case_id="f1:RACK-1",
        facility_id="f1",
        asset_key="RACK-1",
        feature_start=FAULT - FEATURE_WINDOW,
        fault_time=FAULT,
        repair_time=FAULT - timedelta(days=3),  # malformed: repair before fault
    )
    with pytest.raises(LeakageError):
        assert_no_leakage([_event("leak", FAULT - timedelta(days=2))], [case])


def test_corpus_folds_are_leak_free_across_seeds() -> None:
    """The dedicated leakage test: across generated seeds, zero violations."""
    for seed in (CORPUS_SEED, CORPUS_SEED + 1, CORPUS_SEED + 2):
        corpus = generate_corpus(seed, 10)
        folds = [
            fold
            for facility in corpus.facilities
            for fold in build_folds(facility.replay_cases, FEATURE_WINDOW)
        ]
        events = [event for facility in corpus.facilities for event in facility.events]
        check = check_no_leakage(events, folds)
        assert check.ok, (seed, check.violations)
        # Every fold actually used pre-fault features (windows are populated).
        assert check.feature_events_used >= len(folds)
