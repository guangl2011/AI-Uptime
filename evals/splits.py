"""No-leakage time-split fold utilities (testing plan: ``MVP (fold utilities)``).

The split discipline, enforced in code rather than prose: every replay case
gets a feature window ``[feature_start, fault_time)`` and a label window
``[fault_time, repair_time]``; a feature is any event inside the feature
window. Two invariants hold for every fold:

1. **Window discipline** — no event at or after ``fault_time`` is ever
   returned as a feature (you cannot use the future to predict the fault).
2. **No post-repair leakage** — no event timestamped at or after a case's
   ``repair_time`` appears in that case's pre-repair feature window. The
   corpus deliberately logs post-repair telemetry, so a naive
   "all events for this asset" fold WOULD leak; these utilities are what
   keeps that from ever happening.

Scored replay over these folds arrives with WP2 state estimation; the MVP
runs the fold construction and its invariants as an eval gate so the
discipline is already load-bearing before any learned component depends on
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from evals.corpus import CorpusEvent, ReplayCase


class LeakageError(ValueError):
    """A fold construction would leak post-fault/post-repair information."""


@dataclass(frozen=True)
class FoldCase:
    """One replay case's evaluation fold (feature and label windows)."""

    case_id: str
    facility_id: str
    asset_key: str
    feature_start: datetime
    fault_time: datetime
    repair_time: datetime

    @classmethod
    def from_replay(cls, case: ReplayCase, feature_window: timedelta) -> FoldCase:
        """Build a fold whose feature window ends at the fault time."""
        if feature_window <= timedelta(0):
            raise ValueError("feature window must be positive")
        if case.repair_time <= case.fault_time:
            raise ValueError("repair must follow the fault")
        return cls(
            case_id=case.case_id,
            facility_id=case.facility_id,
            asset_key=case.asset_key,
            feature_start=case.fault_time - feature_window,
            fault_time=case.fault_time,
            repair_time=case.repair_time,
        )


@dataclass(frozen=True)
class LeakageCheck:
    """Result of checking every fold against the no-leakage invariants."""

    cases: int
    feature_events_used: int
    post_repair_events_excluded: int
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.violations


def feature_events(events: Sequence[CorpusEvent], case: FoldCase) -> list[CorpusEvent]:
    """Events usable as features for one fold, window-disciplined.

    Returns only events of the case's asset in ``[feature_start, fault_time)``.
    Raises ``LeakageError`` if an in-window candidate is timestamped at or
    after the case's repair — the testing plan's post-repair rule, enforced
    here so a mis-built fold cannot silently score leaked features.
    """
    features: list[CorpusEvent] = []
    for event in events:
        if event.asset_key != case.asset_key or event.facility_id != case.facility_id:
            continue
        if not (case.feature_start <= event.occurred_at < case.fault_time):
            continue
        if event.occurred_at >= case.repair_time:
            raise LeakageError(
                f"post-repair event {event.event_id} "
                f"({event.occurred_at.isoformat()}) inside the feature window "
                f"of case {case.case_id} — repair at {case.repair_time.isoformat()}"
            )
        features.append(event)
    return features


def label_window_events(events: Sequence[CorpusEvent], case: FoldCase) -> list[CorpusEvent]:
    """Events in the label window ``[fault_time, repair_time]`` of one fold."""
    return [
        event
        for event in events
        if event.asset_key == case.asset_key
        and event.facility_id == case.facility_id
        and case.fault_time <= event.occurred_at <= case.repair_time
    ]


def check_no_leakage(
    events: Sequence[CorpusEvent], cases: Sequence[FoldCase]
) -> LeakageCheck:
    """Check every fold against both no-leakage invariants.

    Counts, per fold, the feature events actually used and the post-repair
    events the window discipline excluded; every exclusion that would have
    been a feature under naive asset-level grouping is the leak this check
    exists to prevent. Returns the accumulated violations (empty = pass).
    """
    violations: list[str] = []
    features_used = 0
    post_repair_excluded = 0
    for case in cases:
        asset_events = [
            event
            for event in events
            if event.asset_key == case.asset_key and event.facility_id == case.facility_id
        ]
        try:
            features = feature_events(asset_events, case)
        except LeakageError as error:
            violations.append(str(error))
            continue
        features_used += len(features)
        post_repair_excluded += sum(
            1 for event in asset_events if event.occurred_at >= case.repair_time
        )
        # Invariant 1 restated against the result: no feature at/after fault.
        leaked = [event for event in features if event.occurred_at >= case.fault_time]
        for event in leaked:
            violations.append(
                f"case {case.case_id}: future event {event.event_id} in feature window"
            )
    return LeakageCheck(
        cases=len(cases),
        feature_events_used=features_used,
        post_repair_events_excluded=post_repair_excluded,
        violations=tuple(violations),
    )


def build_folds(
    replay_cases: Sequence[ReplayCase], feature_window: timedelta
) -> list[FoldCase]:
    """Fold every replay case; a zero/negative window or backwards episode
    raises (loudly) rather than producing a leaky fold."""
    return [FoldCase.from_replay(case, feature_window) for case in replay_cases]


def assert_no_leakage(events: Sequence[CorpusEvent], cases: Sequence[FoldCase]) -> LeakageCheck:
    """``check_no_leakage`` that raises on any violation (for direct callers)."""
    check = check_no_leakage(events, cases)
    if not check.ok:
        raise LeakageError("; ".join(check.violations))
    return check
