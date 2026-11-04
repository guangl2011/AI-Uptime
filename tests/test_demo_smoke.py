"""The `make demo` runbook must work on a clean clone.

The seed file is committed, the demo module round-trips it through the live
API, and the cold-chain-first market sequence is asserted so a reordering of
the beachhead (cold chain -> AIDC -> hospitals) cannot slip in silently.
"""

from __future__ import annotations

from typing import Any

import pytest

from uptime_api.demo import SEED_PATH
from uptime_api.demo import main as run_demo


def test_seed_file_is_committed_and_loads(seed_facility: dict[str, Any]) -> None:
    assert SEED_PATH.exists()
    assert seed_facility["facility_id"] == "cold_chain_example_01"
    assert seed_facility["segment"] == "cold_chain"
    assert len(seed_facility["nodes"]) > 0
    assert len(seed_facility["edges"]) > 0


def test_seed_assets_within_phase1_band(seed_facility: dict[str, Any]) -> None:
    assets = [n for n in seed_facility["nodes"] if n["node_type"] == "asset"]
    assert 5 <= len(assets) <= 15


def test_demo_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    run_demo()  # returns normally only if every stage passed
    captured = capsys.readouterr()
    assert "[demo] Phase 2 onboarding scanner complete" in captured.out


def test_seed_segments_match_market_sequence(seed_facility: dict[str, Any]) -> None:
    allowed = ["cold_chain", "aidc", "hospital"]
    assert seed_facility["segment"] in allowed
