"""The drift pin: the frozen canonical table must equal a fresh refit.

These tests re-derive the canonical calibration constants from the corpus
through the real eval path — corpus generation, API loading, the shared
facility-level split, feature extraction, and ``fit_calibration`` — and
compare against the frozen values in ``forecasting/canonical.py``. If the
corpus, split, or feature logic changes, this pin fails and the constants
must be regenerated deliberately (not silently absorbed). Slow-ish by
design: it runs the real corpus once per module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from evals.corpus import CORPUS_SEED, generate_corpus
from evals.harness import load_facility
from fastapi.testclient import TestClient

from uptime_api.app import create_app
from uptime_api.forecasting.calibration import (
    CANONICAL_EVAL_FRACTION,
    CalibrationCase,
    CalibrationTable,
    fit_calibration,
    train_eval_split,
)
from uptime_api.forecasting.canonical import _CELL_CASES, CANONICAL_TABLE
from uptime_api.forecasting.features import (
    ProtectionStatus,
    asset_protection_features,
)
from uptime_api.graph.schemas import GraphSnapshotOut


@pytest.fixture(scope="module")
def refit_table() -> CalibrationTable:
    """Refit the calibration table from the canonical corpus, once."""
    corpus = generate_corpus(CORPUS_SEED, None)
    train_ids, _eval_ids = train_eval_split(
        [facility.facility_id for facility in corpus.facilities],
        seed=CORPUS_SEED,
        eval_fraction=CANONICAL_EVAL_FRACTION,
    )
    cases: list[CalibrationCase] = []
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(f"sqlite:///{Path(tmp) / 'refit.db'}")
        with TestClient(app) as client:
            for facility in corpus.facilities:
                report = load_facility(client, facility)
                version = report["graph_version"]
                assert isinstance(version, int)
                response = client.get(f"/facilities/{facility.facility_id}/graph?version={version}")
                assert response.status_code == 200, response.text
                if facility.facility_id not in train_ids:
                    continue
                snapshot = GraphSnapshotOut.model_validate(response.json())
                for feature in asset_protection_features(snapshot):
                    cases.append(
                        CalibrationCase(
                            facility_id=facility.facility_id,
                            status=feature.status,
                            label=1 if facility.expected_exposed[feature.asset_key] else 0,
                        )
                    )
    return fit_calibration(cases, seed=CORPUS_SEED, train_facilities=len(train_ids))


def test_canonical_table_equals_fresh_refit(refit_table: CalibrationTable) -> None:
    assert CANONICAL_TABLE.seed == refit_table.seed
    assert CANONICAL_TABLE.train_facilities == refit_table.train_facilities
    for status in ProtectionStatus:
        assert CANONICAL_TABLE.cell_for(status) == refit_table.cell_for(status)


def test_canonical_cell_counts_match_the_documented_fit() -> None:
    cells = {
        status: (cell.cases if (cell := CANONICAL_TABLE.cell_for(status)) is not None else 0)
        for status in ProtectionStatus
    }
    assert cells == {
        ProtectionStatus.HAS_UNPROTECTED: 507,
        ProtectionStatus.ALL_PROTECTED: 341,
        ProtectionStatus.NO_OUT_EDGES: 106,
    }
    total = sum(cases for cases, _positives in _CELL_CASES.values())
    assert total == 954  # 507 + 341 + 106 training cases from 60 facilities
