"""Frozen canonical calibration table — the runtime forecaster's constants.

Fit once from the canonical synthetic corpus through the real eval path
(corpus → API load → protection features → ``fit_calibration``) and frozen
here so the feature-flagged runtime model needs no training data. Provenance:

- corpus seed 20261104 (``CORPUS_SEED``), all 80 facilities;
- facility-level holdout ``train_eval_split(seed=20261104)`` at the
  canonical 0.25 eval fraction — 60 training facilities, 20 held out;
- cells: has_unprotected (507 cases, 507 positives), all_protected
  (341, 0), no_out_edges (106, 0).

Synthetic-only by construction: these posteriors describe the corpus's
declared-redundancy physics, not field failure rates — every forecast built
on them says so in its ``basis``, and the eval harness re-fits its own table
from its own train split rather than trusting these constants (a test pins
that re-fitting reproduces this table exactly, so constants and corpus
cannot drift apart silently).
"""

from __future__ import annotations

from uptime_api.forecasting.calibration import CalibrationCell, CalibrationTable
from uptime_api.forecasting.features import ProtectionStatus

_CELL_CASES: dict[ProtectionStatus, tuple[int, int]] = {
    ProtectionStatus.HAS_UNPROTECTED: (507, 507),
    ProtectionStatus.ALL_PROTECTED: (341, 0),
    ProtectionStatus.NO_OUT_EDGES: (106, 0),
}


def _canonical_table() -> CalibrationTable:
    cells = {
        status: CalibrationCell(status=status, cases=cases, positives=positives)
        for status, (cases, positives) in _CELL_CASES.items()
    }
    return CalibrationTable(seed=20261104, train_facilities=60, cells=cells)


CANONICAL_TABLE = _canonical_table()
