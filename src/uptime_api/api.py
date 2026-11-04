"""HTTP surface — Phase 0 stub.

Route groups land here per the MVP spec (art_SSJ84EWf): onboarding/ingestion,
the versioned graph twin, and assessments. Phase 0 ships only the liveness
probe so CI and the demo runbook have a real surface to exercise.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
