"""Enumerations for the assessor layer (Phase 2 onboarding scanner)."""

from __future__ import annotations

import enum


class RiskLevel(enum.StrEnum):
    """Banded criticality: high >= 70, elevated >= 30, low below."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


class FindingKind(enum.StrEnum):
    """Deterministic SPOF / redundancy findings the scanner emits."""

    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    MISSING_REDUNDANCY = "missing_redundancy"
    UNSERVED_SERVICE = "unserved_service"
    LIFE_SAFETY_ISOLATION = "life_safety_isolation"
