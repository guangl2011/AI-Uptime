"""Assessor-layer errors, translated to HTTP status codes in the API layer."""

from __future__ import annotations


class AssessmentNotFound(Exception):
    """No assessment with this id for the facility."""
