"""Persisted assessments: reproducible reports pinned to a graph version.

The report is stored as serialized JSON next to the ``graph_version_id`` it
was computed from — reading an assessment never re-runs the engine, so a
report is byte-for-byte reproducible even after the twin moves on.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from uptime_api.graph.models import Base, utcnow


class Assessment(Base):
    """One scanner run over one pinned graph version."""

    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(ForeignKey("facilities.id"), index=True)
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("graph_versions.id"), index=True)
    graph_version_no: Mapped[int]
    author: Mapped[str]
    result: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
