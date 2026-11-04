"""AssessmentService — run, persist, and retrieve scanner assessments.

A run pins one graph version (the given ``version_no`` or the latest),
computes the report through the pure scoring engine, and stores the
serialized result. Retrieval serves the stored JSON, so a report stays
byte-for-byte reproducible no matter how the twin evolves afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from uptime_api.assessor.errors import AssessmentNotFound
from uptime_api.assessor.models import Assessment
from uptime_api.assessor.schemas import AssessmentReport, AssessmentSummary
from uptime_api.assessor.scoring import build_report
from uptime_api.graph.errors import GraphVersionNotFound
from uptime_api.graph.models import GraphVersion
from uptime_api.graph.service import GraphService

if TYPE_CHECKING:
    from uptime_api.forecasting.interface import ForecastModel


class AssessmentService:
    """Transactional gateway to persisted assessments."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def run_assessment(
        self,
        facility_id: str,
        author: str,
        version_no: int | None = None,
        forecaster: ForecastModel | None = None,
    ) -> tuple[Assessment, AssessmentReport]:
        """Assess one pinned graph version and persist the report."""
        graph_service = GraphService(self._session)
        graph_service.get_facility(facility_id)  # 404 semantics for the API layer
        snapshot = graph_service.get_snapshot(facility_id, version_no)

        report = build_report(snapshot, forecaster)
        assessment = Assessment(
            facility_id=facility_id,
            graph_version_id=self._version_id(facility_id, snapshot.version_no),
            graph_version_no=snapshot.version_no,
            author=author,
            result=report.model_dump(mode="json"),
        )
        self._session.add(assessment)
        self._session.flush()
        return assessment, report

    def list_assessments(self, facility_id: str) -> list[AssessmentSummary]:
        """Assessment history for a facility, oldest first."""
        GraphService(self._session).get_facility(facility_id)
        rows = self._session.scalars(
            select(Assessment)
            .where(Assessment.facility_id == facility_id)
            .order_by(Assessment.id)
        ).all()
        return [AssessmentSummary.model_validate(row) for row in rows]

    def get_report(self, facility_id: str, assessment_id: int) -> AssessmentReport:
        """The stored report for one assessment — never recomputed."""
        row = self._session.scalars(
            select(Assessment).where(
                Assessment.facility_id == facility_id,
                Assessment.id == assessment_id,
            )
        ).first()
        if row is None:
            raise AssessmentNotFound(
                f"assessment {assessment_id} not found for facility {facility_id!r}"
            )
        return AssessmentReport.model_validate(row.result)

    def _version_id(self, facility_id: str, version_no: int) -> int:
        graph_version = self._session.scalars(
            select(GraphVersion).where(
                GraphVersion.facility_id == facility_id,
                GraphVersion.version_no == version_no,
            )
        ).first()
        if graph_version is None:
            raise GraphVersionNotFound(f"facility {facility_id!r} has no version {version_no}")
        return graph_version.id
