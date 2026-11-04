"""Assessment report contracts (Phase 2 onboarding scanner).

These shapes follow the MVP spec's ``AssessmentReport`` interface: per-service
risk with criticality drivers, single points of failure, exposure, evidence
citations, and explicit data gaps. Every citation's ``record_id`` is a
provenance record resolvable through the graph snapshot API — a score or
finding without evidence is invalid by construction.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from uptime_api.assessor.enums import FindingKind, RiskLevel
from uptime_api.forecasting.schema import RiskForecast  # noqa: F401  (report contract)
from uptime_api.graph.enums import AssetType, RedundancyConfig, SourceKind


class EvidenceCitation(BaseModel):
    """One provenance-backed claim: resolvable via the graph snapshot API."""

    record_id: int
    kind: SourceKind
    claim: str


class CriticalityScore(BaseModel):
    """A deterministic 0-100 score with the drivers that produced it."""

    score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    drivers: list[str]


class SpofEntry(BaseModel):
    """One asset whose single failure interrupts the containing service."""

    asset_id: str
    reason: str


class Exposure(BaseModel):
    """Redundancy-aware fault tolerance for one service's supply."""

    first_fault_tolerated: bool
    concurrent_faults_tolerated: int
    unprotected_assets: list[str]


class ServiceAssessment(BaseModel):
    """Scored assessment of one logical service."""

    service_id: str
    name: str
    redundancy: RedundancyConfig
    criticality: CriticalityScore
    single_points_of_failure: list[SpofEntry]
    exposure: Exposure
    evidence: list[EvidenceCitation]
    data_gaps: list[str]
    # Hospital semantics (Phase 4): the service carries the life-safety
    # attribute. Default keeps stored pre-Phase-4 reports readable.
    life_safety: bool = False


class AssetAssessment(BaseModel):
    """Scored assessment of one physical asset."""

    node_key: str
    name: str
    asset_type: AssetType | None
    criticality: CriticalityScore
    exposed_services: list[str]
    tolerated_services: list[str]
    evidence: list[EvidenceCitation]


class Finding(BaseModel):
    """One ranked scanner finding, cited to graph provenance."""

    kind: FindingKind
    subject: str
    title: str
    detail: str
    affected_services: list[str]
    evidence: list[EvidenceCitation]


class EvidenceWindow(BaseModel):
    """Time span of evidence behind an assessment (None until time-windowed
    evidence ingestion lands — topology-only assessments carry no window)."""

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None

    model_config = {"populate_by_name": True}


class AssessmentReport(BaseModel):
    """The full, reproducible scanner output pinned to one graph version."""

    facility_id: str
    graph_version: int
    evidence_window: EvidenceWindow | None
    services: list[ServiceAssessment]
    assets: list[AssetAssessment]
    findings: list[Finding]
    data_gaps: list[str]
    open_questions: list[str]
    # Populated only when the WP2 forecasting flag is enabled (default off).
    forecasts: list[RiskForecast] = []


class AssessmentCreate(BaseModel):
    """Payload to run a new assessment (pins the given or latest version)."""

    author: str = Field(default="scanner_ui", min_length=1, max_length=120)
    version_no: int | None = None


class AssessmentSummary(BaseModel):
    """List entry: identity without the full report payload."""

    model_config = {"from_attributes": True}

    id: int
    facility_id: str
    graph_version_no: int
    created_at: datetime


class InventoryPreviewOut(BaseModel):
    """Dry-run parse result so the UI can fix unresolved rows pre-commit."""

    filename: str
    parseable_nodes: int
    unresolved: list[dict[str, object]]
