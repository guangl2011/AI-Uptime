"""SQLAlchemy models for the versioned facility graph (WP1 core).

Versioning model: every commit produces a NEW immutable graph version holding
a full snapshot of nodes and edges. Nothing in an older version is ever
mutated, so twin history stays reconstructable and a later assessment can pin
a version and be reproducible. Every node and edge carries a provenance
reference — where that element and its criticality value came from.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from uptime_api.graph.enums import (
    AssetType,
    FailureMode,
    NodeType,
    RedundancyConfig,
    Segment,
    SourceKind,
)


def utcnow() -> datetime:
    """Timestamp helper — always UTC, always aware."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all AI-Uptime tables."""


class Facility(Base):
    """One site (a cold-chain warehouse first, then AIDC/hospital examples)."""

    __tablename__ = "facilities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    segment: Mapped[Segment] = mapped_column(
        Enum(Segment, values_callable=lambda e: [m.value for m in e]),
        default=Segment.COLD_CHAIN,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    versions: Mapped[list[GraphVersion]] = relationship(
        back_populates="facility", order_by="GraphVersion.version_no"
    )


class ProvenanceRecord(Base):
    """Where one graph element (or one inventory row) came from.

    One row per node/edge claim: source kind, system, and a resolvable
    reference (e.g. ``inventory.csv#row=12`` or ``OEM datasheet p.4``).
    """

    __tablename__ = "provenance_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(ForeignKey("facilities.id"), index=True)
    source_kind: Mapped[SourceKind] = mapped_column(
        Enum(SourceKind, values_callable=lambda e: [m.value for m in e])
    )
    source_system: Mapped[str] = mapped_column(String(120))
    source_ref: Mapped[str] = mapped_column(String(300))
    payload: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GraphVersion(Base):
    """An immutable snapshot of the twin for one facility (version_no 1, 2, ...)."""

    __tablename__ = "graph_versions"
    __table_args__ = (UniqueConstraint("facility_id", "version_no", name="uq_facility_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(ForeignKey("facilities.id"), index=True)
    version_no: Mapped[int]
    author: Mapped[str] = mapped_column(String(120))
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    facility: Mapped[Facility] = relationship(back_populates="versions")
    nodes: Mapped[list[GraphNode]] = relationship(
        back_populates="graph_version", cascade="all, delete-orphan"
    )
    edges: Mapped[list[GraphEdge]] = relationship(
        back_populates="graph_version", cascade="all, delete-orphan"
    )


class GraphNode(Base):
    """An asset or service node within one graph version."""

    __tablename__ = "graph_nodes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("graph_versions.id"), index=True)
    node_key: Mapped[str] = mapped_column(String(120))
    node_type: Mapped[NodeType] = mapped_column(
        Enum(NodeType, values_callable=lambda e: [m.value for m in e])
    )
    asset_type: Mapped[AssetType | None] = mapped_column(
        Enum(AssetType, values_callable=lambda e: [m.value for m in e]), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200))
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attributes: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    provenance_id: Mapped[int] = mapped_column(ForeignKey("provenance_records.id"))

    graph_version: Mapped[GraphVersion] = relationship(back_populates="nodes")
    provenance: Mapped[ProvenanceRecord] = relationship()


class GraphEdge(Base):
    """A typed dependency between two nodes within one graph version.

    Redundancy membership lives on the edge: edges sharing a ``member_group``
    form a redundant set whose config is ``redundancy_config`` (N / N+1 / 2N).
    """

    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("graph_versions.id"), index=True)
    from_node_id: Mapped[int] = mapped_column(ForeignKey("graph_nodes.id"), index=True)
    to_node_id: Mapped[int] = mapped_column(ForeignKey("graph_nodes.id"), index=True)
    failure_mode: Mapped[FailureMode | None] = mapped_column(
        Enum(FailureMode, values_callable=lambda e: [m.value for m in e]), nullable=True
    )
    redundancy_config: Mapped[RedundancyConfig | None] = mapped_column(
        Enum(RedundancyConfig, values_callable=lambda e: [m.value for m in e]), nullable=True
    )
    member_group: Mapped[str | None] = mapped_column(String(120), nullable=True)
    provenance_id: Mapped[int] = mapped_column(ForeignKey("provenance_records.id"))

    graph_version: Mapped[GraphVersion] = relationship(back_populates="edges")
    from_node: Mapped[GraphNode] = relationship(foreign_keys=[from_node_id])
    to_node: Mapped[GraphNode] = relationship(foreign_keys=[to_node_id])
    provenance: Mapped[ProvenanceRecord] = relationship()
