"""API payload contracts for the versioned graph twin.

These are the frozen Phase 1 shapes: the frontend (Phase 2/3) renders from
this contract, not from backend internals. Every node and edge payload
carries its own provenance — a commit without provenance is invalid.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uptime_api.graph.enums import (
    AssetType,
    FailureMode,
    NodeType,
    RedundancyConfig,
    Segment,
    SourceKind,
)


class ProvenanceIn(BaseModel):
    """Provenance supplied with a new graph element."""

    source_kind: SourceKind
    source_system: str = Field(min_length=1, max_length=120)
    source_ref: str = Field(min_length=1, max_length=300)
    payload: dict[str, object] | None = None


class ProvenanceOut(BaseModel):
    """A stored provenance record, with its id so claims stay resolvable."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    source_kind: SourceKind
    source_system: str
    source_ref: str
    payload: dict[str, object] | None
    created_at: datetime


class NodeIn(BaseModel):
    """One graph node in a commit payload."""

    node_key: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,120}$")
    node_type: NodeType
    asset_type: AssetType | None = None
    name: str = Field(min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=200)
    attributes: dict[str, object] = Field(default_factory=dict)
    provenance: ProvenanceIn

    @model_validator(mode="after")
    def check_asset_type_matches_node_type(self) -> Self:
        """Asset nodes must declare a type; service nodes must not."""
        if self.node_type is NodeType.ASSET and self.asset_type is None:
            raise ValueError("asset nodes must declare asset_type")
        if self.node_type is NodeType.SERVICE and self.asset_type is not None:
            raise ValueError("service nodes must not declare asset_type")
        return self


class NodeOut(BaseModel):
    """A stored graph node with its provenance resolved."""

    model_config = ConfigDict(from_attributes=True)

    node_key: str
    node_type: NodeType
    asset_type: AssetType | None
    name: str
    location: str | None
    attributes: dict[str, object]
    provenance: ProvenanceOut


class EdgeIn(BaseModel):
    """One dependency edge in a commit payload."""

    from_node: str = Field(min_length=1, max_length=120)
    to_node: str = Field(min_length=1, max_length=120)
    failure_mode: FailureMode | None = None
    redundancy_config: RedundancyConfig | None = None
    member_group: str | None = Field(default=None, max_length=120)
    provenance: ProvenanceIn

    @model_validator(mode="after")
    def check_not_self_loop(self) -> Self:
        """A dependency of a node on itself carries no information."""
        if self.from_node == self.to_node:
            raise ValueError("self-loops are not allowed")
        return self


class EdgeOut(BaseModel):
    """A stored dependency edge with endpoints by node key."""

    from_node: str
    to_node: str
    failure_mode: FailureMode | None
    redundancy_config: RedundancyConfig | None
    member_group: str | None
    provenance: ProvenanceOut


class GraphVersionIn(BaseModel):
    """A full graph snapshot committed as a new immutable version."""

    author: str = Field(min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=500)
    nodes: list[NodeIn] = Field(min_length=1)
    edges: list[EdgeIn] = Field(default_factory=list)


class FacilityCreate(BaseModel):
    """Payload to create a facility."""

    id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,62}$")
    name: str = Field(min_length=1, max_length=200)
    segment: Segment = Segment.COLD_CHAIN


class FacilityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    segment: Segment
    created_at: datetime


class GraphVersionOut(BaseModel):
    """Version-history entry (metadata only; read the snapshot for content)."""

    version_no: int
    author: str
    notes: str | None
    created_at: datetime
    node_count: int
    edge_count: int


class GraphSnapshotOut(BaseModel):
    """An immutable graph version read back with provenance intact."""

    facility_id: str
    version_no: int
    author: str
    notes: str | None
    created_at: datetime
    nodes: list[NodeOut]
    edges: list[EdgeOut]


class UnresolvedRowOut(BaseModel):
    """An inventory row that was flagged instead of guessed."""

    row: int
    reason: str
    detail: str


class InventoryImportOut(BaseModel):
    """Result of an equipment-inventory upload that became a graph version."""

    facility_id: str
    version_no: int
    nodes_created: int
    unresolved: list[UnresolvedRowOut]
