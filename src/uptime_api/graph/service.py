"""GraphService — commit immutable graph versions, read them back.

Every commit is a full snapshot: the payload becomes the next version_no for
the facility and nothing older is touched. Provenance rows are created for
every node and edge in the same transaction, so a version can never exist
with an untraceable element.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from uptime_api.graph.enums import Segment
from uptime_api.graph.errors import (
    DuplicateNodeKey,
    FacilityAlreadyExists,
    FacilityNotFound,
    GraphVersionNotFound,
    UnknownEdgeEndpoint,
)
from uptime_api.graph.models import (
    Facility,
    GraphEdge,
    GraphNode,
    GraphVersion,
    ProvenanceRecord,
)
from uptime_api.graph.schemas import (
    EdgeIn,
    EdgeOut,
    GraphSnapshotOut,
    GraphVersionIn,
    GraphVersionOut,
    NodeIn,
    NodeOut,
    ProvenanceIn,
    ProvenanceOut,
)

_GRAPH_EAGER = (
    selectinload(GraphVersion.nodes).selectinload(GraphNode.provenance),
    selectinload(GraphVersion.edges).selectinload(GraphEdge.provenance),
)


class GraphService:
    """Transactional gateway to the versioned twin."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- facilities ---------------------------------------------------------

    def create_facility(self, facility_id: str, name: str, segment: Segment) -> Facility:
        """Create a facility; ids are caller-supplied so seeds are deterministic."""
        if self._session.get(Facility, facility_id) is not None:
            raise FacilityAlreadyExists(f"facility {facility_id!r} already exists")
        facility = Facility(id=facility_id, name=name, segment=segment)
        self._session.add(facility)
        self._session.flush()
        return facility

    def get_facility(self, facility_id: str) -> Facility:
        """Fetch a facility or raise FacilityNotFound."""
        facility = self._session.get(Facility, facility_id)
        if facility is None:
            raise FacilityNotFound(f"facility {facility_id!r} not found")
        return facility

    def list_facilities(self) -> list[Facility]:
        """All facilities, oldest first."""
        return list(
            self._session.scalars(select(Facility).order_by(Facility.created_at)).all()
        )

    # -- versions -----------------------------------------------------------

    def commit_version(self, facility_id: str, payload: GraphVersionIn) -> GraphVersion:
        """Commit a full graph snapshot as the facility's next immutable version."""
        self.get_facility(facility_id)

        node_keys = [node.node_key for node in payload.nodes]
        if len(set(node_keys)) != len(node_keys):
            raise DuplicateNodeKey("commit payload contains duplicate node keys")

        edge_endpoints = {edge.from_node for edge in payload.edges} | {
            edge.to_node for edge in payload.edges
        }
        unknown = edge_endpoints - set(node_keys)
        if unknown:
            raise UnknownEdgeEndpoint(f"edges reference unknown nodes: {sorted(unknown)}")

        version = GraphVersion(
            facility_id=facility_id,
            version_no=self._next_version_no(facility_id),
            author=payload.author,
            notes=payload.notes,
        )
        self._session.add(version)
        self._session.flush()  # assign version.id

        node_ids = {node.node_key: self._add_node(version, node) for node in payload.nodes}
        for edge in payload.edges:
            self._add_edge(version, edge, node_ids[edge.from_node], node_ids[edge.to_node])

        self._session.flush()
        return version

    def list_versions(self, facility_id: str) -> list[GraphVersionOut]:
        """Version history for a facility, oldest first."""
        self.get_facility(facility_id)
        versions = self._session.scalars(
            select(GraphVersion)
            .where(GraphVersion.facility_id == facility_id)
            .options(*_GRAPH_EAGER)
            .order_by(GraphVersion.version_no)
        ).all()
        return [_version_out(version) for version in versions]

    def get_snapshot(self, facility_id: str, version_no: int | None = None) -> GraphSnapshotOut:
        """Read one immutable version (latest when version_no is None)."""
        self.get_facility(facility_id)
        stmt = (
            select(GraphVersion)
            .where(GraphVersion.facility_id == facility_id)
            .options(*_GRAPH_EAGER)
            .order_by(GraphVersion.version_no.desc())
        )
        if version_no is not None:
            stmt = stmt.where(GraphVersion.version_no == version_no)
        version = self._session.scalars(stmt).first()
        if version is None:
            raise GraphVersionNotFound(
                f"facility {facility_id!r} has no graph version"
                if version_no is None
                else f"facility {facility_id!r} has no version {version_no}"
            )
        return GraphSnapshotOut(
            facility_id=facility_id,
            version_no=version.version_no,
            author=version.author,
            notes=version.notes,
            created_at=version.created_at,
            nodes=[_node_out(node) for node in version.nodes],
            edges=[_edge_out(edge) for edge in version.edges],
        )

    # -- internals ----------------------------------------------------------

    def _next_version_no(self, facility_id: str) -> int:
        current_max = self._session.scalar(
            select(func.max(GraphVersion.version_no)).where(
                GraphVersion.facility_id == facility_id
            )
        )
        return (current_max or 0) + 1

    def _add_node(self, version: GraphVersion, node: NodeIn) -> int:
        """Persist one node with its provenance row; returns the node id."""
        provenance = self._add_provenance(version.facility_id, node.provenance)
        graph_node = GraphNode(
            graph_version_id=version.id,
            node_key=node.node_key,
            node_type=node.node_type,
            asset_type=node.asset_type,
            name=node.name,
            location=node.location,
            attributes=node.attributes,
            provenance_id=provenance.id,
        )
        self._session.add(graph_node)
        self._session.flush()
        return graph_node.id

    def _add_edge(self, version: GraphVersion, edge: EdgeIn, from_id: int, to_id: int) -> None:
        """Persist one edge with its provenance row."""
        provenance = self._add_provenance(version.facility_id, edge.provenance)
        self._session.add(
            GraphEdge(
                graph_version_id=version.id,
                from_node_id=from_id,
                to_node_id=to_id,
                failure_mode=edge.failure_mode,
                redundancy_config=edge.redundancy_config,
                member_group=edge.member_group,
                provenance_id=provenance.id,
            )
        )

    def _add_provenance(self, facility_id: str, provenance: ProvenanceIn) -> ProvenanceRecord:
        record = ProvenanceRecord(
            facility_id=facility_id,
            source_kind=provenance.source_kind,
            source_system=provenance.source_system,
            source_ref=provenance.source_ref,
            payload=provenance.payload,
        )
        self._session.add(record)
        self._session.flush()
        return record


def _version_out(version: GraphVersion) -> GraphVersionOut:
    return GraphVersionOut(
        version_no=version.version_no,
        author=version.author,
        notes=version.notes,
        created_at=version.created_at,
        node_count=len(version.nodes),
        edge_count=len(version.edges),
    )


def _node_out(node: GraphNode) -> NodeOut:
    return NodeOut(
        node_key=node.node_key,
        node_type=node.node_type,
        asset_type=node.asset_type,
        name=node.name,
        location=node.location,
        attributes=node.attributes,
        provenance=ProvenanceOut.model_validate(node.provenance),
    )


def _edge_out(edge: GraphEdge) -> EdgeOut:
    return EdgeOut(
        from_node=edge.from_node.node_key,
        to_node=edge.to_node.node_key,
        failure_mode=edge.failure_mode,
        redundancy_config=edge.redundancy_config,
        member_group=edge.member_group,
        provenance=ProvenanceOut.model_validate(edge.provenance),
    )
