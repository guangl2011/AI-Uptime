"""Graph service tests: versioning, immutability, and provenance guarantees."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from uptime_api.graph.enums import (
    RedundancyConfig,
    Segment,
    SourceKind,
)
from uptime_api.graph.errors import (
    DuplicateNodeKey,
    FacilityAlreadyExists,
    FacilityNotFound,
    UnknownEdgeEndpoint,
)
from uptime_api.graph.models import GraphEdge, GraphNode, ProvenanceRecord
from uptime_api.graph.schemas import EdgeIn, GraphVersionIn, NodeIn, ProvenanceIn
from uptime_api.graph.service import GraphService


def _provenance(ref: str, kind: SourceKind = SourceKind.TECHNICIAN_ENTRY) -> ProvenanceIn:
    return ProvenanceIn(source_kind=kind, source_system="test", source_ref=ref)


def _node(key: str, **kwargs: Any) -> NodeIn:
    params: dict[str, Any] = {
        "node_key": key,
        "node_type": "asset",
        "asset_type": "compressor",
        "name": f"node {key}",
        "provenance": _provenance(f"src#{key}"),
    }
    params.update(kwargs)
    return NodeIn(**params)


def _edge(from_key: str, to_key: str, **kwargs: Any) -> EdgeIn:
    params: dict[str, Any] = {
        "from_node": from_key,
        "to_node": to_key,
        "provenance": _provenance(f"edge#{from_key}->{to_key}"),
    }
    params.update(kwargs)
    return EdgeIn(**params)


def _commit(service: GraphService, facility_id: str, nodes: list[NodeIn]) -> int:
    version = service.commit_version(facility_id, GraphVersionIn(author="test", nodes=nodes))
    return version.version_no


def test_create_facility_rejects_duplicate_ids(db_session: Session) -> None:
    service = GraphService(db_session)
    service.create_facility("f1", "Facility One", Segment.COLD_CHAIN)
    with pytest.raises(FacilityAlreadyExists):
        service.create_facility("f1", "Facility One Again", Segment.COLD_CHAIN)


def test_unknown_facility_raises(db_session: Session) -> None:
    service = GraphService(db_session)
    with pytest.raises(FacilityNotFound):
        service.get_facility("missing")


def test_versions_are_sequential_and_older_versions_immutable(db_session: Session) -> None:
    service = GraphService(db_session)
    service.create_facility("f1", "Facility One", Segment.COLD_CHAIN)

    first = _commit(service, "f1", [_node("A"), _node("B")])
    second = _commit(service, "f1", [_node("A"), _node("B"), _node("C")])
    assert (first, second) == (1, 2)

    # The pinned v1 read must show exactly what v1 committed, even after v2.
    v1 = service.get_snapshot("f1", 1)
    v2 = service.get_snapshot("f1", 2)
    assert {n.node_key for n in v1.nodes} == {"A", "B"}
    assert {n.node_key for n in v2.nodes} == {"A", "B", "C"}
    assert {v.version_no for v in service.list_versions("f1")} == {1, 2}


def test_every_committed_element_carries_provenance(db_session: Session) -> None:
    service = GraphService(db_session)
    service.create_facility("f1", "Facility One", Segment.COLD_CHAIN)
    service.commit_version(
        "f1",
        GraphVersionIn(
            author="test",
            nodes=[
                _node("PUMP-1"),
                _node("SVC-1", node_type="service", asset_type=None),
            ],
            edges=[
                _edge(
                    "PUMP-1",
                    "SVC-1",
                    failure_mode="mechanical_failure",
                    redundancy_config=RedundancyConfig.N,
                )
            ],
        ),
    )

    nodes = db_session.query(GraphNode).all()
    edges = db_session.query(GraphEdge).all()
    assert len(nodes) == 2 and len(edges) == 1
    for element in [*nodes, *edges]:
        record = db_session.get(ProvenanceRecord, element.provenance_id)
        assert record is not None, f"untraceable element: {element}"
        assert record.source_system == "test"
        assert record.source_ref


def test_duplicate_node_keys_rejected(db_session: Session) -> None:
    service = GraphService(db_session)
    service.create_facility("f1", "Facility One", Segment.COLD_CHAIN)
    with pytest.raises(DuplicateNodeKey):
        service.commit_version(
            "f1", GraphVersionIn(author="test", nodes=[_node("A"), _node("A")])
        )


def test_edges_may_not_reference_unknown_nodes(db_session: Session) -> None:
    service = GraphService(db_session)
    service.create_facility("f1", "Facility One", Segment.COLD_CHAIN)
    with pytest.raises(UnknownEdgeEndpoint):
        service.commit_version(
            "f1",
            GraphVersionIn(author="test", nodes=[_node("A")], edges=[_edge("A", "GHOST")]),
        )


def test_asset_nodes_require_asset_type_and_services_forbid_it() -> None:
    with pytest.raises(ValidationError):
        NodeIn(
            node_key="X",
            node_type="asset",
            name="no type",
            provenance=_provenance("src"),
        )
    with pytest.raises(ValidationError):
        NodeIn(
            node_key="SVC",
            node_type="service",
            asset_type="compressor",
            name="service with a type",
            provenance=_provenance("src"),
        )


def test_self_loops_rejected() -> None:
    with pytest.raises(ValidationError):
        EdgeIn(from_node="A", to_node="A", provenance=_provenance("src"))
