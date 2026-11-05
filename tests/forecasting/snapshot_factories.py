"""Snapshot/edge/node factories shared by the forecasting test modules.

Bare-module import works because pytest inserts the test directory into
sys.path (no ``__init__.py`` in tests/ — the repo's test convention).
"""

from __future__ import annotations

from datetime import UTC, datetime

from uptime_api.graph.enums import (
    AssetType,
    FailureMode,
    NodeType,
    RedundancyConfig,
    SourceKind,
)
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut, NodeOut, ProvenanceOut

NOW = datetime(2026, 11, 5, tzinfo=UTC)


def provenance(ref: str = "row-1", pid: int = 1) -> ProvenanceOut:
    return ProvenanceOut(
        id=pid,
        source_kind=SourceKind.INVENTORY_IMPORT,
        source_system="test",
        source_ref=ref,
        payload=None,
        created_at=NOW,
    )


def asset(key: str, asset_type: AssetType = AssetType.COLD_ROOM) -> NodeOut:
    return NodeOut(
        node_key=key,
        node_type=NodeType.ASSET,
        asset_type=asset_type,
        name=key,
        location=None,
        attributes={},
        provenance=provenance(f"node-{key}"),
    )


def service(key: str) -> NodeOut:
    return NodeOut(
        node_key=key,
        node_type=NodeType.SERVICE,
        asset_type=None,
        name=key,
        location=None,
        attributes={},
        provenance=provenance(f"node-{key}"),
    )


def edge(
    from_node: str,
    to_node: str,
    config: RedundancyConfig | None = None,
    group: str | None = None,
) -> EdgeOut:
    return EdgeOut(
        from_node=from_node,
        to_node=to_node,
        failure_mode=FailureMode.POWER_LOSS,
        redundancy_config=config,
        member_group=group,
        provenance=provenance(f"edge-{from_node}->{to_node}"),
    )


def snapshot(nodes: list[NodeOut], edges: list[EdgeOut]) -> GraphSnapshotOut:
    return GraphSnapshotOut(
        facility_id="f1",
        version_no=1,
        author="test",
        notes=None,
        created_at=NOW,
        nodes=nodes,
        edges=edges,
    )
