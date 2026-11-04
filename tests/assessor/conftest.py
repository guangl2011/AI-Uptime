"""Fixtures for pure assessor tests: graph-snapshot builders.

Builders are exposed through one namespace fixture (``b``) so tests read
``b.asset("X")`` without any cross-file imports — the tests directory is not
a package, mirroring the rest of the tree.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from uptime_api.graph.enums import AssetType, NodeType, RedundancyConfig, SourceKind
from uptime_api.graph.schemas import (
    EdgeOut,
    GraphSnapshotOut,
    NodeOut,
    ProvenanceOut,
)

_PROV_TIME = datetime(2026, 11, 1, tzinfo=UTC)
_counter = {"provenance": 0}


def _provenance(ref: str) -> ProvenanceOut:
    """A synthetic but resolvable provenance record (unique id per call)."""
    _counter["provenance"] += 1
    return ProvenanceOut(
        id=_counter["provenance"],
        source_kind=SourceKind.TECHNICIAN_ENTRY,
        source_system="test_corpus",
        source_ref=ref,
        payload=None,
        created_at=_PROV_TIME,
    )


def _asset(key: str, asset_type: AssetType = AssetType.UTILITY_FEED) -> NodeOut:
    return NodeOut(
        node_key=key,
        node_type=NodeType.ASSET,
        asset_type=asset_type,
        name=key,
        location=None,
        attributes={},
        provenance=_provenance(f"asset:{key}"),
    )


def _service(key: str) -> NodeOut:
    return NodeOut(
        node_key=key,
        node_type=NodeType.SERVICE,
        asset_type=None,
        name=key,
        location=None,
        attributes={},
        provenance=_provenance(f"service:{key}"),
    )


def _edge(
    from_key: str,
    to_key: str,
    redundancy: RedundancyConfig | None = None,
    member_group: str | None = None,
) -> EdgeOut:
    return EdgeOut(
        from_node=from_key,
        to_node=to_key,
        failure_mode=None,
        redundancy_config=redundancy,
        member_group=member_group,
        provenance=_provenance(f"edge:{from_key}->{to_key}"),
    )


def _snapshot(
    nodes: list[NodeOut], edges: list[EdgeOut], version_no: int = 1
) -> GraphSnapshotOut:
    return GraphSnapshotOut(
        facility_id="test_facility",
        version_no=version_no,
        author="test",
        notes=None,
        created_at=_PROV_TIME,
        nodes=nodes,
        edges=edges,
    )


def _chain(
    *keys: str,
    redundancy: RedundancyConfig | None = None,
    member_group: str | None = None,
) -> tuple[list[NodeOut], list[EdgeOut]]:
    """A linear supply chain feeding a service: A1 -> A2 -> ... -> SVC."""
    assert len(keys) >= 2
    nodes: list[Any] = [_asset(key) for key in keys[:-1]] + [_service(keys[-1])]
    edges = [
        _edge(a, to, redundancy, member_group)
        for a, to in zip(keys[:-1], keys[1:], strict=False)
    ]
    return nodes, edges


@pytest.fixture()
def b() -> SimpleNamespace:
    """Builder namespace for pure graph-snapshot construction."""
    return SimpleNamespace(
        asset=_asset,
        service=_service,
        edge=_edge,
        snapshot=_snapshot,
        chain=_chain,
        asset_type=AssetType,
        redundancy=RedundancyConfig,
    )
