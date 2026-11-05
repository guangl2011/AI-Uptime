"""Unit tests for snapshot-derived protection features (forecasting/features.py)."""

from __future__ import annotations

from datetime import UTC, datetime

from uptime_api.forecasting.features import (
    ProtectionStatus,
    asset_protection_features,
    redundancy_class_of,
)
from uptime_api.graph.enums import (
    AssetType,
    FailureMode,
    NodeType,
    RedundancyConfig,
    SourceKind,
)
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut, NodeOut, ProvenanceOut

_NOW = datetime(2026, 11, 5, tzinfo=UTC)
_PROVENANCE = ProvenanceOut(
    id=1,
    source_kind=SourceKind.INVENTORY_IMPORT,
    source_system="test",
    source_ref="row-1",
    payload=None,
    created_at=_NOW,
)


def _asset(key: str) -> NodeOut:
    return NodeOut(
        node_key=key,
        node_type=NodeType.ASSET,
        asset_type=AssetType.COLD_ROOM,
        name=key,
        location=None,
        attributes={},
        provenance=_PROVENANCE,
    )


def _edge(
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
        provenance=_PROVENANCE,
    )


def _snapshot(nodes: list[NodeOut], edges: list[EdgeOut]) -> GraphSnapshotOut:
    return GraphSnapshotOut(
        facility_id="f1",
        version_no=1,
        author="test",
        notes=None,
        created_at=_NOW,
        nodes=nodes,
        edges=edges,
    )


def test_asset_with_no_out_edges_is_abstain_class() -> None:
    snapshot = _snapshot([_asset("a-1")], [])
    features = asset_protection_features(snapshot)
    assert len(features) == 1
    assert features[0].asset_key == "a-1"
    assert features[0].status is ProtectionStatus.NO_OUT_EDGES
    assert features[0].conflict is None


def test_unprotected_edge_is_has_unprotected() -> None:
    snapshot = _snapshot([_asset("a-1")], [_edge("a-1", "svc-1", RedundancyConfig.N, "rack-1")])
    features = asset_protection_features(snapshot)
    assert features[0].status is ProtectionStatus.HAS_UNPROTECTED
    assert features[0].conflict is None


def test_all_protected_when_every_edge_is_redundant_group_of_two() -> None:
    # A group lives across edges/assets: each N+1/2N edge needs a second
    # member somewhere in the snapshot for the protection to be real.
    snapshot = _snapshot(
        [_asset("a-1"), _asset("a-2")],
        [
            _edge("a-1", "svc-1", RedundancyConfig.N_PLUS_1, "pair-1"),
            _edge("a-2", "svc-1", RedundancyConfig.N_PLUS_1, "pair-1"),
            _edge("a-1", "svc-2", RedundancyConfig.N2, "pair-2"),
            _edge("a-2", "svc-2", RedundancyConfig.N2, "pair-2"),
        ],
    )
    by_key = {f.asset_key: f for f in asset_protection_features(snapshot)}
    assert by_key["a-1"].status is ProtectionStatus.ALL_PROTECTED
    assert by_key["a-2"].status is ProtectionStatus.ALL_PROTECTED


def test_single_member_group_contradicts_declared_redundancy() -> None:
    snapshot = _snapshot(
        [_asset("a-1")],
        [_edge("a-1", "svc-1", RedundancyConfig.N_PLUS_1, "solo-group")],
    )
    features = asset_protection_features(snapshot)
    assert features[0].conflict is not None
    assert "N+1" in features[0].conflict
    assert "solo-group" in features[0].conflict
    assert features[0].status is ProtectionStatus.HAS_UNPROTECTED


def test_declared_redundancy_without_group_conflicts() -> None:
    snapshot = _snapshot(
        [_asset("a-1")],
        [_edge("a-1", "svc-1", RedundancyConfig.N2, None)],
    )
    features = asset_protection_features(snapshot)
    assert features[0].conflict is not None
    assert "names no member group" in features[0].conflict


def test_plain_n_edge_with_group_does_not_conflict() -> None:
    # N declares no fault tolerance, so group size cannot contradict it.
    snapshot = _snapshot(
        [_asset("a-1")],
        [_edge("a-1", "svc-1", RedundancyConfig.N, "lone-group")],
    )
    features = asset_protection_features(snapshot)
    assert features[0].conflict is None
    assert features[0].status is ProtectionStatus.HAS_UNPROTECTED


def test_features_are_key_sorted_and_deterministic() -> None:
    edges = [
        _edge("b-1", "svc-1", RedundancyConfig.N, "g"),
        _edge("a-1", "svc-1", RedundancyConfig.N, "g"),
    ]
    snapshot = _snapshot([_asset("b-1"), _asset("a-1")], edges)
    first = [f.asset_key for f in asset_protection_features(snapshot)]
    second = [f.asset_key for f in asset_protection_features(snapshot)]
    assert first == ["a-1", "b-1"]
    assert first == second


def test_redundancy_class_weakest_link() -> None:
    assert redundancy_class_of([]) is RedundancyConfig.N
    assert (
        redundancy_class_of(
            [
                _edge("a", "s", RedundancyConfig.N2, "g"),
                _edge("a", "s2", RedundancyConfig.N_PLUS_1, "g"),
            ]
        )
        is RedundancyConfig.N_PLUS_1
    )
    assert redundancy_class_of([_edge("a", "s", RedundancyConfig.N2, "g")]) is RedundancyConfig.N2


def test_service_nodes_are_not_issued_features() -> None:
    service = NodeOut(
        node_key="svc-1",
        node_type=NodeType.SERVICE,
        asset_type=None,
        name="svc-1",
        location=None,
        attributes={},
        provenance=_PROVENANCE,
    )
    snapshot = _snapshot([_asset("a-1"), service], [_edge("a-1", "svc-1", RedundancyConfig.N, "g")])
    keys = [f.asset_key for f in asset_protection_features(snapshot)]
    assert keys == ["a-1"]
