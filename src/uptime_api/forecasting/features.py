"""Forecast features extracted from one immutable graph snapshot.

Pure, deterministic reads — the same declared-redundancy physics the corpus
adjudicates from construction intent (a group absorbs a single fault iff it
has >= 2 members and declares N+1/2N), derived here independently from the
snapshot the way the generator derives labels independently of the assessor.
The harness checks the two derivations against each other; agreement on
synthetic data is the eval's calibration signal, not a copied answer.

Two derived cells drive the calibrated model:

- ``redundancy_class`` — the asset's weakest outgoing supply configuration
  (the flat-prior baseline's feature, kept for the abstention fallback);
- ``protection status`` — whether the asset has an outgoing supply edge that
  would NOT tolerate a single fault. This is the discriminating feature:
  an unprotected path means the fault flows; fully-protected or edge-less
  assets interrupt nothing through the declared graph.

A declared N+1/2N whose group membership contradicts it (fewer than two
members, or no group at all) is *conflicting evidence*: the forecast
abstains on such assets with a stated reason rather than silently picking a
side of the contradiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from uptime_api.graph.enums import RedundancyConfig
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut

_SINGLE_FAULT_TOLERANT_CONFIGS = frozenset({RedundancyConfig.N_PLUS_1, RedundancyConfig.N2})


class ProtectionStatus(StrEnum):
    """Effective single-fault protection across an asset's outgoing edges."""

    HAS_UNPROTECTED = "has_unprotected"
    ALL_PROTECTED = "all_protected"
    NO_OUT_EDGES = "no_out_edges"


@dataclass(frozen=True)
class AssetFeatures:
    """One asset's forecast features, read from the snapshot alone."""

    asset_key: str
    redundancy_class: str
    status: ProtectionStatus
    # Stated reason when declared redundancy contradicts group membership;
    # None when the snapshot is internally consistent.
    conflict: str | None


def redundancy_class_of(out_edges: list[EdgeOut]) -> RedundancyConfig:
    """Weakest outgoing supply configuration = the asset's own protection.

    Shared with the flat-prior baseline (kept there as the baseline's own
    feature): unprotected (no group, or N) wins over N+1, which wins over 2N.
    """
    configs = [edge.redundancy_config for edge in out_edges]
    if not out_edges or RedundancyConfig.N in configs or None in configs:
        return RedundancyConfig.N
    if RedundancyConfig.N_PLUS_1 in configs:
        return RedundancyConfig.N_PLUS_1
    return RedundancyConfig.N2


def _conflict_reason(edge: EdgeOut, group_size: int) -> str | None:
    """Reason string when the edge declares redundancy its group contradicts.

    ``group_size`` is the concrete member count of ``edge.member_group``
    (0 when the edge names no group — that branch never reads it), so an
    absent count can never reach the comparison below.
    """
    if edge.redundancy_config not in _SINGLE_FAULT_TOLERANT_CONFIGS:
        return None
    if edge.member_group is None:
        return (
            f"edge {edge.from_node} -> {edge.to_node} declares "
            f"{edge.redundancy_config.value} but names no member group"
        )
    if group_size < 2:
        return (
            f"edge {edge.from_node} -> {edge.to_node} declares "
            f"{edge.redundancy_config.value} but group {edge.member_group} "
            f"has {group_size} member(s)"
        )
    return None


def asset_protection_features(snapshot: GraphSnapshotOut) -> list[AssetFeatures]:
    """Per-asset protection features over every asset node, key-sorted."""
    group_sizes: dict[str, int] = {}
    out_edges: dict[str, list[EdgeOut]] = {}
    for edge in snapshot.edges:
        out_edges.setdefault(edge.from_node, []).append(edge)
        if edge.member_group is not None:
            group_sizes[edge.member_group] = group_sizes.get(edge.member_group, 0) + 1

    features: list[AssetFeatures] = []
    for node in sorted(
        (n for n in snapshot.nodes if n.node_type.value == "asset"),
        key=lambda n: n.node_key,
    ):
        edges = out_edges.get(node.node_key, [])
        conflict: str | None = None
        for edge in sorted(edges, key=_edge_sort_key):
            reason = _conflict_reason(edge, group_sizes.get(edge.member_group or "", 0))
            if reason is not None:
                conflict = reason
                break
        tolerating = [
            edge
            for edge in edges
            if edge.member_group is not None
            and group_sizes.get(edge.member_group, 1) >= 2
            and edge.redundancy_config in _SINGLE_FAULT_TOLERANT_CONFIGS
        ]
        if not edges:
            status = ProtectionStatus.NO_OUT_EDGES
        elif len(tolerating) == len(edges):
            status = ProtectionStatus.ALL_PROTECTED
        else:
            status = ProtectionStatus.HAS_UNPROTECTED
        features.append(
            AssetFeatures(
                asset_key=node.node_key,
                redundancy_class=redundancy_class_of(edges).value,
                status=status,
                conflict=conflict,
            )
        )
    return features


def _edge_sort_key(edge: EdgeOut) -> tuple[str, str]:
    return (edge.from_node, edge.to_node)
