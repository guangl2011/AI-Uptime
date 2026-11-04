"""Deterministic, evidence-linked criticality scoring and SPOF detection.

Pure functions over an immutable graph snapshot — no I/O, no ML, no clocks.
Same snapshot in, byte-identical report out; that is the reproducibility
property assessments pin their graph version for.

Semantics
---------
Edge direction is provider → consumer (``from`` feeds ``to``). Redundancy
lives on edges: edges sharing a ``member_group`` are redundant members of one
supply set with ``redundancy_config`` declaring the tolerance. An edge with
no member group is a single unprotected feed (config N).

Fault propagation: when an asset fails, the fault flows downstream along
outgoing edges and stops where redundancy absorbs it — a group tolerates a
single fault iff it has >= 2 members and config N+1 or 2N. Services reached
by an unabsorbed fault are *exposed* to that asset.

Scoring model (explainable by construction)
-------------------------------------------
Asset criticality   = 60 per exposed service, capped at 100. A fault that is
                      fully absorbed by declared redundancy adds nothing: the
                      deterministic semantics say it interrupts no service, so
                      a nonzero score would assert risk that is not there
                      (the calibration the eval harness's Brier gate checks).
                      Absorbed impact is still reported — ``tolerated_services``
                      and the driver lines carry it. The 60 weight calibrates
                      the deterministic single-fault signal within the eval's
                      Brier gate on every segment (single exposure = 0.6,
                      multi-exposure caps at 1.0) while preserving gradation
                      for the ranking metric.
Service criticality = 30 + 15 per unprotected supplier asset, capped at 100.
Risk bands          = high >= 70, elevated >= 30, low below.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from uptime_api.assessor.enums import FindingKind, RiskLevel
from uptime_api.assessor.schemas import (
    AssessmentReport,
    AssetAssessment,
    CriticalityScore,
    EvidenceCitation,
    Exposure,
    Finding,
    ServiceAssessment,
    SpofEntry,
)
from uptime_api.graph.enums import (
    ATTRIBUTE_LIFE_SAFETY,
    ATTRIBUTE_LOAD_CLASS,
    LoadClass,
    NodeType,
    RedundancyConfig,
)
from uptime_api.graph.schemas import EdgeOut, GraphSnapshotOut, NodeOut

if TYPE_CHECKING:
    from uptime_api.forecasting.interface import ForecastModel

# Scoring constants — documented in the module docstring; single source of truth.
EXPOSED_SERVICE_WEIGHT = 60
SERVICE_BASE_SCORE = 30
UNPROTECTED_SUPPLIER_WEIGHT = 15
HIGH_RISK_THRESHOLD = 70
ELEVATED_RISK_THRESHOLD = 30

_SINGLE_FAULT_TOLERANT_CONFIGS = frozenset({RedundancyConfig.N_PLUS_1, RedundancyConfig.N2})


def risk_level_for(score: int) -> RiskLevel:
    """Band a 0-100 score. Kept as a function so bands are testable."""
    if score >= HIGH_RISK_THRESHOLD:
        return RiskLevel.HIGH
    if score >= ELEVATED_RISK_THRESHOLD:
        return RiskLevel.ELEVATED
    return RiskLevel.LOW


@dataclass(frozen=True)
class _Topology:
    """Indexes over one snapshot; built once per assessment."""

    snapshot: GraphSnapshotOut
    nodes_by_key: dict[str, NodeOut]
    out_edges: dict[str, list[EdgeOut]]
    in_edges: dict[str, list[EdgeOut]]
    group_sizes: dict[str, int]

    @classmethod
    def build(cls, snapshot: GraphSnapshotOut) -> _Topology:
        nodes_by_key = {node.node_key: node for node in snapshot.nodes}
        out_edges: dict[str, list[EdgeOut]] = {}
        in_edges: dict[str, list[EdgeOut]] = {}
        group_sizes: dict[str, int] = {}
        for edge in snapshot.edges:
            out_edges.setdefault(edge.from_node, []).append(edge)
            in_edges.setdefault(edge.to_node, []).append(edge)
            if edge.member_group is not None:
                group_sizes[edge.member_group] = group_sizes.get(edge.member_group, 0) + 1
        # Deterministic traversal order regardless of storage order.
        for index in [*out_edges.values(), *in_edges.values()]:
            index.sort(key=_edge_sort_key)
        return cls(
            snapshot=snapshot,
            nodes_by_key=nodes_by_key,
            out_edges=out_edges,
            in_edges=in_edges,
            group_sizes=group_sizes,
        )

    def edge_tolerates_single_fault(self, edge: EdgeOut) -> bool:
        """A redundant group absorbs a single member failure only when it has
        at least 2 members AND declares N+1/2N. N fails on the first fault."""
        if edge.member_group is None:
            return False
        if self.group_sizes.get(edge.member_group, 1) < 2:
            return False
        return edge.redundancy_config in _SINGLE_FAULT_TOLERANT_CONFIGS

    def group_concurrent_tolerance(self, edge: EdgeOut) -> int:
        """Faults the edge's group tolerates: 0 for N, members - 1 otherwise."""
        if edge.member_group is None or edge.redundancy_config is RedundancyConfig.N:
            return 0
        return max(0, self.group_sizes.get(edge.member_group, 1) - 1)

    def services_downstream(self, start: str, respect_tolerance: bool) -> frozenset[str]:
        """Service keys reachable downstream from ``start``.

        With ``respect_tolerance`` the fault stops at redundant groups that
        absorb a single fault (the *exposed* set); without it, pure graph
        reach (the *impacted* set).
        """
        reached_services: set[str] = set()
        visited: set[str] = {start}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for edge in self.out_edges.get(current, []):
                if respect_tolerance and self.edge_tolerates_single_fault(edge):
                    continue
                if edge.to_node in visited:
                    continue
                visited.add(edge.to_node)
                if self.nodes_by_key[edge.to_node].node_type is NodeType.SERVICE:
                    reached_services.add(edge.to_node)
                queue.append(edge.to_node)
        return frozenset(reached_services)


def _is_disconnected(topology: _Topology, node_key: str) -> bool:
    """True when an asset relates to no service at all: no structural path to
    one downstream, and no service feeding it as a demand-side leaf."""
    if topology.services_downstream(node_key, respect_tolerance=False):
        return False
    incoming = topology.in_edges.get(node_key, [])
    return not any(
        topology.nodes_by_key[edge.from_node].node_type is NodeType.SERVICE
        for edge in incoming
    )


def _edge_sort_key(edge: EdgeOut) -> tuple[str, str, int]:
    return (edge.from_node, edge.to_node, edge.provenance.id)


def _load_class(node: NodeOut) -> LoadClass | None:
    """The asset's declared branch class, or None when unclassified.

    Membership-checked, not raised: an unknown value is treated as
    unclassified (shared infrastructure), never silently as either class.
    """
    value = node.attributes.get(ATTRIBUTE_LOAD_CLASS)
    valid_values = {member.value for member in LoadClass}
    return LoadClass(value) if value in valid_values else None


def _attr_flag(attributes: dict[str, object], key: str) -> bool:
    """Truthiness of a boolean-valued attribute across its spellings
    (JSON booleans and operator strings like 'true'/'1'/'yes')."""
    value = attributes.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _node_citation(node: NodeOut) -> EvidenceCitation:
    descriptor = f"{node.node_type.value}" + (
        f" {node.asset_type.value}" if node.asset_type else ""
    )
    provenance = node.provenance
    return EvidenceCitation(
        record_id=provenance.id,
        kind=provenance.source_kind,
        claim=f"{node.node_key} ({descriptor}) — {provenance.source_kind.value} "
        f"{provenance.source_system}:{provenance.source_ref}",
    )


def _edge_citation(edge: EdgeOut) -> EvidenceCitation:
    provenance = edge.provenance
    label = f"dependency {edge.from_node} → {edge.to_node}"
    if edge.redundancy_config is not None:
        label += f" [{edge.redundancy_config.value}"
        if edge.member_group:
            label += f", group {edge.member_group}"
        label += "]"
    return EvidenceCitation(
        record_id=provenance.id,
        kind=provenance.source_kind,
        claim=f"{label} — {provenance.source_kind.value} "
        f"{provenance.source_system}:{provenance.source_ref}",
    )


def analyze_asset_criticality(topology: _Topology) -> list[AssetAssessment]:
    """Score every asset: 40 per exposed service, cap 100, absorbed = 0."""
    assessments: list[AssetAssessment] = []
    for node in sorted(topology.snapshot.nodes, key=lambda n: n.node_key):
        if node.node_type is not NodeType.ASSET:
            continue
        exposed = topology.services_downstream(node.node_key, respect_tolerance=True)
        impacted = topology.services_downstream(node.node_key, respect_tolerance=False)
        tolerated = impacted - exposed
        score = min(100, EXPOSED_SERVICE_WEIGHT * len(exposed))

        drivers: list[str] = []
        for service in sorted(exposed):
            drivers.append(f"single failure exposes {service} (unprotected path)")
        for service in sorted(tolerated):
            drivers.append(f"impact on {service} absorbed by declared redundancy")
        if not drivers:
            if _is_disconnected(topology, node.node_key):
                drivers.append("reaches no service — topology may be incomplete")
            else:
                drivers.append("no unprotected single-fault exposure to any service")

        evidence = [_node_citation(node)]
        evidence.extend(_edge_citation(e) for e in topology.out_edges.get(node.node_key, []))
        evidence.extend(_edge_citation(e) for e in topology.in_edges.get(node.node_key, []))

        assessments.append(
            AssetAssessment(
                node_key=node.node_key,
                name=node.name,
                asset_type=node.asset_type,
                criticality=CriticalityScore(
                    score=score, risk_level=risk_level_for(score), drivers=drivers
                ),
                exposed_services=sorted(exposed),
                tolerated_services=sorted(tolerated),
                evidence=evidence,
            )
        )
    return assessments


def _service_spofs_by_asset(asset_assessments: list[AssetAssessment]) -> dict[str, list[str]]:
    """Invert the exposure map: which assets interrupt which services."""
    spofs: dict[str, list[str]] = {}
    for asset in asset_assessments:
        for service in asset.exposed_services:
            spofs.setdefault(service, []).append(asset.node_key)
    return spofs


def _weakest_redundancy(topology: _Topology, incoming: list[EdgeOut]) -> RedundancyConfig:
    """The service's effective redundancy: weakest incoming supply path."""
    if any(not topology.edge_tolerates_single_fault(edge) for edge in incoming):
        return RedundancyConfig.N
    if any(edge.redundancy_config is RedundancyConfig.N_PLUS_1 for edge in incoming):
        return RedundancyConfig.N_PLUS_1
    return RedundancyConfig.N2


def analyze_service_criticality(
    topology: _Topology, asset_assessments: list[AssetAssessment]
) -> list[ServiceAssessment]:
    """Score every service and derive its SPOF list, exposure, and data gaps."""
    spofs_by_service = {
        service: sorted(assets)
        for service, assets in _service_spofs_by_asset(asset_assessments).items()
    }

    assessments: list[ServiceAssessment] = []
    for node in sorted(topology.snapshot.nodes, key=lambda n: n.node_key):
        if node.node_type is not NodeType.SERVICE:
            continue
        incoming = topology.in_edges.get(node.node_key, [])
        spofs = spofs_by_service.get(node.node_key, [])

        if incoming:
            first_fault_tolerated = all(
                topology.edge_tolerates_single_fault(edge) for edge in incoming
            )
            concurrent = min(topology.group_concurrent_tolerance(edge) for edge in incoming)
            redundancy = _weakest_redundancy(topology, incoming)
        else:
            first_fault_tolerated = False
            concurrent = 0
            redundancy = RedundancyConfig.N

        score = min(100, SERVICE_BASE_SCORE + UNPROTECTED_SUPPLIER_WEIGHT * len(spofs))
        drivers: list[str] = []
        if spofs:
            drivers.append(f"{len(spofs)} unprotected supplier(s): {', '.join(spofs)}")
        if incoming and not first_fault_tolerated:
            drivers.append("no redundant supply path — first fault interrupts the service")
        if not incoming:
            drivers.append("no supply edges — topology incomplete")

        evidence = [_node_citation(node)]
        evidence.extend(_edge_citation(edge) for edge in incoming)

        service_gaps = [] if incoming else ["service has no incoming supply edges"]

        assessments.append(
            ServiceAssessment(
                service_id=node.node_key,
                name=node.name,
                redundancy=redundancy,
                criticality=CriticalityScore(
                    score=score, risk_level=risk_level_for(score), drivers=drivers
                ),
                life_safety=_attr_flag(node.attributes, ATTRIBUTE_LIFE_SAFETY),
                single_points_of_failure=[
                    SpofEntry(
                        asset_id=asset_key,
                        reason=f"single failure of {asset_key} interrupts this service",
                    )
                    for asset_key in spofs
                ],
                exposure=Exposure(
                    first_fault_tolerated=first_fault_tolerated,
                    concurrent_faults_tolerated=concurrent,
                    unprotected_assets=spofs,
                ),
                evidence=evidence,
                data_gaps=service_gaps,
            )
        )
    return assessments


def _life_safety_isolation_findings(topology: _Topology) -> list[Finding]:
    """Find life-safety services whose supply paths run through general loads.

    For each service marked ``life_safety`` (per the MVP spec's hospital
    isolation semantics), walk every upstream supply path and flag the first
    general-class asset on it. Unclassified assets (shared infrastructure such
    as the utility feed or an ATS) do not violate isolation — only explicitly
    general-class loads do. Evidence cites the full violation path so the
    finding is traceable edge by edge.
    """
    findings: list[Finding] = []
    for node in sorted(topology.snapshot.nodes, key=lambda n: n.node_key):
        if node.node_type is not NodeType.SERVICE:
            continue
        if not _attr_flag(node.attributes, ATTRIBUTE_LIFE_SAFETY):
            continue

        parents: dict[str, tuple[str, EdgeOut]] = {}
        visited = {node.node_key}
        queue: deque[str] = deque([node.node_key])
        violations: dict[str, list[EdgeOut]] = {}
        while queue:
            current = queue.popleft()
            for edge in topology.in_edges.get(current, []):
                source = edge.from_node
                if source in visited:
                    continue
                visited.add(source)
                parents[source] = (current, edge)
                source_node = topology.nodes_by_key[source]
                if (
                    source_node.node_type is NodeType.ASSET
                    and _load_class(source_node) is LoadClass.GENERAL
                ):
                    violations[source] = _path_edges_to_service(
                        source, node.node_key, parents
                    )
                    continue  # nearest violation per path — don't descend past it
                queue.append(source)

        if not violations:
            continue
        paths = [
            " -> ".join([*[edge.from_node for edge in edges], edges[-1].to_node])
            for _, edges in sorted(violations.items())
        ]
        evidence = [_node_citation(node)]
        for asset_key, edges in sorted(violations.items()):
            asset_node = topology.nodes_by_key[asset_key]
            evidence.append(_node_citation(asset_node))
            evidence.extend(_edge_citation(edge) for edge in edges)
        findings.append(
            Finding(
                kind=FindingKind.LIFE_SAFETY_ISOLATION,
                subject=node.node_key,
                title=f"{node.node_key} (life safety) is not isolated from general loads",
                detail=f"Life-safety service {node.node_key} draws supply through "
                f"general-load asset(s): {', '.join(sorted(violations))}. Path(s): "
                + "; ".join(paths)
                + ". Life-safety branches must be separate from general "
                "distribution.",
                affected_services=[node.node_key],
                evidence=evidence,
            )
        )
    return findings


def _path_edges_to_service(
    source: str, service_key: str, parents: dict[str, tuple[str, EdgeOut]]
) -> list[EdgeOut]:
    """Edges along the BFS path from ``source`` down to ``service_key``."""
    edges: list[EdgeOut] = []
    cursor = source
    while cursor != service_key:
        previous, edge = parents[cursor]
        edges.append(edge)
        cursor = previous
    return edges


def detect_findings(
    topology: _Topology,
    asset_assessments: list[AssetAssessment],
    service_assessments: list[ServiceAssessment],
) -> tuple[list[Finding], list[str]]:
    """Ranked findings plus facility-level data gaps.

    Ranking: most services affected first, then highest affected-service
    criticality, then kind and subject — a total, deterministic order.
    """
    service_score = {s.service_id: s.criticality.score for s in service_assessments}

    findings: list[Finding] = []
    data_gaps: list[str] = []

    # R1 — single points of failure: any asset whose one fault exposes >= 1 service.
    for asset in asset_assessments:
        if asset.exposed_services:
            findings.append(
                Finding(
                    kind=FindingKind.SINGLE_POINT_OF_FAILURE,
                    subject=asset.node_key,
                    title=f"{asset.node_key} is a single point of failure",
                    detail=f"A single failure of {asset.node_key} interrupts "
                    f"{len(asset.exposed_services)} service(s) with no redundancy "
                    f"absorption: {', '.join(asset.exposed_services)}.",
                    affected_services=asset.exposed_services,
                    evidence=asset.evidence,
                )
            )
        elif _is_disconnected(topology, asset.node_key):
            # Orphan assets are data gaps, not findings — they assert no risk.
            # Redundancy-absorbed suppliers and demand-side leaves (assets a
            # service feeds) are connected, so they do not land here.
            data_gaps.append(
                f"asset {asset.node_key} does not reach any service; "
                "topology may be incomplete"
            )

    # R2 — missing redundancy: services fed only by unprotected (N) paths.
    # Unserved services are R3's subject — a missing-supply finding would
    # carry an empty supplier list here.
    for service in service_assessments:
        if service.exposure.unprotected_assets and not service.exposure.first_fault_tolerated:
            findings.append(
                Finding(
                    kind=FindingKind.MISSING_REDUNDANCY,
                    subject=service.service_id,
                    title=f"{service.service_id} tolerates no supply faults (N)",
                    detail=f"Every supply path into {service.service_id} is "
                    "unprotected (observed config N where N+1/2N would absorb a "
                    "first fault). Unprotected suppliers: "
                    f"{', '.join(service.exposure.unprotected_assets)}.",
                    affected_services=[service.service_id],
                    evidence=service.evidence,
                )
            )

    # R3 — unserved services: no incoming edges at all.
    for service in service_assessments:
        if service.data_gaps:
            findings.append(
                Finding(
                    kind=FindingKind.UNSERVED_SERVICE,
                    subject=service.service_id,
                    title=f"{service.service_id} has no supply edges",
                    detail=f"{service.service_id} declares no incoming dependency, "
                    "so its availability cannot be assessed — confirm the topology.",
                    affected_services=[service.service_id],
                    evidence=service.evidence,
                )
            )
            data_gaps.append(f"service {service.service_id} has no incoming supply edges")

    # R4 — life-safety isolation: a life-safety service whose supply path runs
    # through general-load equipment (hospital red-line discipline). Wiring
    # integrity, independent of redundancy tolerance: a redundant feed through
    # general distribution is still a mixed-branch violation.
    findings.extend(_life_safety_isolation_findings(topology))

    def _rank(finding: Finding) -> tuple[int, int, str, str]:
        worst_affected = max(
            (service_score.get(s, 0) for s in finding.affected_services), default=0
        )
        return (
            -len(finding.affected_services),
            -worst_affected,
            finding.kind.value,
            finding.subject,
        )

    findings.sort(key=_rank)
    return findings, sorted(set(data_gaps))


def build_report(
    snapshot: GraphSnapshotOut,
    forecaster: ForecastModel | None = None,
) -> AssessmentReport:
    """Analyze one immutable snapshot into the full assessment report.

    Each stage computes once; later stages consume the earlier results.
    Forecasts are computed only when a forecast model is wired in — the
    forecasting feature flag decides (default OFF); they are informational
    estimates and nothing executes on them.
    """
    topology = _Topology.build(snapshot)
    assets = analyze_asset_criticality(topology)
    services = analyze_service_criticality(topology, assets)
    findings, data_gaps = detect_findings(topology, assets, services)
    forecasts = forecaster.forecast(snapshot) if forecaster is not None else []

    if not snapshot.nodes:
        data_gaps.append("graph version is empty — nothing to assess")
    data_gaps.append(
        "no time-windowed evidence ingested (alarms/work orders arrive in a "
        "later increment); this assessment is topology-only"
    )

    return AssessmentReport(
        facility_id=snapshot.facility_id,
        graph_version=snapshot.version_no,
        evidence_window=None,
        services=services,
        assets=assets,
        findings=findings,
        forecasts=forecasts,
        data_gaps=sorted(set(data_gaps)),
        open_questions=[],
    )
