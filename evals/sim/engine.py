"""Sim event engine: pure state transitions over facility topologies.

The assessor answers "what does one fault interrupt on this snapshot"; it
has no state. The engine answers "what does the facility look like after
this sequence of events", so the assessor can be re-run on each answer. It
is a set of pure functions — no I/O, no clocks, no RNG — the determinism
gate holds because every transition is a total function of its input state.

Semantics (one model, matching the corpus adjudication and the assessor):

- **Fault propagation** (Category 2's "cascade dynamics"): a faulted node's
  loss flows downstream along outgoing edges and stops at redundant groups
  that can still absorb it. An edge ``x → y`` in a redundant group absorbs
  while a sibling member of that group (another supplier of ``y``) is still
  live and the edge declares N+1/2N — the dynamic form of the assessor's
  ``group_sizes >= 2`` rule, which it equals on any intact graph.
- **Materialization**: dead nodes stay in the working topology but every
  edge touching a dead node is dropped. The committed snapshot therefore
  shows exactly the live supply structure: dead services become unserved
  (the ``unserved_service`` finding names the loss), and a surviving group
  member's edge count drops, so its tolerance degrades in the report.
- **State purity**: the dead set is a total function of (topology,
  faulted set). ``repair`` simply leaves the faulted set — restoration
  flows out of recomputation, so repair order is adjudicable step by step.

Events (the M1 union): fault, repair, redundancy_change, load_shift.

What-if interventions (M3) ride the same pure-transition discipline but on
branch states, never the timeline: ``apply_intervention`` applies one
counterfactual — ``RemoveAssetEvent`` (the node leaves the facility and its
loss cascades like a fault, but the snapshot omits the node itself) or a
``RedundancyChangeEvent`` (the N → N+1 upgrade: reclassify plus a standby
member, the exact timeline transition). Branch states carry the ``removed``
set so the cascade closure stays a total function of the state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Redundancy configs that tolerate a single member's fault — mirrors the
# assessor's _SINGLE_FAULT_TOLERANT_CONFIGS (canonical API spellings).
TOLERANT_CONFIGS = frozenset({"N+1", "2N"})
VALID_CONFIGS = frozenset({"N", "N+1", "2N"})

NODE_TYPE_ASSET = "asset"
NODE_TYPE_SERVICE = "service"


class SimError(Exception):
    """Base class for simulator errors (structural or state-level)."""


class SimDslError(SimError):
    """A topology payload or scenario file is structurally invalid."""


class SimEventError(SimError):
    """An event is illegal against the current simulation state."""


@dataclass(frozen=True)
class NodeSpec:
    """One graph node with its raw GraphVersionIn payload for re-commit."""

    key: str
    node_type: str
    raw: dict[str, object]


@dataclass(frozen=True)
class EdgeSpec:
    """One dependency edge (provider → consumer) with its raw payload."""

    from_node: str
    to_node: str
    failure_mode: str | None
    redundancy_config: str | None
    member_group: str | None
    raw: dict[str, object]


@dataclass(frozen=True)
class Topology:
    """The working facility topology: nodes and supply edges."""

    nodes: dict[str, NodeSpec]
    edges: tuple[EdgeSpec, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Topology:
        """Parse a GraphVersionIn-compatible payload (corpus or scenario)."""
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise SimDslError("topology payload must carry a non-empty 'nodes' list")
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_edges, list):
            raise SimDslError("topology payload 'edges' must be a list")

        nodes: dict[str, NodeSpec] = {}
        for index, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, dict):
                raise SimDslError(f"node {index} is not an object")
            key = raw_node.get("node_key")
            node_type = raw_node.get("node_type")
            if not isinstance(key, str) or not key:
                raise SimDslError(f"node {index} has no usable node_key")
            if node_type not in (NODE_TYPE_ASSET, NODE_TYPE_SERVICE):
                raise SimDslError(f"node {key}: node_type must be asset or service")
            if key in nodes:
                raise SimDslError(f"duplicate node key {key}")
            nodes[key] = NodeSpec(key=key, node_type=node_type, raw=raw_node)

        edges: list[EdgeSpec] = []
        for index, raw_edge in enumerate(raw_edges):
            if not isinstance(raw_edge, dict):
                raise SimDslError(f"edge {index} is not an object")
            from_node = raw_edge.get("from_node")
            to_node = raw_edge.get("to_node")
            if not isinstance(from_node, str) or not isinstance(to_node, str):
                raise SimDslError(f"edge {index} needs from_node and to_node strings")
            if from_node not in nodes or to_node not in nodes:
                raise SimDslError(f"edge {from_node} -> {to_node} references an unknown node")
            if from_node == to_node:
                raise SimDslError(f"edge {from_node} -> {to_node} is a self-loop")
            config = raw_edge.get("redundancy_config")
            if config is not None and config not in VALID_CONFIGS:
                raise SimDslError(f"edge {from_node} -> {to_node}: unknown redundancy {config!r}")
            edges.append(
                EdgeSpec(
                    from_node=from_node,
                    to_node=to_node,
                    failure_mode=_optional_str(raw_edge.get("failure_mode")),
                    redundancy_config=config,
                    member_group=_optional_str(raw_edge.get("member_group")),
                    raw=raw_edge,
                )
            )
        return cls(nodes=nodes, edges=tuple(edges))

    def service_keys(self) -> frozenset[str]:
        """Declared service node keys."""
        return frozenset(
            key for key, spec in self.nodes.items() if spec.node_type == NODE_TYPE_SERVICE
        )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultEvent:
    """Take one asset out of service (triggers cascade recomputation)."""

    asset_key: str


@dataclass(frozen=True)
class RepairEvent:
    """Restore one previously faulted asset (its own fault, not a cascade)."""

    asset_key: str


@dataclass(frozen=True)
class RedundancyChangeEvent:
    """Reclassify a member group's config and add a standby member to it.

    The standby is explicit (node + edge payloads, scenario style) because
    tolerance needs >= 2 members: re-labeling a lone member would be a
    no-op, so the engine rejects events that would be one.
    """

    member_group: str
    redundancy_config: str
    standby_node: dict[str, object]
    standby_edge: dict[str, object]


@dataclass(frozen=True)
class LoadShiftEvent:
    """Move one unprotected dependency: ``from_node``'s feed of the consumer
    is rewired onto ``to_node``."""

    consumer: str
    from_node: str
    to_node: str


@dataclass(frozen=True)
class RemoveAssetEvent:
    """What-if component removal: the asset leaves the facility entirely.

    Counterfactual semantics, not a fault: the materialized snapshot omits
    the node (the facility is the facility *without this component*), while
    the cascade closure propagates its loss exactly as a fault's — the
    "what breaks if this dies" answer. Branch-only: the timeline union has
    no removal (the plan's event list), so ``apply_event`` rejects it; only
    ``apply_intervention`` accepts it.
    """

    asset_key: str


SimEvent = FaultEvent | RepairEvent | RedundancyChangeEvent | LoadShiftEvent

# Counterfactual interventions (M3): a removal, or the N → N+1 upgrade —
# reusing the timeline's RedundancyChangeEvent wholesale so a what-if upgrade
# is the same transition a scenario's timeline upgrade would be.
WhatIfIntervention = RemoveAssetEvent | RedundancyChangeEvent


def describe_event(event: SimEvent) -> str:
    """Stable one-line description for version notes and trace records."""
    match event:
        case FaultEvent(asset_key=key):
            return f"fault {key}"
        case RepairEvent(asset_key=key):
            return f"repair {key}"
        case RedundancyChangeEvent(member_group=group, redundancy_config=config):
            return f"redundancy_change {group} -> {config}"
        case LoadShiftEvent(consumer=consumer, from_node=src, to_node=dst):
            return f"load_shift {consumer} {src} -> {dst}"
    raise SimEventError(f"unrecognized event {event!r}")  # pragma: no cover


def describe_intervention(intervention: WhatIfIntervention) -> str:
    """Stable one-line description of a counterfactual intervention."""
    match intervention:
        case RemoveAssetEvent(asset_key=key):
            return f"remove {key}"
        case RedundancyChangeEvent(member_group=group, redundancy_config=config):
            return f"upgrade {group} -> {config}"
    raise SimEventError(f"unrecognized intervention {intervention!r}")  # pragma: no cover


def apply_intervention(state: SimState, intervention: WhatIfIntervention) -> SimState:
    """Apply one what-if intervention to a branch copy of ``state``.

    Upgrades are the timeline's own transition (a branch upgrade must be
    exactly what the same event on the timeline would be); removals are
    counterfactual-only.
    """
    match intervention:
        case RemoveAssetEvent(asset_key=key):
            return _apply_remove_asset(state, key)
        case RedundancyChangeEvent():
            return apply_event(state, intervention)
    raise SimEventError(f"unrecognized intervention {intervention!r}")  # pragma: no cover


def _apply_remove_asset(state: SimState, key: str) -> SimState:
    node = state.topology.nodes.get(key)
    if node is None:
        raise SimEventError(f"remove targets unknown node {key!r}")
    if node.node_type != NODE_TYPE_ASSET:
        raise SimEventError(
            f"remove targets {key!r}, which is a service — what-if removal takes assets"
        )
    if key in state.removed:
        raise SimEventError(f"asset {key} is already removed")
    # The working topology keeps the node and its edges so the cascade
    # closure can propagate the loss downstream; materialization is what
    # omits the node. ``removed`` rides the state for exactly that reason.
    return SimState(topology=state.topology, faulted=state.faulted, removed=state.removed | {key})


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimState:
    """One simulation state: the working topology plus faulted asset keys.

    ``removed`` carries what-if removals (M3, branch states only): removed
    nodes stay in the working topology so the closure sees their edges, and
    materialization omits them. The timeline never sets it.
    """

    topology: Topology
    faulted: frozenset[str]
    removed: frozenset[str] = frozenset()


def initial_state(topology: Topology) -> SimState:
    """The state before any event: nothing faulted."""
    return SimState(topology=topology, faulted=frozenset())


def apply_event(state: SimState, event: SimEvent) -> SimState:
    """Apply one event, returning the next state (the input is untouched)."""
    match event:
        case FaultEvent(asset_key=key):
            return _apply_fault(state, key)
        case RepairEvent(asset_key=key):
            return _apply_repair(state, key)
        case RedundancyChangeEvent(
            member_group=group,
            redundancy_config=config,
            standby_node=standby_node,
            standby_edge=standby_edge,
        ):
            return _apply_redundancy_change(state, group, config, standby_node, standby_edge)
        case LoadShiftEvent(consumer=consumer, from_node=src, to_node=dst):
            return _apply_load_shift(state, consumer, src, dst)
    raise SimEventError(f"unrecognized event {event!r}")  # pragma: no cover


def _apply_fault(state: SimState, key: str) -> SimState:
    node = state.topology.nodes.get(key)
    if node is None:
        raise SimEventError(f"fault targets unknown node {key!r}")
    if node.node_type != NODE_TYPE_ASSET:
        raise SimEventError(f"fault targets {key!r}, which is a service — fault events take assets")
    if key in state.faulted:
        raise SimEventError(f"asset {key} is already faulted")
    return SimState(topology=state.topology, faulted=state.faulted | {key})


def _apply_repair(state: SimState, key: str) -> SimState:
    if key not in state.topology.nodes:
        raise SimEventError(f"repair targets unknown node {key!r}")
    if key not in state.faulted:
        raise SimEventError(
            f"repair targets {key}, which was not faulted by an event — "
            "cascade-dead assets return when their supply is restored"
        )
    return SimState(topology=state.topology, faulted=state.faulted - {key})


def _apply_redundancy_change(
    state: SimState,
    group: str,
    config: str,
    standby_node_raw: dict[str, object],
    standby_edge_raw: dict[str, object],
) -> SimState:
    if config not in VALID_CONFIGS:
        raise SimEventError(f"redundancy_change declares unknown config {config!r}")
    group_edges = [edge for edge in state.topology.edges if edge.member_group == group]
    if not group_edges:
        raise SimEventError(f"redundancy_change targets unknown member group {group!r}")

    standby = _parse_standby(standby_node_raw, standby_edge_raw, config, group, group_edges, state)

    # Reclassify every member edge, then add the standby's node and edge.
    reclassified = tuple(
        _edge_with_config(edge, config) if edge.member_group == group else edge
        for edge in state.topology.edges
    )
    topology = Topology(
        nodes={**state.topology.nodes, standby.node.key: standby.node},
        edges=(*reclassified, standby.edge),
    )
    return SimState(topology=topology, faulted=state.faulted)


@dataclass(frozen=True)
class StandbySpec:
    """A validated standby member: node spec plus the edge joining the group."""

    node: NodeSpec
    edge: EdgeSpec


def _parse_standby(
    standby_node_raw: dict[str, object],
    standby_edge_raw: dict[str, object],
    config: str,
    group: str,
    group_edges: list[EdgeSpec],
    state: SimState,
) -> StandbySpec:
    """Validate the standby payloads against the group they join."""
    if standby_node_raw.get("node_type") != NODE_TYPE_ASSET:
        raise SimEventError("redundancy_change standby must be an asset node")
    key = standby_node_raw.get("node_key")
    if not isinstance(key, str) or not key:
        raise SimEventError("redundancy_change standby has no usable node_key")
    if key in state.topology.nodes:
        raise SimEventError(f"redundancy_change standby {key} already exists in the topology")

    edge_from = standby_edge_raw.get("from_node")
    edge_to = standby_edge_raw.get("to_node")
    if edge_from != key:
        raise SimEventError(f"redundancy_change standby edge must originate at {key}")
    targets = {edge.to_node for edge in group_edges}
    if len(targets) != 1:
        raise SimEventError(f"member group {group!r} does not name one consumer — ambiguous target")
    if edge_to != next(iter(targets)):
        raise SimEventError(
            f"redundancy_change standby edge must target "
            f"{next(iter(targets))}, the group's consumer"
        )
    if standby_edge_raw.get("member_group") != group:
        raise SimEventError(f"redundancy_change standby edge must join group {group!r}")
    if standby_edge_raw.get("redundancy_config") != config:
        raise SimEventError(
            f"redundancy_change standby edge must declare the event's config {config!r}"
        )

    edge = EdgeSpec(
        from_node=key,
        to_node=str(edge_to),
        failure_mode=_optional_str(standby_edge_raw.get("failure_mode")),
        redundancy_config=config,
        member_group=group,
        raw=standby_edge_raw,
    )
    return StandbySpec(
        node=NodeSpec(key=key, node_type=NODE_TYPE_ASSET, raw=standby_node_raw), edge=edge
    )


def _edge_with_config(edge: EdgeSpec, config: str) -> EdgeSpec:
    raw = dict(edge.raw)
    raw["redundancy_config"] = config
    return replace(edge, redundancy_config=config, raw=raw)


def _apply_load_shift(state: SimState, consumer: str, from_node: str, to_node: str) -> SimState:
    if consumer not in state.topology.nodes:
        raise SimEventError(f"load_shift targets unknown consumer {consumer!r}")
    if to_node not in state.topology.nodes:
        raise SimEventError(f"load_shift targets unknown destination {to_node!r}")
    if to_node == from_node:
        raise SimEventError("load_shift would rewire a feed onto itself")
    matching = [
        edge
        for edge in state.topology.edges
        if edge.from_node == from_node and edge.to_node == consumer
    ]
    if len(matching) != 1:
        raise SimEventError(
            f"load_shift expects exactly one dependency edge {from_node} -> {consumer}, "
            f"found {len(matching)}"
        )
    edge = matching[0]
    if edge.member_group is not None:
        raise SimEventError(
            f"load_shift cannot move a redundant-group member ({edge.member_group!r}); "
            "M1 shifts unprotected dependencies only"
        )
    raw = dict(edge.raw)
    raw["from_node"] = to_node
    rewired = replace(edge, from_node=to_node, raw=raw)
    edges = tuple(rewired if e is edge else e for e in state.topology.edges)
    return SimState(
        topology=Topology(nodes=state.topology.nodes, edges=edges), faulted=state.faulted
    )


# ---------------------------------------------------------------------------
# Cascade closure and materialization
# ---------------------------------------------------------------------------


def dead_nodes(state: SimState) -> frozenset[str]:
    """The dead set: faulted and removed nodes plus everything their loss
    cascades to.

    Least fixpoint — a live node dies when a supplier edge's provider is
    dead and that edge cannot absorb (no live sibling in its redundant
    group, or the edge declares N/none). Removed nodes seed the closure so
    a what-if removal's loss cascades exactly like a fault's. Iterating in
    sorted key order keeps the fixpoint deterministic regardless of set
    hashing.
    """
    incoming: dict[str, list[EdgeSpec]] = {}
    members: dict[tuple[str, str], set[str]] = {}
    for edge in state.topology.edges:
        incoming.setdefault(edge.to_node, []).append(edge)
        if edge.member_group is not None:
            group_key = (edge.to_node, edge.member_group)
            members.setdefault(group_key, set()).add(edge.from_node)

    dead = set(state.faulted) | set(state.removed)
    changed = True
    while changed:
        changed = False
        for key in sorted(state.topology.nodes):
            if key in dead:
                continue
            for edge in incoming.get(key, ()):
                if edge.from_node not in dead:
                    continue
                if _absorbs(edge, dead, members):
                    continue
                dead.add(key)
                changed = True
                break
    return frozenset(dead)


def _absorbs(
    edge: EdgeSpec, dead: set[str], members: dict[tuple[str, str], set[str]]
) -> bool:
    """True when ``edge``'s redundant group still has a live sibling member.

    The dynamic form of the assessor's ``group_sizes >= 2`` rule: on any
    intact graph the two agree; after faults, only live siblings count.
    """
    if edge.member_group is None or edge.redundancy_config not in TOLERANT_CONFIGS:
        return False
    siblings = members.get((edge.to_node, edge.member_group), set())
    return any(sibling not in dead for sibling in siblings)


def materialize(
    state: SimState, dead: frozenset[str], *, author: str, notes: str
) -> dict[str, object]:
    """Render the state as a GraphVersionIn payload for the real API.

    ``removed`` nodes are omitted outright (the counterfactual facility is
    the facility without that component); every other node is kept, and
    every edge touching a dead node is dropped, so the committed snapshot
    shows the live supply structure exactly: dead services arrive isolated
    (named by the assessor's unserved_service finding), dead assets stay
    visible but path-less, and surviving group members' consumers degrade
    in the report.
    """
    nodes = [
        state.topology.nodes[key].raw
        for key in sorted(state.topology.nodes)
        if key not in state.removed
    ]
    gone = dead | state.removed
    edges = [
        edge.raw
        for edge in state.topology.edges
        if edge.from_node not in gone and edge.to_node not in gone
    ]
    return {"author": author, "notes": notes, "nodes": nodes, "edges": edges}
