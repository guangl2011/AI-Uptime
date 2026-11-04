"""Enumerations for the versioned facility graph (WP1 core).

Cold chain is the beachhead market, so asset types lean refrigeration; the
power-chain types (UPS/ATS/generator) are shared with the AIDC expansion.
"""

from __future__ import annotations

import enum


class Segment(enum.StrEnum):
    """Market segment of a facility — the locked sequence is cold chain first."""

    COLD_CHAIN = "cold_chain"
    AIDC = "aidc"
    HOSPITAL = "hospital"


class NodeType(enum.StrEnum):
    """A graph node is either a physical asset or a logical service it supports."""

    ASSET = "asset"
    SERVICE = "service"


class AssetType(enum.StrEnum):
    """Equipment taxonomy. Cold-chain types first, then the shared power chain."""

    COLD_ROOM = "cold_room"
    REFRIGERATION_RACK = "refrigeration_rack"
    COMPRESSOR = "compressor"
    CONDENSER = "condenser"
    EVAPORATOR = "evaporator"
    UTILITY_FEED = "utility_feed"
    GENERATOR = "generator"
    ATS = "ats"
    UPS = "ups"
    PDU = "pdu"


class RedundancyConfig(enum.StrEnum):
    """How many faults the served service tolerates.

    N fails on the first fault, N+1 tolerates one fault, 2N tolerates a
    full-path loss. Values are the canonical API spelling (MVP spec).
    """

    N = "N"
    N_PLUS_1 = "N+1"
    N2 = "2N"


class FailureMode(enum.StrEnum):
    """How a dependency edge fails — propagation semantics land in Phase 2."""

    POWER_LOSS = "power_loss"
    MECHANICAL_FAILURE = "mechanical_failure"
    REFRIGERANT_LOSS = "refrigerant_loss"
    CONTROL_FAILURE = "control_failure"


class SourceKind(enum.StrEnum):
    """Where a graph element or criticality value came from (provenance)."""

    INVENTORY_IMPORT = "inventory_import"
    TECHNICIAN_ENTRY = "technician_entry"
    OEM_DOC = "oem_doc"
    ALARM = "alarm"
    WORK_ORDER = "work_order"
    SENSOR = "sensor"


class UnresolvedReason(enum.StrEnum):
    """Why an inventory row could not become a graph node — explicit abstention."""

    MISSING_ASSET_ID = "missing_asset_id"
    UNKNOWN_ASSET_TYPE = "unknown_asset_type"
    DUPLICATE_ASSET_ID = "duplicate_asset_id"
