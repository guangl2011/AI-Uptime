"""Enumerations for the versioned facility graph (WP1 core).

Asset types follow the locked market sequence: cold chain (the beachhead),
then the AIDC power chain plus its cooling-distribution units, then hospital
branch equipment. Power-chain types are shared across all three segments.
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
    """Equipment taxonomy: cold-chain types, the AIDC cooling train, hospital
    branch equipment, and the shared electrical power chain."""

    # Cold chain (beachhead).
    COLD_ROOM = "cold_room"
    REFRIGERATION_RACK = "refrigeration_rack"
    COMPRESSOR = "compressor"
    CONDENSER = "condenser"
    EVAPORATOR = "evaporator"
    # AIDC — electrical chain plus the chilled-water cooling train.
    UTILITY_FEED = "utility_feed"
    GENERATOR = "generator"
    ATS = "ats"
    UPS = "ups"
    PDU = "pdu"
    IT_RACK = "it_rack"
    CHILLER = "chiller"
    CDU = "cdu"
    PUMP = "pump"
    CRAH = "crah"
    # Hospitals — branch panels, medical gas, clinical air handling.
    DISTRIBUTION_PANEL = "distribution_panel"
    AIR_HANDLER = "air_handler"
    MED_GAS_MANIFOLD = "med_gas_manifold"
    MEDICAL_IMAGING = "medical_imaging"


class LoadClass(enum.StrEnum):
    """Branch role of an asset (hospital life-safety isolation, Phase 4).

    Set as an asset attribute (``load_class``): ``life_safety`` equipment
    sits on the emergency/red-line branch, ``general`` on the normal branch.
    Shared upstream equipment (utility, generator, ATS) is deliberately
    unclassified — isolation is asserted at the branch level, not the source.
    """

    GENERAL = "general"
    LIFE_SAFETY = "life_safety"


# Attribute keys for the class-aware semantics above. Free-form node
# attributes are the transport; these constants are the single spelling.
ATTRIBUTE_LOAD_CLASS = "load_class"
ATTRIBUTE_LIFE_SAFETY = "life_safety"


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
