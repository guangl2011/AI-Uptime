"""Ingestion layer — normalize external inputs into canonical graph records.

Every parser in this package is deterministic and read-only by construction:
it parses bytes into records with provenance and flags what it cannot resolve.
Nothing here connects to a facility system, holds credentials, or can issue a
write of any kind to equipment.
"""
