"""HTTP surface — facilities, the versioned graph twin, and inventory import.

Phase 1 route groups (MVP spec art_SSJ84EWf): facility creation, graph
version commits/history/snapshots, and read-only equipment-inventory upload.
Assessment routes land in a later phase. Nothing here can reach a facility
system — the only writes are to AI-Uptime's own database.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from uptime_api.graph import schemas
from uptime_api.graph.errors import (
    FacilityAlreadyExists,
    FacilityNotFound,
    GraphError,
    GraphVersionNotFound,
)
from uptime_api.graph.schemas import GraphVersionIn
from uptime_api.graph.service import GraphService
from uptime_api.ingestion.inventory import (
    InventoryFormatError,
    ParsedInventory,
    parse_inventory,
)

router = APIRouter()


def get_session(request: Request) -> Iterator[Session]:
    """One session per request; commit on success, roll back on any error."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _not_found(error: FacilityNotFound | GraphVersionNotFound) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# --- Facilities -----------------------------------------------------------


@router.post("/facilities", status_code=201)
def create_facility(
    payload: schemas.FacilityCreate, session: Session = Depends(get_session)
) -> schemas.FacilityOut:
    """Create a facility. Supply ``id`` for deterministic seeds."""
    facility_id = payload.id or uuid.uuid4().hex
    try:
        facility = GraphService(session).create_facility(
            facility_id, payload.name, payload.segment
        )
    except FacilityAlreadyExists as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return schemas.FacilityOut.model_validate(facility)


@router.get("/facilities")
def list_facilities(
    session: Session = Depends(get_session),
) -> list[schemas.FacilityOut]:
    """All facilities, oldest first."""
    service = GraphService(session)
    return [schemas.FacilityOut.model_validate(f) for f in service.list_facilities()]


@router.get("/facilities/{facility_id}")
def get_facility(
    facility_id: str, session: Session = Depends(get_session)
) -> schemas.FacilityOut:
    """Fetch one facility."""
    try:
        facility = GraphService(session).get_facility(facility_id)
    except FacilityNotFound as error:
        raise _not_found(error) from error
    return schemas.FacilityOut.model_validate(facility)


# --- Graph versions ---------------------------------------------------------


@router.post("/facilities/{facility_id}/graph/versions", status_code=201)
def commit_graph_version(
    facility_id: str,
    payload: GraphVersionIn,
    session: Session = Depends(get_session),
) -> schemas.GraphVersionOut:
    """Commit a full graph snapshot as the facility's next immutable version."""
    service = GraphService(session)
    try:
        version = service.commit_version(facility_id, payload)
        return service.list_versions(facility_id)[version.version_no - 1]
    except FacilityNotFound as error:
        raise _not_found(error) from error
    except GraphError as error:  # invalid topology: duplicates, dangling edges
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/facilities/{facility_id}/graph/versions")
def list_graph_versions(
    facility_id: str, session: Session = Depends(get_session)
) -> list[schemas.GraphVersionOut]:
    """Version history for a facility, oldest first."""
    try:
        return GraphService(session).list_versions(facility_id)
    except FacilityNotFound as error:
        raise _not_found(error) from error


@router.get("/facilities/{facility_id}/graph")
def get_graph(
    facility_id: str,
    version: int | None = None,
    session: Session = Depends(get_session),
) -> schemas.GraphSnapshotOut:
    """Read one immutable graph version (default: latest) with provenance intact."""
    try:
        return GraphService(session).get_snapshot(facility_id, version)
    except (FacilityNotFound, GraphVersionNotFound) as error:
        raise _not_found(error) from error


# --- Inventory ingestion (read-only integrations only) -----------------------


@router.post("/facilities/{facility_id}/inventory", status_code=201)
async def upload_inventory(
    facility_id: str,
    file: UploadFile,
    session: Session = Depends(get_session),
) -> schemas.InventoryImportOut:
    """Import an equipment inventory (CSV/JSON) as the facility's first version.

    Ambiguous rows are flagged in ``unresolved`` with a concrete reason —
    they never become guessed nodes. A facility that already has a graph
    version returns 409: imports don't silently overwrite topology.
    """
    if file.filename is None:
        raise HTTPException(status_code=422, detail="upload needs a filename (.csv or .json)")
    service = GraphService(session)

    try:
        service.get_facility(facility_id)
    except FacilityNotFound as error:
        raise _not_found(error) from error
    if service.list_versions(facility_id):
        raise HTTPException(
            status_code=409,
            detail="facility already has a graph version; edit topology via graph versions",
        )

    content = await file.read()
    try:
        parsed: ParsedInventory = parse_inventory(content, file.filename)
    except InventoryFormatError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not parsed.nodes:
        raise HTTPException(
            status_code=422,
            detail=f"no importable rows: {[u.reason.value for u in parsed.unresolved]}",
        )

    version = service.commit_version(
        facility_id,
        GraphVersionIn(
            author="inventory_import",
            notes=f"equipment inventory import from {file.filename}",
            nodes=parsed.nodes,
            edges=[],
        ),
    )
    return schemas.InventoryImportOut(
        facility_id=facility_id,
        version_no=version.version_no,
        nodes_created=len(parsed.nodes),
        unresolved=[
            schemas.UnresolvedRowOut(row=u.row, reason=u.reason.value, detail=u.detail)
            for u in parsed.unresolved
        ],
    )
