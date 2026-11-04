"""Shared fixtures: the generated seed facility, API clients, and DB sessions."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from uptime_api.app import create_app
from uptime_api.db import build_engine, build_session_factory
from uptime_api.graph.models import Base

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"
SEED_PATH = SEED_DIR / "example_facility.json"


@pytest.fixture(scope="session")
def seed_facility() -> dict[str, Any]:
    """The committed seed facility, exactly as shipped."""
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def load_seed_via_api() -> Callable[[TestClient, dict[str, Any]], None]:
    """Load the seed facility through the real HTTP surface (create + commit)."""

    def _load(client: TestClient, seed: dict[str, Any]) -> None:
        created = client.post(
            "/facilities",
            json={
                "id": seed["facility_id"],
                "name": seed["name"],
                "segment": seed["segment"],
            },
        )
        assert created.status_code == 201, created.text
        committed = client.post(
            f"/facilities/{seed['facility_id']}/graph/versions",
            json={
                "author": seed["author"],
                "notes": seed["notes"],
                "nodes": seed["nodes"],
                "edges": seed["edges"],
            },
        )
        assert committed.status_code == 201, committed.text

    return _load


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client bound to a fresh SQLite database per test."""
    with TestClient(create_app(f"sqlite:///{tmp_path / 'test.db'}")) as test_client:
        yield test_client


@pytest.fixture()
def db_session(tmp_path: Path) -> Iterator[Session]:
    """A raw session for service-level tests (schema created, fresh DB per test)."""
    engine = build_engine(f"sqlite:///{tmp_path / 'service.db'}")
    Base.metadata.create_all(engine)
    session = build_session_factory(engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
