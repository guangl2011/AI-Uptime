"""Shared fixtures: the generated seed facility and API test clients."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uptime_api.app import create_app

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"
SEED_PATH = SEED_DIR / "example_facility.json"


@pytest.fixture(scope="session")
def seed_facility() -> dict[str, Any]:
    """The committed seed facility, exactly as shipped."""
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client bound to a fresh SQLite database per test."""
    with TestClient(create_app(f"sqlite:///{tmp_path / 'test.db'}")) as test_client:
        yield test_client
