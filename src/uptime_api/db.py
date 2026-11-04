"""Database wiring: engine and session-factory construction."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./uptime.db"
DATABASE_URL_ENV = "UPTIME_DATABASE_URL"


def default_database_url() -> str:
    """The database URL from the environment, or the local SQLite default."""
    return os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None) -> Engine:
    """Build the SQLAlchemy engine; SQLite gets its standard thread setting."""
    url = database_url or default_database_url()
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
