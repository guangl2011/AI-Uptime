"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from uptime_api import __version__
from uptime_api.api import router
from uptime_api.db import build_engine, build_session_factory, default_database_url


def create_app(database_url: str | None = None) -> FastAPI:
    """Build the FastAPI application bound to a database (default: local SQLite)."""
    engine = build_engine(database_url or default_database_url())

    app = FastAPI(title="AI-Uptime API", version=__version__)
    app.state.session_factory = build_session_factory(engine)
    app.include_router(router)

    return app


app = create_app()
