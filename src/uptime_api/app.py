"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from uptime_api import __version__
from uptime_api.api import router
from uptime_api.assessor import models as assessor_models  # noqa: F401  (registers tables)
from uptime_api.db import build_engine, build_session_factory, default_database_url
from uptime_api.forecasting import resolve_forecaster
from uptime_api.graph.models import Base


def create_app(
    database_url: str | None = None,
    *,
    enable_forecasting: bool | None = None,
) -> FastAPI:
    """Build the FastAPI application bound to a database (default: local SQLite).

    ``enable_forecasting`` overrides the UPTIME_FORECASTING environment flag;
    both default to OFF. The resolved model (or None) lives on app state and
    only ever *annotates* reports — it never drives behavior.
    """
    engine = build_engine(database_url or default_database_url())

    Base.metadata.create_all(engine)
    app = FastAPI(title="AI-Uptime API", version=__version__)
    app.state.session_factory = build_session_factory(engine)
    app.state.forecaster = resolve_forecaster(enabled=enable_forecasting)
    app.include_router(router)

    return app


app = create_app()
