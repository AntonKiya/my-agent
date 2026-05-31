import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_service.api.health import router as health_router
from agent_service.channels.telegram.routes import router as telegram_router
from agent_service.config import AppSettings, get_settings
from agent_service.container import AppContainer
from agent_service.observability.logfire_integration import configure_logfire
from agent_service.observability.logging import configure_observability

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not hasattr(app.state, "settings"):
        app.state.settings = get_settings()
    if not hasattr(app.state, "container"):
        app.state.container = AppContainer(settings=app.state.settings)
    settings = app.state.settings
    container = app.state.container
    configure_observability(settings)
    configure_logfire(settings, app=app)
    await container.start()
    logger.info(
        "Service startup",
        extra={
            "event": "service_startup",
            "service": settings.service_name,
            "environment": settings.environment,
        },
    )
    try:
        yield
    finally:
        await container.stop()
        logger.info(
            "Service shutdown",
            extra={
                "event": "service_shutdown",
                "service": settings.service_name,
                "environment": settings.environment,
            },
        )


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    # Keep dependency assembly at the app boundary so handlers and workers do not
    # create queues, clients, adapters, or services on their own.
    container = AppContainer(settings=app_settings)

    def settings_dependency() -> AppSettings:
        return app_settings

    app = FastAPI(
        title=app_settings.service_name,
        debug=app_settings.debug,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.container = container
    app.dependency_overrides[get_settings] = settings_dependency
    app.include_router(health_router)
    app.include_router(telegram_router)
    configure_logfire(app_settings, app=app)
    return app
