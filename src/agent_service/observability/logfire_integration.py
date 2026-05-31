import logging
from importlib import import_module
from typing import Any

from fastapi import FastAPI

from agent_service.config import AppSettings

_logfire_configured = False
_global_instrumentation_configured = False
_instrumented_fastapi_apps: set[int] = set()


def configure_logfire(settings: AppSettings, *, app: FastAPI | None = None) -> None:
    token = settings.logfire_token.get_secret_value() if settings.logfire_token else None
    if token is None:
        return

    logfire = _import_logfire()
    _configure_logfire_once(logfire, settings=settings, token=token)
    _install_logfire_logging_handler(logfire, settings=settings)
    _configure_global_instrumentation_once(logfire)
    if app is not None:
        _instrument_fastapi_once(logfire, app)


def _configure_logfire_once(logfire: Any, *, settings: AppSettings, token: str) -> None:
    global _logfire_configured
    if _logfire_configured:
        return

    logfire.configure(
        token=token,
        service_name=settings.service_name,
        environment=settings.environment,
        send_to_logfire=True,
        console=False,
        inspect_arguments=False,
    )
    _logfire_configured = True


def _configure_global_instrumentation_once(logfire: Any) -> None:
    global _global_instrumentation_configured
    if _global_instrumentation_configured:
        return

    logfire.instrument_httpx(
        capture_headers=False,
        capture_request_body=False,
        capture_response_body=False,
    )
    logfire.instrument_asyncpg(capture_parameters=False)
    logfire.instrument_redis(capture_statement=False)
    logfire.instrument_pydantic_ai(include_content=False)
    _global_instrumentation_configured = True


def _install_logfire_logging_handler(logfire: Any, *, settings: AppSettings) -> None:
    root_logger = logging.getLogger()
    if any(
        getattr(handler, "_agent_service_logfire_handler", False)
        for handler in root_logger.handlers
    ):
        return

    handler = logfire.LogfireLoggingHandler(
        level=settings.log_level,
        fallback=logging.NullHandler(),
    )
    handler._agent_service_logfire_handler = True
    root_logger.addHandler(handler)


def _instrument_fastapi_once(logfire: Any, app: FastAPI) -> None:
    app_id = id(app)
    if app_id in _instrumented_fastapi_apps:
        return

    logfire.instrument_fastapi(
        app,
        capture_headers=False,
        record_send_receive=False,
    )
    _instrumented_fastapi_apps.add(app_id)


def _import_logfire() -> Any:
    return import_module("logfire")


def reset_logfire_integration_for_tests() -> None:
    global _logfire_configured, _global_instrumentation_configured
    _logfire_configured = False
    _global_instrumentation_configured = False
    _instrumented_fastapi_apps.clear()
