import json
import logging
import os
import sys
import warnings
from typing import Any, cast

import pytest
from fastapi import FastAPI
from opentelemetry import context as otel_context
from pydantic import SecretStr

import agent_service.observability.events as events_module
from agent_service.config import AppSettings
from agent_service.observability import logfire_integration
from agent_service.observability.events import elapsed_ms, log_event, start_timer
from agent_service.observability.logfire_integration import (
    configure_logfire,
    reset_logfire_integration_for_tests,
)
from agent_service.observability.logging import JsonLogFormatter, configure_logging
from agent_service.observability.tracing import (
    create_trace_id,
    get_trace_id,
    reset_trace_id,
    set_trace_id,
)


class FakeLogfireLoggingHandler(logging.NullHandler):
    _agent_service_logfire_handler: bool

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs


def test_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SERVICE_ENVIRONMENT", "test")
    monkeypatch.setenv("AGENT_SERVICE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENT_SERVICE_PORT", "9000")
    monkeypatch.setenv("AGENT_SERVICE_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS", "3.5")

    settings = AppSettings()

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.port == 9000
    assert settings.graceful_shutdown_timeout_seconds == 3.5


def test_trace_id_context_can_be_set_and_reset() -> None:
    trace_id = create_trace_id()

    token = set_trace_id(trace_id)

    assert get_trace_id() == trace_id

    reset_trace_id(token)

    assert get_trace_id() is None


def test_configure_logging_outputs_json_with_trace_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = AppSettings(environment="test", log_level="INFO")
    configure_logging(settings)

    trace_id = create_trace_id()
    token = set_trace_id(trace_id)
    try:
        logging.getLogger("agent_service.tests").info(
            "Observed event",
            extra={"event": "test_event"},
        )
    finally:
        reset_trace_id(token)

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert payload["level"] == "INFO"
    assert payload["message"] == "Observed event"
    assert payload["event"] == "test_event"
    assert payload["trace_id"] == trace_id


def test_log_event_omits_none_fields_and_preserves_safe_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = AppSettings(environment="test", log_level="INFO")
    configure_logging(settings)

    log_event(
        logging.getLogger("agent_service.tests"),
        logging.INFO,
        "Observed structured event",
        event="structured_event",
        conversation_id="conversation-1",
        user_id=None,
        duration_ms=1.25,
    )

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert payload["event"] == "structured_event"
    assert payload["conversation_id"] == "conversation-1"
    assert payload["duration_ms"] == 1.25
    assert "user_id" not in payload


def test_json_log_formatter_outputs_valid_exception_info() -> None:
    formatter = JsonLogFormatter()
    try:
        raise RuntimeError("observability failed")
    except RuntimeError:
        record = logging.LogRecord(
            name="agent_service.tests",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="Observed exception",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "ERROR"
    assert payload["message"] == "Observed exception"
    assert "RuntimeError: observability failed" in payload["exception"]


def test_json_log_formatter_ignores_invalid_exc_info_without_crashing() -> None:
    record = logging.LogRecord(
        name="agent_service.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=0,
        msg="Task was destroyed but it is pending",
        args=(),
        exc_info=cast(Any, True),
    )

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert payload["message"] == "Task was destroyed but it is pending"
    assert "exception" not in payload


def test_elapsed_ms_returns_non_negative_duration() -> None:
    started_at = start_timer()

    assert elapsed_ms(started_at) >= 0


def test_attached_trace_context_suppresses_logfire_propagated_context_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extract(_carrier: dict[str, str]) -> otel_context.Context:
        warnings.warn("Found propagated trace context.", RuntimeWarning, stacklevel=2)
        return otel_context.get_current()

    monkeypatch.setattr(events_module, "extract", fake_extract)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with events_module.attached_trace_context(
            {events_module.TRACE_CONTEXT_METADATA_KEY: {"traceparent": "value"}}
        ):
            pass

    assert captured == []


class FakeLogfire:
    def __init__(self) -> None:
        self.configure_calls: list[dict[str, object]] = []
        self.httpx_calls: list[dict[str, object]] = []
        self.asyncpg_calls: list[dict[str, object]] = []
        self.redis_calls: list[dict[str, object]] = []
        self.pydantic_ai_calls: list[dict[str, object]] = []
        self.fastapi_apps: list[FastAPI] = []
        self.fastapi_calls: list[dict[str, object]] = []
        self.logging_handlers: list[FakeLogfireLoggingHandler] = []

    def configure(self, **kwargs: object) -> None:
        self.configure_calls.append(kwargs)

    def instrument_httpx(self, **kwargs: object) -> None:
        self.httpx_calls.append(kwargs)

    def instrument_asyncpg(self, **kwargs: object) -> None:
        self.asyncpg_calls.append(kwargs)

    def instrument_redis(self, **kwargs: object) -> None:
        self.redis_calls.append(kwargs)

    def instrument_pydantic_ai(self, **kwargs: object) -> None:
        self.pydantic_ai_calls.append(kwargs)

    def instrument_fastapi(self, app: FastAPI, **kwargs: object) -> None:
        self.fastapi_apps.append(app)
        self.fastapi_calls.append(kwargs)

    def LogfireLoggingHandler(self, **kwargs: Any) -> FakeLogfireLoggingHandler:
        handler = FakeLogfireLoggingHandler(**kwargs)
        self.logging_handlers.append(handler)
        return handler


def test_configure_logfire_does_nothing_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_logfire_integration_for_tests()

    def fail_import() -> object:
        raise AssertionError("logfire should not be imported without a token")

    monkeypatch.setattr(logfire_integration, "_import_logfire", fail_import)

    configure_logfire(AppSettings(environment="test"))


def test_configure_logfire_installs_safe_instrumentation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_logfire_integration_for_tests()
    monkeypatch.delenv("OTEL_PYTHON_HTTPX_EXCLUDED_URLS", raising=False)
    fake_logfire = FakeLogfire()
    app = FastAPI()
    settings = AppSettings(
        environment="test",
        service_name="agent-service-test",
        logfire_token=SecretStr("token"),
    )

    monkeypatch.setattr(logfire_integration, "_import_logfire", lambda: fake_logfire)

    configure_logfire(settings, app=app)
    configure_logfire(settings, app=app)

    assert fake_logfire.configure_calls == [
        {
            "token": "token",
            "service_name": "agent-service-test",
            "environment": "test",
            "send_to_logfire": True,
            "console": False,
            "inspect_arguments": False,
        }
    ]
    assert len(fake_logfire.logging_handlers) == 1
    assert fake_logfire.logging_handlers[0]._agent_service_logfire_handler is True
    assert fake_logfire.logging_handlers[0].kwargs["level"] == "INFO"
    assert isinstance(fake_logfire.logging_handlers[0].kwargs["fallback"], logging.NullHandler)
    assert fake_logfire.httpx_calls == [
        {
            "capture_headers": False,
            "capture_request_body": False,
            "capture_response_body": False,
        }
    ]
    assert "https://api\\.telegram\\.org/bot[^/]+/.*" in os.environ[
        "OTEL_PYTHON_HTTPX_EXCLUDED_URLS"
    ]
    assert fake_logfire.asyncpg_calls == [{"capture_parameters": False}]
    assert fake_logfire.redis_calls == [{"capture_statement": False}]
    assert fake_logfire.pydantic_ai_calls == [{"include_content": False}]
    assert fake_logfire.fastapi_apps == [app]
    assert fake_logfire.fastapi_calls == [
        {
            "capture_headers": False,
            "record_send_receive": False,
        }
    ]
