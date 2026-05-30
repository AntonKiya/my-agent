import json
import logging

import pytest

from agent_service.config import AppSettings
from agent_service.observability.events import elapsed_ms, log_event, start_timer
from agent_service.observability.logging import configure_logging
from agent_service.observability.tracing import (
    create_trace_id,
    get_trace_id,
    reset_trace_id,
    set_trace_id,
)


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


def test_elapsed_ms_returns_non_negative_duration() -> None:
    started_at = start_timer()

    assert elapsed_ms(started_at) >= 0
