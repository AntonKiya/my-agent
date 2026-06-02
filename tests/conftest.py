from collections.abc import Generator

import pytest

from agent_service.config import AppSettings, get_settings
from agent_service.observability.events import disable_business_spans_for_tests


@pytest.fixture(autouse=True)
def isolate_app_settings_from_local_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    get_settings.cache_clear()
    disable_business_spans_for_tests()
    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    monkeypatch.setenv("AGENT_SERVICE_POSTGRES_DSN", "")
    monkeypatch.setenv("AGENT_SERVICE_REDIS_DSN", "")
    monkeypatch.setenv("AGENT_SERVICE_TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("AGENT_SERVICE_TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
    monkeypatch.setenv("AGENT_SERVICE_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("AGENT_SERVICE_LOGFIRE_TOKEN", "")
    monkeypatch.setenv("AGENT_SERVICE_MEMORY_COMPACTION_ENABLED", "false")
    yield
    get_settings.cache_clear()
    disable_business_spans_for_tests()
