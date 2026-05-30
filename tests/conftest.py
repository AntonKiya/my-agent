import pytest


@pytest.fixture(autouse=True)
def isolate_app_settings_from_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SERVICE_POSTGRES_DSN", "")
    monkeypatch.setenv("AGENT_SERVICE_REDIS_DSN", "")
    monkeypatch.setenv("AGENT_SERVICE_TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("AGENT_SERVICE_LOGFIRE_TOKEN", "")
    monkeypatch.setenv("AGENT_SERVICE_MEMORY_COMPACTION_ENABLED", "false")
