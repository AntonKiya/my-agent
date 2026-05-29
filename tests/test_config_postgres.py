import pytest
from pydantic import ValidationError

from agent_service.config import AppSettings


def test_postgres_pool_settings_have_safe_defaults() -> None:
    settings = AppSettings(environment="test")

    assert settings.postgres_dsn is None
    assert settings.telegram_bot_token is None
    assert settings.inbound_worker_count == 8
    assert settings.agent_retry_max_attempts == 3
    assert settings.agent_retry_backoff_seconds == (1.0, 5.0, 15.0)
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 10
    assert settings.postgres_command_timeout_seconds == 30.0


def test_postgres_pool_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        AppSettings(environment="test", postgres_pool_max_size=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", postgres_command_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", inbound_worker_count=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_max_attempts=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_backoff_seconds=(-1.0,))

    with pytest.raises(ValidationError):
        AppSettings(
            environment="test",
            postgres_pool_min_size=20,
            postgres_pool_max_size=10,
        )
