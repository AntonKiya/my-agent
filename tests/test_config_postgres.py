import pytest
from pydantic import ValidationError

from agent_service.config import AppSettings


def test_postgres_pool_settings_have_safe_defaults() -> None:
    settings = AppSettings(environment="test")

    assert settings.postgres_dsn is None
    assert settings.telegram_bot_token is None
    assert settings.inbound_worker_count == 8
    assert settings.inbound_publish_timeout_seconds == 1.0
    assert settings.inbound_worker_error_backoff_seconds == 0.1
    assert settings.agent_retry_max_attempts == 3
    assert settings.agent_retry_backoff_seconds == (1.0, 5.0, 15.0)
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 10
    assert settings.postgres_command_timeout_seconds == 30.0
    assert settings.redis_dsn is None
    assert settings.redis_context_snapshot_ttl_seconds == 24 * 60 * 60
    assert not settings.memory_compaction_enabled
    assert settings.memory_model_context_window_tokens == 196_600
    assert settings.memory_reserved_output_tokens == 16_384
    assert settings.memory_compaction_trigger_fraction == 0.80
    assert settings.memory_recent_tail_fraction == 0.30
    assert settings.memory_compaction_queue_maxsize == 1000
    assert settings.memory_compaction_worker_count == 0
    assert settings.memory_compaction_worker_error_backoff_seconds == 0.1
    assert settings.memory_compaction_publish_timeout_seconds == 0.1
    assert settings.memory_compaction_target_summary_tokens == 1000


def test_postgres_pool_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        AppSettings(environment="test", postgres_pool_max_size=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", postgres_command_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", inbound_worker_count=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", inbound_publish_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", inbound_worker_error_backoff_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_max_attempts=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_backoff_seconds=(-1.0,))

    with pytest.raises(ValidationError):
        AppSettings(environment="test", redis_context_snapshot_ttl_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_model_context_window_tokens=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_reserved_output_tokens=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_trigger_fraction=1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_recent_tail_fraction=0)

    with pytest.raises(ValidationError):
        AppSettings(
            environment="test",
            memory_model_context_window_tokens=100,
            memory_reserved_output_tokens=100,
        )

    with pytest.raises(ValidationError):
        AppSettings(
            environment="test",
            memory_compaction_trigger_fraction=0.30,
            memory_recent_tail_fraction=0.30,
        )

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_queue_maxsize=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_worker_count=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_worker_error_backoff_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_publish_timeout_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_target_summary_tokens=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", memory_compaction_target_summary_tokens=1201)

    with pytest.raises(ValidationError):
        AppSettings(
            environment="test",
            postgres_pool_min_size=20,
            postgres_pool_max_size=10,
        )
