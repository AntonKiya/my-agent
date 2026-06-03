from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from agent_service.config import AppSettings


def test_postgres_pool_settings_have_safe_defaults() -> None:
    settings = cast(Any, AppSettings)(environment="test", _env_file=None)

    assert settings.postgres_dsn is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_webhook_secret_token is None
    assert settings.telegram_http_connect_timeout_seconds == 10.0
    assert settings.telegram_http_read_timeout_seconds == 15.0
    assert settings.telegram_http_write_timeout_seconds == 10.0
    assert settings.telegram_http_pool_timeout_seconds == 10.0
    assert settings.telegram_http_keepalive_expiry_seconds == 60.0
    assert not settings.telegram_thinking_draft_enabled
    assert settings.telegram_thinking_draft_timeout_seconds == 1.0
    assert settings.openrouter_http_connect_timeout_seconds == 10.0
    assert settings.openrouter_http_write_timeout_seconds == 10.0
    assert settings.openrouter_http_pool_timeout_seconds == 10.0
    assert settings.openrouter_http_keepalive_expiry_seconds == 60.0
    assert settings.inbound_worker_count == 8
    assert settings.inbound_publish_timeout_seconds == 1.0
    assert settings.inbound_worker_error_backoff_seconds == 0.1
    assert settings.delivery_worker_count == 4
    assert settings.delivery_worker_error_backoff_seconds == 0.1
    assert settings.delivery_retry_max_attempts == 3
    assert settings.delivery_retry_backoff_seconds == (1.0, 5.0, 15.0)
    assert settings.agent_retry_max_attempts == 3
    assert settings.agent_retry_backoff_seconds == (1.0, 5.0, 15.0)
    assert settings.agent_provider is None
    assert settings.agent_model is None
    assert settings.agent_timeout_seconds == 60.0
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 10
    assert settings.postgres_command_timeout_seconds == 30.0
    assert settings.redis_dsn is None
    assert settings.redis_context_snapshot_ttl_seconds == 24 * 60 * 60
    assert settings.recent_message_limit == 100
    assert not settings.memory_compaction_enabled
    assert settings.memory_model_context_window_tokens == 196_600
    assert settings.memory_reserved_output_tokens == 16_384
    assert settings.memory_compaction_trigger_fraction == 0.80
    assert settings.memory_recent_tail_fraction == 0.30
    assert settings.memory_compaction_queue_maxsize == 1000
    assert settings.memory_compaction_worker_count == 0
    assert settings.memory_compaction_worker_error_backoff_seconds == 0.1
    assert settings.memory_compaction_publish_timeout_seconds == 0.1
    assert settings.memory_compaction_timeout_seconds == 120.0
    assert settings.memory_compaction_target_summary_tokens == 1000
    assert settings.memory_compaction_model is None


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
        AppSettings(environment="test", delivery_worker_count=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", delivery_worker_error_backoff_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", delivery_retry_max_attempts=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", delivery_retry_backoff_seconds=(-1.0,))

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_max_attempts=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_retry_backoff_seconds=(-1.0,))

    with pytest.raises(ValidationError):
        AppSettings(environment="test", agent_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", redis_context_snapshot_ttl_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", recent_message_limit=0)

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
        AppSettings(environment="test", memory_compaction_timeout_seconds=0)

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

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_http_connect_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_http_read_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_http_write_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_http_pool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_http_keepalive_expiry_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", telegram_thinking_draft_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_connect_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_write_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_pool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_keepalive_expiry_seconds=-1)


def test_agent_settings_accept_openrouter_configuration() -> None:
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        memory_compaction_model="openai/gpt-4.1-mini",
    )

    assert settings.agent_provider == "openrouter"
    assert settings.agent_model == "openai/gpt-4.1-mini"
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "secret"
    assert settings.memory_compaction_model == "openai/gpt-4.1-mini"


def test_agent_settings_accept_partial_provider_configuration() -> None:
    settings = AppSettings(environment="test", agent_provider="openrouter")

    assert settings.agent_provider == "openrouter"
    assert settings.agent_model is None
    assert settings.openrouter_api_key is None


def test_vkusvill_mcp_settings_accept_stdio_configuration() -> None:
    settings = AppSettings(
        environment="test",
        vkusvill_mcp_command="uvx",
        vkusvill_mcp_args=("vkusvill-mcp",),
        vkusvill_mcp_env={"TOKEN": "secret"},
    )

    assert settings.vkusvill_mcp_command == "uvx"
    assert settings.vkusvill_mcp_args == ("vkusvill-mcp",)
    assert settings.vkusvill_mcp_env == {"TOKEN": "secret"}
    assert settings.vkusvill_mcp_url is None


def test_vkusvill_mcp_settings_accept_url_configuration() -> None:
    settings = AppSettings(
        environment="test",
        vkusvill_mcp_url="http://localhost:8765/mcp",
        vkusvill_mcp_headers={"Authorization": "Bearer token"},
    )

    assert settings.vkusvill_mcp_url == "http://localhost:8765/mcp"
    assert settings.vkusvill_mcp_headers == {"Authorization": "Bearer token"}
    assert settings.vkusvill_mcp_command is None


def test_vkusvill_mcp_settings_reject_conflicting_transports() -> None:
    with pytest.raises(ValidationError, match="either vkusvill_mcp_command or vkusvill_mcp_url"):
        AppSettings(
            environment="test",
            vkusvill_mcp_command="uvx",
            vkusvill_mcp_url="http://localhost:8765/mcp",
        )


def test_vkusvill_mcp_settings_reject_stdio_options_without_command() -> None:
    with pytest.raises(ValidationError, match="vkusvill_mcp_args requires"):
        AppSettings(environment="test", vkusvill_mcp_args=("vkusvill-mcp",))

    with pytest.raises(ValidationError, match="vkusvill_mcp_env requires"):
        AppSettings(environment="test", vkusvill_mcp_env={"TOKEN": "secret"})


def test_prod_telegram_bot_requires_webhook_secret() -> None:
    with pytest.raises(ValidationError, match="telegram_webhook_secret_token is required"):
        AppSettings(environment="prod", telegram_bot_token=SecretStr("bot-token"))

    settings = AppSettings(
        environment="prod",
        telegram_bot_token=SecretStr("bot-token"),
        telegram_webhook_secret_token=SecretStr("webhook-secret"),
    )

    assert settings.telegram_webhook_secret_token is not None
    assert settings.telegram_webhook_secret_token.get_secret_value() == "webhook-secret"
