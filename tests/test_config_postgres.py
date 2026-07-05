from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from agent_service.config import AppSettings
from agent_service.memory_settings import (
    DEFAULT_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
    MAX_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
)


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
    assert not settings.telegram_rich_messages_enabled
    assert not settings.telegram_thinking_draft_enabled
    assert settings.telegram_thinking_draft_timeout_seconds == 1.0
    assert settings.telegram_thinking_draft_refresh_seconds == 8.0
    assert settings.telegram_media_group_debounce_seconds == 2.0
    assert settings.telegram_media_group_ttl_seconds == 60
    assert settings.telegram_media_group_flush_interval_seconds == 0.5
    assert settings.telegram_media_group_lock_ttl_seconds == 10.0
    assert settings.openrouter_http_connect_timeout_seconds == 10.0
    assert settings.openrouter_http_write_timeout_seconds == 10.0
    assert settings.openrouter_http_pool_timeout_seconds == 10.0
    assert settings.openrouter_http_keepalive_expiry_seconds == 60.0
    assert settings.openrouter_provider_sort is None
    assert settings.openrouter_provider_preferred_max_latency_p90 is None
    assert settings.openrouter_provider_preferred_max_latency_p99 is None
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
    assert (
        settings.memory_compaction_target_summary_tokens
        == DEFAULT_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS
    )
    assert settings.memory_compaction_model is None
    assert settings.transcription_audio_enabled
    assert settings.transcription_model == "whisper-large-v3-turbo"
    assert settings.transcription_timeout_seconds == 30.0
    assert settings.transcription_retry_max_attempts == 3
    assert settings.transcription_retry_backoff_seconds == (1.0, 5.0)
    assert settings.transcription_max_audio_size_bytes == 25_000_000
    assert settings.transcription_audio_temp_dir is None
    assert settings.image_analysis_enabled
    assert settings.image_analysis_model is None
    assert settings.image_analysis_timeout_seconds == 60.0
    assert settings.image_analysis_tool_timeout_seconds == 90.0
    assert settings.image_analysis_max_images == 5
    assert settings.image_max_size_bytes == 10_000_000
    assert str(settings.image_media_dir) == "var/media/images"
    assert settings.image_generation_enabled
    assert settings.image_generation_model == "google/gemini-2.5-flash-image"
    assert settings.image_generation_timeout_seconds == 120.0
    assert settings.image_generation_tool_timeout_seconds == 150.0
    assert settings.image_generation_max_source_images == 5
    assert settings.image_generation_max_output_images == 1
    assert settings.image_generation_max_output_size_bytes == 10_000_000
    assert str(settings.image_generation_media_dir) == "var/media/generated-images"
    assert settings.document_reading_enabled
    assert settings.document_reading_tool_timeout_seconds == 10.0
    assert settings.document_max_size_bytes == 2_000_000
    assert settings.document_max_extracted_chars == 80_000
    assert str(settings.document_media_dir) == "var/media/documents"
    assert settings.groq_api_key is None
    assert settings.groq_http_connect_timeout_seconds == 10.0
    assert settings.groq_http_read_timeout_seconds == 30.0
    assert settings.groq_http_write_timeout_seconds == 30.0
    assert settings.groq_http_pool_timeout_seconds == 10.0
    assert settings.groq_http_keepalive_expiry_seconds == 60.0


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
        AppSettings(
            environment="test",
            memory_compaction_target_summary_tokens=(
                MAX_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS + 1
            ),
        )

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
        AppSettings(environment="test", telegram_thinking_draft_refresh_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_connect_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_write_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_pool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", openrouter_http_keepalive_expiry_seconds=-1)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", transcription_model="")

    with pytest.raises(ValidationError):
        AppSettings(environment="test", transcription_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", transcription_retry_max_attempts=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", transcription_retry_backoff_seconds=(-1.0,))

    with pytest.raises(ValidationError):
        AppSettings(environment="test", transcription_max_audio_size_bytes=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_analysis_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_analysis_tool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_analysis_max_images=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_max_size_bytes=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_model="")

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_tool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_max_source_images=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_max_output_images=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", image_generation_max_output_size_bytes=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", groq_http_connect_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", groq_http_read_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", groq_http_write_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", groq_http_pool_timeout_seconds=0)

    with pytest.raises(ValidationError):
        AppSettings(environment="test", groq_http_keepalive_expiry_seconds=-1)


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


def test_agent_settings_read_openrouter_routing_configuration_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_SERVICE_OPENROUTER_PROVIDER_SORT", "throughput")
    monkeypatch.setenv("AGENT_SERVICE_OPENROUTER_PROVIDER_PREFERRED_MAX_LATENCY_P90", "3")
    monkeypatch.setenv("AGENT_SERVICE_OPENROUTER_PROVIDER_PREFERRED_MAX_LATENCY_P99", "6")

    settings = AppSettings(environment="test")

    assert settings.openrouter_provider_sort == "throughput"
    assert settings.openrouter_provider_preferred_max_latency_p90 == 3.0
    assert settings.openrouter_provider_preferred_max_latency_p99 == 6.0


def test_agent_settings_validate_openrouter_latency_configuration() -> None:
    with pytest.raises(ValidationError, match="p90 and p99 must be configured together"):
        AppSettings(
            environment="test",
            openrouter_provider_preferred_max_latency_p90=3.0,
        )

    with pytest.raises(ValidationError, match="p90 must be less than or equal to p99"):
        AppSettings(
            environment="test",
            openrouter_provider_preferred_max_latency_p90=7.0,
            openrouter_provider_preferred_max_latency_p99=6.0,
        )


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


def test_tutu_mcp_settings_accept_stdio_configuration() -> None:
    settings = AppSettings(
        environment="test",
        tutu_mcp_command="uvx",
        tutu_mcp_args=("tutu-mcp",),
        tutu_mcp_env={"TOKEN": "secret"},
    )

    assert settings.tutu_mcp_command == "uvx"
    assert settings.tutu_mcp_args == ("tutu-mcp",)
    assert settings.tutu_mcp_env == {"TOKEN": "secret"}
    assert settings.tutu_mcp_url is None


def test_tutu_mcp_settings_accept_url_configuration() -> None:
    settings = AppSettings(
        environment="test",
        tutu_mcp_url="https://mcp.tutu.ru/mcp",
        tutu_mcp_headers={"Authorization": "Bearer token"},
    )

    assert settings.tutu_mcp_url == "https://mcp.tutu.ru/mcp"
    assert settings.tutu_mcp_headers == {"Authorization": "Bearer token"}
    assert settings.tutu_mcp_command is None


def test_tutu_mcp_settings_reject_conflicting_transports() -> None:
    with pytest.raises(ValidationError, match="either tutu_mcp_command or tutu_mcp_url"):
        AppSettings(
            environment="test",
            tutu_mcp_command="uvx",
            tutu_mcp_url="https://mcp.tutu.ru/mcp",
        )


def test_tutu_mcp_settings_reject_stdio_options_without_command() -> None:
    with pytest.raises(ValidationError, match="tutu_mcp_args requires"):
        AppSettings(environment="test", tutu_mcp_args=("tutu-mcp",))

    with pytest.raises(ValidationError, match="tutu_mcp_env requires"):
        AppSettings(environment="test", tutu_mcp_env={"TOKEN": "secret"})


def test_weather_forecast_settings_accept_timeout_configuration() -> None:
    settings = AppSettings(
        environment="test",
        weather_forecast_enabled=False,
        weather_forecast_tool_timeout_seconds=12.0,
        weather_http_connect_timeout_seconds=2.0,
        weather_http_read_timeout_seconds=8.0,
        weather_http_write_timeout_seconds=3.0,
        weather_http_pool_timeout_seconds=4.0,
        weather_http_keepalive_expiry_seconds=30.0,
    )

    assert settings.weather_forecast_enabled is False
    assert settings.weather_forecast_tool_timeout_seconds == 12.0
    assert settings.weather_http_connect_timeout_seconds == 2.0
    assert settings.weather_http_read_timeout_seconds == 8.0
    assert settings.weather_http_write_timeout_seconds == 3.0
    assert settings.weather_http_pool_timeout_seconds == 4.0
    assert settings.weather_http_keepalive_expiry_seconds == 30.0


def test_web_research_settings_accept_timeout_configuration() -> None:
    settings = AppSettings(
        environment="test",
        web_research_enabled=False,
        web_research_tool_timeout_seconds=22.0,
        web_research_search_depth="basic",
        web_research_extract_depth="advanced",
        web_research_max_content_chars_per_source=1234,
        tavily_api_key=SecretStr("tvly-test"),
        tavily_http_connect_timeout_seconds=2.0,
        tavily_http_read_timeout_seconds=18.0,
        tavily_http_write_timeout_seconds=3.0,
        tavily_http_pool_timeout_seconds=4.0,
        tavily_http_keepalive_expiry_seconds=30.0,
    )

    assert settings.web_research_enabled is False
    assert settings.web_research_tool_timeout_seconds == 22.0
    assert settings.web_research_search_depth == "basic"
    assert settings.web_research_extract_depth == "advanced"
    assert settings.web_research_max_content_chars_per_source == 1234
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "tvly-test"
    assert settings.tavily_http_connect_timeout_seconds == 2.0
    assert settings.tavily_http_read_timeout_seconds == 18.0
    assert settings.tavily_http_write_timeout_seconds == 3.0
    assert settings.tavily_http_pool_timeout_seconds == 4.0
    assert settings.tavily_http_keepalive_expiry_seconds == 30.0


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
