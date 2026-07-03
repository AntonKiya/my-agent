from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_service.memory_settings import (
    DEFAULT_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
    MAX_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
)

Environment = Literal["local", "dev", "staging", "prod", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
AgentProvider = Literal["openrouter"]
WebResearchSearchDepth = Literal["ultra-fast", "fast", "basic", "advanced"]
WebResearchExtractDepth = Literal["basic", "advanced"]
DEFAULT_IMAGE_GENERATION_MODEL = "google/gemini-2.5-flash-image"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_prefix="AGENT_SERVICE_",
        extra="ignore",
    )

    service_name: str = "my-agent"
    environment: Environment = "local"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "INFO"
    graceful_shutdown_timeout_seconds: float = Field(default=10.0, gt=0)
    inbound_queue_maxsize: int = Field(default=5000, ge=0)
    outbound_queue_maxsize: int = Field(default=5000, ge=0)
    inbound_publish_timeout_seconds: float = Field(default=1.0, gt=0)
    outbound_publish_timeout_seconds: float = Field(default=5.0, ge=0)
    inbound_worker_count: int = Field(default=8, ge=0)
    inbound_worker_error_backoff_seconds: float = Field(default=0.1, ge=0)
    delivery_worker_count: int = Field(default=4, ge=0)
    delivery_worker_error_backoff_seconds: float = Field(default=0.1, ge=0)
    delivery_retry_max_attempts: int = Field(default=3, ge=1)
    delivery_retry_backoff_seconds: tuple[float, ...] = (1.0, 5.0, 15.0)
    reminders_enabled: bool = True
    reminders_tool_timeout_seconds: float = Field(default=10.0, gt=0)
    reminder_worker_count: int = Field(default=1, ge=0)
    reminder_worker_poll_interval_seconds: float = Field(default=30.0, gt=0)
    reminder_worker_batch_size: int = Field(default=500, gt=0)
    reminder_worker_error_backoff_seconds: float = Field(default=1.0, ge=0)
    notification_outbox_worker_count: int = Field(default=1, ge=0)
    notification_outbox_poll_interval_seconds: float = Field(default=1.0, gt=0)
    notification_outbox_batch_size: int = Field(default=100, gt=0)
    notification_outbox_lease_seconds: float = Field(default=60.0, gt=0)
    notification_outbox_worker_error_backoff_seconds: float = Field(default=1.0, ge=0)
    reminders_max_active_per_user: int = Field(default=100, gt=0)
    reminders_min_interval_minutes: int = Field(default=30, gt=0)
    reminders_max_created_per_message: int = Field(default=3, gt=0)
    agent_retry_max_attempts: int = Field(default=3, ge=1)
    agent_retry_backoff_seconds: tuple[float, ...] = (1.0, 5.0, 15.0)
    agent_provider: AgentProvider | None = None
    agent_model: str | None = None
    agent_timeout_seconds: float = Field(default=60.0, gt=0)
    vkusvill_mcp_command: str | None = None
    vkusvill_mcp_args: tuple[str, ...] = ()
    vkusvill_mcp_env: dict[str, str] = Field(default_factory=dict)
    vkusvill_mcp_url: str | None = None
    vkusvill_mcp_headers: dict[str, str] = Field(default_factory=dict)
    vkusvill_mcp_init_timeout_seconds: float = Field(default=5.0, gt=0)
    vkusvill_mcp_read_timeout_seconds: float = Field(default=300.0, gt=0)
    tutu_mcp_command: str | None = None
    tutu_mcp_args: tuple[str, ...] = ()
    tutu_mcp_env: dict[str, str] = Field(default_factory=dict)
    tutu_mcp_url: str | None = None
    tutu_mcp_headers: dict[str, str] = Field(default_factory=dict)
    tutu_mcp_init_timeout_seconds: float = Field(default=5.0, gt=0)
    tutu_mcp_read_timeout_seconds: float = Field(default=300.0, gt=0)
    weather_forecast_enabled: bool = True
    weather_forecast_tool_timeout_seconds: float = Field(default=15.0, gt=0)
    weather_http_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    weather_http_read_timeout_seconds: float = Field(default=10.0, gt=0)
    weather_http_write_timeout_seconds: float = Field(default=5.0, gt=0)
    weather_http_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    weather_http_keepalive_expiry_seconds: float = Field(default=60.0, ge=0)
    web_research_enabled: bool = True
    web_research_tool_timeout_seconds: float = Field(default=45.0, gt=0)
    web_research_search_depth: WebResearchSearchDepth = "advanced"
    web_research_extract_depth: WebResearchExtractDepth = "basic"
    web_research_max_content_chars_per_source: int = Field(default=20_000, gt=0)
    tavily_api_key: SecretStr | None = None
    tavily_http_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    tavily_http_read_timeout_seconds: float = Field(default=20.0, gt=0)
    tavily_http_write_timeout_seconds: float = Field(default=5.0, gt=0)
    tavily_http_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    tavily_http_keepalive_expiry_seconds: float = Field(default=60.0, ge=0)
    transcription_audio_enabled: bool = True
    transcription_model: str = Field(default="whisper-large-v3-turbo", min_length=1)
    transcription_timeout_seconds: float = Field(default=30.0, gt=0)
    transcription_retry_max_attempts: int = Field(default=3, ge=1)
    transcription_retry_backoff_seconds: tuple[float, ...] = (1.0, 5.0)
    transcription_max_audio_size_bytes: int = Field(default=25_000_000, gt=0)
    transcription_audio_temp_dir: Path | None = None
    image_analysis_enabled: bool = True
    image_analysis_model: str | None = None
    image_analysis_timeout_seconds: float = Field(default=60.0, gt=0)
    image_analysis_tool_timeout_seconds: float = Field(default=90.0, gt=0)
    image_analysis_max_images: int = Field(default=5, gt=0)
    image_max_size_bytes: int = Field(default=10_000_000, gt=0)
    image_media_dir: Path = Path("var/media/images")
    image_generation_enabled: bool = True
    image_generation_model: str = Field(default=DEFAULT_IMAGE_GENERATION_MODEL, min_length=1)
    image_generation_timeout_seconds: float = Field(default=120.0, gt=0)
    image_generation_tool_timeout_seconds: float = Field(default=150.0, gt=0)
    image_generation_max_source_images: int = Field(default=5, gt=0)
    image_generation_max_output_images: int = Field(default=1, gt=0)
    image_generation_max_output_size_bytes: int = Field(default=10_000_000, gt=0)
    image_generation_media_dir: Path = Path("var/media/generated-images")
    document_reading_enabled: bool = True
    document_reading_tool_timeout_seconds: float = Field(default=10.0, gt=0)
    document_max_size_bytes: int = Field(default=2_000_000, gt=0)
    document_max_extracted_chars: int = Field(default=80_000, gt=0)
    document_media_dir: Path = Path("var/media/documents")
    groq_api_key: SecretStr | None = None
    groq_http_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    groq_http_read_timeout_seconds: float = Field(default=30.0, gt=0)
    groq_http_write_timeout_seconds: float = Field(default=30.0, gt=0)
    groq_http_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    groq_http_keepalive_expiry_seconds: float = Field(default=60.0, ge=0)

    postgres_dsn: str | None = None
    postgres_pool_min_size: int = Field(default=1, ge=0)
    postgres_pool_max_size: int = Field(default=10, ge=1)
    postgres_command_timeout_seconds: float = Field(default=30.0, gt=0)
    redis_dsn: str | None = None
    redis_context_snapshot_ttl_seconds: int = Field(default=24 * 60 * 60, gt=0)
    recent_message_limit: int = Field(default=100, gt=0)
    memory_compaction_enabled: bool = False
    memory_model_context_window_tokens: int = Field(default=196_600, gt=0)
    memory_reserved_output_tokens: int = Field(default=16_384, ge=0)
    memory_compaction_trigger_fraction: float = Field(default=0.80, gt=0, lt=1)
    memory_recent_tail_fraction: float = Field(default=0.30, gt=0, lt=1)
    memory_compaction_queue_maxsize: int = Field(default=1000, ge=0)
    memory_compaction_worker_count: int = Field(default=0, ge=0)
    memory_compaction_worker_error_backoff_seconds: float = Field(default=0.1, ge=0)
    memory_compaction_publish_timeout_seconds: float = Field(default=0.1, ge=0)
    memory_compaction_timeout_seconds: float = Field(default=120.0, gt=0)
    memory_compaction_target_summary_tokens: int = Field(
        default=DEFAULT_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
        gt=0,
        le=MAX_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS,
    )
    memory_compaction_model: str | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_webhook_secret_token: SecretStr | None = None
    telegram_render_markdown: bool = False
    telegram_rich_messages_enabled: bool = False
    telegram_thinking_draft_enabled: bool = False
    telegram_thinking_draft_timeout_seconds: float = Field(default=1.0, gt=0)
    telegram_media_group_debounce_seconds: float = Field(default=2.0, gt=0)
    telegram_media_group_ttl_seconds: int = Field(default=60, gt=0)
    telegram_media_group_flush_interval_seconds: float = Field(default=0.5, gt=0)
    telegram_media_group_lock_ttl_seconds: float = Field(default=10.0, gt=0)
    telegram_http_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    telegram_http_read_timeout_seconds: float = Field(default=15.0, gt=0)
    telegram_http_write_timeout_seconds: float = Field(default=10.0, gt=0)
    telegram_http_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    telegram_http_keepalive_expiry_seconds: float = Field(default=60.0, ge=0)
    openrouter_http_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    openrouter_http_write_timeout_seconds: float = Field(default=10.0, gt=0)
    openrouter_http_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    openrouter_http_keepalive_expiry_seconds: float = Field(default=60.0, ge=0)
    openrouter_provider_sort: str | None = None
    openrouter_provider_preferred_max_latency_p90: float | None = Field(default=None, gt=0)
    openrouter_provider_preferred_max_latency_p99: float | None = Field(default=None, gt=0)
    openrouter_api_key: SecretStr | None = None
    logfire_token: SecretStr | None = None

    @field_validator(
        "agent_provider",
        "agent_model",
        "vkusvill_mcp_command",
        "vkusvill_mcp_url",
        "tutu_mcp_command",
        "tutu_mcp_url",
        "postgres_dsn",
        "redis_dsn",
        "memory_compaction_model",
        "image_analysis_model",
        "openrouter_provider_sort",
        "telegram_bot_token",
        "telegram_webhook_secret_token",
        "openrouter_api_key",
        "tavily_api_key",
        "groq_api_key",
        "logfire_token",
        mode="before",
    )
    @classmethod
    def empty_optional_value_must_be_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def postgres_pool_min_size_must_not_exceed_max_size(self) -> Self:
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError("postgres_pool_min_size must be less than or equal to max size")
        if self.memory_reserved_output_tokens >= self.memory_model_context_window_tokens:
            raise ValueError("memory_reserved_output_tokens must be less than model context window")
        if self.memory_recent_tail_fraction >= self.memory_compaction_trigger_fraction:
            raise ValueError(
                "memory_recent_tail_fraction must be less than compaction trigger fraction"
            )
        latency_p90 = self.openrouter_provider_preferred_max_latency_p90
        latency_p99 = self.openrouter_provider_preferred_max_latency_p99
        if (latency_p90 is None) != (latency_p99 is None):
            raise ValueError(
                "openrouter_provider_preferred_max_latency_p90 and p99 must be configured together"
            )
        if latency_p90 is not None and latency_p99 is not None and latency_p90 > latency_p99:
            raise ValueError(
                "openrouter_provider_preferred_max_latency_p90 must be less than or equal to p99"
            )
        if self.environment != "test" and self.telegram_webhook_secret_token is None:
            raise ValueError(
                "telegram_webhook_secret_token is required outside the test environment"
            )
        if self.vkusvill_mcp_command is not None and self.vkusvill_mcp_url is not None:
            raise ValueError("configure either vkusvill_mcp_command or vkusvill_mcp_url, not both")
        if self.vkusvill_mcp_args and self.vkusvill_mcp_command is None:
            raise ValueError("vkusvill_mcp_args requires vkusvill_mcp_command")
        if self.vkusvill_mcp_env and self.vkusvill_mcp_command is None:
            raise ValueError("vkusvill_mcp_env requires vkusvill_mcp_command")
        if self.tutu_mcp_command is not None and self.tutu_mcp_url is not None:
            raise ValueError("configure either tutu_mcp_command or tutu_mcp_url, not both")
        if self.tutu_mcp_args and self.tutu_mcp_command is None:
            raise ValueError("tutu_mcp_args requires tutu_mcp_command")
        if self.tutu_mcp_env and self.tutu_mcp_command is None:
            raise ValueError("tutu_mcp_env requires tutu_mcp_command")
        return self

    @field_validator(
        "agent_retry_backoff_seconds",
        "delivery_retry_backoff_seconds",
        "transcription_retry_backoff_seconds",
    )
    @classmethod
    def retry_backoff_seconds_must_not_be_negative(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        if any(delay < 0 for delay in value):
            raise ValueError("retry backoff seconds must not contain negative values")
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
