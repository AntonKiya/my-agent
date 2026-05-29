from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "prod", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENT_SERVICE_",
        extra="ignore",
    )

    service_name: str = "agent-service"
    environment: Environment = "local"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "INFO"
    graceful_shutdown_timeout_seconds: float = Field(default=10.0, gt=0)
    inbound_queue_maxsize: int = Field(default=5000, ge=0)
    outbound_queue_maxsize: int = Field(default=5000, ge=0)
    inbound_worker_count: int = Field(default=8, ge=0)
    agent_retry_max_attempts: int = Field(default=3, ge=1)
    agent_retry_backoff_seconds: tuple[float, ...] = (1.0, 5.0, 15.0)

    postgres_dsn: str | None = None
    postgres_pool_min_size: int = Field(default=1, ge=0)
    postgres_pool_max_size: int = Field(default=10, ge=1)
    postgres_command_timeout_seconds: float = Field(default=30.0, gt=0)
    redis_dsn: str | None = None
    telegram_bot_token: SecretStr | None = None
    logfire_token: SecretStr | None = None

    @field_validator(
        "postgres_dsn",
        "redis_dsn",
        "telegram_bot_token",
        "logfire_token",
        mode="before",
    )
    @classmethod
    def empty_optional_secret_or_dsn_must_be_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def postgres_pool_min_size_must_not_exceed_max_size(self) -> Self:
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError("postgres_pool_min_size must be less than or equal to max size")
        return self

    @field_validator("agent_retry_backoff_seconds")
    @classmethod
    def agent_retry_backoff_seconds_must_not_be_negative(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        if any(delay < 0 for delay in value):
            raise ValueError("agent_retry_backoff_seconds must not contain negative values")
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
