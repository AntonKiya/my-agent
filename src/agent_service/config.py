from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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

    postgres_dsn: str | None = None
    redis_dsn: str | None = None
    telegram_bot_token: SecretStr | None = None
    logfire_token: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
