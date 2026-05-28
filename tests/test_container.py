import pytest
from pydantic import SecretStr

from agent_service.channels import (
    ChannelAdapterNotFoundError,
    ChannelAdapterRegistry,
    InMemoryChannelAdapterRegistry,
)
from agent_service.channels.telegram import TelegramAdapter
from agent_service.config import AppSettings
from agent_service.container import AppContainer
from agent_service.messaging import InboundQueue, OutboundQueue


async def test_container_tracks_lifecycle_state() -> None:
    settings = AppSettings(environment="test", graceful_shutdown_timeout_seconds=0.25)
    container = AppContainer(settings=settings)

    assert not container.started
    assert container.task_supervisor.shutdown_timeout_seconds == 0.25
    assert isinstance(container.inbound_queue, InboundQueue)
    assert isinstance(container.outbound_queue, OutboundQueue)
    assert isinstance(container.channel_adapters, ChannelAdapterRegistry)
    assert isinstance(container.channel_adapters, InMemoryChannelAdapterRegistry)
    assert container.telegram_adapter is None

    with pytest.raises(ChannelAdapterNotFoundError):
        container.channel_adapters.get("telegram")

    await container.start()

    assert container.started

    await container.stop()

    assert not container.started


async def test_container_registers_telegram_adapter_when_token_is_configured() -> None:
    settings = AppSettings(environment="test", telegram_bot_token=SecretStr("token"))
    container = AppContainer(settings=settings)

    assert isinstance(container.telegram_adapter, TelegramAdapter)
    assert container.channel_adapters.get("telegram") is container.telegram_adapter

    await container.start()
    await container.stop()

    assert container._telegram_http_client is not None
    assert container._telegram_http_client.is_closed
