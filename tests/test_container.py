from contextlib import AbstractAsyncContextManager
from uuid import UUID

import pytest
from pydantic import SecretStr

from agent_service.agents import AgentRequest, AgentResponse
from agent_service.channels import (
    ChannelAdapterNotFoundError,
    ChannelAdapterRegistry,
    InMemoryChannelAdapterRegistry,
)
from agent_service.channels.telegram import TelegramAdapter
from agent_service.config import AppSettings
from agent_service.container import AppContainer
from agent_service.conversations import Conversation, ConversationResolverProtocol
from agent_service.inbound import InboundIntake
from agent_service.memory import ConversationMemoryMessage, PreparedConversationContext
from agent_service.messaging import InboundQueue, OutboundQueue
from agent_service.users import PostgresConnection


class FakeManagedPostgresPool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise NotImplementedError


class FakeAgentBoundary:
    async def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(text=request.text or "ok", trace_id=request.trace_id)


class FakeMemoryService:
    async def record_user_message(
        self,
        *,
        conversation: Conversation,
        event: object,
    ) -> ConversationMemoryMessage:
        raise NotImplementedError

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        raise NotImplementedError

    async def record_assistant_message(
        self,
        *,
        conversation: Conversation,
        response: AgentResponse,
        trace_id: str | None = None,
        outbound_event_id: UUID | None = None,
    ) -> ConversationMemoryMessage:
        raise NotImplementedError


async def test_container_tracks_lifecycle_state() -> None:
    settings = AppSettings(environment="test", graceful_shutdown_timeout_seconds=0.25)
    container = AppContainer(settings=settings)

    assert not container.started
    assert container.task_supervisor.shutdown_timeout_seconds == 0.25
    assert isinstance(container.inbound_queue, InboundQueue)
    assert isinstance(container.outbound_queue, OutboundQueue)
    assert container.inbound_intake_service is None
    assert container.conversation_resolver is None
    assert container.memory_service is None
    assert container.agent_boundary is None
    assert container.task_supervisor.task_count == 0
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


async def test_container_wires_postgres_user_intake_when_dsn_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()
    create_pool_kwargs: dict[str, object] = {}

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        create_pool_kwargs.update(kwargs)
        return fake_pool

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        postgres_pool_min_size=2,
        postgres_pool_max_size=8,
        postgres_command_timeout_seconds=12.5,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert container.started
    assert container._postgres_pool is fake_pool
    assert isinstance(container.inbound_intake_service, InboundIntake)
    assert isinstance(container.conversation_resolver, ConversationResolverProtocol)
    assert container.task_supervisor.task_count == 0
    assert create_pool_kwargs == {
        "dsn": "postgresql://agent:secret@localhost:5432/agent",
        "min_size": 2,
        "max_size": 8,
        "command_timeout": 12.5,
    }

    await container.stop()

    assert fake_pool.closed
    assert container._postgres_pool is None
    assert container.inbound_intake_service is None
    assert container.conversation_resolver is None


async def test_container_starts_configured_inbound_worker_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        inbound_worker_count=3,
        graceful_shutdown_timeout_seconds=0.1,
    )
    container = AppContainer(settings=settings)
    container.memory_service = FakeMemoryService()
    container.agent_boundary = FakeAgentBoundary()

    await container.start()

    assert container.started
    assert container.task_supervisor.task_count == 3

    await container.stop()

    assert container.task_supervisor.task_count == 0
    assert fake_pool.closed
