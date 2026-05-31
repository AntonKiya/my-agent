import asyncio
from contextlib import AbstractAsyncContextManager
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from agent_service.agents import AgentRequest, AgentResponse, PydanticAIAgentBoundary
from agent_service.channels import (
    ChannelAdapterNotFoundError,
    ChannelAdapterRegistry,
    InMemoryChannelAdapterRegistry,
)
from agent_service.channels.telegram import TelegramAdapter
from agent_service.config import AppSettings
from agent_service.container import AppContainer
from agent_service.conversations import Conversation, ConversationResolverProtocol
from agent_service.delivery import DeliveryResult, DeliveryStatus
from agent_service.inbound import InboundIntake
from agent_service.memory import (
    ConversationCompactionDecision,
    ConversationCompactionPolicy,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationCompactionStore,
    ConversationContextSnapshotStore,
    ConversationMemoryMessage,
    ConversationMemoryStore,
    ConversationSummary,
    DefaultConversationMemoryService,
    PreparedConversationContext,
)
from agent_service.messaging.interfaces import CompactionQueue, InboundQueue
from agent_service.outbound import OutboundEvent, OutboundQueue
from agent_service.users import PostgresConnection


class FakeManagedPostgresPool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise NotImplementedError


class FakeManagedRedisClient:
    def __init__(self, *, ping_error: BaseException | None = None) -> None:
        self.closed = False
        self.pinged = False
        self.ping_error = ping_error

    async def get(self, name: str) -> bytes | str | None:
        return None

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
    ) -> object:
        return True

    async def delete(self, *names: str) -> object:
        return len(names)

    async def ping(self) -> object:
        self.pinged = True
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def aclose(self) -> None:
        self.closed = True


class FakeAgentBoundary:
    async def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(text=request.text or "ok", trace_id=request.trace_id)


class FakeCompactor:
    async def compact(
        self,
        *,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionResult:
        return ConversationCompactionResult(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            summary=request.previous_summary or "summary",
            last_compacted_sequence=request.last_compacted_sequence,
        )


class FakeDeliveryAdapter:
    channel = "telegram"

    def __init__(self) -> None:
        self.events: list[OutboundEvent] = []
        self.started = asyncio.Event()

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        self.events.append(event)
        self.started.set()
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=DeliveryStatus.SENT,
            external_message_ids=["501"],
        )


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

    async def prepare_compaction_request(
        self,
        *,
        conversation: Conversation,
        compact_through_sequence: int | None = None,
    ) -> ConversationCompactionRequest:
        raise NotImplementedError

    async def evaluate_compaction(
        self,
        *,
        conversation: Conversation,
        policy: object,
    ) -> ConversationCompactionDecision:
        raise NotImplementedError

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        raise NotImplementedError


async def test_container_tracks_lifecycle_state() -> None:
    settings = AppSettings(environment="test", graceful_shutdown_timeout_seconds=0.25)
    container = AppContainer(settings=settings)

    assert not container.started
    assert container.task_supervisor.shutdown_timeout_seconds == 0.25
    assert isinstance(container.inbound_queue, InboundQueue)
    assert isinstance(container.outbound_queue, OutboundQueue)
    assert isinstance(container.compaction_queue, CompactionQueue)
    assert container.inbound_intake_service is None
    assert container.conversation_resolver is None
    assert container.conversation_memory_store is None
    assert container.conversation_snapshot_store is None
    assert container.conversation_compaction_store is None
    assert isinstance(container.conversation_compaction_policy, ConversationCompactionPolicy)
    assert not container.conversation_compaction_policy.enabled
    assert container.conversation_compactor is None
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
    telegram_http_client = container._telegram_http_client

    await container.start()
    await container.stop()

    assert telegram_http_client is not None
    assert telegram_http_client.is_closed
    assert container._telegram_http_client is None


async def test_container_does_not_build_agent_boundary_without_complete_agent_config() -> None:
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
    )
    container = AppContainer(settings=settings)

    assert container.agent_boundary is None
    assert container._agent_http_client is None

    await container.start()

    assert container.started
    assert container.agent_boundary is None
    assert container._agent_http_client is None

    await container.stop()


async def test_container_builds_openrouter_agent_boundary_when_configured() -> None:
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        agent_timeout_seconds=12.5,
    )
    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert container.agent_boundary.timeout_seconds == 12.5
    assert container._agent_http_client is not None
    assert not container._agent_http_client.is_closed

    await container.stop()

    assert container._agent_http_client is None
    assert container.agent_boundary is None


async def test_new_container_gets_fresh_openrouter_agent_http_client() -> None:
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
    )
    first_container = AppContainer(settings=settings)
    first_http_client = first_container._agent_http_client

    await first_container.stop()
    second_container = AppContainer(settings=settings)

    assert first_http_client is not None
    assert first_http_client.is_closed
    assert isinstance(second_container.agent_boundary, PydanticAIAgentBoundary)
    assert second_container._agent_http_client is not None
    assert second_container._agent_http_client is not first_http_client

    await second_container.stop()


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
    assert isinstance(container.conversation_memory_store, ConversationMemoryStore)
    assert isinstance(container.conversation_compaction_store, ConversationCompactionStore)
    assert container.conversation_snapshot_store is None
    assert isinstance(container.memory_service, DefaultConversationMemoryService)
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
    assert container.conversation_memory_store is None
    assert container.conversation_compaction_store is None
    assert container.conversation_snapshot_store is None


async def test_container_wires_redis_snapshot_store_when_dsn_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeManagedRedisClient()
    redis_kwargs: dict[str, object] = {}

    def fake_from_url(url: str, **kwargs: object) -> FakeManagedRedisClient:
        redis_kwargs.update({"url": url, **kwargs})
        return fake_client

    monkeypatch.setattr("agent_service.container.redis.from_url", fake_from_url)
    settings = AppSettings(
        environment="test",
        redis_dsn="redis://127.0.0.1:6379/0",
        redis_context_snapshot_ttl_seconds=60,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert container.started
    assert fake_client.pinged
    assert isinstance(container.conversation_snapshot_store, ConversationContextSnapshotStore)
    assert redis_kwargs == {
        "url": "redis://127.0.0.1:6379/0",
        "decode_responses": True,
        "socket_connect_timeout": 5.0,
        "socket_timeout": 5.0,
    }

    await container.stop()

    assert fake_client.closed
    assert container._redis_client is None
    assert container.conversation_snapshot_store is None


async def test_container_passes_redis_snapshot_store_to_memory_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()
    fake_client = FakeManagedRedisClient()

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    def fake_from_url(url: str, **kwargs: object) -> FakeManagedRedisClient:
        return fake_client

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr("agent_service.container.redis.from_url", fake_from_url)
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        redis_dsn="redis://127.0.0.1:6379/0",
        memory_compaction_enabled=True,
        memory_model_context_window_tokens=100_000,
        memory_reserved_output_tokens=8_000,
        memory_compaction_trigger_fraction=0.75,
        memory_recent_tail_fraction=0.25,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert isinstance(container.memory_service, DefaultConversationMemoryService)
    assert container.memory_service.snapshot_store is container.conversation_snapshot_store
    assert container.memory_service.compaction_store is container.conversation_compaction_store
    assert container.conversation_compaction_policy.enabled
    assert container.conversation_compaction_policy.context_window_tokens == 100_000
    assert container.conversation_compaction_policy.reserved_output_tokens == 8_000
    assert container.conversation_compaction_policy.trigger_fraction == 0.75
    assert container.conversation_compaction_policy.recent_tail_fraction == 0.25

    await container.stop()


async def test_container_closes_redis_client_when_ping_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeManagedRedisClient(ping_error=RuntimeError("redis down"))

    def fake_from_url(url: str, **kwargs: object) -> FakeManagedRedisClient:
        return fake_client

    monkeypatch.setattr("agent_service.container.redis.from_url", fake_from_url)
    container = AppContainer(
        settings=AppSettings(
            environment="test",
            redis_dsn="redis://127.0.0.1:6379/0",
        )
    )

    with pytest.raises(RuntimeError, match="redis down"):
        await container.start()

    assert fake_client.closed
    assert not container.started
    assert container._redis_client is None
    assert container.conversation_snapshot_store is None


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


async def test_container_starts_configured_delivery_worker_tasks() -> None:
    settings = AppSettings(
        environment="test",
        telegram_bot_token=SecretStr("token"),
        delivery_worker_count=3,
        graceful_shutdown_timeout_seconds=0.1,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert container.started
    assert container.task_supervisor.task_count == 3

    await container.stop()

    assert container.task_supervisor.task_count == 0


async def test_container_stop_drains_generated_outbound_events_before_delivery_shutdown() -> None:
    settings = AppSettings(
        environment="test",
        delivery_worker_count=1,
        graceful_shutdown_timeout_seconds=0.5,
    )
    container = AppContainer(settings=settings)
    adapter = FakeDeliveryAdapter()
    container.channel_adapters.register(adapter)
    event = OutboundEvent(
        channel="telegram",
        user_id=uuid4(),
        conversation_id=uuid4(),
        external_chat_id="12345",
        text="generated response",
    )
    await container.outbound_queue.publish(event)

    await container.start()
    await container.stop()

    assert adapter.events == [event]
    assert event.status is DeliveryStatus.SENT
    assert container.outbound_queue.stats.size == 0
    assert container.task_supervisor.task_count == 0


async def test_container_starts_inbound_workers_with_configured_openrouter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        inbound_worker_count=2,
        graceful_shutdown_timeout_seconds=0.1,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert container.started
    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert container.task_supervisor.task_count == 2

    await container.stop()

    assert container.task_supervisor.task_count == 0
    assert fake_pool.closed


async def test_container_starts_configured_compaction_worker_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        inbound_worker_count=0,
        memory_compaction_enabled=True,
        memory_compaction_worker_count=2,
        graceful_shutdown_timeout_seconds=0.1,
    )
    container = AppContainer(settings=settings)
    container.conversation_compactor = FakeCompactor()

    await container.start()

    assert container.started
    assert container.task_supervisor.task_count == 2

    await container.stop()

    assert container.task_supervisor.task_count == 0
    assert fake_pool.closed
