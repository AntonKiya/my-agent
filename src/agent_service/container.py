from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Protocol, cast

import asyncpg
import httpx

from agent_service.agents import AgentBoundary
from agent_service.channels import ChannelAdapterRegistry, InMemoryChannelAdapterRegistry
from agent_service.channels.telegram import TelegramAdapter
from agent_service.config import AppSettings
from agent_service.conversations import (
    AsyncioConversationLockManager,
    ConversationLockManager,
    ConversationResolver,
    ConversationResolverProtocol,
    PostgresConversationStore,
)
from agent_service.conversations import (
    PostgresPool as ConversationPostgresPool,
)
from agent_service.inbound import (
    AgentRetryPolicy,
    InboundIntake,
    InboundIntakeService,
    InboundWorker,
)
from agent_service.memory import ConversationMemoryService
from agent_service.messaging import (
    AsyncioInboundQueue,
    AsyncioOutboundQueue,
    InboundQueue,
    OutboundQueue,
)
from agent_service.runtime.lifecycle import TaskSupervisor
from agent_service.users import PostgresPool, PostgresUserStore, UserResolver


class ManagedPostgresPool(PostgresPool, Protocol):
    async def close(self) -> None:
        """Close the underlying Postgres connection pool."""


@dataclass(slots=True)
class AppContainer:
    # Central assembly point for infrastructure dependencies and their lifecycle.
    # It should not contain message processing, agent logic, or channel behavior.
    settings: AppSettings
    task_supervisor: TaskSupervisor = field(init=False)
    inbound_queue: InboundQueue = field(init=False)
    outbound_queue: OutboundQueue = field(init=False)
    inbound_intake_service: InboundIntake | None = field(init=False)
    conversation_resolver: ConversationResolverProtocol | None = field(init=False)
    conversation_lock_manager: ConversationLockManager = field(init=False)
    memory_service: ConversationMemoryService | None = field(init=False)
    agent_boundary: AgentBoundary | None = field(init=False)
    channel_adapters: ChannelAdapterRegistry = field(init=False)
    telegram_adapter: TelegramAdapter | None = field(init=False)
    _postgres_pool: ManagedPostgresPool | None = field(default=None, init=False, repr=False)
    _telegram_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.task_supervisor = TaskSupervisor(
            shutdown_timeout_seconds=self.settings.graceful_shutdown_timeout_seconds,
        )
        self.inbound_queue = AsyncioInboundQueue(
            maxsize=self.settings.inbound_queue_maxsize,
        )
        self.outbound_queue = AsyncioOutboundQueue(
            maxsize=self.settings.outbound_queue_maxsize,
        )
        self.inbound_intake_service = None
        self.conversation_resolver = None
        self.conversation_lock_manager = AsyncioConversationLockManager()
        self.memory_service = None
        self.agent_boundary = None
        self.channel_adapters = InMemoryChannelAdapterRegistry()
        self.telegram_adapter = self._build_telegram_adapter()
        if self.telegram_adapter is not None:
            self.channel_adapters.register(self.telegram_adapter)

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        # The container owns infrastructure lifecycle, not business processing.
        if self._started:
            return
        await self._start_postgres_dependencies()
        self._start_inbound_workers()
        self._started = True

    async def stop(self) -> None:
        await self.task_supervisor.stop()
        if self._telegram_http_client is not None:
            await self._telegram_http_client.aclose()
        if self._postgres_pool is not None:
            await self._postgres_pool.close()
            self._postgres_pool = None
            self.inbound_intake_service = None
            self.conversation_resolver = None
        self._started = False

    def _build_telegram_adapter(self) -> TelegramAdapter | None:
        if self.settings.telegram_bot_token is None:
            return None

        self._telegram_http_client = httpx.AsyncClient()
        return TelegramAdapter(
            bot_token=self.settings.telegram_bot_token,
            client=self._telegram_http_client,
        )

    async def _start_postgres_dependencies(self) -> None:
        if self.settings.postgres_dsn is None:
            return

        self._postgres_pool = await _create_managed_postgres_pool(self.settings)
        user_store = PostgresUserStore(self._postgres_pool)
        user_resolver = UserResolver(user_store)
        self.inbound_intake_service = InboundIntakeService(
            user_resolver=user_resolver,
            inbound_queue=self.inbound_queue,
        )
        conversation_store = PostgresConversationStore(
            cast(ConversationPostgresPool, self._postgres_pool)
        )
        self.conversation_resolver = ConversationResolver(conversation_store)

    def _start_inbound_workers(self) -> None:
        if self.settings.inbound_worker_count == 0:
            return
        if self.conversation_resolver is None:
            return
        if self.memory_service is None or self.agent_boundary is None:
            return

        retry_policy = AgentRetryPolicy(
            max_attempts=self.settings.agent_retry_max_attempts,
            backoff_seconds=self.settings.agent_retry_backoff_seconds,
        )
        for index in range(self.settings.inbound_worker_count):
            worker = InboundWorker(
                inbound_queue=self.inbound_queue,
                outbound_queue=self.outbound_queue,
                conversation_resolver=self.conversation_resolver,
                memory_service=self.memory_service,
                agent_boundary=self.agent_boundary,
                lock_manager=self.conversation_lock_manager,
                retry_policy=retry_policy,
            )
            self.task_supervisor.create_task(
                worker.run_forever(),
                name=f"inbound-worker-{index + 1}",
            )


async def _create_managed_postgres_pool(settings: AppSettings) -> ManagedPostgresPool:
    pool = await cast(
        Awaitable[object],
        asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            command_timeout=settings.postgres_command_timeout_seconds,
        ),
    )
    return cast(ManagedPostgresPool, pool)
