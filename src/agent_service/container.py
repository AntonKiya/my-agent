import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Protocol, cast

import asyncpg
import httpx
import redis.asyncio as redis

from agent_service.agents import AgentBoundary, build_openrouter_agent_boundary
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
from agent_service.delivery import DeliveryRetryPolicy, DeliveryWorker
from agent_service.inbound import (
    AgentRetryPolicy,
    InboundIntake,
    InboundIntakeService,
    InboundWorker,
)
from agent_service.memory import (
    ConversationCompactionPolicy,
    ConversationCompactionStore,
    ConversationCompactionWorker,
    ConversationCompactor,
    ConversationContextSnapshotStore,
    ConversationMemoryService,
    ConversationMemoryStore,
    DefaultConversationMemoryService,
    PostgresConversationCompactionStore,
    PostgresConversationMemoryStore,
    RedisConversationContextSnapshotStore,
)
from agent_service.memory import PostgresPool as MemoryPostgresPool
from agent_service.messaging.in_memory import (
    AsyncioCompactionQueue,
    AsyncioInboundQueue,
    AsyncioOutboundQueue,
)
from agent_service.messaging.interfaces import CompactionQueue, InboundQueue
from agent_service.outbound import OutboundQueue
from agent_service.runtime.lifecycle import TaskSupervisor
from agent_service.users import PostgresPool, PostgresUserStore, UserResolver

logger = logging.getLogger(__name__)


class ManagedPostgresPool(PostgresPool, Protocol):
    async def close(self) -> None:
        """Close the underlying Postgres connection pool."""


class ManagedRedisClient(Protocol):
    async def get(self, name: str) -> bytes | str | None:
        """Return a Redis string value."""

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
    ) -> object:
        """Set a Redis string value."""

    async def delete(self, *names: str) -> object:
        """Delete Redis keys."""

    async def ping(self) -> object:
        """Verify that Redis is reachable."""

    async def aclose(self) -> None:
        """Close the Redis client."""


@dataclass(slots=True)
class AppContainer:
    # Central assembly point for infrastructure dependencies and their lifecycle.
    # It should not contain message processing, agent logic, or channel behavior.
    settings: AppSettings
    task_supervisor: TaskSupervisor = field(init=False)
    inbound_queue: InboundQueue = field(init=False)
    outbound_queue: OutboundQueue = field(init=False)
    compaction_queue: CompactionQueue = field(init=False)
    inbound_intake_service: InboundIntake | None = field(init=False)
    conversation_resolver: ConversationResolverProtocol | None = field(init=False)
    conversation_lock_manager: ConversationLockManager = field(init=False)
    delivery_lock_manager: ConversationLockManager = field(init=False)
    conversation_memory_store: ConversationMemoryStore | None = field(init=False)
    conversation_snapshot_store: ConversationContextSnapshotStore | None = field(init=False)
    conversation_compaction_store: ConversationCompactionStore | None = field(init=False)
    conversation_compaction_policy: ConversationCompactionPolicy = field(init=False)
    conversation_compactor: ConversationCompactor | None = field(init=False)
    memory_service: ConversationMemoryService | None = field(init=False)
    agent_boundary: AgentBoundary | None = field(init=False)
    channel_adapters: ChannelAdapterRegistry = field(init=False)
    telegram_adapter: TelegramAdapter | None = field(init=False)
    _postgres_pool: ManagedPostgresPool | None = field(default=None, init=False, repr=False)
    _redis_client: ManagedRedisClient | None = field(default=None, init=False, repr=False)
    _telegram_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _agent_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
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
        self.compaction_queue = AsyncioCompactionQueue(
            maxsize=self.settings.memory_compaction_queue_maxsize,
        )
        self.inbound_intake_service = None
        self.conversation_resolver = None
        self.conversation_lock_manager = AsyncioConversationLockManager()
        self.delivery_lock_manager = AsyncioConversationLockManager()
        self.conversation_memory_store = None
        self.conversation_snapshot_store = None
        self.conversation_compaction_store = None
        self.conversation_compaction_policy = ConversationCompactionPolicy(
            enabled=self.settings.memory_compaction_enabled,
            context_window_tokens=self.settings.memory_model_context_window_tokens,
            reserved_output_tokens=self.settings.memory_reserved_output_tokens,
            trigger_fraction=self.settings.memory_compaction_trigger_fraction,
            recent_tail_fraction=self.settings.memory_recent_tail_fraction,
        )
        self.conversation_compactor = None
        self.memory_service = None
        self.agent_boundary = self._build_agent_boundary()
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
        try:
            if self.agent_boundary is None:
                self.agent_boundary = self._build_agent_boundary()
            await self._start_redis_dependencies()
            await self._start_postgres_dependencies()
            self._start_inbound_workers()
            self._start_delivery_workers()
            self._start_compaction_workers()
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        await self.task_supervisor.stop(group="inbound")
        await self.task_supervisor.stop(group="compaction")
        await self._drain_outbound_queue()
        await self.task_supervisor.stop(group="delivery")
        await self.task_supervisor.stop()
        if self._telegram_http_client is not None:
            await self._telegram_http_client.aclose()
            self._telegram_http_client = None
        if self._agent_http_client is not None:
            await self._agent_http_client.aclose()
            self._agent_http_client = None
            self.agent_boundary = None
        if self._postgres_pool is not None:
            await self._postgres_pool.close()
            self._postgres_pool = None
            self.inbound_intake_service = None
            self.conversation_resolver = None
            self.conversation_memory_store = None
            self.conversation_compaction_store = None
            self.memory_service = None
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._redis_client = None
            self.conversation_snapshot_store = None
        self._started = False

    def _build_telegram_adapter(self) -> TelegramAdapter | None:
        if self.settings.telegram_bot_token is None:
            return None

        self._telegram_http_client = httpx.AsyncClient()
        return TelegramAdapter(
            bot_token=self.settings.telegram_bot_token,
            client=self._telegram_http_client,
        )

    def _build_agent_boundary(self) -> AgentBoundary | None:
        if self.settings.agent_provider != "openrouter":
            return None
        if self.settings.agent_model is None or self.settings.openrouter_api_key is None:
            return None

        self._agent_http_client = httpx.AsyncClient()
        return build_openrouter_agent_boundary(
            model_name=self.settings.agent_model,
            api_key=self.settings.openrouter_api_key.get_secret_value(),
            http_client=self._agent_http_client,
            timeout_seconds=self.settings.agent_timeout_seconds,
        )

    async def _start_redis_dependencies(self) -> None:
        if self.settings.redis_dsn is None:
            return

        self._redis_client = await _create_managed_redis_client(self.settings)
        self.conversation_snapshot_store = RedisConversationContextSnapshotStore(
            self._redis_client,
            ttl_seconds=self.settings.redis_context_snapshot_ttl_seconds,
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
            publish_timeout_seconds=self.settings.inbound_publish_timeout_seconds,
        )
        conversation_store = PostgresConversationStore(
            cast(ConversationPostgresPool, self._postgres_pool)
        )
        self.conversation_resolver = ConversationResolver(conversation_store)
        self.conversation_memory_store = PostgresConversationMemoryStore(
            cast(MemoryPostgresPool, self._postgres_pool)
        )
        self.conversation_compaction_store = PostgresConversationCompactionStore(
            cast(MemoryPostgresPool, self._postgres_pool)
        )
        self.memory_service = DefaultConversationMemoryService(
            memory_store=self.conversation_memory_store,
            snapshot_store=self.conversation_snapshot_store,
            compaction_store=self.conversation_compaction_store,
        )

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
                error_backoff_seconds=self.settings.inbound_worker_error_backoff_seconds,
                compaction_queue=(
                    self.compaction_queue if self._compaction_processing_enabled() else None
                ),
                compaction_policy=(
                    self.conversation_compaction_policy
                    if self._compaction_processing_enabled()
                    else None
                ),
                compaction_publish_timeout_seconds=(
                    self.settings.memory_compaction_publish_timeout_seconds
                ),
            )
            self.task_supervisor.create_task(
                worker.run_forever(),
                name=f"inbound-worker-{index + 1}",
                group="inbound",
            )

    def _start_compaction_workers(self) -> None:
        if not self._compaction_processing_enabled():
            return
        if self.memory_service is None or self.conversation_compactor is None:
            return

        for index in range(self.settings.memory_compaction_worker_count):
            worker = ConversationCompactionWorker(
                compaction_queue=self.compaction_queue,
                memory_service=self.memory_service,
                compactor=self.conversation_compactor,
                lock_manager=self.conversation_lock_manager,
                error_backoff_seconds=(
                    self.settings.memory_compaction_worker_error_backoff_seconds
                ),
            )
            self.task_supervisor.create_task(
                worker.run_forever(),
                name=f"compaction-worker-{index + 1}",
                group="compaction",
            )

    def _start_delivery_workers(self) -> None:
        if self.settings.delivery_worker_count == 0:
            return
        if not getattr(self.channel_adapters, "channels", ()):
            return

        retry_policy = DeliveryRetryPolicy(
            max_attempts=self.settings.delivery_retry_max_attempts,
            backoff_seconds=self.settings.delivery_retry_backoff_seconds,
        )
        for index in range(self.settings.delivery_worker_count):
            worker = DeliveryWorker(
                outbound_queue=self.outbound_queue,
                channel_adapters=self.channel_adapters,
                lock_manager=self.delivery_lock_manager,
                retry_policy=retry_policy,
                error_backoff_seconds=self.settings.delivery_worker_error_backoff_seconds,
            )
            self.task_supervisor.create_task(
                worker.run_forever(),
                name=f"delivery-worker-{index + 1}",
                group="delivery",
            )

    def _compaction_processing_enabled(self) -> bool:
        return (
            self.conversation_compaction_policy.enabled
            and self.settings.memory_compaction_worker_count > 0
            and self.conversation_compactor is not None
        )

    async def _drain_outbound_queue(self) -> None:
        if self.task_supervisor.task_count_for_group("delivery") == 0:
            if self.outbound_queue.stats.size > 0:
                logger.warning(
                    "Outbound queue was not drained because no delivery workers are running",
                    extra={
                        "event": "outbound_queue_drain_skipped",
                        "queue_size": self.outbound_queue.stats.size,
                    },
                )
            return

        try:
            await asyncio.wait_for(
                self.outbound_queue.join(),
                timeout=self.settings.graceful_shutdown_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Outbound queue did not drain before shutdown timeout",
                extra={
                    "event": "outbound_queue_drain_timeout",
                    "queue_size": self.outbound_queue.stats.size,
                    "shutdown_timeout_seconds": self.settings.graceful_shutdown_timeout_seconds,
                },
            )
        else:
            logger.info(
                "Outbound queue drained before shutdown",
                extra={
                    "event": "outbound_queue_drained",
                    "queue_size": self.outbound_queue.stats.size,
                },
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


async def _create_managed_redis_client(settings: AppSettings) -> ManagedRedisClient:
    if settings.redis_dsn is None:
        raise ValueError("redis_dsn must be configured before creating a Redis client")
    client = cast(
        ManagedRedisClient,
        redis.from_url(
            settings.redis_dsn,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
        ),
    )
    try:
        await client.ping()
    except BaseException:
        await client.aclose()
        raise
    return client
