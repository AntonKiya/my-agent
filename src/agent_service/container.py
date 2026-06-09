import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import asyncpg
import httpx
import redis.asyncio as redis
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.toolsets import AgentToolset

from agent_service.agents import AgentBoundary, build_openrouter_agent_boundary
from agent_service.channels import ChannelAdapterRegistry, InMemoryChannelAdapterRegistry
from agent_service.channels.telegram import TelegramAdapter, TelegramMediaFetcher
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
from agent_service.image_analysis import (
    IMAGE_ANALYSIS_TOOLSET_ID,
    OpenRouterVisionAnalyzer,
    build_image_analysis_toolsets,
)
from agent_service.inbound import (
    AgentRetryPolicy,
    ContentProcessingRetryPolicy,
    InboundContentPreprocessor,
    InboundIdempotencyStore,
    InboundIntake,
    InboundIntakeService,
    InboundWorker,
    PostgresInboundIdempotencyStore,
)
from agent_service.mcp import VKUSVILL_SHOPPING_SKILL_ID, build_vkusvill_mcp_toolsets
from agent_service.media import (
    InMemoryMediaFetcherRegistry,
    MediaAssetStore,
    PersistentFileMediaStore,
    PostgresMediaAssetStore,
    TempFileMediaStore,
)
from agent_service.memory import (
    ConversationCompactionPolicy,
    ConversationCompactionStore,
    ConversationCompactionWorker,
    ConversationCompactor,
    ConversationContextSnapshotStore,
    ConversationMemoryService,
    ConversationMemoryStore,
    ConversationSummaryAgent,
    ConversationSummaryOutput,
    DefaultConversationMemoryService,
    PostgresConversationCompactionStore,
    PostgresConversationMemoryStore,
    PydanticAIConversationCompactor,
    RedisConversationContextSnapshotStore,
)
from agent_service.memory import PostgresPool as MemoryPostgresPool
from agent_service.messaging.in_memory import (
    AsyncioCompactionQueue,
    AsyncioInboundQueue,
    AsyncioOutboundQueue,
)
from agent_service.messaging.interfaces import CompactionQueue, InboundQueue
from agent_service.observability.events import elapsed_ms, log_event, start_timer
from agent_service.outbound import OutboundQueue
from agent_service.runtime.lifecycle import TaskSupervisor
from agent_service.transcription import GroqWhisperTranscriber
from agent_service.users import PostgresPool, PostgresUserStore, UserResolver
from agent_service.weather import (
    WEATHER_FORECAST_SKILL_ID,
    build_weather_forecast_toolsets,
)
from agent_service.web_research import build_web_research_toolsets

logger = logging.getLogger(__name__)

OPENROUTER_ERROR_RESPONSE_LOG_BODY_LIMIT = 12000
OPENROUTER_REQUEST_STARTED_AT_EXTENSION = "agent_service_openrouter_started_at"


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
    inbound_idempotency_store: InboundIdempotencyStore | None = field(init=False)
    conversation_resolver: ConversationResolverProtocol | None = field(init=False)
    conversation_lock_manager: ConversationLockManager = field(init=False)
    delivery_lock_manager: ConversationLockManager = field(init=False)
    conversation_memory_store: ConversationMemoryStore | None = field(init=False)
    conversation_snapshot_store: ConversationContextSnapshotStore | None = field(init=False)
    conversation_compaction_store: ConversationCompactionStore | None = field(init=False)
    conversation_compaction_policy: ConversationCompactionPolicy = field(init=False)
    conversation_compactor: ConversationCompactor | None = field(init=False)
    memory_service: ConversationMemoryService | None = field(init=False)
    media_asset_store: MediaAssetStore | None = field(init=False)
    agent_boundary: AgentBoundary | None = field(init=False)
    content_preprocessor: InboundContentPreprocessor | None = field(init=False)
    channel_adapters: ChannelAdapterRegistry = field(init=False)
    telegram_adapter: TelegramAdapter | None = field(init=False)
    _postgres_pool: ManagedPostgresPool | None = field(default=None, init=False, repr=False)
    _redis_client: ManagedRedisClient | None = field(default=None, init=False, repr=False)
    _telegram_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _agent_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _image_analysis_http_client: httpx.AsyncClient | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _groq_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _weather_http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _web_research_http_client: httpx.AsyncClient | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _compaction_http_client: httpx.AsyncClient | None = field(
        default=None,
        init=False,
        repr=False,
    )
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
        self.inbound_idempotency_store = None
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
        self.conversation_compactor = self._build_conversation_compactor()
        self.memory_service = None
        self.media_asset_store = None
        self.agent_boundary = self._build_agent_boundary()
        self.channel_adapters = InMemoryChannelAdapterRegistry()
        self.telegram_adapter = self._build_telegram_adapter()
        if self.telegram_adapter is not None:
            self.channel_adapters.register(self.telegram_adapter)
        self.content_preprocessor = self._build_content_preprocessor()

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
            if self.content_preprocessor is None:
                self.content_preprocessor = self._build_content_preprocessor()
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
        if self._image_analysis_http_client is not None:
            await self._image_analysis_http_client.aclose()
            self._image_analysis_http_client = None
        if self._groq_http_client is not None:
            await self._groq_http_client.aclose()
            self._groq_http_client = None
        self.content_preprocessor = None
        if self._weather_http_client is not None:
            await self._weather_http_client.aclose()
            self._weather_http_client = None
        if self._web_research_http_client is not None:
            await self._web_research_http_client.aclose()
            self._web_research_http_client = None
        if self._compaction_http_client is not None:
            await self._compaction_http_client.aclose()
            self._compaction_http_client = None
            self.conversation_compactor = None
        if self._postgres_pool is not None:
            await self._postgres_pool.close()
            self._postgres_pool = None
            self.inbound_intake_service = None
            self.inbound_idempotency_store = None
            self.conversation_resolver = None
            self.conversation_memory_store = None
            self.conversation_compaction_store = None
            self.memory_service = None
            self.media_asset_store = None
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._redis_client = None
            self.conversation_snapshot_store = None
        self._started = False

    def _build_telegram_adapter(self) -> TelegramAdapter | None:
        if self.settings.telegram_bot_token is None:
            return None

        self._telegram_http_client = _create_telegram_http_client(self.settings)
        return TelegramAdapter(
            bot_token=self.settings.telegram_bot_token,
            client=self._telegram_http_client,
            render_markdown=self.settings.telegram_render_markdown,
        )

    def _build_agent_boundary(self) -> AgentBoundary | None:
        if self.settings.agent_provider != "openrouter":
            return None
        if self.settings.agent_model is None or self.settings.openrouter_api_key is None:
            return None

        self._agent_http_client = _create_openrouter_http_client(
            self.settings,
            read_timeout_seconds=self.settings.agent_timeout_seconds,
            worker_count=self.settings.inbound_worker_count,
        )

        enabled_skill_ids: set[str] = set()
        capability_toolsets = {}
        direct_toolsets: list[AgentToolset[Any]] = []

        weather_toolsets = self._build_weather_forecast_toolsets()
        if weather_toolsets:
            enabled_skill_ids.add(WEATHER_FORECAST_SKILL_ID)
            capability_toolsets[WEATHER_FORECAST_SKILL_ID] = weather_toolsets

        vkusvill_toolsets = build_vkusvill_mcp_toolsets(self.settings)
        if vkusvill_toolsets:
            enabled_skill_ids.add(VKUSVILL_SHOPPING_SKILL_ID)
            capability_toolsets[VKUSVILL_SHOPPING_SKILL_ID] = vkusvill_toolsets

        image_analysis_toolsets = self._build_image_analysis_toolsets()
        if image_analysis_toolsets:
            enabled_skill_ids.add(IMAGE_ANALYSIS_TOOLSET_ID)
            capability_toolsets[IMAGE_ANALYSIS_TOOLSET_ID] = image_analysis_toolsets

        direct_toolsets.extend(self._build_web_research_toolsets())
        return build_openrouter_agent_boundary(
            model_name=self.settings.agent_model,
            api_key=self.settings.openrouter_api_key.get_secret_value(),
            http_client=self._agent_http_client,
            model_settings=_build_openrouter_model_settings(self.settings),
            timeout_seconds=self.settings.agent_timeout_seconds,
            capability_toolsets=capability_toolsets or None,
            toolsets=tuple(direct_toolsets) or None,
            enabled_skill_ids=enabled_skill_ids,
        )

    def _build_content_preprocessor(self) -> InboundContentPreprocessor | None:
        media_fetchers = InMemoryMediaFetcherRegistry()
        if self.settings.telegram_bot_token is not None and self._telegram_http_client is not None:
            media_fetchers.register(
                TelegramMediaFetcher(
                    bot_token=self.settings.telegram_bot_token,
                    client=self._telegram_http_client,
                    max_file_size_bytes=self.settings.transcription_max_audio_size_bytes,
                )
            )
        if not media_fetchers.channels:
            return None

        audio_media_store = None
        audio_transcriber = None
        if self.settings.transcription_audio_enabled and self.settings.groq_api_key is not None:
            self._groq_http_client = _create_groq_http_client(
                self.settings,
                read_timeout_seconds=self.settings.groq_http_read_timeout_seconds,
                worker_count=self.settings.inbound_worker_count,
            )
            audio_media_store = TempFileMediaStore(
                base_dir=self.settings.transcription_audio_temp_dir,
            )
            audio_transcriber = GroqWhisperTranscriber(
                api_key=self.settings.groq_api_key.get_secret_value(),
                client=self._groq_http_client,
                model=self.settings.transcription_model,
                timeout_seconds=self.settings.transcription_timeout_seconds,
            )

        image_media_store = None
        if self.settings.image_analysis_enabled and self.media_asset_store is not None:
            image_media_store = PersistentFileMediaStore(base_dir=self.settings.image_media_dir)

        if audio_transcriber is None and image_media_store is None:
            return None

        return InboundContentPreprocessor(
            media_fetchers=media_fetchers,
            audio_media_store=audio_media_store,
            audio_transcriber=audio_transcriber,
            image_media_store=image_media_store,
            media_asset_store=self.media_asset_store,
            retry_policy=ContentProcessingRetryPolicy(
                max_attempts=self.settings.transcription_retry_max_attempts,
                backoff_seconds=self.settings.transcription_retry_backoff_seconds,
            ),
            max_audio_size_bytes=self.settings.transcription_max_audio_size_bytes,
            max_image_size_bytes=self.settings.image_max_size_bytes,
        )

    def _build_weather_forecast_toolsets(self) -> tuple[AgentToolset[Any], ...]:
        if not self.settings.weather_forecast_enabled:
            return ()

        weather_http_client = _create_weather_http_client(self.settings)
        self._weather_http_client = weather_http_client
        return build_weather_forecast_toolsets(
            self.settings,
            http_client=weather_http_client,
        )

    def _build_web_research_toolsets(self) -> tuple[AgentToolset[Any], ...]:
        if not self.settings.web_research_enabled:
            return ()
        if self.settings.tavily_api_key is None:
            return ()

        web_research_http_client = _create_tavily_http_client(self.settings)
        self._web_research_http_client = web_research_http_client
        return build_web_research_toolsets(
            self.settings,
            http_client=web_research_http_client,
        )

    def _build_image_analysis_toolsets(self) -> tuple[AgentToolset[Any], ...]:
        if not self.settings.image_analysis_enabled:
            return ()
        if self.settings.image_analysis_model is None:
            return ()
        if self.settings.openrouter_api_key is None:
            return ()
        if self.media_asset_store is None:
            return ()

        image_analysis_http_client = _create_openrouter_http_client(
            self.settings,
            read_timeout_seconds=self.settings.image_analysis_timeout_seconds,
            worker_count=self.settings.inbound_worker_count,
        )
        self._image_analysis_http_client = image_analysis_http_client
        analyzer = OpenRouterVisionAnalyzer(
            api_key=self.settings.openrouter_api_key.get_secret_value(),
            client=image_analysis_http_client,
            model=self.settings.image_analysis_model,
            timeout_seconds=self.settings.image_analysis_timeout_seconds,
        )
        return build_image_analysis_toolsets(
            self.settings,
            analyzer=analyzer,
            media_asset_store=self.media_asset_store,
        )

    def _build_conversation_compactor(self) -> ConversationCompactor | None:
        if not self.settings.memory_compaction_enabled:
            return None
        if self.settings.memory_compaction_model is None:
            return None
        if self.settings.openrouter_api_key is None:
            return None

        self._compaction_http_client = _create_openrouter_http_client(
            self.settings,
            read_timeout_seconds=self.settings.memory_compaction_timeout_seconds,
            worker_count=self.settings.memory_compaction_worker_count,
        )
        model = OpenRouterModel(
            self.settings.memory_compaction_model,
            provider=OpenRouterProvider(
                api_key=self.settings.openrouter_api_key.get_secret_value(),
                http_client=self._compaction_http_client,
            ),
        )
        return PydanticAIConversationCompactor(
            agent=cast(
                ConversationSummaryAgent,
                Agent(model, output_type=ConversationSummaryOutput),
            ),
            target_summary_tokens=self.settings.memory_compaction_target_summary_tokens,
            timeout_seconds=self.settings.memory_compaction_timeout_seconds,
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
        self.inbound_idempotency_store = PostgresInboundIdempotencyStore(
            self._postgres_pool,
        )
        self.inbound_intake_service = InboundIntakeService(
            user_resolver=user_resolver,
            inbound_queue=self.inbound_queue,
            idempotency_store=self.inbound_idempotency_store,
            publish_timeout_seconds=self.settings.inbound_publish_timeout_seconds,
        )
        conversation_store = PostgresConversationStore(
            cast(ConversationPostgresPool, self._postgres_pool)
        )
        self.conversation_resolver = ConversationResolver(conversation_store)
        self.conversation_memory_store = PostgresConversationMemoryStore(
            cast(MemoryPostgresPool, self._postgres_pool)
        )
        self.media_asset_store = PostgresMediaAssetStore(self._postgres_pool)
        self.conversation_compaction_store = PostgresConversationCompactionStore(
            cast(MemoryPostgresPool, self._postgres_pool)
        )
        self.memory_service = DefaultConversationMemoryService(
            memory_store=self.conversation_memory_store,
            snapshot_store=self.conversation_snapshot_store,
            compaction_store=self.conversation_compaction_store,
            recent_message_limit=self.settings.recent_message_limit,
        )
        await self._rebuild_runtime_dependencies_after_postgres()

    async def _rebuild_runtime_dependencies_after_postgres(self) -> None:
        if self._groq_http_client is not None:
            await self._groq_http_client.aclose()
            self._groq_http_client = None
        self.content_preprocessor = self._build_content_preprocessor()

        if (
            self.settings.agent_provider == "openrouter"
            and self.settings.agent_model is not None
            and self.settings.openrouter_api_key is not None
        ):
            if self._agent_http_client is not None:
                await self._agent_http_client.aclose()
                self._agent_http_client = None
            if self._image_analysis_http_client is not None:
                await self._image_analysis_http_client.aclose()
                self._image_analysis_http_client = None
            self.agent_boundary = self._build_agent_boundary()

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
                idempotency_store=self.inbound_idempotency_store,
                retry_policy=retry_policy,
                error_backoff_seconds=self.settings.inbound_worker_error_backoff_seconds,
                outbound_publish_timeout_seconds=self.settings.outbound_publish_timeout_seconds,
                thinking_indicator_sender=(
                    self.telegram_adapter if self.settings.telegram_thinking_draft_enabled else None
                ),
                thinking_indicator_timeout_seconds=(
                    self.settings.telegram_thinking_draft_timeout_seconds
                ),
                content_preprocessor=self.content_preprocessor,
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


def _create_telegram_http_client(settings: AppSettings) -> httpx.AsyncClient:
    return _create_external_http_client(
        connect_timeout_seconds=settings.telegram_http_connect_timeout_seconds,
        read_timeout_seconds=settings.telegram_http_read_timeout_seconds,
        write_timeout_seconds=settings.telegram_http_write_timeout_seconds,
        pool_timeout_seconds=settings.telegram_http_pool_timeout_seconds,
        keepalive_expiry_seconds=settings.telegram_http_keepalive_expiry_seconds,
        worker_count=settings.delivery_worker_count,
    )


def _create_openrouter_http_client(
    settings: AppSettings,
    *,
    read_timeout_seconds: float,
    worker_count: int,
) -> httpx.AsyncClient:
    return _create_external_http_client(
        connect_timeout_seconds=settings.openrouter_http_connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=settings.openrouter_http_write_timeout_seconds,
        pool_timeout_seconds=settings.openrouter_http_pool_timeout_seconds,
        keepalive_expiry_seconds=settings.openrouter_http_keepalive_expiry_seconds,
        worker_count=worker_count,
        event_hooks={
            "request": [_mark_openrouter_request_started],
            "response": [
                _log_openrouter_response_timing,
                _log_openrouter_error_response,
            ],
        },
    )


def _build_openrouter_model_settings(settings: AppSettings) -> OpenRouterModelSettings | None:
    provider: dict[str, object] = {}
    if settings.openrouter_provider_sort is not None:
        provider["sort"] = settings.openrouter_provider_sort

    latency_p90 = settings.openrouter_provider_preferred_max_latency_p90
    latency_p99 = settings.openrouter_provider_preferred_max_latency_p99
    if latency_p90 is not None and latency_p99 is not None:
        provider["preferred_max_latency"] = {
            "p90": latency_p90,
            "p99": latency_p99,
        }

    extra_body: dict[str, object] = {}
    if provider:
        extra_body["provider"] = provider

    if not extra_body:
        return None
    return OpenRouterModelSettings(extra_body=extra_body)


def _create_weather_http_client(settings: AppSettings) -> httpx.AsyncClient:
    return _create_external_http_client(
        connect_timeout_seconds=settings.weather_http_connect_timeout_seconds,
        read_timeout_seconds=settings.weather_http_read_timeout_seconds,
        write_timeout_seconds=settings.weather_http_write_timeout_seconds,
        pool_timeout_seconds=settings.weather_http_pool_timeout_seconds,
        keepalive_expiry_seconds=settings.weather_http_keepalive_expiry_seconds,
        worker_count=settings.inbound_worker_count,
    )


def _create_tavily_http_client(settings: AppSettings) -> httpx.AsyncClient:
    return _create_external_http_client(
        connect_timeout_seconds=settings.tavily_http_connect_timeout_seconds,
        read_timeout_seconds=settings.tavily_http_read_timeout_seconds,
        write_timeout_seconds=settings.tavily_http_write_timeout_seconds,
        pool_timeout_seconds=settings.tavily_http_pool_timeout_seconds,
        keepalive_expiry_seconds=settings.tavily_http_keepalive_expiry_seconds,
        worker_count=settings.inbound_worker_count,
    )


def _create_groq_http_client(
    settings: AppSettings,
    *,
    read_timeout_seconds: float,
    worker_count: int,
) -> httpx.AsyncClient:
    return _create_external_http_client(
        connect_timeout_seconds=settings.groq_http_connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=settings.groq_http_write_timeout_seconds,
        pool_timeout_seconds=settings.groq_http_pool_timeout_seconds,
        keepalive_expiry_seconds=settings.groq_http_keepalive_expiry_seconds,
        worker_count=worker_count,
    )


def _create_external_http_client(
    *,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    pool_timeout_seconds: float,
    keepalive_expiry_seconds: float,
    worker_count: int,
    event_hooks: dict[str, list[Callable[..., object]]] | None = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=write_timeout_seconds,
            pool=pool_timeout_seconds,
        ),
        limits=_external_http_limits(
            worker_count=worker_count,
            keepalive_expiry_seconds=keepalive_expiry_seconds,
        ),
        event_hooks=event_hooks,
    )


async def _mark_openrouter_request_started(request: httpx.Request) -> None:
    request.extensions[OPENROUTER_REQUEST_STARTED_AT_EXTENSION] = start_timer()
    log_event(
        logger,
        logging.INFO,
        "OpenRouter HTTP request started",
        event="openrouter_http_request_started",
        http_method=request.method,
        http_url_path=request.url.path,
    )


async def _log_openrouter_response_timing(response: httpx.Response) -> None:
    await response.aread()
    started_at = response.request.extensions.get(OPENROUTER_REQUEST_STARTED_AT_EXTENSION)
    duration_ms = elapsed_ms(started_at) if isinstance(started_at, float) else None
    log_event(
        logger,
        logging.INFO,
        "OpenRouter HTTP response received",
        event="openrouter_http_response_received",
        http_method=response.request.method,
        http_url_path=response.request.url.path,
        http_status_code=response.status_code,
        duration_ms=duration_ms,
        http_response_content_length=_response_content_length(response),
        **_openrouter_response_header_fields(response),
    )


async def _log_openrouter_error_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        body = await response.aread()
        error = _openrouter_response_error(body)
        if error is None:
            return
    else:
        body = await response.aread()
        error = _openrouter_response_error(body)

    text = body.decode(response.encoding or "utf-8", errors="replace")
    logger.warning(
        "OpenRouter error response received",
        extra={
            "event": "openrouter_error_response_received",
            "http_method": response.request.method,
            "http_url_path": response.request.url.path,
            "http_status_code": response.status_code,
            "openrouter_error_code": _mapping_value(error, "code"),
            "openrouter_error_message": _mapping_value(error, "message"),
            "openrouter_response_body": _truncate_log_text(
                text,
                max_length=OPENROUTER_ERROR_RESPONSE_LOG_BODY_LIMIT,
            ),
            **_openrouter_response_header_fields(response),
        },
    )


def _openrouter_response_error(body: bytes) -> object | None:
    if not body:
        return None
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded.get("error")


def _mapping_value(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _openrouter_response_header_fields(response: httpx.Response) -> dict[str, str]:
    fields: dict[str, str] = {}
    for header_name in (
        "x-request-id",
        "x-openrouter-request-id",
        "cf-ray",
    ):
        header_value = response.headers.get(header_name)
        if header_value:
            fields[f"http_response_header_{header_name.replace('-', '_')}"] = header_value
    return fields


def _response_content_length(response: httpx.Response) -> int | None:
    content_length = response.headers.get("content-length")
    if content_length is None:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def _truncate_log_text(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    omitted = len(value) - max_length
    return f"{value[:max_length]}...[truncated {omitted} chars]"


def _external_http_limits(
    *,
    worker_count: int,
    keepalive_expiry_seconds: float,
) -> httpx.Limits:
    return httpx.Limits(
        max_connections=max(worker_count * 2, 10),
        max_keepalive_connections=max(worker_count, 4),
        keepalive_expiry=keepalive_expiry_seconds,
    )


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
