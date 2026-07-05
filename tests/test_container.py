import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr

import agent_service.container as container_module
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
from agent_service.inbound import InboundContentPreprocessor, InboundIntake
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
    PydanticAIConversationCompactor,
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
        px: int | None = None,
        nx: bool = False,
    ) -> object:
        return True

    async def delete(self, *names: str) -> object:
        return len(names)

    async def scan_iter(
        self,
        match: str | None = None,
        count: int | None = None,
    ) -> AsyncIterator[bytes | str]:
        if False:
            yield ""

    async def ping(self) -> object:
        self.pinged = True
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def aclose(self) -> None:
        self.closed = True


class DelayedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.was_read = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(0.001)
        self.was_read = True
        yield self.body


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
    assert container.inbound_idempotency_store is None
    assert container.conversation_resolver is None
    assert container.conversation_memory_store is None
    assert container.conversation_snapshot_store is None
    assert container.conversation_compaction_store is None
    assert isinstance(container.conversation_compaction_policy, ConversationCompactionPolicy)
    assert not container.conversation_compaction_policy.enabled
    assert container.conversation_compactor is None
    assert container.memory_service is None
    assert container.agent_boundary is None
    assert container.content_preprocessor is None
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
    settings = AppSettings(
        environment="test",
        telegram_bot_token=SecretStr("token"),
        telegram_http_connect_timeout_seconds=11.0,
        telegram_http_read_timeout_seconds=31.0,
        telegram_http_write_timeout_seconds=12.0,
        telegram_http_pool_timeout_seconds=13.0,
        telegram_rich_messages_enabled=True,
    )
    container = AppContainer(settings=settings)

    assert isinstance(container.telegram_adapter, TelegramAdapter)
    assert container.telegram_adapter.rich_messages_enabled
    assert container.channel_adapters.get("telegram") is container.telegram_adapter
    telegram_http_client = container._telegram_http_client
    assert telegram_http_client is not None
    assert telegram_http_client.timeout.connect == 11.0
    assert telegram_http_client.timeout.read == 31.0
    assert telegram_http_client.timeout.write == 12.0
    assert telegram_http_client.timeout.pool == 13.0

    await container.start()
    await container.stop()

    assert telegram_http_client.is_closed
    assert container._telegram_http_client is None


async def test_container_wires_telegram_thinking_draft_sender_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_workers: list[dict[str, Any]] = []

    class CapturingInboundWorker:
        def __init__(self, **kwargs: Any) -> None:
            captured_workers.append(kwargs)

        async def run_forever(self) -> None:
            return None

    monkeypatch.setattr("agent_service.container.InboundWorker", CapturingInboundWorker)
    settings = AppSettings(
        environment="test",
        telegram_bot_token=SecretStr("token"),
        telegram_thinking_draft_enabled=True,
        telegram_thinking_draft_timeout_seconds=0.25,
        telegram_thinking_draft_refresh_seconds=6.5,
        inbound_worker_count=1,
    )
    container = AppContainer(settings=settings)
    container.conversation_resolver = cast(ConversationResolverProtocol, object())
    container.memory_service = FakeMemoryService()
    container.agent_boundary = FakeAgentBoundary()

    container._start_inbound_workers()

    assert len(captured_workers) == 1
    assert captured_workers[0]["thinking_indicator_sender"] is container.telegram_adapter
    assert captured_workers[0]["thinking_indicator_timeout_seconds"] == 0.25
    assert captured_workers[0]["thinking_indicator_refresh_seconds"] == 6.5

    await container.stop()


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


async def test_container_builds_audio_content_preprocessor_when_configured() -> None:
    settings = AppSettings(
        environment="test",
        telegram_bot_token=SecretStr("telegram-token"),
        groq_api_key=SecretStr("groq-secret"),
        transcription_model="whisper-large-v3-turbo",
        transcription_timeout_seconds=33.0,
        transcription_retry_max_attempts=2,
        transcription_retry_backoff_seconds=(0.1,),
        transcription_max_audio_size_bytes=123_456,
        groq_http_connect_timeout_seconds=11.0,
        groq_http_read_timeout_seconds=22.0,
        groq_http_write_timeout_seconds=12.0,
        groq_http_pool_timeout_seconds=13.0,
    )
    container = AppContainer(settings=settings)

    assert isinstance(container.content_preprocessor, InboundContentPreprocessor)
    assert container._groq_http_client is not None
    assert container._groq_http_client.timeout.connect == 11.0
    assert container._groq_http_client.timeout.read == 22.0
    assert container._groq_http_client.timeout.write == 12.0
    assert container._groq_http_client.timeout.pool == 13.0
    assert container.content_preprocessor.max_audio_size_bytes == 123_456
    assert container.content_preprocessor.retry_policy.max_attempts == 2

    groq_client = container._groq_http_client

    await container.stop()

    assert groq_client.is_closed
    assert container._groq_http_client is None
    assert container.content_preprocessor is None


async def test_container_builds_openrouter_agent_boundary_when_configured() -> None:
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        agent_timeout_seconds=12.5,
        openrouter_http_connect_timeout_seconds=14.0,
        openrouter_http_write_timeout_seconds=15.0,
        openrouter_http_pool_timeout_seconds=16.0,
    )
    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert container.agent_boundary.timeout_seconds == 12.5
    assert container._agent_http_client is not None
    assert container._agent_http_client.timeout.connect == 14.0
    assert container._agent_http_client.timeout.read == 12.5
    assert container._agent_http_client.timeout.write == 15.0
    assert container._agent_http_client.timeout.pool == 16.0
    assert not container._agent_http_client.is_closed

    await container.stop()

    assert container._agent_http_client is None
    assert container.agent_boundary is None


async def test_openrouter_http_client_wires_timing_and_error_hooks() -> None:
    settings = AppSettings(environment="test")

    client = container_module._create_openrouter_http_client(
        settings,
        read_timeout_seconds=12.5,
        worker_count=1,
    )

    try:
        assert client.event_hooks["request"] == [container_module._mark_openrouter_request_started]
        assert client.event_hooks["response"] == [
            container_module._log_openrouter_response_timing,
            container_module._log_openrouter_error_response,
        ]
    finally:
        await client.aclose()


async def test_openrouter_response_timing_hook_logs_duration_and_request_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    stream = DelayedAsyncByteStream(b'{"ok":true}')
    response = httpx.Response(
        200,
        headers={
            "content-length": "123",
            "x-openrouter-request-id": "or-request-1",
            "cf-ray": "cf-ray-1",
        },
        stream=stream,
        request=request,
    )

    caplog.set_level(logging.INFO, logger="agent_service.container")
    await container_module._mark_openrouter_request_started(request)
    await container_module._log_openrouter_response_timing(response)

    assert stream.was_read
    assert response.content == b'{"ok":true}'

    timing_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "openrouter_http_response_received"
    )
    timing_record_data = cast(Any, timing_record)
    duration_ms = timing_record_data.duration_ms
    assert timing_record_data.http_method == "POST"
    assert timing_record_data.http_url_path == "/api/v1/chat/completions"
    assert timing_record_data.http_status_code == 200
    assert isinstance(duration_ms, float)
    assert duration_ms >= 0
    assert timing_record_data.http_response_content_length == 123
    assert timing_record_data.http_response_header_x_openrouter_request_id == "or-request-1"
    assert timing_record_data.http_response_header_cf_ray == "cf-ray-1"


async def test_container_passes_vkusvill_mcp_toolsets_to_openrouter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        vkusvill_mcp_url="http://localhost:8765/mcp",
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["model_name"] == "openai/gpt-4.1-mini"
    assert captured["timeout_seconds"] == 60.0
    assert captured["enabled_skill_ids"] == {"weather-forecast", "vkusvill-shopping"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    weather_toolsets = capability_toolsets["weather-forecast"]
    assert isinstance(weather_toolsets, tuple)
    assert len(weather_toolsets) == 1
    vkusvill_toolsets = capability_toolsets["vkusvill-shopping"]
    assert isinstance(vkusvill_toolsets, tuple)
    assert len(vkusvill_toolsets) == 1

    await container.stop()


async def test_container_passes_tutu_mcp_toolsets_to_openrouter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        tutu_mcp_url="https://mcp.tutu.ru/mcp",
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["enabled_skill_ids"] == {"tutu-travel", "weather-forecast"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    weather_toolsets = capability_toolsets["weather-forecast"]
    assert isinstance(weather_toolsets, tuple)
    assert len(weather_toolsets) == 1
    tutu_toolsets = capability_toolsets["tutu-travel"]
    assert isinstance(tutu_toolsets, tuple)
    assert len(tutu_toolsets) == 1

    await container.stop()


async def test_container_passes_openrouter_model_settings_to_agent_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="minimax/minimax-m2.5",
        openrouter_api_key=SecretStr("secret"),
        openrouter_provider_sort="throughput",
        openrouter_provider_preferred_max_latency_p90=3.0,
        openrouter_provider_preferred_max_latency_p99=6.0,
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["model_name"] == "minimax/minimax-m2.5"
    assert captured["model_settings"] == {
        "extra_body": {
            "provider": {
                "sort": "throughput",
                "preferred_max_latency": {
                    "p90": 3.0,
                    "p99": 6.0,
                },
            },
        },
    }

    await container.stop()


async def test_container_disables_vkusvill_skill_without_mcp_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["enabled_skill_ids"] == {"weather-forecast"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    assert set(capability_toolsets) == {"weather-forecast"}

    await container.stop()


async def test_container_can_disable_weather_forecast_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        weather_forecast_enabled=False,
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["capability_toolsets"] is None
    assert captured["enabled_skill_ids"] == set()

    await container.stop()


async def test_container_passes_web_research_as_direct_openrouter_toolset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        tavily_api_key=SecretStr("tvly-test"),
        tavily_http_connect_timeout_seconds=3.0,
        tavily_http_read_timeout_seconds=21.0,
        tavily_http_write_timeout_seconds=4.0,
        tavily_http_pool_timeout_seconds=5.0,
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    assert captured["enabled_skill_ids"] == {"weather-forecast"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    assert set(capability_toolsets) == {"weather-forecast"}
    direct_toolsets = captured["toolsets"]
    assert isinstance(direct_toolsets, tuple)
    assert {cast(Any, toolset).id for toolset in direct_toolsets} == {
        "time",
        "web_research",
    }
    assert container._web_research_http_client is not None
    assert container._web_research_http_client.timeout.connect == 3.0
    assert container._web_research_http_client.timeout.read == 21.0
    assert container._web_research_http_client.timeout.write == 4.0
    assert container._web_research_http_client.timeout.pool == 5.0

    web_research_http_client = container._web_research_http_client
    await container.stop()

    assert web_research_http_client.is_closed
    assert container._web_research_http_client is None


async def test_container_passes_image_analysis_as_direct_openrouter_toolset_after_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()
    captured: dict[str, object] = {}

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        image_analysis_model="openai/gpt-4.1-mini",
        inbound_worker_count=0,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert captured["enabled_skill_ids"] == {"weather-forecast"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    assert set(capability_toolsets) == {"weather-forecast"}
    direct_toolsets = captured["toolsets"]
    assert isinstance(direct_toolsets, tuple)
    assert {cast(Any, toolset).id for toolset in direct_toolsets} == {
        "time",
        "image-analysis",
        "image-generation",
        "file-reading",
    }
    assert container._image_analysis_http_client is not None
    assert container._image_generation_http_client is not None

    image_analysis_http_client = container._image_analysis_http_client
    image_generation_http_client = container._image_generation_http_client
    await container.stop()

    assert image_analysis_http_client.is_closed
    assert image_generation_http_client.is_closed
    assert container._image_analysis_http_client is None
    assert container._image_generation_http_client is None


async def test_container_passes_reminders_as_deferred_capability_after_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool = FakeManagedPostgresPool()
    captured: dict[str, object] = {}

    async def fake_create_pool(**kwargs: object) -> FakeManagedPostgresPool:
        return fake_pool

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr("agent_service.container.asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        postgres_dsn="postgresql://agent:secret@localhost:5432/agent",
        telegram_bot_token=SecretStr("token"),
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
        inbound_worker_count=0,
        delivery_worker_count=0,
        reminder_worker_count=0,
        notification_outbox_worker_count=0,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert captured["enabled_skill_ids"] == {"weather-forecast", "reminders"}
    capability_toolsets = captured["capability_toolsets"]
    assert isinstance(capability_toolsets, dict)
    assert set(capability_toolsets) == {"weather-forecast", "reminders"}
    reminder_toolsets = capability_toolsets["reminders"]
    assert isinstance(reminder_toolsets, tuple)
    assert len(reminder_toolsets) == 1
    direct_toolsets = captured["toolsets"]
    assert isinstance(direct_toolsets, tuple)
    assert {cast(Any, toolset).id for toolset in direct_toolsets} == {
        "time",
        "image-generation",
        "file-reading",
    }

    await container.stop()


async def test_container_skips_web_research_without_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_openrouter_agent_boundary(**kwargs: object) -> PydanticAIAgentBoundary:
        captured.update(kwargs)
        return PydanticAIAgentBoundary(agent=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "build_openrouter_agent_boundary",
        fake_build_openrouter_agent_boundary,
    )
    settings = AppSettings(
        environment="test",
        agent_provider="openrouter",
        agent_model="openai/gpt-4.1-mini",
        openrouter_api_key=SecretStr("secret"),
    )

    container = AppContainer(settings=settings)

    assert isinstance(container.agent_boundary, PydanticAIAgentBoundary)
    direct_toolsets = captured["toolsets"]
    assert isinstance(direct_toolsets, tuple)
    assert len(direct_toolsets) == 1
    assert cast(Any, direct_toolsets[0]).id == "time"
    assert container._web_research_http_client is None

    await container.stop()


async def test_container_builds_pydantic_ai_compactor_when_configured() -> None:
    settings = AppSettings(
        environment="test",
        memory_compaction_enabled=True,
        memory_compaction_model="openai/gpt-4.1-mini",
        memory_compaction_target_summary_tokens=900,
        memory_compaction_timeout_seconds=88.0,
        openrouter_http_connect_timeout_seconds=14.0,
        openrouter_http_write_timeout_seconds=15.0,
        openrouter_http_pool_timeout_seconds=16.0,
        openrouter_api_key=SecretStr("secret"),
    )
    container = AppContainer(settings=settings)

    assert isinstance(container.conversation_compactor, PydanticAIConversationCompactor)
    assert container.conversation_compactor.target_summary_tokens == 900
    assert container.conversation_compactor.timeout_seconds == 88.0
    assert container._compaction_http_client is not None
    assert container._compaction_http_client.timeout.connect == 14.0
    assert container._compaction_http_client.timeout.read == 88.0
    assert container._compaction_http_client.timeout.write == 15.0
    assert container._compaction_http_client.timeout.pool == 16.0
    assert not container._compaction_http_client.is_closed

    await container.stop()

    assert container._compaction_http_client is None
    assert container.conversation_compactor is None


async def test_container_does_not_build_compactor_without_complete_config() -> None:
    missing_model = AppContainer(
        settings=AppSettings(
            environment="test",
            memory_compaction_enabled=True,
            openrouter_api_key=SecretStr("secret"),
        )
    )
    missing_key = AppContainer(
        settings=AppSettings(
            environment="test",
            memory_compaction_enabled=True,
            memory_compaction_model="openai/gpt-4.1-mini",
        )
    )

    assert missing_model.conversation_compactor is None
    assert missing_model._compaction_http_client is None
    assert missing_key.conversation_compactor is None
    assert missing_key._compaction_http_client is None

    await missing_model.stop()
    await missing_key.stop()


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
    assert container.inbound_idempotency_store is not None
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
    assert container.inbound_idempotency_store is None
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
    assert container.media_group_aggregator is not None
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
    assert container.media_group_aggregator is None


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
        recent_message_limit=37,
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
    assert container.memory_service.recent_message_limit == 37
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


async def test_container_starts_compaction_workers_with_configured_pydantic_ai_compactor(
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
        memory_compaction_model="openai/gpt-4.1-mini",
        memory_compaction_worker_count=2,
        openrouter_api_key=SecretStr("secret"),
        graceful_shutdown_timeout_seconds=0.1,
    )
    container = AppContainer(settings=settings)

    await container.start()

    assert container.started
    assert isinstance(container.conversation_compactor, PydanticAIConversationCompactor)
    assert container.task_supervisor.task_count == 2

    await container.stop()

    assert container.task_supervisor.task_count == 0
    assert container._compaction_http_client is None
    assert container.conversation_compactor is None
    assert fake_pool.closed
