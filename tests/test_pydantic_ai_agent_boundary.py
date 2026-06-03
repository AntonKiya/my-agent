import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RequestUsage

import agent_service.agents.pydantic_ai as pydantic_ai_module
from agent_service.agents import (
    AgentRequest,
    EmptyAgentResponseError,
    PydanticAIAgentBoundary,
    PydanticAIRunContext,
    UnsupportedAgentRequestError,
    build_openrouter_agent_boundary,
)
from agent_service.channels import Attachment, AttachmentType


@dataclass(slots=True)
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    details: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class FakeRunResult:
    output: Any
    run_usage: FakeUsage = field(default_factory=FakeUsage)
    messages: list[ModelMessage] = field(default_factory=list)

    def usage(self) -> FakeUsage:
        return self.run_usage

    def new_messages(self) -> list[ModelMessage]:
        return self.messages


@dataclass(slots=True)
class CallableUsage(FakeUsage):
    called: bool = False

    def __call__(self) -> "CallableUsage":
        self.called = True
        return self


@dataclass(slots=True)
class FakePydanticAIAgent:
    result: FakeRunResult = field(default_factory=lambda: FakeRunResult("ok"))
    error: BaseException | None = None
    delay_seconds: float = 0
    entered: asyncio.Event | None = None
    release: asyncio.Event | None = None
    active_count_reached: asyncio.Event | None = None
    active_count_target: int = 1
    calls: list[dict[str, Any]] = field(default_factory=list)
    active_count: int = 0
    max_active_count: int = 0

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        output_type: type[str] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        conversation_id: str | None = None,
        instructions: str | Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FakeRunResult:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "output_type": output_type,
                "message_history": message_history,
                "conversation_id": conversation_id,
                "instructions": instructions,
                "metadata": metadata,
            }
        )
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)
        if (
            self.active_count_reached is not None
            and self.active_count >= self.active_count_target
        ):
            self.active_count_reached.set()
        if self.entered is not None:
            self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.active_count -= 1


def agent_request(
    *,
    text: str | None = "hello",
    pydantic_ai: PydanticAIRunContext | None = None,
    attachments: list[Attachment] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRequest:
    return AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        text=text,
        attachments=attachments or [],
        pydantic_ai=pydantic_ai,
        metadata=metadata if metadata is not None else {"idempotency_key": "telegram:123:42"},
        trace_id="trace-1",
    )


def test_build_openrouter_agent_boundary_wires_builtin_skill_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_agents: list[dict[str, Any]] = []

    def fake_agent_factory(model: object, **kwargs: Any) -> object:
        created_agents.append({"model": model, **kwargs})
        return FakePydanticAIAgent()

    monkeypatch.setattr(pydantic_ai_module, "Agent", fake_agent_factory)
    toolsets = [object()]

    boundary = build_openrouter_agent_boundary(
        model_name="openai/gpt-4o-mini",
        api_key="key",
        capability_toolsets={"vkusvill-shopping": toolsets},  # type: ignore[dict-item]
    )

    assert isinstance(boundary, PydanticAIAgentBoundary)
    assert len(created_agents) == 1
    assert created_agents[0]["output_type"] is str
    capabilities = created_agents[0]["capabilities"]
    assert len(capabilities) == 1
    assert capabilities[0].id == "vkusvill-shopping"
    assert capabilities[0].defer_loading is True
    assert capabilities[0].toolsets == tuple(toolsets)


def test_build_openrouter_agent_boundary_respects_enabled_skill_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_agents: list[dict[str, Any]] = []

    def fake_agent_factory(model: object, **kwargs: Any) -> object:
        created_agents.append({"model": model, **kwargs})
        return FakePydanticAIAgent()

    monkeypatch.setattr(pydantic_ai_module, "Agent", fake_agent_factory)

    boundary = build_openrouter_agent_boundary(
        model_name="openai/gpt-4o-mini",
        api_key="key",
        enabled_skill_ids=set(),
    )

    assert isinstance(boundary, PydanticAIAgentBoundary)
    assert created_agents[0]["capabilities"] == ()


async def test_pydantic_ai_agent_boundary_passes_prepared_context() -> None:
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="previous")])]
    agent = FakePydanticAIAgent(
        result=FakeRunResult(
            output=" answer ",
            run_usage=FakeUsage(
                input_tokens=10,
                output_tokens=5,
                requests=1,
                details={"cached": 2},
            ),
        )
    )
    boundary = PydanticAIAgentBoundary(agent=agent)
    request = agent_request(
        pydantic_ai=PydanticAIRunContext(
            user_prompt="hello from memory",
            message_history=history,
            conversation_id="conversation-1",
            instructions="compressed memory",
            metadata={"snapshot_version": 2},
        )
    )

    response = await boundary.run(request)

    assert response.text == "answer"
    assert response.trace_id == "trace-1"
    assert response.metadata == {
        "agent": "pydantic_ai",
        "model_conversation_id": "conversation-1",
    }
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.usage.metadata == {"requests": 1, "details": {"cached": 2}}
    assert agent.calls == [
        {
            "user_prompt": "hello from memory",
            "output_type": str,
            "message_history": history,
            "conversation_id": "conversation-1",
            "instructions": "compressed memory",
            "metadata": {
                "snapshot_version": 2,
                "user_id": str(request.user_id),
                "conversation_id": str(request.conversation_id),
                "inbound_event_id": str(request.inbound_event_id),
                "channel": "telegram",
                "trace_id": "trace-1",
            },
        }
    ]


async def test_pydantic_ai_agent_boundary_returns_new_messages() -> None:
    new_messages = [ModelResponse(parts=[TextPart(content="ok")])]
    agent = FakePydanticAIAgent(
        result=FakeRunResult(output="ok", messages=new_messages),
    )
    boundary = PydanticAIAgentBoundary(agent=agent)

    response = await boundary.run(agent_request())

    assert response.pydantic_ai_new_messages == new_messages


async def test_pydantic_ai_agent_boundary_uses_latest_response_usage() -> None:
    new_messages = [
        ModelResponse(
            parts=[TextPart(content="tool calls")],
            usage=RequestUsage(input_tokens=20_783, output_tokens=262),
        ),
        ModelRequest(parts=[UserPromptPart(content="tool results")]),
        ModelResponse(
            parts=[TextPart(content="final")],
            usage=RequestUsage(input_tokens=23_904, output_tokens=179),
        ),
    ]
    agent = FakePydanticAIAgent(
        result=FakeRunResult(
            output="ok",
            run_usage=FakeUsage(input_tokens=68_450, output_tokens=753, requests=3),
            messages=new_messages,
        ),
    )
    boundary = PydanticAIAgentBoundary(agent=agent)

    response = await boundary.run(agent_request())

    assert response.usage is not None
    assert response.usage.input_tokens == 23_904
    assert response.usage.output_tokens == 179
    assert response.usage.total_tokens == 24_083


async def test_pydantic_ai_agent_boundary_does_not_call_usage_property_object() -> None:
    usage = CallableUsage(input_tokens=3, output_tokens=4, requests=1)
    agent = FakePydanticAIAgent(result=FakeRunResult(output="ok", run_usage=usage))
    boundary = PydanticAIAgentBoundary(agent=agent)

    response = await boundary.run(agent_request())

    assert usage.called is False
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 7


async def test_pydantic_ai_agent_boundary_falls_back_to_request_text() -> None:
    agent = FakePydanticAIAgent(result=FakeRunResult(output="ok"))
    boundary = PydanticAIAgentBoundary(agent=agent)
    request = agent_request(text="plain text", pydantic_ai=None)

    response = await boundary.run(request)

    assert response.text == "ok"
    assert agent.calls[0]["user_prompt"] == "plain text"
    assert agent.calls[0]["message_history"] == []
    assert agent.calls[0]["conversation_id"] == str(request.conversation_id)


async def test_pydantic_ai_agent_boundary_does_not_pass_transport_metadata() -> None:
    agent = FakePydanticAIAgent(result=FakeRunResult(output="ok"))
    boundary = PydanticAIAgentBoundary(agent=agent)
    request = agent_request(
        metadata={
            "idempotency_key": "telegram:123:42",
            "external_chat_id": "123",
            "external_message_id": "42",
            "raw_update": {"message": {"text": "secret raw payload"}},
            "retry_attempt": 1,
        },
        pydantic_ai=PydanticAIRunContext(
            user_prompt="hello",
            conversation_id="conversation-1",
            metadata={
                "snapshot_version": 2,
                "current_sequence": 10,
                "last_seen_sequence": 9,
                "raw_context": {"text": "must not cross boundary"},
            },
        ),
    )

    await boundary.run(request)

    assert agent.calls[0]["metadata"] == {
        "retry_attempt": 1,
        "snapshot_version": 2,
        "current_sequence": 10,
        "last_seen_sequence": 9,
        "user_id": str(request.user_id),
        "conversation_id": str(request.conversation_id),
        "inbound_event_id": str(request.inbound_event_id),
        "channel": "telegram",
        "trace_id": "trace-1",
    }


async def test_pydantic_ai_agent_boundary_rejects_attachment_requests() -> None:
    boundary = PydanticAIAgentBoundary(agent=FakePydanticAIAgent())
    attachment_only_request = agent_request(
        text=None,
        attachments=[
            Attachment(
                attachment_type=AttachmentType.DOCUMENT,
                external_id="file-1",
            )
        ],
    )
    mixed_request = agent_request(
        text="hello",
        attachments=[
            Attachment(
                attachment_type=AttachmentType.DOCUMENT,
                external_id="file-1",
            )
        ],
    )

    with pytest.raises(UnsupportedAgentRequestError):
        await boundary.run(attachment_only_request)
    with pytest.raises(UnsupportedAgentRequestError):
        await boundary.run(mixed_request)


async def test_pydantic_ai_agent_boundary_rejects_empty_model_output() -> None:
    boundary = PydanticAIAgentBoundary(
        agent=FakePydanticAIAgent(result=FakeRunResult(output="   "))
    )

    with pytest.raises(EmptyAgentResponseError):
        await boundary.run(agent_request())


async def test_pydantic_ai_agent_boundary_rejects_non_text_model_output() -> None:
    boundary = PydanticAIAgentBoundary(agent=FakePydanticAIAgent(result=FakeRunResult(output=123)))

    with pytest.raises(TypeError):
        await boundary.run(agent_request())


async def test_pydantic_ai_agent_boundary_times_out_provider_call() -> None:
    boundary = PydanticAIAgentBoundary(
        agent=FakePydanticAIAgent(delay_seconds=0.05),
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await boundary.run(agent_request())


def test_pydantic_ai_agent_boundary_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        PydanticAIAgentBoundary(agent=FakePydanticAIAgent(), timeout_seconds=0)


async def test_pydantic_ai_agent_boundary_propagates_provider_errors() -> None:
    boundary = PydanticAIAgentBoundary(
        agent=FakePydanticAIAgent(error=RuntimeError("provider failed"))
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        await boundary.run(agent_request())


async def test_pydantic_ai_agent_boundary_timeout_cancels_provider_call() -> None:
    agent = FakePydanticAIAgent(delay_seconds=0.05)
    boundary = PydanticAIAgentBoundary(
        agent=agent,
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await boundary.run(agent_request(text="first"))

    agent.delay_seconds = 0
    response = await boundary.run(agent_request(text="second"))

    assert response.text == "ok"
    assert agent.active_count == 0
    assert len(agent.calls) == 2


async def test_pydantic_ai_agent_boundary_does_not_serialize_provider_calls() -> None:
    both_calls_entered = asyncio.Event()
    release = asyncio.Event()
    agent = FakePydanticAIAgent(
        active_count_reached=both_calls_entered,
        active_count_target=2,
        release=release,
    )
    boundary = PydanticAIAgentBoundary(agent=agent)

    first = asyncio.create_task(boundary.run(agent_request(text="first")))
    second = asyncio.create_task(boundary.run(agent_request(text="second")))
    await asyncio.wait_for(both_calls_entered.wait(), timeout=0.1)

    assert agent.max_active_count == 2
    assert len(agent.calls) == 2

    release.set()
    await first
    await second

    assert agent.active_count == 0
