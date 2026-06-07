import asyncio
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.toolsets import AgentToolset

from agent_service.agents.interfaces import AgentBoundary
from agent_service.agents.models import AgentRequest, AgentResponse, AgentUsage
from agent_service.instructions import load_base_agent_instructions
from agent_service.skills import load_builtin_skill_capabilities

SAFE_REQUEST_METADATA_KEYS = frozenset({"retry_attempt"})
SAFE_CONTEXT_METADATA_KEYS = frozenset(
    {
        "current_sequence",
        "last_seen_sequence",
        "snapshot_version",
    }
)


class AgentBoundaryError(RuntimeError):
    """Base error raised by concrete agent boundary implementations."""


class UnsupportedAgentRequestError(AgentBoundaryError):
    """Raised when a request cannot be represented safely for the agent."""


class EmptyAgentResponseError(AgentBoundaryError):
    """Raised when the model returns no usable assistant text."""


class PydanticAIRunResult(Protocol):
    output: Any

    def usage(self) -> Any:
        """Return Pydantic AI usage details when available."""
        ...

    def new_messages(self) -> list[ModelMessage]:
        """Return messages produced during this Pydantic AI run."""
        ...


class PydanticAIAgent(Protocol):
    async def run(
        self,
        user_prompt: str | None = None,
        *,
        output_type: type[str] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        conversation_id: str | None = None,
        instructions: str | Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PydanticAIRunResult:
        """Run a Pydantic AI agent with a channel-neutral prompt and context."""
        ...


@dataclass(slots=True)
class PydanticAIAgentBoundary(AgentBoundary):
    agent: PydanticAIAgent
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

    async def run(self, request: AgentRequest) -> AgentResponse:
        run_context = request.pydantic_ai
        if request.attachments:
            raise UnsupportedAgentRequestError(
                "Pydantic AI agent boundary currently supports text requests only"
            )
        user_prompt = run_context.user_prompt if run_context is not None else request.text
        if user_prompt is None or not user_prompt.strip():
            raise UnsupportedAgentRequestError("Agent request must include non-empty text")

        message_history = run_context.message_history if run_context is not None else []
        conversation_id = (
            run_context.conversation_id if run_context is not None else str(request.conversation_id)
        )
        instructions = run_context.instructions if run_context is not None else None
        metadata = _run_metadata(request)

        async with asyncio.timeout(self.timeout_seconds):
            result = await self.agent.run(
                user_prompt,
                output_type=str,
                message_history=message_history,
                conversation_id=conversation_id,
                instructions=instructions,
                metadata=metadata,
            )

        usage = _result_usage(result)
        new_messages = _agent_new_messages(result)
        response_usage = _latest_model_response_usage(new_messages) or usage
        text = _response_text(result.output)
        return AgentResponse(
            text=text,
            metadata={
                "agent": "pydantic_ai",
                **_safe_response_metadata(request),
            },
            usage=_agent_usage_from_usage(response_usage),
            pydantic_ai_new_messages=new_messages,
            trace_id=request.trace_id,
        )


def build_openrouter_agent_boundary(
    *,
    model_name: str,
    api_key: str,
    http_client: httpx.AsyncClient | None = None,
    model_settings: OpenRouterModelSettings | None = None,
    timeout_seconds: float = 60.0,
    capability_toolsets: Mapping[str, Sequence[AgentToolset[Any]]] | None = None,
    toolsets: Sequence[AgentToolset[Any]] | None = None,
    enabled_skill_ids: Collection[str] | None = None,
) -> PydanticAIAgentBoundary:
    model = OpenRouterModel(
        model_name,
        provider=OpenRouterProvider(api_key=api_key, http_client=http_client),
        settings=model_settings,
    )
    return PydanticAIAgentBoundary(
        agent=cast(
            PydanticAIAgent,
            Agent(
                model,
                output_type=str,
                instructions=load_base_agent_instructions(),
                capabilities=load_builtin_skill_capabilities(
                    toolsets_by_skill_id=capability_toolsets,
                    enabled_skill_ids=enabled_skill_ids,
                ),
                toolsets=toolsets,
            ),
        ),
        timeout_seconds=timeout_seconds,
    )


def _run_metadata(request: AgentRequest) -> dict[str, Any]:
    metadata = {
        **_safe_metadata_subset(request.metadata, SAFE_REQUEST_METADATA_KEYS),
        **_safe_metadata_subset(
            request.pydantic_ai.metadata if request.pydantic_ai is not None else {},
            SAFE_CONTEXT_METADATA_KEYS,
        ),
        "user_id": str(request.user_id),
        "conversation_id": str(request.conversation_id),
        "inbound_event_id": str(request.inbound_event_id),
        "channel": request.channel,
        "trace_id": request.trace_id,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _safe_response_metadata(request: AgentRequest) -> dict[str, Any]:
    metadata = {
        "model_conversation_id": (
            request.pydantic_ai.conversation_id if request.pydantic_ai is not None else None
        ),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _response_text(output: Any) -> str:
    if not isinstance(output, str):
        raise TypeError("Pydantic AI agent output must be text")
    text = output.strip()
    if not text:
        raise EmptyAgentResponseError("Pydantic AI agent returned empty text")
    return text


def _agent_usage(result: PydanticAIRunResult) -> AgentUsage | None:
    return _agent_usage_from_usage(_result_usage(result))


def _result_usage(result: PydanticAIRunResult) -> Any:
    usage = result.usage
    if callable(usage) and not _has_usage_details(usage):
        usage = usage()
    return usage


def _agent_usage_from_usage(usage: Any) -> AgentUsage | None:
    input_tokens = _optional_int_attr(usage, "input_tokens")
    output_tokens = _optional_int_attr(usage, "output_tokens")
    total_tokens = _optional_int_attr(usage, "total_tokens")
    metadata = _usage_metadata(usage)
    if input_tokens is None and output_tokens is None and total_tokens is None and not metadata:
        return None
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        metadata=metadata,
    )


def _has_usage_details(value: object) -> bool:
    return any(
        getattr(value, attr, None) is not None
        for attr in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "requests",
            "tool_calls",
            "details",
        )
    )


def _optional_int_attr(value: object, attr: str) -> int | None:
    item = getattr(value, attr, None)
    if isinstance(item, int):
        return item
    if callable(item):
        result = item()
        if isinstance(result, int):
            return result
    return None


def _usage_metadata(usage: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attr in ("requests", "tool_calls", "details"):
        value = getattr(usage, attr, None)
        if value:
            metadata[attr] = value
    return metadata


def _agent_new_messages(result: PydanticAIRunResult) -> list[ModelMessage]:
    new_messages = getattr(result, "new_messages", None)
    if not callable(new_messages):
        return []
    messages = new_messages()
    if not isinstance(messages, list):
        return []
    return messages


def _latest_model_response_usage(messages: list[ModelMessage]) -> object | None:
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        if _usage_has_tokens(message.usage):
            return message.usage
    return None


def _usage_has_tokens(usage: object) -> bool:
    return any(
        (value is not None and value > 0)
        for value in (
            _optional_int_attr(usage, "input_tokens"),
            _optional_int_attr(usage, "output_tokens"),
            _optional_int_attr(usage, "total_tokens"),
        )
    )


def _safe_metadata_subset(metadata: dict[str, Any], allowed_keys: frozenset[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key in allowed_keys and _is_safe_metadata_value(value)
    }


def _is_safe_metadata_value(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)
