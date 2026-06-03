from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_core import to_jsonable_python

from agent_service.memory.models import ConversationMemoryMessage, ConversationMemoryRole

CAPABILITY_LOAD_TOOL_KIND = "capability-load"
TOOL_RESULT_CONTENT_METADATA_KEY = "content"


def pydantic_ai_message_from_memory(message: ConversationMemoryMessage) -> ModelMessage:
    """Convert stored memory into the message shape accepted by Pydantic AI."""
    if message.role is ConversationMemoryRole.USER:
        return ModelRequest(
            parts=[UserPromptPart(content=_message_text(message), timestamp=message.created_at)],
            timestamp=message.created_at,
            conversation_id=str(message.conversation_id),
            metadata=_metadata(message),
        )
    if message.role is ConversationMemoryRole.ASSISTANT:
        return ModelResponse(
            parts=[TextPart(content=_message_text(message))],
            timestamp=message.created_at,
            conversation_id=str(message.conversation_id),
            metadata=_metadata(message),
        )
    if message.role is ConversationMemoryRole.TOOL_CALL:
        if _is_load_capability_message(message):
            return ModelResponse(
                parts=[
                    LoadCapabilityCallPart(
                        args=cast(Any, _tool_args(message)),
                        tool_call_id=message.tool_call_id or str(message.id),
                    )
                ],
                timestamp=message.created_at,
                conversation_id=str(message.conversation_id),
                metadata=_metadata(message),
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=message.tool_name or "unknown",
                    args=_tool_args(message),
                    tool_call_id=message.tool_call_id or str(message.id),
                )
            ],
            timestamp=message.created_at,
            conversation_id=str(message.conversation_id),
            metadata=_metadata(message),
        )
    if _is_load_capability_message(message):
        return ModelRequest(
            parts=[
                LoadCapabilityReturnPart(
                    content=cast(Any, _load_capability_return_content(message)),
                    tool_call_id=message.tool_call_id or str(message.id),
                    timestamp=message.created_at,
                    metadata=_metadata(message),
                )
            ],
            timestamp=message.created_at,
            conversation_id=str(message.conversation_id),
            metadata=_metadata(message),
        )
    return ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name=message.tool_name or "unknown",
                content=_tool_result_content(message),
                tool_call_id=message.tool_call_id or str(message.id),
                timestamp=message.created_at,
                metadata=_metadata(message),
            )
        ],
        timestamp=message.created_at,
        conversation_id=str(message.conversation_id),
        metadata=_metadata(message),
    )


def pydantic_ai_history_from_memory(
    messages: list[ConversationMemoryMessage],
) -> list[ModelMessage]:
    return [pydantic_ai_message_from_memory(message) for message in messages]


def pydantic_ai_tool_messages_to_memory(
    messages: list[ModelMessage],
    *,
    conversation_id: UUID,
    user_id: UUID,
    trace_id: str | None = None,
) -> list[ConversationMemoryMessage]:
    memory_messages: list[ConversationMemoryMessage] = []
    for message in messages:
        created_at = message.timestamp or datetime.now(UTC)
        for part in message.parts:
            if isinstance(part, UserPromptPart | TextPart):
                continue
            if isinstance(part, ToolCallPart):
                memory_messages.append(
                    ConversationMemoryMessage(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role=ConversationMemoryRole.TOOL_CALL,
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        trace_id=trace_id,
                        metadata=_tool_call_metadata(part),
                        created_at=created_at,
                    )
                )
            elif isinstance(part, ToolReturnPart):
                memory_messages.append(
                    ConversationMemoryMessage(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role=ConversationMemoryRole.TOOL_RESULT,
                        text=part.content if isinstance(part.content, str) else None,
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        trace_id=trace_id,
                        metadata=_tool_return_metadata(part),
                        created_at=part.timestamp,
                    )
                )
    return memory_messages


def _metadata(message: ConversationMemoryMessage) -> dict[str, Any]:
    return {
        "message_id": str(message.id),
        "sequence": message.sequence,
        "trace_id": message.trace_id,
        **message.metadata,
    }


def _message_text(message: ConversationMemoryMessage) -> str:
    if message.text:
        return message.text
    if message.attachments:
        return "[attachments]"
    if message.role is ConversationMemoryRole.TOOL_CALL:
        return f"Tool call: {message.tool_name or 'unknown'}"
    if message.role is ConversationMemoryRole.TOOL_RESULT:
        return f"Tool result: {message.tool_name or 'unknown'}"
    return "[empty message]"


def _tool_args(message: ConversationMemoryMessage) -> str | dict[str, Any] | None:
    args = message.metadata.get("args")
    if isinstance(args, str) or isinstance(args, dict):
        return args
    return None


def _tool_result_content(message: ConversationMemoryMessage) -> Any:
    if message.text is not None:
        return message.text
    content = message.metadata.get(TOOL_RESULT_CONTENT_METADATA_KEY)
    if content is not None:
        return content
    return message.metadata


def _is_load_capability_message(message: ConversationMemoryMessage) -> bool:
    return (
        message.tool_name == "load_capability"
        and message.metadata.get("tool_kind") == CAPABILITY_LOAD_TOOL_KIND
    )


def _load_capability_return_content(message: ConversationMemoryMessage) -> dict[str, Any]:
    content = _tool_result_content(message)
    if isinstance(content, dict):
        instructions = content.get("instructions")
        if isinstance(instructions, str):
            return {"instructions": instructions}
    return {}


def _tool_call_metadata(part: ToolCallPart) -> dict[str, Any]:
    metadata: dict[str, Any] = {"args": _jsonable(part.args)}
    if part.tool_kind is not None:
        metadata["tool_kind"] = part.tool_kind
    if isinstance(part, LoadCapabilityCallPart):
        metadata["tool_kind"] = CAPABILITY_LOAD_TOOL_KIND
    if part.id is not None:
        metadata["part_id"] = part.id
    if part.provider_name is not None:
        metadata["provider_name"] = part.provider_name
    if part.provider_details is not None:
        metadata["provider_details"] = _jsonable(part.provider_details)
    return metadata


def _tool_return_metadata(part: ToolReturnPart) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not isinstance(part.content, str):
        metadata[TOOL_RESULT_CONTENT_METADATA_KEY] = _jsonable(part.content)
    if part.tool_kind is not None:
        metadata["tool_kind"] = part.tool_kind
    if isinstance(part, LoadCapabilityReturnPart):
        metadata["tool_kind"] = CAPABILITY_LOAD_TOOL_KIND
    if part.metadata is not None:
        metadata["part_metadata"] = _jsonable(part.metadata)
    if part.outcome != "success":
        metadata["outcome"] = part.outcome
    return metadata


def _jsonable(value: Any) -> Any:
    return to_jsonable_python(value)
