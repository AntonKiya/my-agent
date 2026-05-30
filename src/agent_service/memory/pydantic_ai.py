from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agent_service.memory.models import ConversationMemoryMessage, ConversationMemoryRole


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


def _tool_result_content(message: ConversationMemoryMessage) -> str | dict[str, Any]:
    if message.text is not None:
        return message.text
    return message.metadata
