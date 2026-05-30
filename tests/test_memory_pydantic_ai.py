from datetime import UTC, datetime
from uuid import uuid4

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agent_service.memory import (
    ConversationMemoryMessage,
    ConversationMemoryRole,
    pydantic_ai_history_from_memory,
    pydantic_ai_message_from_memory,
)


def memory_message(
    *,
    role: ConversationMemoryRole,
    text: str | None = "hello",
) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        conversation_id=uuid4(),
        user_id=uuid4(),
        sequence=3,
        role=role,
        text=text,
        tool_name="search" if role in _TOOL_ROLES else None,
        tool_call_id="call-1" if role in _TOOL_ROLES else None,
        trace_id="trace-1",
        metadata={"args": {"query": "weather"}} if role is ConversationMemoryRole.TOOL_CALL else {},
        created_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    )


_TOOL_ROLES = {
    ConversationMemoryRole.TOOL_CALL,
    ConversationMemoryRole.TOOL_RESULT,
}


def test_user_memory_message_becomes_pydantic_ai_request() -> None:
    message = memory_message(role=ConversationMemoryRole.USER, text="question")

    converted = pydantic_ai_message_from_memory(message)

    assert isinstance(converted, ModelRequest)
    assert converted.conversation_id == str(message.conversation_id)
    assert converted.metadata is not None
    assert converted.metadata["message_id"] == str(message.id)
    assert converted.metadata["sequence"] == 3
    assert isinstance(converted.parts[0], UserPromptPart)
    assert converted.parts[0].content == "question"


def test_assistant_memory_message_becomes_pydantic_ai_response() -> None:
    message = memory_message(role=ConversationMemoryRole.ASSISTANT, text="answer")

    converted = pydantic_ai_message_from_memory(message)

    assert isinstance(converted, ModelResponse)
    assert converted.conversation_id == str(message.conversation_id)
    assert converted.timestamp == message.created_at
    assert isinstance(converted.parts[0], TextPart)
    assert converted.parts[0].content == "answer"


def test_tool_memory_messages_keep_tool_call_linkage() -> None:
    tool_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    tool_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text="sunny")

    converted_call = pydantic_ai_message_from_memory(tool_call)
    converted_result = pydantic_ai_message_from_memory(tool_result)

    assert isinstance(converted_call, ModelResponse)
    assert isinstance(converted_call.parts[0], ToolCallPart)
    assert converted_call.parts[0].tool_name == "search"
    assert converted_call.parts[0].tool_call_id == "call-1"
    assert converted_call.parts[0].args == {"query": "weather"}
    assert isinstance(converted_result, ModelRequest)
    assert isinstance(converted_result.parts[0], ToolReturnPart)
    assert converted_result.parts[0].tool_name == "search"
    assert converted_result.parts[0].tool_call_id == "call-1"
    assert converted_result.parts[0].content == "sunny"


def test_pydantic_ai_history_preserves_memory_order() -> None:
    user_message = memory_message(role=ConversationMemoryRole.USER, text="first")
    assistant_message = memory_message(role=ConversationMemoryRole.ASSISTANT, text="second")

    history = pydantic_ai_history_from_memory([user_message, assistant_message])

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)
