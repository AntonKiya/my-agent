from datetime import UTC, datetime
from uuid import uuid4

from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
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
    pydantic_ai_tool_messages_to_memory,
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


def test_load_capability_memory_messages_restore_typed_parts() -> None:
    tool_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    tool_call.tool_name = "load_capability"
    tool_call.metadata = {
        "tool_kind": "capability-load",
        "args": {"id": "vkusvill-shopping"},
    }
    tool_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text=None)
    tool_result.tool_name = "load_capability"
    tool_result.metadata = {
        "tool_kind": "capability-load",
        "content": {"instructions": "Use VkusVill tools."},
    }

    converted_call = pydantic_ai_message_from_memory(tool_call)
    converted_result = pydantic_ai_message_from_memory(tool_result)

    assert isinstance(converted_call, ModelResponse)
    assert isinstance(converted_call.parts[0], LoadCapabilityCallPart)
    assert converted_call.parts[0].capability_id == "vkusvill-shopping"
    assert isinstance(converted_result, ModelRequest)
    assert isinstance(converted_result.parts[0], LoadCapabilityReturnPart)
    assert converted_result.parts[0].instructions == "Use VkusVill tools."


def test_pydantic_ai_tool_messages_to_memory_extracts_only_tool_parts() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    messages = [
        ModelRequest(parts=[UserPromptPart(content="add juice")]),
        ModelResponse(
            parts=[
                LoadCapabilityCallPart(
                    args={"id": "vkusvill-shopping"},
                    tool_call_id="load-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                LoadCapabilityReturnPart(
                    content={"instructions": "Use VkusVill tools."},
                    tool_call_id="load-1",
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="mcp_vkusvill_vkusvill_products_search",
                    args={"query": "сок"},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="mcp_vkusvill_vkusvill_products_search",
                    content={"items": [{"xml_id": "123", "name": "Сок"}]},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]

    memory_messages = pydantic_ai_tool_messages_to_memory(
        messages,
        conversation_id=conversation_id,
        user_id=user_id,
        trace_id="trace-1",
    )

    assert [message.role for message in memory_messages] == [
        ConversationMemoryRole.TOOL_CALL,
        ConversationMemoryRole.TOOL_RESULT,
        ConversationMemoryRole.TOOL_CALL,
        ConversationMemoryRole.TOOL_RESULT,
    ]
    assert memory_messages[0].tool_name == "load_capability"
    assert memory_messages[0].metadata["tool_kind"] == "capability-load"
    assert memory_messages[1].metadata["content"] == {"instructions": "Use VkusVill tools."}
    assert memory_messages[2].metadata["args"] == {"query": "сок"}
    assert memory_messages[3].metadata["content"] == {"items": [{"xml_id": "123", "name": "Сок"}]}


def test_pydantic_ai_history_preserves_memory_order() -> None:
    user_message = memory_message(role=ConversationMemoryRole.USER, text="first")
    assistant_message = memory_message(role=ConversationMemoryRole.ASSISTANT, text="second")

    history = pydantic_ai_history_from_memory([user_message, assistant_message])

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)


def test_pydantic_ai_history_groups_complete_parallel_tool_round() -> None:
    first_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    first_call.tool_call_id = "call-a"
    first_call.metadata = {"args": {"query": "juice"}}
    second_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    second_call.tool_call_id = "call-b"
    second_call.metadata = {"args": {"query": "milk"}}
    first_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text="juice")
    first_result.tool_call_id = "call-a"
    second_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text="milk")
    second_result.tool_call_id = "call-b"

    history = pydantic_ai_history_from_memory(
        [first_call, second_call, first_result, second_result]
    )

    assert len(history) == 2
    assert isinstance(history[0], ModelResponse)
    assert [part.tool_call_id for part in history[0].parts] == ["call-a", "call-b"]
    assert isinstance(history[1], ModelRequest)
    assert [part.tool_call_id for part in history[1].parts] == ["call-a", "call-b"]


def test_pydantic_ai_history_preserves_separate_tool_rounds() -> None:
    load_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    load_call.tool_name = "load_capability"
    load_call.tool_call_id = "load-1"
    load_call.metadata = {
        "tool_kind": "capability-load",
        "args": {"id": "vkusvill-shopping"},
    }
    load_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text=None)
    load_result.tool_name = "load_capability"
    load_result.tool_call_id = "load-1"
    load_result.metadata = {
        "tool_kind": "capability-load",
        "content": {"instructions": "Use VkusVill tools."},
    }
    search_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    search_call.tool_call_id = "call-1"
    search_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text="found")
    search_result.tool_call_id = "call-1"

    history = pydantic_ai_history_from_memory([load_call, load_result, search_call, search_result])

    assert len(history) == 4
    assert isinstance(history[0], ModelResponse)
    assert isinstance(history[0].parts[0], LoadCapabilityCallPart)
    assert isinstance(history[1], ModelRequest)
    assert isinstance(history[1].parts[0], LoadCapabilityReturnPart)
    assert isinstance(history[2], ModelResponse)
    assert isinstance(history[2].parts[0], ToolCallPart)
    assert isinstance(history[3], ModelRequest)
    assert isinstance(history[3].parts[0], ToolReturnPart)


def test_pydantic_ai_history_drops_incomplete_tool_pairs() -> None:
    orphan_result = memory_message(role=ConversationMemoryRole.TOOL_RESULT, text="orphan")
    orphan_result.tool_call_id = "missing-call"
    unmatched_call = memory_message(role=ConversationMemoryRole.TOOL_CALL, text=None)
    unmatched_call.tool_call_id = "missing-result"
    user_message = memory_message(role=ConversationMemoryRole.USER, text="next")

    history = pydantic_ai_history_from_memory([orphan_result, unmatched_call, user_message])

    assert len(history) == 1
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert history[0].parts[0].content == "next"
