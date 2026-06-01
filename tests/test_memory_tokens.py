from uuid import uuid4

from agent_service.memory import (
    ConversationMemoryMessage,
    ConversationMemoryRole,
    estimate_message_tokens,
    usage_token_count,
    usage_total_token_count,
)


def test_token_estimator_counts_multilingual_message_content() -> None:
    message = ConversationMemoryMessage(
        conversation_id=uuid4(),
        user_id=uuid4(),
        role=ConversationMemoryRole.USER,
        text="Привет, summarize this JSON: {'ok': true}",
    )

    assert estimate_message_tokens(message) > 0


def test_token_estimator_counts_tool_payloads() -> None:
    tool_call = ConversationMemoryMessage(
        conversation_id=uuid4(),
        user_id=uuid4(),
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
        metadata={"args": {"query": "weather in Moscow"}},
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=tool_call.conversation_id,
        user_id=tool_call.user_id,
        role=ConversationMemoryRole.TOOL_RESULT,
        tool_name="search",
        tool_call_id="call-1",
        text="sunny",
    )

    assert estimate_message_tokens(tool_call) > estimate_message_tokens(tool_result)


def test_usage_helpers_prefer_total_and_fallback_to_input_plus_output() -> None:
    assert usage_total_token_count(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 20}
    ) == 20
    assert usage_total_token_count({"input_tokens": 10, "output_tokens": 5}) == 15
    assert usage_token_count({"input_tokens": 10}, "input_tokens") == 10
