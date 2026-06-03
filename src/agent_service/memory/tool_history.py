from agent_service.memory.models import ConversationMemoryMessage, ConversationMemoryRole

TOOL_HISTORY_OVERFETCH_FACTOR = 2


def is_tool_message(message: ConversationMemoryMessage) -> bool:
    return message.role in {
        ConversationMemoryRole.TOOL_CALL,
        ConversationMemoryRole.TOOL_RESULT,
    }


def tool_history_fetch_limit(limit: int) -> int:
    return max(limit, limit * TOOL_HISTORY_OVERFETCH_FACTOR)


def retain_recent_messages_preserving_tool_runs(
    messages: list[ConversationMemoryMessage],
    *,
    limit: int,
) -> list[ConversationMemoryMessage]:
    if len(messages) <= limit:
        return list(messages)
    return expand_tail_to_tool_run_start(messages, messages[-limit:])


def expand_tail_to_tool_run_start(
    messages: list[ConversationMemoryMessage],
    retained_tail: list[ConversationMemoryMessage],
) -> list[ConversationMemoryMessage]:
    if not retained_tail:
        return []
    first_retained_id = retained_tail[0].id
    first_retained_index = next(
        (index for index, message in enumerate(messages) if message.id == first_retained_id),
        None,
    )
    if first_retained_index is None:
        return list(retained_tail)
    if not is_tool_message(messages[first_retained_index]):
        return list(retained_tail)

    start_index = first_retained_index
    while start_index > 0 and is_tool_message(messages[start_index - 1]):
        start_index -= 1
    return list(messages[start_index:])
