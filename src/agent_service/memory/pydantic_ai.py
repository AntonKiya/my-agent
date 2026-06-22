import logging
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_core import to_jsonable_python

from agent_service.channels import Attachment
from agent_service.memory.models import ConversationMemoryMessage, ConversationMemoryRole
from agent_service.memory.tool_history import is_tool_message
from agent_service.observability.events import log_event

CAPABILITY_LOAD_TOOL_KIND = "capability-load"
TOOL_RESULT_CONTENT_METADATA_KEY = "content"
logger = logging.getLogger(__name__)


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
    history: list[ModelMessage] = []
    repair_stats = _new_tool_history_repair_stats()
    index = 0
    while index < len(messages):
        message = messages[index]
        if not is_tool_message(message):
            history.append(pydantic_ai_message_from_memory(message))
            index += 1
            continue

        tool_run: list[ConversationMemoryMessage] = []
        while index < len(messages) and is_tool_message(messages[index]):
            tool_run.append(messages[index])
            index += 1
        history.extend(_pydantic_ai_tool_run_from_memory(tool_run, repair_stats=repair_stats))

    if _tool_history_was_repaired(repair_stats):
        log_event(
            logger,
            logging.INFO,
            "Pydantic AI tool history repaired",
            event="pydantic_ai_tool_history_repaired",
            memory_message_count=len(messages),
            model_message_count=len(history),
            **repair_stats,
        )
    return history


def pydantic_ai_history_from_context(
    *,
    summary: str | None,
    messages: list[ConversationMemoryMessage],
    conversation_id: UUID,
) -> list[ModelMessage]:
    """Build model history from compacted summary plus persisted recent messages."""
    history = pydantic_ai_history_from_memory(messages)
    if summary is None or not summary.strip():
        return history
    return [
        ModelRequest(
            parts=[SystemPromptPart(content=summary)],
            conversation_id=str(conversation_id),
            metadata={"source": "conversation_summary"},
        ),
        *history,
    ]


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


def _group_metadata(messages: list[ConversationMemoryMessage]) -> dict[str, Any]:
    if len(messages) == 1:
        return _metadata(messages[0])
    return {
        "message_ids": [str(message.id) for message in messages],
        "sequences": [message.sequence for message in messages],
        "trace_ids": sorted(
            {message.trace_id for message in messages if message.trace_id is not None}
        ),
    }


def _pydantic_ai_tool_run_from_memory(
    messages: list[ConversationMemoryMessage],
    *,
    repair_stats: dict[str, int],
) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    index = 0
    while index < len(messages):
        if messages[index].role is ConversationMemoryRole.TOOL_RESULT:
            repair_stats["dropped_orphan_tool_results"] += 1
            index += 1
            continue

        tool_calls: list[ConversationMemoryMessage] = []
        while index < len(messages) and messages[index].role is ConversationMemoryRole.TOOL_CALL:
            tool_calls.append(messages[index])
            index += 1

        tool_results: list[ConversationMemoryMessage] = []
        while index < len(messages) and messages[index].role is ConversationMemoryRole.TOOL_RESULT:
            tool_results.append(messages[index])
            index += 1

        history.extend(
            _pydantic_ai_tool_round_from_memory(
                tool_calls,
                tool_results,
                repair_stats=repair_stats,
            )
        )
    return history


def _pydantic_ai_tool_round_from_memory(
    tool_calls: list[ConversationMemoryMessage],
    tool_results: list[ConversationMemoryMessage],
    *,
    repair_stats: dict[str, int],
) -> list[ModelMessage]:
    calls_by_id: dict[str, ConversationMemoryMessage] = {}
    call_ids: list[str] = []
    for tool_call in tool_calls:
        tool_call_id = _safe_tool_call_id(tool_call)
        if tool_call_id is None:
            repair_stats["dropped_unmatched_tool_calls"] += 1
            continue
        if tool_call_id in calls_by_id:
            repair_stats["dropped_duplicate_tool_calls"] += 1
            continue
        calls_by_id[tool_call_id] = tool_call
        call_ids.append(tool_call_id)

    results_by_id: dict[str, ConversationMemoryMessage] = {}
    for tool_result in tool_results:
        tool_call_id = _safe_tool_call_id(tool_result)
        if tool_call_id is None or tool_call_id not in calls_by_id:
            repair_stats["dropped_orphan_tool_results"] += 1
            continue
        if tool_call_id in results_by_id:
            repair_stats["dropped_duplicate_tool_results"] += 1
            continue
        results_by_id[tool_call_id] = tool_result

    paired_call_ids = [tool_call_id for tool_call_id in call_ids if tool_call_id in results_by_id]
    repair_stats["dropped_unmatched_tool_calls"] += len(call_ids) - len(paired_call_ids)
    if not paired_call_ids:
        return []

    paired_calls = [calls_by_id[tool_call_id] for tool_call_id in paired_call_ids]
    paired_results = [results_by_id[tool_call_id] for tool_call_id in paired_call_ids]
    return [
        ModelResponse(
            parts=[_tool_call_part_from_memory(message) for message in paired_calls],
            timestamp=paired_calls[0].created_at,
            conversation_id=str(paired_calls[0].conversation_id),
            metadata=_group_metadata(paired_calls),
        ),
        ModelRequest(
            parts=[_tool_return_part_from_memory(message) for message in paired_results],
            timestamp=paired_results[0].created_at,
            conversation_id=str(paired_results[0].conversation_id),
            metadata=_group_metadata(paired_results),
        ),
    ]


def _tool_call_part_from_memory(message: ConversationMemoryMessage) -> ToolCallPart:
    tool_call_id = _safe_tool_call_id(message) or str(message.id)
    if _is_load_capability_message(message):
        return LoadCapabilityCallPart(
            args=cast(Any, _tool_args(message)),
            tool_call_id=tool_call_id,
        )
    return ToolCallPart(
        tool_name=message.tool_name or "unknown",
        args=_tool_args(message),
        tool_call_id=tool_call_id,
    )


def _tool_return_part_from_memory(message: ConversationMemoryMessage) -> ToolReturnPart:
    tool_call_id = _safe_tool_call_id(message) or str(message.id)
    if _is_load_capability_message(message):
        return LoadCapabilityReturnPart(
            content=cast(Any, _load_capability_return_content(message)),
            tool_call_id=tool_call_id,
            timestamp=message.created_at,
            metadata=_metadata(message),
        )
    return ToolReturnPart(
        tool_name=message.tool_name or "unknown",
        content=_tool_result_content(message),
        tool_call_id=tool_call_id,
        timestamp=message.created_at,
        metadata=_metadata(message),
    )


def _safe_tool_call_id(message: ConversationMemoryMessage) -> str | None:
    if isinstance(message.tool_call_id, str) and message.tool_call_id.strip():
        return message.tool_call_id
    return None


def _new_tool_history_repair_stats() -> dict[str, int]:
    return {
        "dropped_orphan_tool_results": 0,
        "dropped_unmatched_tool_calls": 0,
        "dropped_duplicate_tool_calls": 0,
        "dropped_duplicate_tool_results": 0,
    }


def _tool_history_was_repaired(stats: dict[str, int]) -> bool:
    return any(value > 0 for value in stats.values())


def _message_text(message: ConversationMemoryMessage) -> str:
    attachment_markers = _attachment_markers(message.attachments)
    if message.text:
        if attachment_markers:
            return "\n".join([message.text, *attachment_markers])
        return message.text
    if attachment_markers:
        return "\n".join(attachment_markers)
    if message.role is ConversationMemoryRole.TOOL_CALL:
        return f"Tool call: {message.tool_name or 'unknown'}"
    if message.role is ConversationMemoryRole.TOOL_RESULT:
        return f"Tool result: {message.tool_name or 'unknown'}"
    return "[empty message]"


def _attachment_markers(attachments: list[Attachment]) -> list[str]:
    markers: list[str] = []
    image_index = 0
    image_count = sum(
        1 for attachment in attachments if attachment.attachment_type.value == "image"
    )
    for attachment in attachments:
        if attachment.attachment_type.value != "image":
            continue
        media_id = _attachment_media_id(attachment)
        if media_id is None:
            continue
        image_index += 1
        label = (
            "Generated image" if attachment.metadata.get("generated") is True else "Attached image"
        )
        if image_count == 1:
            markers.append(f'[{label}: media_id="{media_id}"]')
        else:
            markers.append(f'[{label} {image_index}: media_id="{media_id}"]')
    return markers


def _attachment_media_id(attachment: Attachment) -> str | None:
    media_id = attachment.metadata.get("media_id")
    if isinstance(media_id, str) and media_id.strip():
        return media_id.strip()
    if attachment.attachment_id is not None and attachment.attachment_id.strip():
        return attachment.attachment_id.strip()
    return None


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
