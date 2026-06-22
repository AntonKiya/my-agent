import json
import logging
from functools import lru_cache
from typing import Any

from agent_service.channels import Attachment
from agent_service.memory.models import ConversationMemoryMessage, ConversationMemoryRole
from agent_service.observability.events import log_event

DEFAULT_TOKEN_ENCODING = "o200k_base"
logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is declared, fallback is defensive.
    tiktoken = None  # type: ignore[assignment]


def estimate_text_tokens(
    text: str | None,
    *,
    encoding_name: str = DEFAULT_TOKEN_ENCODING,
) -> int:
    if not text:
        return 0
    if tiktoken is None:
        _log_token_estimator_fallback(reason="tiktoken_unavailable", encoding_name=encoding_name)
        return _fallback_token_estimate(text)
    try:
        encoding = _encoding(encoding_name)
    except Exception as exc:
        _log_token_estimator_fallback(
            reason="encoding_unavailable",
            encoding_name=encoding_name,
            error_type=type(exc).__name__,
        )
        return _fallback_token_estimate(text)
    return len(encoding.encode(text))


def estimate_message_tokens(
    message: ConversationMemoryMessage,
    *,
    encoding_name: str = DEFAULT_TOKEN_ENCODING,
) -> int:
    return estimate_text_tokens(
        _message_payload_for_estimate(message),
        encoding_name=encoding_name,
    )


def estimate_messages_tokens(
    messages: list[ConversationMemoryMessage],
    *,
    encoding_name: str = DEFAULT_TOKEN_ENCODING,
) -> int:
    return sum(
        estimate_message_tokens(message, encoding_name=encoding_name) for message in messages
    )


def usage_token_count(usage: object, name: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
        if callable(value):
            value = value()
    return value if isinstance(value, int) and value >= 0 else None


def usage_total_token_count(usage: object) -> int | None:
    total_tokens = usage_token_count(usage, "total_tokens")
    if total_tokens is not None:
        return total_tokens

    input_tokens = usage_token_count(usage, "input_tokens")
    output_tokens = usage_token_count(usage, "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0) + (output_tokens or 0)


@lru_cache(maxsize=16)
def _encoding(encoding_name: str) -> Any:
    if tiktoken is None:
        raise RuntimeError("tiktoken is not available")
    return tiktoken.get_encoding(encoding_name)


def _message_payload_for_estimate(message: ConversationMemoryMessage) -> str:
    if message.role is ConversationMemoryRole.TOOL_CALL:
        return _compact_json(
            {
                "role": message.role.value,
                "tool_name": message.tool_name or "unknown",
                "tool_call_id": message.tool_call_id or str(message.id),
                "args": _tool_args(message),
            }
        )
    if message.role is ConversationMemoryRole.TOOL_RESULT:
        return _compact_json(
            {
                "role": message.role.value,
                "tool_name": message.tool_name or "unknown",
                "tool_call_id": message.tool_call_id or str(message.id),
                "content": _tool_result_content(message),
            }
        )
    return _compact_json(
        {
            "role": message.role.value,
            "content": _message_text(message),
        }
    )


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


def _tool_result_content(message: ConversationMemoryMessage) -> str | dict[str, Any]:
    if message.text is not None:
        return message.text
    return message.metadata


def _compact_json(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _fallback_token_estimate(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@lru_cache(maxsize=16)
def _log_token_estimator_fallback(
    *,
    reason: str,
    encoding_name: str,
    error_type: str | None = None,
) -> None:
    log_event(
        logger,
        logging.WARNING,
        "Token estimator fell back to byte-length approximation",
        event="memory_token_estimator_fallback",
        reason=reason,
        encoding_name=encoding_name,
        error_type=error_type,
    )
