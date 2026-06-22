from typing import Any
from uuid import UUID

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.config import AppSettings
from agent_service.document_reading.readers import (
    DocumentReaderRegistry,
    DocumentReadError,
)
from agent_service.media import MediaAssetStore, MediaAssetType

FILE_READING_TOOLSET_ID = "file-reading"
FILE_READING_TOOLSET_INSTRUCTIONS = (
    "For any question about an attached or previously attached file/document, call "
    "readFileContent before answering. Use media_id values from current file markers or recent "
    "context, preferring the latest relevant file. Do not infer file contents from filename, "
    "memory, or prior summaries alone."
)


def build_file_reading_toolsets(
    settings: AppSettings,
    *,
    media_asset_store: MediaAssetStore,
    reader_registry: DocumentReaderRegistry | None = None,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.document_reading_enabled:
        return ()

    readers = reader_registry or DocumentReaderRegistry()

    async def readFileContent(
        ctx: RunContext[dict[str, Any]],
        media_ids: list[str],
    ) -> dict[str, Any]:
        """Read text content from attached TXT, Markdown, DOCX, PDF, or PPTX files.

        Args:
            media_ids: One or more media_id values from the attached file markers.
        """
        deps = ctx.deps or {}
        user_id = _uuid_dep(deps.get("user_id"))
        conversation_id = _uuid_dep(deps.get("conversation_id"))
        if user_id is None or conversation_id is None:
            return _error("context_unavailable")

        normalized_media_ids = _normalized_media_ids(media_ids)
        if not normalized_media_ids:
            return _error("empty_media_ids")

        files: list[dict[str, Any]] = []
        for media_id in normalized_media_ids:
            asset = await media_asset_store.get(
                media_id=media_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if asset is None or asset.media_type is not MediaAssetType.DOCUMENT:
                return _error("file_not_found_or_not_accessible")
            if asset.size_bytes > settings.document_max_size_bytes:
                return _error("file_too_large")

            try:
                content = await readers.read(
                    asset=asset,
                    max_chars=settings.document_max_extracted_chars,
                )
            except DocumentReadError as exc:
                return _error(exc.error_code or "file_read_failed")

            files.append(
                {
                    "media_id": content.media_id,
                    "filename": content.filename,
                    "content_type": content.content_type,
                    "content": content.content,
                    "truncated": content.truncated,
                    "size_bytes": content.size_bytes,
                    "char_count": content.char_count,
                }
            )

        return {
            "success": True,
            "data": {
                "files": files,
            },
        }

    return (
        FunctionToolset(
            [readFileContent],
            id=FILE_READING_TOOLSET_ID,
            instructions=FILE_READING_TOOLSET_INSTRUCTIONS,
            timeout=settings.document_reading_tool_timeout_seconds,
            require_parameter_descriptions=True,
        ),
    )


def _normalized_media_ids(media_ids: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for media_id in media_ids:
        value = media_id.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _uuid_dep(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _error(error_code: str) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
    }
