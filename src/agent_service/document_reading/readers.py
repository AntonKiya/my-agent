import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_service.media import MediaAsset

SUPPORTED_DOCUMENT_CONTENT_TYPES = frozenset(
    {
        "text/markdown",
        "text/plain",
        "text/x-markdown",
    }
)
SUPPORTED_DOCUMENT_SUFFIXES = frozenset({".md", ".txt"})


class DocumentReadError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class DocumentContent:
    media_id: str
    filename: str | None
    content_type: str | None
    content: str
    truncated: bool
    size_bytes: int
    char_count: int


class DocumentReader(Protocol):
    def supports(self, asset: MediaAsset) -> bool:
        """Return whether this reader can extract content from the asset."""
        ...

    async def read(self, *, asset: MediaAsset, max_chars: int) -> DocumentContent:
        """Extract text content from the asset."""
        ...


@dataclass(frozen=True, slots=True)
class PlainTextDocumentReader:
    def supports(self, asset: MediaAsset) -> bool:
        filename = document_filename(asset)
        if not is_supported_document_payload(filename=filename, content_type=asset.content_type):
            return False
        suffix = _filename_suffix(filename)
        return suffix == ".txt" or (
            suffix is None
            and _normalized_content_type(asset.content_type) == "text/plain"
        )

    async def read(self, *, asset: MediaAsset, max_chars: int) -> DocumentContent:
        return await _read_text_asset(asset=asset, max_chars=max_chars)


@dataclass(frozen=True, slots=True)
class MarkdownDocumentReader:
    def supports(self, asset: MediaAsset) -> bool:
        filename = document_filename(asset)
        if not is_supported_document_payload(filename=filename, content_type=asset.content_type):
            return False
        suffix = _filename_suffix(filename)
        return suffix == ".md" or (
            suffix is None
            and _normalized_content_type(asset.content_type) in {"text/markdown", "text/x-markdown"}
        )

    async def read(self, *, asset: MediaAsset, max_chars: int) -> DocumentContent:
        return await _read_text_asset(asset=asset, max_chars=max_chars)


@dataclass(frozen=True, slots=True)
class DocumentReaderRegistry:
    readers: tuple[DocumentReader, ...] = (
        PlainTextDocumentReader(),
        MarkdownDocumentReader(),
    )

    async def read(self, *, asset: MediaAsset, max_chars: int) -> DocumentContent:
        for reader in self.readers:
            if reader.supports(asset):
                return await reader.read(asset=asset, max_chars=max_chars)
        raise DocumentReadError(
            "Document format is not supported",
            error_code="unsupported_file_type",
        )


def is_supported_document_payload(*, filename: str | None, content_type: str | None) -> bool:
    suffix = _filename_suffix(filename)
    normalized_content_type = _normalized_content_type(content_type)
    if suffix is not None and suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        return False
    if normalized_content_type is not None:
        return normalized_content_type in SUPPORTED_DOCUMENT_CONTENT_TYPES
    return suffix in SUPPORTED_DOCUMENT_SUFFIXES


def document_filename(asset: MediaAsset) -> str | None:
    filename = asset.metadata.get("original_filename")
    if isinstance(filename, str) and filename:
        return filename
    candidate = Path(asset.storage_key).name
    return candidate or None


async def _read_text_asset(*, asset: MediaAsset, max_chars: int) -> DocumentContent:
    path = Path(asset.storage_key)
    try:
        content_bytes = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise DocumentReadError(
            "Document file could not be read",
            error_code="file_read_failed",
        ) from exc

    content = _decode_text(content_bytes)
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return DocumentContent(
        media_id=asset.media_id,
        filename=document_filename(asset),
        content_type=asset.content_type,
        content=content,
        truncated=truncated,
        size_bytes=asset.size_bytes,
        char_count=len(content),
    )


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _filename_suffix(filename: str | None) -> str | None:
    if filename is None:
        return None
    suffix = Path(filename).suffix.lower()
    return suffix or None


def _normalized_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized or None
