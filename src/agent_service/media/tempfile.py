import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_service.media.interfaces import MediaStorageError, MediaStore, PersistentMediaStore
from agent_service.media.models import MediaPayload, StoredMedia


@dataclass(slots=True)
class TempFileMediaStore(MediaStore):
    base_dir: Path | None = None

    async def store(self, payload: MediaPayload) -> StoredMedia:
        try:
            path = await asyncio.to_thread(self._write_payload, payload)
        except Exception as exc:
            raise MediaStorageError(
                "Temporary media file could not be written",
                retryable=True,
                error_code="media_temp_write_failed",
            ) from exc
        return StoredMedia(
            path=path,
            content_type=payload.content_type,
            filename=payload.filename,
            size_bytes=payload.size_bytes,
            metadata=dict(payload.metadata),
        )

    async def delete(self, media: StoredMedia) -> None:
        try:
            await asyncio.to_thread(media.path.unlink, missing_ok=True)
        except Exception as exc:
            raise MediaStorageError(
                "Temporary media file could not be deleted",
                retryable=False,
                error_code="media_temp_delete_failed",
            ) from exc

    def _write_payload(self, payload: MediaPayload) -> Path:
        base_dir = self.base_dir or Path(tempfile.gettempdir()) / "agent-service-media"
        base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        suffix = _safe_suffix(payload.filename, payload.content_type)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="inbound-",
            suffix=suffix,
            dir=base_dir,
            delete=False,
        ) as file:
            os.chmod(file.name, 0o600)
            file.write(payload.content)
            return Path(file.name)


def _safe_suffix(filename: str | None, content_type: str | None) -> str:
    if filename is not None:
        suffix = Path(filename).suffix
        if suffix and suffix.isascii() and len(suffix) <= 16:
            return suffix.lower()
    if content_type == "audio/ogg":
        return ".ogg"
    if content_type == "audio/mpeg":
        return ".mp3"
    if content_type in {"audio/mp4", "audio/x-m4a"}:
        return ".m4a"
    if content_type == "audio/wav":
        return ".wav"
    if content_type == "audio/webm":
        return ".webm"
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    if content_type == "image/gif":
        return ".gif"
    return ".bin"


@dataclass(slots=True)
class PersistentFileMediaStore(PersistentMediaStore):
    base_dir: Path

    async def store(self, *, media_id: str, payload: MediaPayload) -> StoredMedia:
        try:
            path = await asyncio.to_thread(self._write_payload, media_id, payload)
        except Exception as exc:
            raise MediaStorageError(
                "Persistent media file could not be written",
                retryable=True,
                error_code="media_persistent_write_failed",
            ) from exc
        return StoredMedia(
            path=path,
            content_type=payload.content_type,
            filename=payload.filename,
            size_bytes=payload.size_bytes,
            metadata=dict(payload.metadata),
        )

    def _write_payload(self, media_id: str, payload: MediaPayload) -> Path:
        self.base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        suffix = _safe_suffix(payload.filename, payload.content_type)
        path = self.base_dir / f"{media_id}{suffix}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(payload.content)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
