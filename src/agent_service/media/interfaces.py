from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_service.channels.models import Attachment, ChannelName, InboundEvent
from agent_service.media.models import MediaAsset, MediaPayload, StoredMedia


class MediaError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code


class MediaFetchError(MediaError):
    """Raised when channel media cannot be fetched."""


class MediaStorageError(MediaError):
    """Raised when temporary media cannot be stored or removed."""


@runtime_checkable
class ChannelMediaFetcher(Protocol):
    channel: ChannelName

    async def fetch(self, *, event: InboundEvent, attachment: Attachment) -> MediaPayload:
        """Fetch a channel attachment into a channel-neutral media payload."""
        ...


@runtime_checkable
class MediaFetcherRegistry(Protocol):
    def register(self, fetcher: ChannelMediaFetcher) -> None:
        """Register a fetcher for its channel."""
        ...

    def get(self, channel: ChannelName) -> ChannelMediaFetcher:
        """Return the fetcher for a channel."""
        ...


@runtime_checkable
class MediaStore(Protocol):
    async def store(self, payload: MediaPayload) -> StoredMedia:
        """Persist media temporarily for processing."""
        ...

    async def delete(self, media: StoredMedia) -> None:
        """Delete temporary media after processing reaches a final outcome."""
        ...


@runtime_checkable
class PersistentMediaStore(Protocol):
    async def store(self, *, media_id: str, payload: MediaPayload) -> StoredMedia:
        """Persist media for future tool access."""
        ...


@runtime_checkable
class MediaAssetStore(Protocol):
    async def create(self, *, asset: MediaAsset) -> MediaAsset:
        """Persist a media asset index row."""
        ...

    async def get(
        self,
        *,
        media_id: str,
        user_id: UUID,
        conversation_id: UUID,
    ) -> MediaAsset | None:
        """Load a media asset only when it belongs to the given user and conversation."""
        ...
