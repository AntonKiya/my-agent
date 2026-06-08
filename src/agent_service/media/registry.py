from dataclasses import dataclass, field

from agent_service.channels.models import ChannelName
from agent_service.media.interfaces import ChannelMediaFetcher


class MediaFetcherNotFoundError(LookupError):
    def __init__(self, channel: ChannelName) -> None:
        super().__init__(f"No media fetcher registered for channel {channel!r}")
        self.channel = channel


@dataclass(slots=True)
class InMemoryMediaFetcherRegistry:
    _fetchers: dict[ChannelName, ChannelMediaFetcher] = field(default_factory=dict)

    @property
    def channels(self) -> tuple[ChannelName, ...]:
        return tuple(self._fetchers)

    def register(self, fetcher: ChannelMediaFetcher) -> None:
        if fetcher.channel in self._fetchers:
            raise ValueError(f"Media fetcher already registered for channel {fetcher.channel!r}")
        self._fetchers[fetcher.channel] = fetcher

    def get(self, channel: ChannelName) -> ChannelMediaFetcher:
        try:
            return self._fetchers[channel]
        except KeyError as exc:
            raise MediaFetcherNotFoundError(channel) from exc
