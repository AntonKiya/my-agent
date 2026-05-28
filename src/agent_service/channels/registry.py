from dataclasses import dataclass, field

from agent_service.channels.errors import ChannelAdapterNotFoundError
from agent_service.channels.interfaces import ChannelAdapter, ChannelAdapterRegistry
from agent_service.channels.models import ChannelName


@dataclass(slots=True)
class InMemoryChannelAdapterRegistry(ChannelAdapterRegistry):
    _adapters: dict[ChannelName, ChannelAdapter] = field(default_factory=dict, init=False)

    def register(self, adapter: ChannelAdapter) -> None:
        if not adapter.channel:
            raise ValueError("Channel adapter name must not be empty")

        existing = self._adapters.get(adapter.channel)
        if existing is not None and existing is not adapter:
            raise ValueError(f"Channel adapter is already registered for {adapter.channel!r}")

        self._adapters[adapter.channel] = adapter

    def get(self, channel: ChannelName) -> ChannelAdapter:
        try:
            return self._adapters[channel]
        except KeyError as exc:
            raise ChannelAdapterNotFoundError(channel) from exc

    @property
    def channels(self) -> tuple[ChannelName, ...]:
        return tuple(self._adapters)
