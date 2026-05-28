class ChannelError(Exception):
    """Base error for channel infrastructure failures."""


class ChannelAdapterNotFoundError(ChannelError):
    def __init__(self, channel: str) -> None:
        super().__init__(f"Channel adapter is not registered for channel {channel!r}")
        self.channel = channel
