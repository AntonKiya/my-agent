import pytest

from agent_service.channels import (
    ChannelAdapterNotFoundError,
    DeliveryResult,
    DeliveryStatus,
    InMemoryChannelAdapterRegistry,
    OutboundEvent,
)


class FakeAdapter:
    def __init__(self, channel: str = "telegram") -> None:
        self.channel = channel

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=DeliveryStatus.SENT,
        )


def test_in_memory_channel_adapter_registry_resolves_registered_adapter() -> None:
    registry = InMemoryChannelAdapterRegistry()
    adapter = FakeAdapter()

    registry.register(adapter)

    assert registry.get("telegram") is adapter
    assert registry.channels == ("telegram",)


def test_in_memory_channel_adapter_registry_rejects_missing_adapter() -> None:
    registry = InMemoryChannelAdapterRegistry()

    with pytest.raises(ChannelAdapterNotFoundError) as exc_info:
        registry.get("telegram")

    assert exc_info.value.channel == "telegram"


def test_in_memory_channel_adapter_registry_rejects_duplicate_channel() -> None:
    registry = InMemoryChannelAdapterRegistry()
    registry.register(FakeAdapter("telegram"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeAdapter("telegram"))
