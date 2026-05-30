import asyncio

import agent_service
from agent_service import inbound, memory, messaging


def test_package_exposes_main() -> None:
    assert callable(agent_service.main)


def test_memory_package_exposes_runtime_contracts() -> None:
    assert memory.ConversationMemoryStore.__name__ == "ConversationMemoryStore"
    assert memory.ConversationContextSnapshotStore.__name__ == "ConversationContextSnapshotStore"
    assert memory.ConversationCompactor.__name__ == "ConversationCompactor"
    assert memory.DefaultConversationMemoryService.__name__ == "DefaultConversationMemoryService"
    assert memory.NoopConversationCompactor.__name__ == "NoopConversationCompactor"
    assert (
        memory.RedisConversationContextSnapshotStore.__name__
        == "RedisConversationContextSnapshotStore"
    )
    assert memory.PostgresConversationMemoryStore.__name__ == "PostgresConversationMemoryStore"
    assert callable(memory.pydantic_ai_message_from_memory)
    assert callable(memory.compaction_request_from_snapshot)


def test_messaging_and_inbound_packages_expose_high_load_contracts() -> None:
    assert messaging.QueueStats.__name__ == "QueueStats"
    assert inbound.InboundIntakeStatus.OVERLOADED.value == "overloaded"


async def test_async_test_runtime_is_configured() -> None:
    await asyncio.sleep(0)
