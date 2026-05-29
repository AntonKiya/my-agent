import asyncio
from uuid import UUID, uuid4

import pytest

from agent_service.conversations import (
    AsyncioConversationLockManager,
    ConversationLockManager,
    ConversationLockTimeoutError,
)


async def test_conversation_lock_manager_returns_lease_and_cleans_idle_lock() -> None:
    manager = AsyncioConversationLockManager()
    conversation_id = uuid4()

    async with manager.acquire(conversation_id) as lease:
        assert lease.conversation_id == conversation_id
        assert manager.tracked_lock_count == 1

    assert isinstance(manager, ConversationLockManager)
    assert manager.tracked_lock_count == 0


async def test_same_conversation_is_processed_sequentially() -> None:
    manager = AsyncioConversationLockManager()
    conversation_id = uuid4()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    order: list[str] = []

    async def first_worker() -> None:
        async with manager.acquire(conversation_id):
            order.append("first-enter")
            first_entered.set()
            await release_first.wait()
            order.append("first-exit")

    async def second_worker() -> None:
        async with manager.acquire(conversation_id):
            order.append("second-enter")
            second_entered.set()

    first_task = asyncio.create_task(first_worker())
    await asyncio.wait_for(first_entered.wait(), timeout=0.1)
    second_task = asyncio.create_task(second_worker())
    await asyncio.sleep(0)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=0.1)

    assert order == ["first-enter", "first-exit", "second-enter"]
    assert manager.tracked_lock_count == 0


async def test_different_conversations_can_run_in_parallel() -> None:
    manager = AsyncioConversationLockManager()
    first_conversation_id = uuid4()
    second_conversation_id = uuid4()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release_both = asyncio.Event()

    async def worker(conversation_id: UUID, entered: asyncio.Event) -> None:
        async with manager.acquire(conversation_id):
            entered.set()
            await release_both.wait()

    first_task = asyncio.create_task(worker(first_conversation_id, first_entered))
    second_task = asyncio.create_task(worker(second_conversation_id, second_entered))

    await asyncio.wait_for(first_entered.wait(), timeout=0.1)
    await asyncio.wait_for(second_entered.wait(), timeout=0.1)

    release_both.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=0.1)

    assert manager.tracked_lock_count == 0


async def test_lock_acquire_timeout_does_not_leak_waiter_reference() -> None:
    manager = AsyncioConversationLockManager()
    conversation_id = uuid4()

    async with manager.acquire(conversation_id):
        with pytest.raises(ConversationLockTimeoutError):
            async with manager.acquire(conversation_id, timeout_seconds=0.001):
                pass
        assert manager.tracked_lock_count == 1

    assert manager.tracked_lock_count == 0


async def test_waiter_cancellation_does_not_leak_lock_reference() -> None:
    manager = AsyncioConversationLockManager()
    conversation_id = uuid4()
    waiter_started = asyncio.Event()

    async def waiter() -> None:
        waiter_started.set()
        async with manager.acquire(conversation_id):
            pass

    async with manager.acquire(conversation_id):
        task = asyncio.create_task(waiter())
        await asyncio.wait_for(waiter_started.wait(), timeout=0.1)
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert manager.tracked_lock_count == 1

    await asyncio.sleep(0)
    assert manager.tracked_lock_count == 0


async def test_lock_timeout_must_be_positive() -> None:
    manager = AsyncioConversationLockManager()

    with pytest.raises(ValueError):
        async with manager.acquire(uuid4(), timeout_seconds=0):
            pass
