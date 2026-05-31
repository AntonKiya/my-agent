import asyncio

from agent_service.runtime.lifecycle import TaskSupervisor


async def test_task_supervisor_removes_completed_tasks() -> None:
    supervisor = TaskSupervisor(shutdown_timeout_seconds=0.1)
    completed = asyncio.Event()

    async def worker() -> None:
        completed.set()

    task = supervisor.create_task(worker(), name="test-completed-worker")

    await task
    await asyncio.sleep(0)

    assert completed.is_set()
    assert supervisor.task_count == 0


async def test_task_supervisor_cancels_running_tasks_on_stop() -> None:
    supervisor = TaskSupervisor(shutdown_timeout_seconds=0.1)
    cancelled = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    supervisor.create_task(worker(), name="test-running-worker")
    await asyncio.sleep(0)

    assert supervisor.task_count == 1

    await supervisor.stop()
    await asyncio.sleep(0)

    assert cancelled.is_set()
    assert supervisor.task_count == 0


async def test_task_supervisor_can_stop_one_task_group() -> None:
    supervisor = TaskSupervisor(shutdown_timeout_seconds=0.1)
    stopped_first = asyncio.Event()
    stopped_second = asyncio.Event()

    async def worker(stopped: asyncio.Event) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            raise

    supervisor.create_task(worker(stopped_first), name="first-worker", group="first")
    supervisor.create_task(worker(stopped_second), name="second-worker", group="second")
    await asyncio.sleep(0)

    await supervisor.stop(group="first")
    await asyncio.sleep(0)

    assert stopped_first.is_set()
    assert not stopped_second.is_set()
    assert supervisor.task_count == 1
    assert supervisor.task_count_for_group("second") == 1

    await supervisor.stop()
    await asyncio.sleep(0)

    assert stopped_second.is_set()
    assert supervisor.task_count == 0
