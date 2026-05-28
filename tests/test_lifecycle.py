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
