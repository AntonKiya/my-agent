from agent_service.config import AppSettings
from agent_service.container import AppContainer


async def test_container_tracks_lifecycle_state() -> None:
    settings = AppSettings(environment="test", graceful_shutdown_timeout_seconds=0.25)
    container = AppContainer(settings=settings)

    assert not container.started
    assert container.task_supervisor.shutdown_timeout_seconds == 0.25

    await container.start()

    assert container.started

    await container.stop()

    assert not container.started
