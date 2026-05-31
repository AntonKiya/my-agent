import uvicorn

from agent_service.config import get_settings
from agent_service.observability.logging import configure_observability


def main() -> None:
    settings = get_settings()
    configure_observability(settings)
    uvicorn.run(
        "agent_service.app:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        log_config=None,
        log_level=settings.log_level.lower(),
    )
