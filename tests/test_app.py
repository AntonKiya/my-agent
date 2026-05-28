from starlette.routing import Route

from agent_service.api.health import health, ready
from agent_service.app import create_app, lifespan
from agent_service.config import AppSettings


def test_create_app_registers_health_routes() -> None:
    settings = AppSettings(environment="test")

    app = create_app(settings)
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}

    assert app.state.settings is settings
    assert app.state.container.settings is settings
    assert {"/health", "/ready", "/webhooks/telegram"}.issubset(route_paths)


async def test_app_lifespan_starts_and_stops_container() -> None:
    app = create_app(AppSettings(environment="test"))
    container = app.state.container

    assert not container.started

    async with lifespan(app):
        assert container.started

    assert not container.started


async def test_health_endpoints_return_service_status() -> None:
    settings = AppSettings(environment="test", service_name="test-agent-service")

    health_response = await health(settings)
    ready_response = await ready(settings)

    assert health_response.status == "ok"
    assert ready_response.status == "ok"
    assert health_response.service == "test-agent-service"
    assert ready_response.environment == "test"
