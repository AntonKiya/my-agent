from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_service.time_tools import TIME_TOOLSET_ID, build_time_toolsets


@pytest.mark.asyncio
async def test_get_current_time_returns_utc_without_timezone() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]

    result = await tool.function(SimpleNamespace(deps={}), timezone=None)

    assert result["success"] is True
    data = result["data"]
    assert data["now_utc"].endswith("Z")
    assert data["timezone"] is None
    assert data["now_local"] is None


@pytest.mark.asyncio
async def test_get_current_time_returns_local_time_for_timezone() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]

    result = await tool.function(SimpleNamespace(deps={}), timezone="Europe/Moscow")

    assert result["success"] is True
    data = result["data"]
    assert data["timezone"] == "Europe/Moscow"
    assert isinstance(data["now_local"], str)


@pytest.mark.asyncio
async def test_get_current_time_uses_context_timezone() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]

    result = await tool.function(
        SimpleNamespace(deps={"user_timezone": "Europe/Moscow"}),
        timezone=None,
    )

    assert result["success"] is True
    assert result["data"]["timezone"] == "Europe/Moscow"


@pytest.mark.asyncio
async def test_get_current_time_rejects_invalid_timezone() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]

    result = await tool.function(SimpleNamespace(deps={}), timezone="Invalid/Zone")

    assert result == {
        "success": False,
        "error_code": "invalid_timezone",
        "message": "timezone must be a valid IANA timezone",
    }


def test_time_toolset_has_expected_id() -> None:
    (toolset,) = build_time_toolsets()

    assert cast(Any, toolset).id == TIME_TOOLSET_ID
