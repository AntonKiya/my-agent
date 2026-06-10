from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_service.time_tools import TIME_TOOLSET_ID, build_time_toolsets


@pytest.mark.asyncio
async def test_get_current_time_requires_timezone() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]

    result = await tool.function(SimpleNamespace(deps={}), timezone=None)

    assert result == {
        "success": False,
        "error_code": "timezone_required",
        "message": "timezone must be provided as a valid IANA timezone",
    }


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


def test_get_current_time_requires_timezone_in_tool_schema() -> None:
    (toolset,) = build_time_toolsets()
    tool = cast(Any, toolset).tools["get_current_time"]
    schema = tool.function_schema.json_schema

    assert "timezone" in schema["required"]
    assert schema["properties"]["timezone"] == {
        "description": "Required IANA timezone such as Europe/Moscow or Europe/Sofia.",
        "type": "string",
    }
