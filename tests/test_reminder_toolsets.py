from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_service.config import AppSettings
from agent_service.reminders.toolsets import build_reminder_toolsets


@pytest.mark.asyncio
async def test_reminder_toolset_does_not_emit_global_instructions() -> None:
    store = SimpleNamespace()

    toolsets = build_reminder_toolsets(
        AppSettings(environment="test", reminders_enabled=True),
        reminder_store=store,
    )
    toolset = cast(Any, toolsets[0])

    instructions = await toolset.get_instructions(SimpleNamespace(deps={}))

    assert instructions is None


def test_reminder_create_tools_require_timezone_in_tool_schema() -> None:
    store = SimpleNamespace()

    toolsets = build_reminder_toolsets(
        AppSettings(environment="test", reminders_enabled=True),
        reminder_store=store,
    )
    tools = cast(Any, toolsets[0]).tools

    for tool_name in (
        "create_once_reminder",
        "create_weekly_reminder",
        "create_interval_window_reminder",
    ):
        schema = tools[tool_name].function_schema.json_schema

        assert "timezone" in schema["required"]
        assert schema["properties"]["timezone"] == {
            "description": "IANA timezone, for example Europe/Moscow or Europe/Sofia.",
            "type": "string",
        }
