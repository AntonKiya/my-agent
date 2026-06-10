from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.tools import ToolDefinition

from agent_service.config import AppSettings
from agent_service.reminders.toolsets import (
    _prepare_reminder_tool_definition,
    build_reminder_toolsets,
)


def test_reminder_tool_prepare_adds_runtime_context_to_tool_description() -> None:
    ctx = SimpleNamespace(deps={"user_timezone": "Europe/Moscow"})
    tool_def = ToolDefinition(
        name="create_once_reminder",
        description="Create a one-time reminder.",
    )

    prepared = _prepare_reminder_tool_definition(ctx, tool_def)  # type: ignore[arg-type]

    assert prepared.description is not None
    assert prepared.description.startswith("Create a one-time reminder.")
    assert "Reminder runtime context:" in prepared.description
    assert "current UTC time:" in prepared.description
    assert "user profile timezone: Europe/Moscow" in prepared.description
    assert "user profile local time:" in prepared.description
    assert "do not ask the user what time it is now" in prepared.description


def test_reminder_tool_prepare_handles_missing_timezone() -> None:
    ctx = SimpleNamespace(deps={})
    tool_def = ToolDefinition(name="create_once_reminder", description="Create reminder.")

    prepared = _prepare_reminder_tool_definition(ctx, tool_def)  # type: ignore[arg-type]

    assert prepared.description is not None
    assert "user profile timezone: unknown" in prepared.description


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
