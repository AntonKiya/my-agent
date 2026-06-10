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
