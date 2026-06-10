from types import SimpleNamespace

from agent_service.reminders.toolsets import _reminder_runtime_instructions


def test_reminder_runtime_instructions_include_current_time_only_for_toolset() -> None:
    ctx = SimpleNamespace(deps={"user_timezone": "Europe/Moscow"})

    instructions = _reminder_runtime_instructions(ctx)  # type: ignore[arg-type]

    assert instructions.startswith("Reminder runtime context:")
    assert "current UTC time:" in instructions
    assert "user profile timezone: Europe/Moscow" in instructions
    assert "user profile local time:" in instructions
    assert "do not ask the user what time it is now" in instructions


def test_reminder_runtime_instructions_handle_missing_timezone() -> None:
    ctx = SimpleNamespace(deps={})

    instructions = _reminder_runtime_instructions(ctx)  # type: ignore[arg-type]

    assert "user profile timezone: unknown" in instructions
