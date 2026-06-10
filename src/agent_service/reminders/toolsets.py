from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.config import AppSettings
from agent_service.reminders.compiler import (
    ReminderScheduleError,
    format_local_fire_times,
    next_fire_at_utc,
    preview_next_fire_times,
    validate_timezone,
)
from agent_service.reminders.interfaces import ReminderStore
from agent_service.reminders.models import Reminder, ReminderSchedule, ReminderStatus

REMINDER_TOOLSET_ID = "reminders"
REMINDER_SKILL_ID = "reminders"
REMINDER_TOOLSET_INSTRUCTIONS = """
Use reminders tools when the user asks to create, list, or cancel reminders.
Never invent a timezone silently. If the user timezone is not available in context and the
user did not provide one, ask one short clarification question. Use IANA timezone names only.
For vague times, use these defaults and mention them in assumptions: morning=10:00,
middle of day/daytime=13:00, after lunch=14:00, afternoon=16:00, evening=18:00,
daytime window=09:00-18:00. If frequency or time is missing entirely, ask a clarification
question instead of creating a reminder. Keep medication, money, deadline, and dosage text exact.
"""


def build_reminder_toolsets(
    settings: AppSettings,
    *,
    reminder_store: ReminderStore,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.reminders_enabled:
        return ()

    async def create_reminder(
        ctx: RunContext[dict[str, Any]],
        message: str,
        schedule: ReminderSchedule,
        timezone: str | None = None,
        assumptions: list[str] | None = None,
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Create one reminder from a validated structured schedule.

        Args:
            message: Exact reminder text, without creative rewriting.
            schedule: Structured reminder schedule. Use once, weekly, or interval_window.
            timezone: IANA timezone, for example Europe/Moscow or Europe/Sofia.
                Omit only if user_timezone is available in context.
            assumptions: Any defaults used for vague phrases, such as evening=18:00.
            source_text: The user's original reminder request.
        """
        deps = ctx.deps or {}
        context = _tool_context(deps)
        if context is None:
            return _error("context_unavailable")
        if context.conversation_type != "private":
            return _error("unsupported_conversation_type")
        clean_message = message.strip()
        if not clean_message:
            return _error("empty_message")
        effective_timezone = (timezone or context.user_timezone or "").strip()
        if not effective_timezone:
            return _error("timezone_required")
        try:
            validate_timezone(effective_timezone)
        except ReminderScheduleError as exc:
            return _error("invalid_timezone", str(exc))
        if schedule.type == "interval_window":
            if schedule.interval_minutes < settings.reminders_min_interval_minutes:
                return _error(
                    "interval_too_short",
                    f"interval_minutes must be at least {settings.reminders_min_interval_minutes}",
                )

        active_count = await reminder_store.count_active_for_user(user_id=context.user_id)
        if active_count >= settings.reminders_max_active_per_user:
            return _error("too_many_active_reminders")
        created_from_message = await reminder_store.count_created_for_source_event(
            source_inbound_event_id=context.inbound_event_id,
        )
        if created_from_message >= settings.reminders_max_created_per_message:
            return _error("too_many_reminders_from_message")

        now = datetime.now(UTC)
        try:
            next_fire = next_fire_at_utc(
                schedule,
                timezone=effective_timezone,
                after_utc=now,
            )
            preview = preview_next_fire_times(
                schedule,
                timezone=effective_timezone,
                after_utc=now,
                limit=3,
            )
        except ReminderScheduleError as exc:
            return _error("invalid_schedule", str(exc))
        if next_fire is None:
            return _error("no_future_occurrences")

        clean_source_text = (
            source_text.strip()
            if source_text is not None and source_text.strip()
            else None
        )
        reminder = Reminder(
            user_id=context.user_id,
            channel=context.channel,
            external_chat_id=context.external_chat_id,
            thread_id=context.thread_id,
            source_conversation_id=context.conversation_id,
            source_inbound_event_id=context.inbound_event_id,
            timezone=effective_timezone,
            message=clean_message,
            schedule=schedule,
            source_text=clean_source_text,
            assumptions=[item.strip() for item in assumptions or () if item.strip()],
            next_fire_at_utc=next_fire,
            created_at=now,
            updated_at=now,
        )
        await reminder_store.create(reminder)
        return {
            "success": True,
            "data": {
                "reminder_id": str(reminder.id),
                "message": reminder.message,
                "timezone": reminder.timezone,
                "next_fire_at_utc": next_fire.isoformat(),
                "next_fire_local": format_local_fire_times(
                    preview,
                    timezone=effective_timezone,
                ),
                "assumptions": reminder.assumptions,
            },
        }

    async def list_reminders(ctx: RunContext[dict[str, Any]]) -> dict[str, Any]:
        """List the user's active and paused reminders.

        Args:
            ctx: Runtime context provided by the agent.
        """
        deps = ctx.deps or {}
        context = _tool_context(deps)
        if context is None:
            return _error("context_unavailable")
        reminders = await reminder_store.list_for_user(
            user_id=context.user_id,
            statuses=(ReminderStatus.ACTIVE, ReminderStatus.PAUSED),
            limit=20,
        )
        return {
            "success": True,
            "data": {
                "reminders": [
                    {
                        "reminder_id": str(reminder.id),
                        "message": reminder.message,
                        "status": reminder.status.value,
                        "timezone": reminder.timezone,
                        "next_fire_at_utc": (
                            reminder.next_fire_at_utc.isoformat()
                            if reminder.next_fire_at_utc is not None
                            else None
                        ),
                        "next_fire_local": (
                            format_local_fire_times(
                                (reminder.next_fire_at_utc,),
                                timezone=reminder.timezone,
                            )[0]
                            if reminder.next_fire_at_utc is not None
                            else None
                        ),
                    }
                    for reminder in reminders
                ],
            },
        }

    async def cancel_reminder(
        ctx: RunContext[dict[str, Any]],
        reminder_id: str,
    ) -> dict[str, Any]:
        """Cancel a reminder by id.

        Args:
            reminder_id: Reminder id from list_reminders or create_reminder.
        """
        deps = ctx.deps or {}
        context = _tool_context(deps)
        if context is None:
            return _error("context_unavailable")
        try:
            parsed_id = UUID(reminder_id)
        except ValueError:
            return _error("invalid_reminder_id")
        deleted = await reminder_store.mark_deleted(
            reminder_id=parsed_id,
            user_id=context.user_id,
        )
        if not deleted:
            return _error("reminder_not_found")
        return {"success": True, "data": {"reminder_id": str(parsed_id), "status": "deleted"}}

    return (
        FunctionToolset(
            [create_reminder, list_reminders, cancel_reminder],
            id=REMINDER_TOOLSET_ID,
            instructions=REMINDER_TOOLSET_INSTRUCTIONS,
            timeout=settings.reminders_tool_timeout_seconds,
            require_parameter_descriptions=True,
        ),
    )


class _ReminderToolContext:
    def __init__(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        inbound_event_id: UUID,
        channel: str,
        external_chat_id: str,
        thread_id: str | None,
        user_timezone: str | None,
        conversation_type: str,
    ) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.inbound_event_id = inbound_event_id
        self.channel = channel
        self.external_chat_id = external_chat_id
        self.thread_id = thread_id
        self.user_timezone = user_timezone
        self.conversation_type = conversation_type


def _tool_context(deps: dict[str, Any]) -> _ReminderToolContext | None:
    user_id = _uuid_dep(deps.get("user_id"))
    conversation_id = _uuid_dep(deps.get("conversation_id"))
    inbound_event_id = _uuid_dep(deps.get("inbound_event_id"))
    channel = _str_dep(deps.get("channel"))
    external_chat_id = _str_dep(deps.get("external_chat_id"))
    thread_id = _optional_str_dep(deps.get("thread_id"))
    user_timezone = _optional_str_dep(deps.get("user_timezone"))
    conversation_type = _str_dep(deps.get("conversation_type")) or "private"
    if (
        user_id is None
        or conversation_id is None
        or inbound_event_id is None
        or channel is None
        or external_chat_id is None
    ):
        return None
    return _ReminderToolContext(
        user_id=user_id,
        conversation_id=conversation_id,
        inbound_event_id=inbound_event_id,
        channel=channel,
        external_chat_id=external_chat_id,
        thread_id=thread_id,
        user_timezone=user_timezone,
        conversation_type=conversation_type,
    )


def _uuid_dep(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _str_dep(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_str_dep(value: object) -> str | None:
    if value is None:
        return None
    return _str_dep(value)


def _error(error_code: str, message: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"success": False, "error_code": error_code}
    if message is not None:
        payload["message"] = message
    return payload
