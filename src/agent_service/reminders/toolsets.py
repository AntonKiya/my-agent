from dataclasses import replace
from datetime import UTC, date, datetime, time
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition
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
from agent_service.reminders.models import (
    IntervalWindowSchedule,
    OnceSchedule,
    Reminder,
    ReminderSchedule,
    ReminderStatus,
    Weekday,
    WeeklySchedule,
)

REMINDER_TOOLSET_ID = "reminders"
REMINDER_SKILL_ID = "reminders"


def build_reminder_toolsets(
    settings: AppSettings,
    *,
    reminder_store: ReminderStore,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.reminders_enabled:
        return ()

    async def create_once_reminder(
        ctx: RunContext[dict[str, Any]],
        message: str,
        local_datetime: datetime,
        timezone: str | None = None,
        assumptions: list[str] | None = None,
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Create a one-time reminder at a local date and time.

        Args:
            message: Exact reminder text, without creative rewriting.
            local_datetime: Local date and time without timezone, for example
                2026-06-10T18:00:00. For relative requests, compute it from runtime current time.
            timezone: IANA timezone, for example Europe/Moscow or Europe/Sofia.
                Omit only if user_timezone is available in context.
            assumptions: Any defaults used for vague phrases, such as evening=18:00.
            source_text: The user's original reminder request.
        """
        schedule = OnceSchedule(local_datetime=local_datetime)
        return await _create_reminder(
            ctx,
            message=message,
            schedule=schedule,
            timezone=timezone,
            assumptions=assumptions,
            source_text=source_text,
        )

    async def create_weekly_reminder(
        ctx: RunContext[dict[str, Any]],
        message: str,
        days_of_week: list[Weekday],
        times: list[time],
        timezone: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        assumptions: list[str] | None = None,
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Create a weekly repeating reminder on selected days and local times.

        Args:
            message: Exact reminder text, without creative rewriting.
            days_of_week: Weekdays such as MO, TU, WE, TH, FR, SA, SU.
            times: Local times without timezone, for example 10:00 or 18:30.
            timezone: IANA timezone, for example Europe/Moscow or Europe/Sofia.
                Omit only if user_timezone is available in context.
            start_date: Optional local start date.
            end_date: Optional local end date.
            assumptions: Any defaults used for vague phrases, such as after lunch=14:00.
            source_text: The user's original reminder request.
        """
        schedule = WeeklySchedule(
            days_of_week=days_of_week,
            times=times,
            start_date=start_date,
            end_date=end_date,
        )
        return await _create_reminder(
            ctx,
            message=message,
            schedule=schedule,
            timezone=timezone,
            assumptions=assumptions,
            source_text=source_text,
        )

    async def create_interval_window_reminder(
        ctx: RunContext[dict[str, Any]],
        message: str,
        interval_minutes: int,
        window_start: time,
        window_end: time,
        timezone: str | None = None,
        days_of_week: list[Weekday] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        assumptions: list[str] | None = None,
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Create a repeating reminder every N minutes inside a local time window.

        Args:
            message: Exact reminder text, without creative rewriting.
            interval_minutes: Interval in minutes between reminders.
            window_start: Local window start time without timezone, for example 09:00.
            window_end: Local window end time without timezone, for example 18:00.
            timezone: IANA timezone, for example Europe/Moscow or Europe/Sofia.
                Omit only if user_timezone is available in context.
            days_of_week: Optional weekdays such as MO, TU, WE, TH, FR, SA, SU.
            start_date: Optional local start date.
            end_date: Optional local end date.
            assumptions: Any defaults used for vague phrases, such as daytime window=09:00-18:00.
            source_text: The user's original reminder request.
        """
        schedule = IntervalWindowSchedule(
            interval_minutes=interval_minutes,
            days_of_week=days_of_week,
            window_start=window_start,
            window_end=window_end,
            start_date=start_date,
            end_date=end_date,
        )
        return await _create_reminder(
            ctx,
            message=message,
            schedule=schedule,
            timezone=timezone,
            assumptions=assumptions,
            source_text=source_text,
        )

    async def _create_reminder(
        ctx: RunContext[dict[str, Any]],
        *,
        message: str,
        schedule: ReminderSchedule,
        timezone: str | None,
        assumptions: list[str] | None,
        source_text: str | None,
    ) -> dict[str, Any]:
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
            reminder_id: Reminder id from list_reminders or a successful create tool result.
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
            [
                Tool(
                    create_once_reminder,
                    prepare=_prepare_reminder_tool_definition,
                    require_parameter_descriptions=True,
                ),
                Tool(
                    create_weekly_reminder,
                    prepare=_prepare_reminder_tool_definition,
                    require_parameter_descriptions=True,
                ),
                Tool(
                    create_interval_window_reminder,
                    prepare=_prepare_reminder_tool_definition,
                    require_parameter_descriptions=True,
                ),
                list_reminders,
                cancel_reminder,
            ],
            id=REMINDER_TOOLSET_ID,
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


def _prepare_reminder_tool_definition(
    ctx: RunContext[dict[str, Any]],
    tool_def: ToolDefinition,
) -> ToolDefinition:
    description = tool_def.description or ""
    return replace(
        tool_def,
        description=f"{description}\n\n{_reminder_runtime_context(ctx)}",
    )


def _reminder_runtime_context(ctx: RunContext[dict[str, Any]]) -> str:
    deps = ctx.deps or {}
    now_utc = datetime.now(UTC).replace(second=0, microsecond=0)
    lines = [
        "Reminder runtime context:",
        f"- current UTC time: {now_utc.isoformat().replace('+00:00', 'Z')}",
    ]
    user_timezone = _optional_str_dep(deps.get("user_timezone"))
    if user_timezone is None:
        lines.append("- user profile timezone: unknown")
    else:
        lines.append(f"- user profile timezone: {user_timezone}")
        try:
            now_local = now_utc.astimezone(ZoneInfo(user_timezone)).replace(tzinfo=None)
        except ZoneInfoNotFoundError:
            lines.append("- user profile local time: unavailable because timezone is invalid")
        else:
            lines.append(f"- user profile local time: {now_local.isoformat()}")
    lines.append(
        "Use this current time for relative reminder requests; "
        "do not ask the user what time it is now."
    )
    return "\n".join(lines)


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
