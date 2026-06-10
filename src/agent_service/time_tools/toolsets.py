from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

TIME_TOOLSET_ID = "time"


def build_time_toolsets() -> tuple[AgentToolset[Any], ...]:
    async def get_current_time(
        ctx: RunContext[dict[str, Any]],
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Get the current time in UTC and, optionally, in a user's local timezone.

        Args:
            timezone: Optional IANA timezone such as Europe/Moscow. If omitted, the tool
                uses user_timezone from runtime context when available.
        """
        deps = ctx.deps or {}
        effective_timezone = _clean_timezone(timezone) or _clean_timezone(deps.get("user_timezone"))
        now_utc = datetime.now(UTC).replace(second=0, microsecond=0)
        payload: dict[str, Any] = {
            "now_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "timezone": effective_timezone,
            "now_local": None,
        }
        if effective_timezone is None:
            return {"success": True, "data": payload}
        try:
            tz = ZoneInfo(effective_timezone)
        except ZoneInfoNotFoundError:
            return {
                "success": False,
                "error_code": "invalid_timezone",
                "message": "timezone must be a valid IANA timezone",
            }
        payload["now_local"] = now_utc.astimezone(tz).replace(tzinfo=None).isoformat()
        return {"success": True, "data": payload}

    return (
        FunctionToolset(
            [get_current_time],
            id=TIME_TOOLSET_ID,
            require_parameter_descriptions=True,
        ),
    )


def _clean_timezone(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
