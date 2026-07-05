from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AbstractToolset

from agent_service.config import AppSettings
from agent_service.mcp.toolsets import (
    PrefixedMCPToolsetConfig,
    ToolCallValidator,
    ToolResultTransformer,
    TransformingToolset,
    build_prefixed_mcp_toolset,
    prefixed_tool_names,
)
from agent_service.mcp.tutu_refs import (
    build_tutu_checkout_link_call_transformer,
    build_tutu_search_result_transformer,
    tutu_checkout_link_definition_transformers,
)
from agent_service.tool_refs import (
    InMemoryToolResultReferenceStore,
    ToolResultReferenceStore,
)

TUTU_MCP_SERVER_ID = "tutu"
TUTU_TRAVEL_SKILL_ID = "tutu-travel"
TUTU_MCP_TOOL_PREFIX = "mcp_tutu"
TUTU_MCP_RAW_TOOL_NAMES = frozenset(
    {
        "search_hotels",
        "search_avia",
        "search_rail",
        "search_bus",
        "search_etrain",
        "search_multitransport",
        "get_offer_details",
        "get_rail_seatmap",
        "create_checkout_link",
    }
)
TUTU_MCP_TOOL_NAMES = prefixed_tool_names(
    TUTU_MCP_TOOL_PREFIX,
    TUTU_MCP_RAW_TOOL_NAMES,
)
TUTU_MCP_RESULT_TRANSFORMERS: dict[str, ToolResultTransformer] = {}
DateProvider = Callable[[], date]

TUTU_SEARCH_HOTELS_TOOL_NAME = f"{TUTU_MCP_TOOL_PREFIX}_search_hotels"
TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME = f"{TUTU_MCP_TOOL_PREFIX}_create_checkout_link"
TUTU_TRANSPORT_SEARCH_TOOL_NAMES = frozenset(
    {
        f"{TUTU_MCP_TOOL_PREFIX}_search_avia",
        f"{TUTU_MCP_TOOL_PREFIX}_search_rail",
        f"{TUTU_MCP_TOOL_PREFIX}_search_bus",
        f"{TUTU_MCP_TOOL_PREFIX}_search_etrain",
        f"{TUTU_MCP_TOOL_PREFIX}_search_multitransport",
    }
)
TUTU_SEARCH_RESULT_TOOL_NAMES = TUTU_TRANSPORT_SEARCH_TOOL_NAMES | {TUTU_SEARCH_HOTELS_TOOL_NAME}


def build_tutu_mcp_toolsets(
    settings: AppSettings,
    *,
    today_provider: DateProvider | None = None,
    tool_result_reference_store: ToolResultReferenceStore | None = None,
) -> tuple[AbstractToolset[Any], ...]:
    if settings.tutu_mcp_command is None and settings.tutu_mcp_url is None:
        return ()

    reference_store = tool_result_reference_store or InMemoryToolResultReferenceStore()
    search_result_transformer = build_tutu_search_result_transformer(reference_store)
    toolset = build_prefixed_mcp_toolset(
        PrefixedMCPToolsetConfig(
            server_id=TUTU_MCP_SERVER_ID,
            prefix=TUTU_MCP_TOOL_PREFIX,
            command=settings.tutu_mcp_command,
            args=settings.tutu_mcp_args,
            env=settings.tutu_mcp_env,
            url=settings.tutu_mcp_url,
            headers=settings.tutu_mcp_headers,
            init_timeout_seconds=settings.tutu_mcp_init_timeout_seconds,
            read_timeout_seconds=settings.tutu_mcp_read_timeout_seconds,
            allowed_raw_tool_names=TUTU_MCP_RAW_TOOL_NAMES,
        )
    )
    return (
        TransformingToolset(
            toolset,
            TUTU_MCP_RESULT_TRANSFORMERS,
            contextual_result_transformers={
                name: search_result_transformer for name in TUTU_SEARCH_RESULT_TOOL_NAMES
            },
            call_transformers={
                TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME: (
                    build_tutu_checkout_link_call_transformer(reference_store)
                ),
            },
            tool_definition_transformers=tutu_checkout_link_definition_transformers(
                TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME
            ),
            return_error_results_for_tool_names=TUTU_MCP_TOOL_NAMES,
            log_error_args_for_tool_names=TUTU_MCP_TOOL_NAMES,
            pre_call_validators=(build_tutu_mcp_pre_call_validator(today_provider=today_provider),),
        ),
    )


def build_tutu_mcp_pre_call_validator(
    *,
    today_provider: DateProvider | None = None,
) -> ToolCallValidator:
    current_date = today_provider or date.today

    def validate(
        name: str,
        tool_args: Mapping[str, Any],
        _ctx: RunContext[Any],
    ) -> Any | None:
        return validate_tutu_mcp_tool_call(name, tool_args, today=current_date())

    return validate


def validate_tutu_mcp_tool_call(
    name: str,
    tool_args: Mapping[str, Any],
    *,
    today: date | None = None,
) -> dict[str, Any] | None:
    current_date = today or date.today()
    if name == TUTU_SEARCH_HOTELS_TOOL_NAME:
        return _validate_hotel_search_tool_call(name, tool_args, current_date)
    if name in TUTU_TRANSPORT_SEARCH_TOOL_NAMES:
        return _validate_transport_search_tool_call(name, tool_args, current_date)
    return None


def _validate_hotel_search_tool_call(
    name: str,
    tool_args: Mapping[str, Any],
    today: date,
) -> dict[str, Any] | None:
    missing_labels: list[str] = []
    missing_fields: list[str] = []

    if not _any_arg_has_value(tool_args, ("city_name", "geo_id")):
        missing_labels.append("город или локацию")
        missing_fields.extend(["city_name", "geo_id"])

    check_in_arg = _date_arg(tool_args, ("check_in", "checkin_date"))
    check_out_arg = _date_arg(tool_args, ("check_out", "checkout_date"))
    if check_in_arg.missing or check_out_arg.missing:
        missing_labels.append("даты заезда и выезда")
        if check_in_arg.missing:
            missing_fields.append("check_in")
        if check_out_arg.missing:
            missing_fields.append("check_out")

    if not _positive_int_arg(tool_args.get("adults")):
        missing_labels.append("количество гостей")
        missing_fields.append("adults")

    if missing_labels:
        return _preflight_error_result(
            name,
            code="missing_hotel_search_fields",
            message="search_hotels requires location, check-in, check-out, and adult count",
            user_message=(
                f"Уточните {_format_russian_list(missing_labels)} для поиска отелей на Туту."
            ),
            invalid_fields=missing_fields,
        )

    if check_in_arg.invalid or check_out_arg.invalid:
        invalid_fields = [
            field for field in (check_in_arg.field, check_out_arg.field) if field is not None
        ]
        return _preflight_error_result(
            name,
            code="invalid_hotel_dates",
            message="check_in/check_out must be valid YYYY-MM-DD dates",
            user_message="Уточните даты заезда и выезда в формате ГГГГ-ММ-ДД.",
            invalid_fields=invalid_fields,
        )

    assert check_in_arg.value is not None
    assert check_out_arg.value is not None
    if check_in_arg.value < today or check_out_arg.value < today:
        return _preflight_error_result(
            name,
            code="hotel_date_in_past",
            message="check_in/check_out must not be in the past",
            user_message=(
                f"Даты {_format_date_range(check_in_arg.value, check_out_arg.value)} "
                "уже прошли. Уточните будущие даты или год поездки."
            ),
            invalid_fields=[check_in_arg.field or "check_in", check_out_arg.field or "check_out"],
        )

    if check_out_arg.value <= check_in_arg.value:
        return _preflight_error_result(
            name,
            code="hotel_checkout_not_after_checkin",
            message="check_out must be after check_in",
            user_message="Дата выезда должна быть позже даты заезда. Уточните даты проживания.",
            invalid_fields=[check_in_arg.field or "check_in", check_out_arg.field or "check_out"],
        )

    return None


def _validate_transport_search_tool_call(
    name: str,
    tool_args: Mapping[str, Any],
    today: date,
) -> dict[str, Any] | None:
    missing_labels: list[str] = []
    missing_fields: list[str] = []

    if not _any_arg_has_value(tool_args, ("origin", "from_city", "city_from")):
        missing_labels.append("пункт отправления")
        missing_fields.append("origin")
    if not _any_arg_has_value(tool_args, ("destination", "to_city", "city_to")):
        missing_labels.append("пункт назначения")
        missing_fields.append("destination")

    departure_date_arg = _date_arg(tool_args, ("departure_date", "date"))
    if departure_date_arg.missing:
        missing_labels.append("дату поездки")
        missing_fields.append("departure_date")

    if missing_labels:
        return _preflight_error_result(
            name,
            code="missing_transport_search_fields",
            message="transport search requires origin, destination, and departure date",
            user_message=(
                f"Уточните {_format_russian_list(missing_labels)} для поиска билетов на Туту."
            ),
            invalid_fields=missing_fields,
        )

    return_date_arg = _date_arg(tool_args, ("return_date",), required=False)
    invalid_fields = [
        field
        for field, invalid in (
            (departure_date_arg.field, departure_date_arg.invalid),
            (return_date_arg.field, return_date_arg.invalid),
        )
        if field is not None and invalid
    ]
    if invalid_fields:
        return _preflight_error_result(
            name,
            code="invalid_transport_dates",
            message="departure_date/return_date must be valid YYYY-MM-DD dates",
            user_message="Уточните дату поездки в формате ГГГГ-ММ-ДД.",
            invalid_fields=invalid_fields,
        )

    assert departure_date_arg.value is not None
    if departure_date_arg.value < today:
        return _preflight_error_result(
            name,
            code="transport_date_in_past",
            message="departure_date must not be in the past",
            user_message=(
                f"Дата поездки {departure_date_arg.value.isoformat()} уже прошла. "
                "Уточните будущую дату или год поездки."
            ),
            invalid_fields=[departure_date_arg.field or "departure_date"],
        )

    if return_date_arg.value is not None:
        if return_date_arg.value < today:
            return _preflight_error_result(
                name,
                code="transport_return_date_in_past",
                message="return_date must not be in the past",
                user_message=(
                    f"Дата обратной поездки {return_date_arg.value.isoformat()} уже прошла. "
                    "Уточните будущую дату или год поездки."
                ),
                invalid_fields=[return_date_arg.field or "return_date"],
            )
        if return_date_arg.value < departure_date_arg.value:
            return _preflight_error_result(
                name,
                code="transport_return_date_before_departure",
                message="return_date must not be before departure_date",
                user_message=(
                    "Дата обратной поездки должна быть не раньше даты отправления. "
                    "Уточните даты поездки."
                ),
                invalid_fields=[
                    departure_date_arg.field or "departure_date",
                    return_date_arg.field or "return_date",
                ],
            )

    return None


class _ParsedDateArg:
    __slots__ = ("field", "invalid", "missing", "value")

    def __init__(
        self,
        *,
        field: str | None,
        value: date | None,
        missing: bool,
        invalid: bool,
    ) -> None:
        self.field = field
        self.value = value
        self.missing = missing
        self.invalid = invalid


def _date_arg(
    tool_args: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    required: bool = True,
) -> _ParsedDateArg:
    for field in field_names:
        value = tool_args.get(field)
        if not _arg_has_value(value):
            continue
        if not isinstance(value, str):
            return _ParsedDateArg(field=field, value=None, missing=False, invalid=True)
        try:
            return _ParsedDateArg(
                field=field,
                value=date.fromisoformat(value),
                missing=False,
                invalid=False,
            )
        except ValueError:
            return _ParsedDateArg(field=field, value=None, missing=False, invalid=True)
    return _ParsedDateArg(field=None, value=None, missing=required, invalid=False)


def _preflight_error_result(
    tool_name: str,
    *,
    code: str,
    message: str,
    user_message: str,
    invalid_fields: list[str],
) -> dict[str, Any]:
    return {
        "ok": False,
        "message": user_message,
        "instruction": "Ask the user this question now. Do not call tools again.",
        "missing_fields": invalid_fields,
        "diagnostic": {
            "code": code,
            "message": message,
            "tool_name": tool_name,
        },
    }


def _any_arg_has_value(
    tool_args: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> bool:
    return any(_arg_has_value(tool_args.get(field)) for field in field_names)


def _arg_has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _positive_int_arg(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _format_russian_list(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} и {items[1]}"
    return f"{', '.join(items[:-1])} и {items[-1]}"


def _format_date_range(start: date, end: date) -> str:
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()} — {end.isoformat()}"
