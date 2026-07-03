from datetime import date
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FilteredToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage

from agent_service.config import AppSettings
from agent_service.mcp import (
    TUTU_MCP_RAW_TOOL_NAMES,
    TUTU_MCP_RESULT_TRANSFORMERS,
    TUTU_MCP_TOOL_NAMES,
    TUTU_MCP_TOOL_PREFIX,
    TransformingToolset,
    build_tutu_mcp_toolsets,
)
from agent_service.mcp.tutu import (
    TUTU_SEARCH_HOTELS_TOOL_NAME,
    TUTU_TRANSPORT_SEARCH_TOOL_NAMES,
    validate_tutu_mcp_tool_call,
)


def _run_context() -> RunContext[Any]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_tutu_mcp_toolsets_are_disabled_without_transport_config() -> None:
    assert build_tutu_mcp_toolsets(AppSettings(environment="test")) == ()


def test_tutu_mcp_toolset_uses_skill_tool_names_as_allowlist() -> None:
    assert TUTU_MCP_RAW_TOOL_NAMES == {
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
    assert TUTU_MCP_TOOL_NAMES == {
        "mcp_tutu_search_hotels",
        "mcp_tutu_search_avia",
        "mcp_tutu_search_rail",
        "mcp_tutu_search_bus",
        "mcp_tutu_search_etrain",
        "mcp_tutu_search_multitransport",
        "mcp_tutu_get_offer_details",
        "mcp_tutu_get_rail_seatmap",
        "mcp_tutu_create_checkout_link",
    }


def test_tutu_mcp_tool_filter_allows_only_wrapped_skill_tools() -> None:
    toolset = build_tutu_mcp_toolsets(
        AppSettings(environment="test", tutu_mcp_url="https://mcp.tutu.ru/mcp")
    )[0]
    assert isinstance(toolset, TransformingToolset)
    assert toolset.return_error_results_for_tool_names == TUTU_MCP_TOOL_NAMES
    assert toolset.log_error_args_for_tool_names == TUTU_MCP_TOOL_NAMES
    assert len(toolset.pre_call_validators) == 1
    filtered = toolset.wrapped
    assert isinstance(filtered, FilteredToolset)

    allowed = ToolDefinition(name="mcp_tutu_search_hotels")
    raw = ToolDefinition(name="search_hotels")
    unrelated = ToolDefinition(name="mcp_tutu_get_personal_context")
    ctx = _run_context()

    assert filtered.filter_func(ctx, allowed) is True
    assert filtered.filter_func(ctx, raw) is False
    assert filtered.filter_func(ctx, unrelated) is False


def test_tutu_mcp_url_toolset_is_prefixed_then_filtered() -> None:
    toolsets = build_tutu_mcp_toolsets(
        AppSettings(
            environment="test",
            tutu_mcp_url="https://mcp.tutu.ru/mcp",
        )
    )

    assert len(toolsets) == 1
    transforming = toolsets[0]
    assert isinstance(transforming, TransformingToolset)
    filtered = transforming.wrapped
    assert isinstance(filtered, FilteredToolset)
    prefixed = filtered.wrapped
    assert isinstance(prefixed, PrefixedToolset)
    assert prefixed.prefix == TUTU_MCP_TOOL_PREFIX


def test_tutu_mcp_stdio_toolset_is_prefixed_then_filtered() -> None:
    toolsets = build_tutu_mcp_toolsets(
        AppSettings(
            environment="test",
            tutu_mcp_command="uvx",
            tutu_mcp_args=("tutu-mcp",),
        )
    )

    assert len(toolsets) == 1
    transforming = toolsets[0]
    assert isinstance(transforming, TransformingToolset)
    filtered = transforming.wrapped
    assert isinstance(filtered, FilteredToolset)
    prefixed = filtered.wrapped
    assert isinstance(prefixed, PrefixedToolset)
    assert prefixed.prefix == TUTU_MCP_TOOL_PREFIX


def test_tutu_result_transformers_are_empty_for_initial_integration() -> None:
    assert TUTU_MCP_RESULT_TRANSFORMERS == {}


def test_tutu_hotel_preflight_requires_core_search_fields() -> None:
    result = validate_tutu_mcp_tool_call(
        TUTU_SEARCH_HOTELS_TOOL_NAME,
        {
            "check_in": None,
            "check_out": None,
            "city_name": None,
            "geo_id": None,
            "adults": None,
        },
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert result["error"]["code"] == "missing_hotel_search_fields"
    assert result["error"]["next_action"] == "ask_user"
    assert result["error"]["retryable"] is False
    assert "город или локацию" in result["error"]["user_message"]
    assert "даты заезда и выезда" in result["error"]["user_message"]
    assert "количество гостей" in result["error"]["user_message"]


def test_tutu_hotel_preflight_rejects_past_dates() -> None:
    result = validate_tutu_mcp_tool_call(
        TUTU_SEARCH_HOTELS_TOOL_NAME,
        {
            "check_in": "2026-06-10",
            "check_out": "2026-06-17",
            "city_name": "Санкт-Петербург",
            "adults": 2,
        },
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert result["error"]["code"] == "hotel_date_in_past"
    assert "2026-06-10 — 2026-06-17" in result["error"]["user_message"]


def test_tutu_hotel_preflight_accepts_valid_future_dates() -> None:
    result = validate_tutu_mcp_tool_call(
        TUTU_SEARCH_HOTELS_TOOL_NAME,
        {
            "check_in": "2026-07-10",
            "check_out": "2026-07-17",
            "city_name": "Санкт-Петербург",
            "adults": 2,
        },
        today=date(2026, 7, 3),
    )

    assert result is None


def test_tutu_transport_preflight_requires_route_and_date() -> None:
    tool_name = next(iter(TUTU_TRANSPORT_SEARCH_TOOL_NAMES))
    result = validate_tutu_mcp_tool_call(
        tool_name,
        {"origin": None, "destination": None, "departure_date": None},
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert result["error"]["code"] == "missing_transport_search_fields"
    assert "пункт отправления" in result["error"]["user_message"]
    assert "пункт назначения" in result["error"]["user_message"]
    assert "дату поездки" in result["error"]["user_message"]


async def test_tutu_toolset_preflight_returns_error_without_remote_mcp_call() -> None:
    toolset = build_tutu_mcp_toolsets(
        AppSettings(environment="test", tutu_mcp_url="https://mcp.tutu.ru/mcp"),
        today_provider=lambda: date(2026, 7, 3),
    )[0]

    result = await toolset.call_tool(
        TUTU_SEARCH_HOTELS_TOOL_NAME,
        {
            "check_in": None,
            "check_out": None,
            "city_name": None,
            "geo_id": None,
            "adults": None,
        },
        _run_context(),
        cast(Any, object()),
    )

    assert result["error"]["code"] == "missing_hotel_search_fields"
