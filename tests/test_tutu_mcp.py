import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, PrefixedToolset
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
    TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME,
    TUTU_SEARCH_HOTELS_TOOL_NAME,
    TUTU_SEARCH_RESULT_TOOL_NAMES,
    TUTU_TRANSPORT_SEARCH_TOOL_NAMES,
    validate_tutu_mcp_tool_call,
)
from agent_service.mcp.tutu_refs import (
    build_tutu_checkout_link_call_transformer,
    build_tutu_search_result_transformer,
)
from agent_service.tool_refs import InMemoryToolResultReferenceStore, ToolResultReference


def _run_context(deps: dict[str, Any] | None = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


@dataclass
class FakeToolset(AbstractToolset[Any]):
    result: Any
    calls: int = 0
    last_args: dict[str, Any] | None = None

    @property
    def id(self) -> str | None:
        return None

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        return {}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: Any,
    ) -> Any:
        self.calls += 1
        self.last_args = dict(tool_args)
        return self.result


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


def test_tutu_result_transformers_keep_sync_transformer_slot_empty() -> None:
    assert TUTU_MCP_RESULT_TRANSFORMERS == {}


def test_tutu_search_result_transformer_scope_matches_search_tools() -> None:
    toolset = build_tutu_mcp_toolsets(
        AppSettings(environment="test", tutu_mcp_url="https://mcp.tutu.ru/mcp")
    )[0]

    assert isinstance(toolset, TransformingToolset)
    assert toolset.contextual_result_transformers.keys() == TUTU_SEARCH_RESULT_TOOL_NAMES
    assert toolset.call_transformers.keys() == {TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME}
    assert toolset.tool_definition_transformers.keys() == {TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME}


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


async def test_tutu_search_result_masking_persists_checkout_ref_and_removes_raw_handles() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    store = InMemoryToolResultReferenceStore(
        now_provider=lambda: datetime(2026, 7, 5, 11, 0, tzinfo=UTC)
    )
    raw_payload: dict[str, Any] = {
        "meta": {
            "transport": "avia",
            "total_returned": 1,
            "from": {"name": "Москва", "geo_id": "2657260"},
            "to": {"name": "Сочи", "geo_id": "2656918"},
        },
        "offers": [
            {
                "transport": "avia",
                "price": {"amount": 26140.0, "currency": "RUB"},
                "departure_at": "2026-07-15T19:30:00+03:00",
                "arrival_at": "2026-07-15T23:10:00+03:00",
                "duration_min": 220,
                "checkout_ref": {
                    "transport": "avia",
                    "offer_hash": {"opaque": "hash"},
                    "departure_at": "2026-07-15T19:30:00+03:00",
                },
                "details_ref": {"opaque": "details"},
                "checkout_url": "https://drop.example/checkout",
                "search_results_url": "https://drop.example/search",
                "legs": [
                    {
                        "from": "Москва — Шереметьево (SVO)",
                        "to": "Сочи, AER",
                        "departure_at": "2026-07-15T19:30:00+03:00",
                        "arrival_at": "2026-07-15T23:10:00+03:00",
                        "duration_min": 220,
                        "segments": [
                            {
                                "from": "Москва — Шереметьево (SVO)",
                                "to": "Сочи, AER",
                                "carrier": "Победа",
                                "departure_at": "2026-07-15T19:30:00+03:00",
                                "arrival_at": "2026-07-15T23:10:00+03:00",
                                "duration_min": 220,
                            }
                        ],
                    }
                ],
                "variants": [
                    {
                        "price": {"amount": 26140.0, "currency": "RUB"},
                        "offer_hash": {"raw": "variant_hash"},
                        "service_class": "ECONOMIC",
                        "conditions": {
                            "fare_family": "Базовый",
                            "refundable": False,
                            "baggage": {"pieces": 0, "kg": 0},
                        },
                    }
                ],
            }
        ],
    }
    toolset = TransformingToolset(
        FakeToolset(raw_payload),
        contextual_result_transformers={
            "mcp_tutu_search_avia": build_tutu_search_result_transformer(
                store,
                selection_id_provider=lambda: "sel_testavia0001",
                now_provider=lambda: datetime(2026, 7, 5, 11, 0, tzinfo=UTC),
            )
        },
    )

    result = await toolset.call_tool(
        "mcp_tutu_search_avia",
        {},
        _run_context({"user_id": user_id, "conversation_id": conversation_id}),
        cast(Any, object()),
    )

    offer = result["offers"][0]
    assert offer["selection_id"] == "sel_testavia0001"
    compact_json = json.dumps(result, ensure_ascii=False)
    assert "checkout_ref" not in compact_json
    assert "offer_hash" not in compact_json
    assert "checkout_url" not in compact_json
    assert "https://drop.example" not in compact_json
    assert offer["fare_options"] == [
        {
            "price": {"amount": 26140.0, "currency": "RUB"},
            "service_class": "ECONOMIC",
            "fare_family": "Базовый",
            "refundable": False,
            "baggage": {"pieces": 0, "kg": 0},
        }
    ]

    reference = await store.get(
        selection_id="sel_testavia0001",
        user_id=user_id,
        conversation_id=conversation_id,
        provider="tutu",
    )
    assert reference is not None
    assert reference.ref_payload["checkout_ref"] == raw_payload["offers"][0]["checkout_ref"]
    assert reference.ref_payload["details_ref"] == {"opaque": "details"}
    assert reference.display_snapshot["selection_id"] == "sel_testavia0001"


async def test_tutu_checkout_link_transformer_resolves_selection_id_to_checkout_ref() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    store = InMemoryToolResultReferenceStore(
        now_provider=lambda: datetime(2026, 7, 5, 11, 0, tzinfo=UTC)
    )
    checkout_ref = {
        "transport": "avia",
        "offer_hash": {"opaque": "hash"},
        "departure_geo_city_id": 2657260,
        "arrival_geo_city_id": 2656918,
        "departure_at": "2026-07-15T19:30:00+03:00",
        "service_class": "ECONOMIC",
    }
    await store.create(
        reference=ToolResultReference(
            selection_id="sel_checkout0001",
            provider="tutu",
            source_tool_name="mcp_tutu_search_avia",
            user_id=user_id,
            conversation_id=conversation_id,
            item_kind="avia",
            item_index=0,
            label="Москва -> Сочи",
            display_snapshot={"selection_id": "sel_checkout0001"},
            ref_payload={"checkout_ref": checkout_ref},
            expires_at=datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
            created_at=datetime(2026, 7, 5, 10, 0, tzinfo=UTC),
        )
    )
    wrapped = FakeToolset({"ok": True})
    toolset = TransformingToolset(
        wrapped,
        call_transformers={
            TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME: build_tutu_checkout_link_call_transformer(store)
        },
    )

    await toolset.call_tool(
        TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME,
        {"selection_id": "sel_checkout0001", "offer_hash": {"wrong": "ignored"}},
        _run_context({"user_id": user_id, "conversation_id": conversation_id}),
        cast(Any, object()),
    )

    assert wrapped.calls == 1
    assert wrapped.last_args == checkout_ref


async def test_tutu_checkout_link_transformer_rejects_expired_selection_id() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    store = InMemoryToolResultReferenceStore(
        now_provider=lambda: datetime(2026, 7, 5, 13, 0, tzinfo=UTC)
    )
    await store.create(
        reference=ToolResultReference(
            selection_id="sel_expired0001",
            provider="tutu",
            source_tool_name="mcp_tutu_search_avia",
            user_id=user_id,
            conversation_id=conversation_id,
            item_kind="avia",
            item_index=0,
            label=None,
            display_snapshot={"selection_id": "sel_expired0001"},
            ref_payload={"checkout_ref": {"transport": "avia"}},
            expires_at=datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
            created_at=datetime(2026, 7, 5, 10, 0, tzinfo=UTC),
        )
    )
    wrapped = FakeToolset({"ok": True})
    toolset = TransformingToolset(
        wrapped,
        call_transformers={
            TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME: build_tutu_checkout_link_call_transformer(store)
        },
    )

    result = await toolset.call_tool(
        TUTU_CREATE_CHECKOUT_LINK_TOOL_NAME,
        {"selection_id": "sel_expired0001"},
        _run_context({"user_id": user_id, "conversation_id": conversation_id}),
        cast(Any, object()),
    )

    assert wrapped.calls == 0
    assert result["error"]["code"] == "selection_not_found_or_expired"
