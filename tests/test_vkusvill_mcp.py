import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage

from agent_service.config import AppSettings
from agent_service.mcp import (
    VKUSVILL_MCP_RAW_TOOL_NAMES,
    VKUSVILL_MCP_RESULT_TRANSFORMERS,
    VKUSVILL_MCP_TOOL_NAMES,
    VKUSVILL_MCP_TOOL_PREFIX,
    PrefixedMCPToolsetConfig,
    TransformingToolset,
    build_prefixed_mcp_toolset,
    build_vkusvill_mcp_toolsets,
    prefixed_tool_names,
)
from agent_service.mcp.vkusvill_compaction import (
    compact_vkusvill_product_analogs_result,
    compact_vkusvill_products_search_result,
)


def _run_context() -> RunContext[Any]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


@dataclass
class FakeToolset(AbstractToolset[Any]):
    result: Any
    calls: int = 0

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
        return self.result


@dataclass
class FailingToolset(AbstractToolset[Any]):
    error: Exception

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
        raise self.error


def test_vkusvill_mcp_toolsets_are_disabled_without_transport_config() -> None:
    assert build_vkusvill_mcp_toolsets(AppSettings(environment="test")) == ()


def test_vkusvill_mcp_toolset_uses_skill_tool_names_as_allowlist() -> None:
    assert VKUSVILL_MCP_RAW_TOOL_NAMES == {
        "vkusvill_products_search",
        "vkusvill_product_details",
        "vkusvill_product_analogs",
        "vkusvill_cart_link_create",
    }
    assert VKUSVILL_MCP_TOOL_NAMES == {
        "mcp_vkusvill_vkusvill_products_search",
        "mcp_vkusvill_vkusvill_product_details",
        "mcp_vkusvill_vkusvill_product_analogs",
        "mcp_vkusvill_vkusvill_cart_link_create",
    }


def test_prefixed_tool_names_builds_wrapped_mcp_names() -> None:
    assert prefixed_tool_names("mcp_demo", {"search", "details"}) == {
        "mcp_demo_search",
        "mcp_demo_details",
    }


def test_prefixed_mcp_toolset_filters_allowlisted_tools() -> None:
    toolset = build_prefixed_mcp_toolset(
        PrefixedMCPToolsetConfig(
            server_id="demo",
            prefix="mcp_demo",
            url="http://localhost:8765/mcp",
            allowed_raw_tool_names={"search"},
        )
    )

    assert isinstance(toolset, FilteredToolset)
    ctx = _run_context()
    assert toolset.filter_func(ctx, ToolDefinition(name="mcp_demo_search")) is True
    assert toolset.filter_func(ctx, ToolDefinition(name="search")) is False
    assert toolset.filter_func(ctx, ToolDefinition(name="mcp_demo_other")) is False


def test_vkusvill_mcp_tool_filter_allows_only_wrapped_skill_tools() -> None:
    toolset = build_vkusvill_mcp_toolsets(
        AppSettings(environment="test", vkusvill_mcp_url="http://localhost:8765/mcp")
    )[0]
    assert isinstance(toolset, TransformingToolset)
    filtered = toolset.wrapped
    assert isinstance(filtered, FilteredToolset)

    allowed = ToolDefinition(name="mcp_vkusvill_vkusvill_products_search")
    raw = ToolDefinition(name="vkusvill_products_search")
    unrelated = ToolDefinition(name="mcp_vkusvill_vkusvill_profile")
    ctx = _run_context()

    assert filtered.filter_func(ctx, allowed) is True
    assert filtered.filter_func(ctx, raw) is False
    assert filtered.filter_func(ctx, unrelated) is False


def test_vkusvill_mcp_url_toolset_is_prefixed_then_filtered() -> None:
    toolsets = build_vkusvill_mcp_toolsets(
        AppSettings(
            environment="test",
            vkusvill_mcp_url="http://localhost:8765/mcp",
        )
    )

    assert len(toolsets) == 1
    transforming = toolsets[0]
    assert isinstance(transforming, TransformingToolset)
    filtered = transforming.wrapped
    assert isinstance(filtered, FilteredToolset)
    prefixed = filtered.wrapped
    assert isinstance(prefixed, PrefixedToolset)
    assert prefixed.prefix == VKUSVILL_MCP_TOOL_PREFIX


def test_vkusvill_mcp_stdio_toolset_is_prefixed_then_filtered() -> None:
    toolsets = build_vkusvill_mcp_toolsets(
        AppSettings(
            environment="test",
            vkusvill_mcp_command="uvx",
            vkusvill_mcp_args=("vkusvill-mcp",),
        )
    )

    assert len(toolsets) == 1
    transforming = toolsets[0]
    assert isinstance(transforming, TransformingToolset)
    filtered = transforming.wrapped
    assert isinstance(filtered, FilteredToolset)
    prefixed = filtered.wrapped
    assert isinstance(prefixed, PrefixedToolset)
    assert prefixed.prefix == VKUSVILL_MCP_TOOL_PREFIX


async def test_transforming_toolset_applies_named_result_transformers() -> None:
    toolset = TransformingToolset(
        FakeToolset("raw-result"),
        {"demo_tool": lambda value: f"{value}-compact"},
    )

    result = await toolset.call_tool(
        "demo_tool",
        {},
        _run_context(),
        cast(Any, object()),
    )

    assert result == "raw-result-compact"


async def test_transforming_toolset_leaves_unmapped_tools_unchanged() -> None:
    toolset = TransformingToolset(
        FakeToolset("raw-result"),
        {"demo_tool": lambda value: f"{value}-compact"},
    )

    result = await toolset.call_tool(
        "other_tool",
        {},
        _run_context(),
        cast(Any, object()),
    )

    assert result == "raw-result"


async def test_transforming_toolset_returns_preflight_result_without_calling_wrapped() -> None:
    wrapped = FakeToolset("raw-result")
    preflight_result = {
        "ok": False,
        "error": {
            "code": "missing_required_fields",
            "user_message": "Уточните параметры.",
        },
    }

    def preflight_validator(
        _name: str,
        _args: Mapping[str, Any],
        _ctx: RunContext[Any],
    ) -> dict[str, Any]:
        return preflight_result

    toolset = TransformingToolset(
        wrapped,
        pre_call_validators=(preflight_validator,),
    )

    result = await toolset.call_tool(
        "demo_tool",
        {},
        _run_context(),
        cast(Any, object()),
    )

    assert result == preflight_result
    assert wrapped.calls == 0


async def test_transforming_toolset_raises_tool_errors_by_default() -> None:
    toolset = TransformingToolset(FailingToolset(RuntimeError("upstream failed")))

    try:
        await toolset.call_tool(
            "demo_tool",
            {},
            _run_context(),
            cast(Any, object()),
        )
    except RuntimeError as exc:
        assert str(exc) == "upstream failed"
    else:  # pragma: no cover
        raise AssertionError("expected tool error to be raised")


async def test_transforming_toolset_can_return_tool_errors_as_results() -> None:
    toolset = TransformingToolset(
        FailingToolset(RuntimeError("upstream failed")),
        return_error_results_for_tool_names={"demo_tool"},
    )

    result = await toolset.call_tool(
        "demo_tool",
        {},
        _run_context(),
        cast(Any, object()),
    )

    assert result == {
        "ok": False,
        "error": {
            "tool_name": "demo_tool",
            "type": "RuntimeError",
            "message": "upstream failed",
            "hint": (
                "The upstream MCP service rejected the call. Do not retry the same "
                "parameters; ask the user for the smallest useful correction."
            ),
        },
    }


async def test_transforming_toolset_logs_error_args_for_configured_tools(
    caplog: Any,
) -> None:
    caplog.set_level(logging.WARNING, logger="agent_service.mcp.toolsets")
    toolset = TransformingToolset(
        FailingToolset(RuntimeError("upstream failed")),
        return_error_results_for_tool_names={"demo_tool"},
        log_error_args_for_tool_names={"demo_tool"},
    )

    await toolset.call_tool(
        "demo_tool",
        {"city_name": "Санкт-Петербург", "check_in": "2026-06-10"},
        _run_context(),
        cast(Any, object()),
    )

    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "mcp_tool_call_failed"
    )
    assert failure_record.tool_args_json == (
        '{"city_name":"Санкт-Петербург","check_in":"2026-06-10"}'
    )


def test_vkusvill_result_transformers_match_nanobot_scope() -> None:
    assert VKUSVILL_MCP_RESULT_TRANSFORMERS.keys() == {
        "mcp_vkusvill_vkusvill_products_search",
        "mcp_vkusvill_vkusvill_product_analogs",
    }


def test_vkusvill_products_search_compaction_keeps_only_model_relevant_fields() -> None:
    raw_payload = {
        "ok": True,
        "debug": "drop",
        "data": {
            "meta": {"query": "сок", "total": 2},
            "items": [
                {
                    "id": 10,
                    "xml_id": "100",
                    "name": "Апельсиновый сок",
                    "description": "1 л",
                    "price": {"current": 150, "old": None, "discount_percent": 0},
                    "unit": "шт",
                    "weight": {"value": 1, "unit": "л", "extra": "drop"},
                    "rating": {"average": 4.8, "count": 42, "extra": "drop"},
                    "pictures": ["drop"],
                    "properties": [
                        {"name": "Состав", "value": "апельсины"},
                        {
                            "name": "Пищевая и энергетическая ценность в 100 г",
                            "value": "45 ккал",
                        },
                        {"name": "Срок годности", "value": "drop"},
                    ],
                },
                {"id": 11, "name": "Без xml_id"},
            ],
        },
    }

    compacted = compact_vkusvill_products_search_result(
        json.dumps(raw_payload, ensure_ascii=False)
    )

    assert isinstance(compacted, str)
    payload = json.loads(compacted)
    assert payload == {
        "ok": True,
        "data": {
            "meta": {"query": "сок", "total": 2},
            "items": [
                {
                    "id": 10,
                    "xml_id": "100",
                    "name": "Апельсиновый сок",
                    "price": {"current": 150, "discount_percent": 0},
                    "unit": "шт",
                    "weight": {"value": 1, "unit": "л"},
                    "properties": [
                        {"name": "Состав", "value": "апельсины"},
                        {
                            "name": "Пищевая и энергетическая ценность в 100 г",
                            "value": "45 ккал",
                        },
                    ],
                }
            ],
        },
    }


def test_vkusvill_product_analogs_compaction_accepts_structured_payload() -> None:
    raw_payload = {
        "ok": True,
        "data": {
            "product_id": "100",
            "total": 1,
            "products": [
                {
                    "id": 20,
                    "xml_id": "200",
                    "name": "Похожий сок",
                    "description": "",
                    "price": {"current": 120, "old": 140, "discount_percent": 14},
                    "unit": "шт",
                    "weight": {"value": None, "unit": None},
                    "rating": {"average": None, "count": None},
                    "properties": [{"name": "Бренд", "value": "drop"}],
                    "images": ["drop"],
                }
            ],
        },
    }

    assert compact_vkusvill_product_analogs_result(raw_payload) == {
        "ok": True,
        "data": {
            "product_id": "100",
            "total": 1,
            "products": [
                {
                    "id": 20,
                    "xml_id": "200",
                    "name": "Похожий сок",
                    "price": {"current": 120, "old": 140, "discount_percent": 14},
                    "unit": "шт",
                }
            ],
        },
    }


def test_vkusvill_compaction_drops_description_rating_and_empty_properties() -> None:
    payload = {
        "ok": True,
        "data": {
            "meta": {"query": "чипсы"},
            "items": [
                {
                    "id": 10,
                    "xml_id": "100",
                    "name": "Чипсы",
                    "description": "long marketing text",
                    "price": {"current": 150},
                    "rating": {"average": 4.8, "count": 42},
                    "properties": [
                        {"name": "Состав", "value": None},
                        {
                            "name": "Пищевая и энергетическая ценность в 100 г",
                            "value": "",
                        },
                    ],
                }
            ],
        },
    }

    compacted = compact_vkusvill_products_search_result(payload)

    assert compacted == {
        "ok": True,
        "data": {
            "meta": {"query": "чипсы"},
            "items": [
                {
                    "id": 10,
                    "xml_id": "100",
                    "name": "Чипсы",
                    "price": {"current": 150},
                }
            ],
        },
    }


def test_vkusvill_compaction_returns_original_for_unexpected_payloads() -> None:
    invalid_json = "{not json"
    unexpected_payload = {"ok": True, "data": {"items": []}}

    assert compact_vkusvill_products_search_result(invalid_json) == invalid_json
    assert compact_vkusvill_products_search_result(unexpected_payload) is unexpected_payload
