import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from agent_service.config import AppSettings
from agent_service.mcp.toolsets import (
    PrefixedMCPToolsetConfig,
    TransformingToolset,
    build_prefixed_mcp_toolset,
    prefixed_tool_names,
)
from agent_service.mcp.vkusvill_compaction import (
    compact_vkusvill_product_analogs_result,
    compact_vkusvill_products_search_result,
)

VKUSVILL_MCP_SERVER_ID = "vkusvill"
VKUSVILL_SHOPPING_SKILL_ID = "vkusvill-shopping"
VKUSVILL_MCP_TOOL_PREFIX = "mcp_vkusvill"
VKUSVILL_PRODUCTS_SEARCH_RAW_TOOL_NAME = "vkusvill_products_search"
VKUSVILL_PRODUCTS_BATCH_SEARCH_RAW_TOOL_NAME = "vkusvill_products_batch_search"
VKUSVILL_PRODUCTS_SEARCH_TOOL_NAME = (
    f"{VKUSVILL_MCP_TOOL_PREFIX}_{VKUSVILL_PRODUCTS_SEARCH_RAW_TOOL_NAME}"
)
VKUSVILL_PRODUCTS_BATCH_SEARCH_TOOL_NAME = (
    f"{VKUSVILL_MCP_TOOL_PREFIX}_{VKUSVILL_PRODUCTS_BATCH_SEARCH_RAW_TOOL_NAME}"
)
VKUSVILL_MCP_RAW_TOOL_NAMES = frozenset(
    {
        VKUSVILL_PRODUCTS_SEARCH_RAW_TOOL_NAME,
        "vkusvill_product_details",
        "vkusvill_product_analogs",
        "vkusvill_cart_link_create",
    }
)
VKUSVILL_MCP_TOOL_NAMES = prefixed_tool_names(
    VKUSVILL_MCP_TOOL_PREFIX,
    VKUSVILL_MCP_RAW_TOOL_NAMES - {VKUSVILL_PRODUCTS_SEARCH_RAW_TOOL_NAME},
) | {VKUSVILL_PRODUCTS_BATCH_SEARCH_TOOL_NAME}
VKUSVILL_MCP_RESULT_TRANSFORMERS = {
    VKUSVILL_PRODUCTS_SEARCH_TOOL_NAME: compact_vkusvill_products_search_result,
    f"{VKUSVILL_MCP_TOOL_PREFIX}_vkusvill_product_analogs": (
        compact_vkusvill_product_analogs_result
    ),
}
VKUSVILL_PRODUCTS_BATCH_SEARCH_DESCRIPTION = (
    "Run multiple VkusVill product searches in one tool call. Use this for product "
    "discovery, including when there is only one item. Each item accepts the same "
    "search parameters as vkusvill_products_search plus item_id, which is echoed back "
    "with that item's compacted search result."
)
VkusVillSearchSort = Literal["price_asc", "price_desc", "rating", "popularity", "new"]


class VkusVillProductsBatchSearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(
        min_length=1,
        description=(
            "Stable caller-provided id for this requested product, such as meat, fish, "
            "bread, or item_1."
        ),
    )
    q: str | None = Field(default=None, description="Search query.")
    page: int | None = Field(default=None, ge=1, description="Search page number.")
    sort: VkusVillSearchSort | None = Field(
        default=None,
        description=("Sort order: price_asc, price_desc, rating, popularity, or new."),
    )


class VkusVillProductsBatchSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[VkusVillProductsBatchSearchItem] = Field(
        min_length=1,
        max_length=20,
        description="Product searches to run. Use one item for a single-product search.",
    )


@dataclass(slots=True)
class VkusVillBatchSearchToolset(WrapperToolset[Any]):
    batch_tool_name: str = VKUSVILL_PRODUCTS_BATCH_SEARCH_TOOL_NAME
    search_tool_name: str = VKUSVILL_PRODUCTS_SEARCH_TOOL_NAME
    args_validator: Any = field(
        default=VkusVillProductsBatchSearchArgs.__pydantic_validator__,
        repr=False,
    )
    parameters_json_schema: dict[str, Any] = field(
        default_factory=VkusVillProductsBatchSearchArgs.model_json_schema,
        repr=False,
    )

    async def get_tools(
        self,
        ctx: RunContext[Any],
    ) -> dict[str, ToolsetTool[Any]]:
        tools = dict(await self.wrapped.get_tools(ctx))
        search_tool = tools.pop(self.search_tool_name, None)
        if search_tool is None:
            return tools

        tools[self.batch_tool_name] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=self.batch_tool_name,
                description=VKUSVILL_PRODUCTS_BATCH_SEARCH_DESCRIPTION,
                parameters_json_schema=self.parameters_json_schema,
            ),
            max_retries=search_tool.max_retries,
            args_validator=self.args_validator,
        )
        return tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        if name != self.batch_tool_name:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        args = VkusVillProductsBatchSearchArgs.model_validate(tool_args)
        tools = await self.wrapped.get_tools(ctx)
        search_tool = tools[self.search_tool_name]

        results = await asyncio.gather(
            *(self._search_item(item, ctx, search_tool) for item in args.items)
        )
        return {"ok": True, "results": list(results)}

    async def _search_item(
        self,
        item: VkusVillProductsBatchSearchItem,
        ctx: RunContext[Any],
        search_tool: ToolsetTool[Any],
    ) -> dict[str, Any]:
        result = await self.wrapped.call_tool(
            self.search_tool_name,
            _search_item_args(item),
            ctx,
            search_tool,
        )
        return {
            "item_id": item.item_id,
            "result": _structured_compact_products_search_result(result),
        }


def _search_item_args(item: VkusVillProductsBatchSearchItem) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.model_dump(exclude={"item_id"}).items()
        if value is not None
    }


def _structured_compact_products_search_result(value: Any) -> Any:
    compacted = compact_vkusvill_products_search_result(value)
    if not isinstance(compacted, str):
        return compacted

    try:
        return json.loads(compacted)
    except (TypeError, ValueError):
        return compacted


def build_vkusvill_mcp_toolsets(settings: AppSettings) -> tuple[AbstractToolset[Any], ...]:
    if settings.vkusvill_mcp_command is None and settings.vkusvill_mcp_url is None:
        return ()

    toolset = build_prefixed_mcp_toolset(
        PrefixedMCPToolsetConfig(
            server_id=VKUSVILL_MCP_SERVER_ID,
            prefix=VKUSVILL_MCP_TOOL_PREFIX,
            command=settings.vkusvill_mcp_command,
            args=settings.vkusvill_mcp_args,
            env=settings.vkusvill_mcp_env,
            url=settings.vkusvill_mcp_url,
            headers=settings.vkusvill_mcp_headers,
            init_timeout_seconds=settings.vkusvill_mcp_init_timeout_seconds,
            read_timeout_seconds=settings.vkusvill_mcp_read_timeout_seconds,
            allowed_raw_tool_names=VKUSVILL_MCP_RAW_TOOL_NAMES,
        )
    )
    transformed = TransformingToolset(
        toolset,
        VKUSVILL_MCP_RESULT_TRANSFORMERS,
    )
    return (
        VkusVillBatchSearchToolset(
            transformed,
        ),
    )
