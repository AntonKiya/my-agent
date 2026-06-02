from typing import Any

from pydantic_ai.toolsets import AbstractToolset

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
VKUSVILL_MCP_RAW_TOOL_NAMES = frozenset(
    {
        "vkusvill_products_search",
        "vkusvill_product_details",
        "vkusvill_product_analogs",
        "vkusvill_cart_link_create",
    }
)
VKUSVILL_MCP_TOOL_NAMES = prefixed_tool_names(
    VKUSVILL_MCP_TOOL_PREFIX,
    VKUSVILL_MCP_RAW_TOOL_NAMES,
)
VKUSVILL_MCP_RESULT_TRANSFORMERS = {
    f"{VKUSVILL_MCP_TOOL_PREFIX}_vkusvill_products_search": (
        compact_vkusvill_products_search_result
    ),
    f"{VKUSVILL_MCP_TOOL_PREFIX}_vkusvill_product_analogs": (
        compact_vkusvill_product_analogs_result
    ),
}


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
    return (
        TransformingToolset(
            toolset,
            VKUSVILL_MCP_RESULT_TRANSFORMERS,
        ),
    )
