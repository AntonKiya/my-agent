from typing import Any

from pydantic_ai.toolsets import AbstractToolset

from agent_service.config import AppSettings
from agent_service.mcp.toolsets import (
    PrefixedMCPToolsetConfig,
    TransformingToolset,
    build_prefixed_mcp_toolset,
    prefixed_tool_names,
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
TUTU_MCP_RESULT_TRANSFORMERS = {}


def build_tutu_mcp_toolsets(settings: AppSettings) -> tuple[AbstractToolset[Any], ...]:
    if settings.tutu_mcp_command is None and settings.tutu_mcp_url is None:
        return ()

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
            return_error_results_for_tool_names=TUTU_MCP_TOOL_NAMES,
            log_error_args_for_tool_names=TUTU_MCP_TOOL_NAMES,
        ),
    )
