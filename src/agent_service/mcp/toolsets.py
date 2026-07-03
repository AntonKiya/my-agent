import json
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastmcp.client.transports import StdioTransport
from pydantic_ai import RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)
ToolResultTransformer = Callable[[Any], Any]
ToolCallValidator = Callable[[str, Mapping[str, Any], RunContext[Any]], Any | None]
MAX_LOG_ERROR_MESSAGE_CHARS = 500
MAX_LOG_TOOL_ARGS_CHARS = 2000


@dataclass(frozen=True, slots=True)
class PrefixedMCPToolsetConfig:
    server_id: str
    prefix: str
    command: str | None = None
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    init_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 300.0
    allowed_raw_tool_names: Collection[str] | None = None


def build_prefixed_mcp_toolset(config: PrefixedMCPToolsetConfig) -> AbstractToolset[Any]:
    if config.command is not None and config.url is not None:
        raise ValueError("configure either command or url, not both")
    if config.command is None and config.url is None:
        raise ValueError("configure command or url")

    if config.command is not None:
        base_toolset = MCPToolset(
            StdioTransport(
                command=config.command,
                args=list(config.args),
                env=dict(config.env) or None,
            ),
            id=config.server_id,
            init_timeout=config.init_timeout_seconds,
            read_timeout=config.read_timeout_seconds,
        )
    else:
        assert config.url is not None
        base_toolset = MCPToolset(
            config.url,
            id=config.server_id,
            headers=dict(config.headers) or None,
            init_timeout=config.init_timeout_seconds,
            read_timeout=config.read_timeout_seconds,
        )

    prefixed = base_toolset.prefixed(config.prefix)
    if config.allowed_raw_tool_names is None:
        return prefixed

    allowed_tool_names = prefixed_tool_names(config.prefix, config.allowed_raw_tool_names)
    return prefixed.filtered(_allow_tool_names(allowed_tool_names))


@dataclass(slots=True)
class TransformingToolset(WrapperToolset[Any]):
    result_transformers: Mapping[str, ToolResultTransformer] = field(default_factory=dict)
    return_error_results_for_tool_names: Collection[str] = field(default_factory=frozenset)
    log_error_args_for_tool_names: Collection[str] = field(default_factory=frozenset)
    pre_call_validators: Sequence[ToolCallValidator] = field(default_factory=tuple)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        started_at = start_timer()
        tool_args_size = _payload_size(tool_args)
        log_event(
            logger,
            logging.INFO,
            "MCP tool call started",
            event="mcp_tool_call_started",
            tool_name=name,
            tool_args_keys=sorted(tool_args),
            tool_args_size=tool_args_size,
        )

        preflight_result = self._preflight_result(name, tool_args, ctx)
        if preflight_result is not None:
            log_event(
                logger,
                logging.INFO,
                "MCP tool call rejected by preflight validator",
                event="mcp_tool_call_preflight_rejected",
                tool_name=name,
                tool_args_keys=sorted(tool_args),
                tool_args_size=tool_args_size,
                duration_ms=elapsed_ms(started_at),
                result_size=_payload_size(preflight_result),
            )
            return preflight_result

        try:
            result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
        except Exception as exc:
            failure_fields = {
                "event": "mcp_tool_call_failed",
                "tool_name": name,
                "tool_args_keys": sorted(tool_args),
                "tool_args_size": tool_args_size,
                "duration_ms": elapsed_ms(started_at),
                "error_type": type(exc).__name__,
                "error_message": _safe_error_message(exc),
            }
            if name in self.log_error_args_for_tool_names:
                failure_fields["tool_args_json"] = _safe_log_payload(tool_args)
            log_event(
                logger,
                logging.WARNING,
                "MCP tool call failed",
                **failure_fields,
            )
            if name in self.return_error_results_for_tool_names:
                error_result = _tool_error_result(name, exc)
                log_event(
                    logger,
                    logging.INFO,
                    "MCP tool error returned as result",
                    event="mcp_tool_error_returned_as_result",
                    tool_name=name,
                    duration_ms=elapsed_ms(started_at),
                    result_size=_payload_size(error_result),
                )
                return error_result
            raise

        transformer = self.result_transformers.get(name)
        original_size = _payload_size(result)
        if transformer is None:
            log_event(
                logger,
                logging.INFO,
                "MCP tool call completed",
                event="mcp_tool_call_completed",
                tool_name=name,
                duration_ms=elapsed_ms(started_at),
                original_size=original_size,
                transformed_size=original_size,
                transformed=False,
            )
            return result

        try:
            transformed = transformer(result)
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "MCP tool result transform failed",
                event="mcp_tool_result_transform_failed",
                tool_name=name,
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
                original_size=original_size,
            )
            log_event(
                logger,
                logging.INFO,
                "MCP tool call completed",
                event="mcp_tool_call_completed",
                tool_name=name,
                duration_ms=elapsed_ms(started_at),
                original_size=original_size,
                transformed_size=original_size,
                transformed=False,
            )
            return result

        transformed_size = _payload_size(transformed)
        log_event(
            logger,
            logging.INFO,
            "MCP tool result transformed",
            event="mcp_tool_result_transformed",
            tool_name=name,
            original_size=original_size,
            transformed_size=transformed_size,
            transformed=transformed != result,
        )
        log_event(
            logger,
            logging.INFO,
            "MCP tool call completed",
            event="mcp_tool_call_completed",
            tool_name=name,
            duration_ms=elapsed_ms(started_at),
            original_size=original_size,
            transformed_size=transformed_size,
            transformed=transformed != result,
        )
        return transformed

    def _preflight_result(
        self,
        name: str,
        tool_args: Mapping[str, Any],
        ctx: RunContext[Any],
    ) -> Any | None:
        for validator in self.pre_call_validators:
            try:
                result = validator(name, tool_args, ctx)
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "MCP tool preflight validator failed",
                    event="mcp_tool_preflight_validator_failed",
                    tool_name=name,
                    error_type=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                )
                continue
            if result is not None:
                return result
        return None


def prefixed_tool_names(prefix: str, raw_tool_names: Collection[str]) -> frozenset[str]:
    return frozenset(f"{prefix}_{name}" for name in raw_tool_names)


def _allow_tool_names(
    allowed_tool_names: Collection[str],
) -> Callable[[object, ToolDefinition], bool]:
    def is_allowed(_ctx: object, tool_definition: ToolDefinition) -> bool:
        return tool_definition.name in allowed_tool_names

    return is_allowed


def _payload_size(value: Any) -> int | None:
    text = _payload_text(value)
    if text is None:
        return None
    return len(text)


def _payload_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _safe_error_message(exc: Exception) -> str | None:
    message = str(exc).strip()
    if not message:
        return None
    if len(message) <= MAX_LOG_ERROR_MESSAGE_CHARS:
        return message
    return f"{message[:MAX_LOG_ERROR_MESSAGE_CHARS]}..."


def _tool_error_result(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "tool_name": name,
            "type": type(exc).__name__,
            "message": _safe_error_message(exc) or type(exc).__name__,
            "hint": (
                "The upstream MCP service rejected the call. Do not retry the same "
                "parameters; ask the user for the smallest useful correction."
            ),
        },
    }


def _safe_log_payload(value: Any) -> str | None:
    text = _payload_text(value)
    if text is None:
        return None
    if len(text) <= MAX_LOG_TOOL_ARGS_CHARS:
        return text
    return f"{text[:MAX_LOG_TOOL_ARGS_CHARS]}..."
