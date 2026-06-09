from typing import Any
from uuid import UUID

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.config import AppSettings
from agent_service.image_analysis.interfaces import ImageAnalysisError, ImageAnalyzer
from agent_service.image_analysis.models import ImageAnalysisRequest
from agent_service.media import MediaAssetStore, MediaAssetType

IMAGE_ANALYSIS_TOOLSET_ID = "image-analysis"
IMAGE_ANALYSIS_TOOLSET_INSTRUCTIONS = (
    "For any question about the visual content of an attached or previously attached image, "
    "call analyzeImage before answering, even if the image was analyzed earlier. Use media_id "
    "values from current markers or recent conversation context, preferring the latest relevant "
    "image. Do not guess from memory or claim images are inaccessible when a media_id is "
    "available. If the target image is unclear, ask one short clarification."
)


def build_image_analysis_toolsets(
    settings: AppSettings,
    *,
    analyzer: ImageAnalyzer,
    media_asset_store: MediaAssetStore,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.image_analysis_enabled:
        return ()

    async def analyzeImage(
        ctx: RunContext[dict[str, Any]],
        prompt: str,
        media_ids: list[str],
    ) -> dict[str, Any]:
        """Analyze images attached in the current conversation.

        Args:
            prompt: The user's question or instruction about the images.
            media_ids: One or more media_id values from the attached image markers.
        """
        deps = ctx.deps or {}
        user_id = _uuid_dep(deps.get("user_id"))
        conversation_id = _uuid_dep(deps.get("conversation_id"))
        if user_id is None or conversation_id is None:
            return _error("context_unavailable")
        normalized_media_ids = _normalized_media_ids(media_ids)
        if not prompt.strip():
            return _error("empty_prompt")
        if not normalized_media_ids:
            return _error("empty_media_ids")
        if len(normalized_media_ids) > settings.image_analysis_max_images:
            return _error("too_many_images")

        assets = []
        for media_id in normalized_media_ids:
            asset = await media_asset_store.get(
                media_id=media_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if asset is None or asset.media_type is not MediaAssetType.IMAGE:
                return _error("image_not_found_or_not_accessible")
            assets.append(asset)

        try:
            result = await analyzer.analyze(
                ImageAnalysisRequest(
                    prompt=prompt.strip(),
                    assets=tuple(assets),
                )
            )
        except ImageAnalysisError as exc:
            return _error(exc.error_code or "image_analysis_failed")

        return {
            "success": True,
            "data": {
                "analysis": result.analysis,
                "provider": result.provider,
                "model": result.model,
            },
        }

    return (
        FunctionToolset(
            [analyzeImage],
            id=IMAGE_ANALYSIS_TOOLSET_ID,
            instructions=IMAGE_ANALYSIS_TOOLSET_INSTRUCTIONS,
            timeout=settings.image_analysis_tool_timeout_seconds,
            require_parameter_descriptions=True,
        ),
    )


def _normalized_media_ids(media_ids: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for media_id in media_ids:
        value = media_id.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _uuid_dep(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _error(error_code: str) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
    }
