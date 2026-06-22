import logging
import secrets
from typing import Any
from uuid import UUID

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.channels.models import Attachment, AttachmentType
from agent_service.config import AppSettings
from agent_service.image_generation.interfaces import ImageGenerationError, ImageGenerator
from agent_service.image_generation.models import ImageGenerationRequest
from agent_service.media import (
    MediaAsset,
    MediaAssetStore,
    MediaAssetType,
    MediaPayload,
    PersistentMediaStore,
)
from agent_service.observability.events import log_event
from agent_service.quotas import (
    QuotaConfigurationError,
    QuotaMetric,
    QuotaPeriod,
    QuotaReservationRequest,
    QuotaService,
)

logger = logging.getLogger(__name__)

IMAGE_GENERATION_SKILL_ID = "image-generation"
IMAGE_GENERATION_TOOLSET_ID = IMAGE_GENERATION_SKILL_ID
IMAGE_GENERATION_TOOL_NAME = "generateImage"
IMAGE_GENERATION_TOOLSET_INSTRUCTIONS = (
    "Call generateImage for image creation or image editing. "
    "For new images, omit source_media_ids. "
    "For edits or follow-ups on an existing image, pass the relevant media_id values from image "
    "markers as source_media_ids. "
    "For edits, describe the requested change and what must stay unchanged. "
    "Ask one short clarification only if the target image is unclear."
)


def build_image_generation_toolsets(
    settings: AppSettings,
    *,
    generator: ImageGenerator,
    media_asset_store: MediaAssetStore,
    media_store: PersistentMediaStore,
    quota_service: QuotaService,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.image_generation_enabled:
        return ()

    async def generateImage(
        ctx: RunContext[dict[str, Any]],
        prompt: str,
        source_media_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new image or edit images from the current conversation.

        Args:
            prompt: The user's image creation or editing instruction.
            source_media_ids: Image media_id values from image markers. Required for edits or
                follow-ups on an existing image; omit only for new images.
        """
        deps = ctx.deps or {}
        user_id = _uuid_dep(deps.get("user_id"))
        conversation_id = _uuid_dep(deps.get("conversation_id"))
        inbound_event_id = _uuid_dep(deps.get("inbound_event_id"))
        channel = _str_dep(deps.get("channel"))
        if user_id is None or conversation_id is None:
            return _error("context_unavailable")

        clean_prompt = prompt.strip()
        if not clean_prompt:
            return _error("empty_prompt")

        normalized_source_media_ids = _normalized_media_ids(source_media_ids or [])
        if len(normalized_source_media_ids) > settings.image_generation_max_source_images:
            return _error("too_many_source_images")

        source_assets = []
        for media_id in normalized_source_media_ids:
            asset = await media_asset_store.get(
                media_id=media_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if asset is None or asset.media_type is not MediaAssetType.IMAGE:
                return _error("image_not_found_or_not_accessible")
            source_assets.append(asset)

        try:
            quota = await quota_service.reserve(
                QuotaReservationRequest(
                    user_id=user_id,
                    metric=QuotaMetric.IMAGE_GENERATION,
                    period=QuotaPeriod.DAY,
                )
            )
        except QuotaConfigurationError:
            log_event(
                logger,
                logging.ERROR,
                "Image generation quota policy is not configured",
                event="image_generation_quota_policy_missing",
                user_id=str(user_id),
                conversation_id=str(conversation_id),
            )
            return _error("quota_unavailable")
        if not quota.allowed:
            return {
                "success": False,
                "error_code": "quota_exceeded",
                "data": {
                    "quota_metric": quota.metric.value,
                    "quota_period": quota.period.value,
                    "quota_reset_at": quota.period_end.isoformat(),
                    "quota_limit_count": quota.limit_count,
                    "quota_used_count": quota.used_count,
                },
            }

        try:
            result = await generator.generate(
                ImageGenerationRequest(
                    prompt=clean_prompt,
                    source_assets=tuple(source_assets),
                )
            )
        except ImageGenerationError as exc:
            return _error(exc.error_code or "image_generation_failed")

        selected_images = result.images[: settings.image_generation_max_output_images]
        if not selected_images:
            return _error("empty_image_generation")
        if any(
            image.size_bytes > settings.image_generation_max_output_size_bytes
            for image in selected_images
        ):
            return _error("generated_image_too_large")

        stored_images: list[dict[str, Any]] = []
        for index, image in enumerate(selected_images):
            media_id = _new_media_id()
            payload = MediaPayload(
                attachment=Attachment(
                    attachment_id=media_id,
                    attachment_type=AttachmentType.IMAGE,
                    content_type=image.content_type,
                    metadata={"media_id": media_id, "generated": True},
                ),
                content=image.content,
                content_type=image.content_type,
                filename=image.filename,
                metadata=dict(image.metadata),
            )
            stored_media = await media_store.store(media_id=media_id, payload=payload)
            await media_asset_store.create(
                asset=MediaAsset(
                    media_id=media_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    media_type=MediaAssetType.IMAGE,
                    storage_key=str(stored_media.path),
                    content_type=stored_media.content_type,
                    size_bytes=stored_media.size_bytes,
                    source_channel=channel or "agent",
                    source_inbound_event_id=inbound_event_id,
                    metadata={
                        "source": "image_generation",
                        "provider": result.provider,
                        "model": result.model,
                        "prompt": clean_prompt,
                        "source_media_ids": normalized_source_media_ids,
                        "generated_index": index,
                        "original_filename": stored_media.filename,
                    },
                )
            )
            stored_images.append(
                {
                    "media_id": media_id,
                    "content_type": stored_media.content_type,
                    "size_bytes": stored_media.size_bytes,
                    "filename": stored_media.filename,
                }
            )

        return {
            "success": True,
            "data": {
                "generated_images": stored_images,
                "provider": result.provider,
                "model": result.model,
                "text": result.text,
                "quota_remaining_count": quota.remaining_count,
            },
        }

    return (
        FunctionToolset(
            [generateImage],
            id=IMAGE_GENERATION_TOOLSET_ID,
            instructions=IMAGE_GENERATION_TOOLSET_INSTRUCTIONS,
            timeout=settings.image_generation_tool_timeout_seconds,
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


def _str_dep(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _new_media_id() -> str:
    return secrets.token_urlsafe(9)


def _error(error_code: str) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
    }
