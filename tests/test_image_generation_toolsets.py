import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agent_service.config import AppSettings
from agent_service.image_generation import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from agent_service.image_generation.toolsets import build_image_generation_toolsets
from agent_service.media import MediaAsset, MediaAssetType, PersistentFileMediaStore
from agent_service.quotas import (
    QuotaMetric,
    QuotaPeriod,
    QuotaReservationRequest,
    QuotaReservationResult,
)


@dataclass(slots=True)
class FakeGenerator:
    requests: list[ImageGenerationRequest] = field(default_factory=list)

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        self.requests.append(request)
        return ImageGenerationResult(
            images=(
                GeneratedImage(
                    content=b"generated-image",
                    content_type="image/png",
                    filename="generated.png",
                ),
            ),
            provider="test",
            model="image-test",
            text="ok",
        )


@dataclass(slots=True)
class FakeMediaAssetStore:
    assets: dict[tuple[str, UUID, UUID], MediaAsset] = field(default_factory=dict)

    async def create(self, *, asset: MediaAsset) -> MediaAsset:
        self.assets[(asset.media_id, asset.user_id, asset.conversation_id)] = asset
        return asset

    async def get(
        self,
        *,
        media_id: str,
        user_id: UUID,
        conversation_id: UUID,
    ) -> MediaAsset | None:
        return self.assets.get((media_id, user_id, conversation_id))


@dataclass(slots=True)
class FakeQuotaService:
    allowed: bool = True
    requests: list[QuotaReservationRequest] = field(default_factory=list)

    async def reserve(self, request: QuotaReservationRequest) -> QuotaReservationResult:
        self.requests.append(request)
        return QuotaReservationResult(
            allowed=self.allowed,
            user_id=request.user_id,
            metric=request.metric,
            period=request.period,
            period_start=datetime(2026, 6, 13, tzinfo=UTC),
            period_end=datetime(2026, 6, 14, tzinfo=UTC),
            used_count=1,
            limit_count=1,
        )


@dataclass(slots=True)
class FakeRunContext:
    deps: dict[str, Any]


async def test_image_generation_toolset_includes_follow_up_edit_instructions(
    tmp_path: Path,
) -> None:
    toolsets = build_image_generation_toolsets(
        AppSettings(environment="test"),
        generator=FakeGenerator(),
        media_asset_store=FakeMediaAssetStore(),
        media_store=PersistentFileMediaStore(base_dir=tmp_path),
        quota_service=FakeQuotaService(),
    )

    instructions = await cast(Any, toolsets[0]).get_instructions(
        RunContext(deps={}, model=TestModel(), usage=RunUsage())
    )

    assert instructions is not None
    content = instructions[0].content
    assert "create a new image or edit" in content
    assert "previously attached" in content
    assert "Do not ask the user to re-upload" in content


async def test_generate_image_checks_ownership_and_stores_generated_asset(
    tmp_path: Path,
) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    other_user_id = uuid4()
    inbound_event_id = uuid4()
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source-image")
    store = FakeMediaAssetStore()
    await store.create(
        asset=MediaAsset(
            media_id="owned",
            user_id=user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.IMAGE,
            storage_key=str(source_path),
            content_type="image/jpeg",
            source_channel="telegram",
        )
    )
    await store.create(
        asset=MediaAsset(
            media_id="other",
            user_id=other_user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.IMAGE,
            storage_key=str(source_path),
            content_type="image/jpeg",
            source_channel="telegram",
        )
    )
    generator = FakeGenerator()
    quota = FakeQuotaService()
    toolsets = build_image_generation_toolsets(
        AppSettings(environment="test"),
        generator=generator,
        media_asset_store=store,
        media_store=PersistentFileMediaStore(base_dir=tmp_path / "generated"),
        quota_service=quota,
    )
    tool = cast(Any, toolsets[0]).tools["generateImage"]
    ctx = cast(
        Any,
        FakeRunContext(
            deps={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "inbound_event_id": inbound_event_id,
                "channel": "telegram",
            }
        ),
    )

    denied = await tool.function(ctx, prompt="Сделай светлее", source_media_ids=["other"])
    allowed = await tool.function(ctx, prompt="Сделай светлее", source_media_ids=["owned"])

    assert denied == {
        "success": False,
        "error_code": "image_not_found_or_not_accessible",
    }
    assert allowed["success"] is True
    generated = allowed["data"]["generated_images"][0]
    assert generated["content_type"] == "image/png"
    assert len(generator.requests) == 1
    assert generator.requests[0].source_assets[0].media_id == "owned"
    assert quota.requests == [
        QuotaReservationRequest(
            user_id=user_id,
            metric=QuotaMetric.IMAGE_GENERATION,
            period=QuotaPeriod.DAY,
        )
    ]
    generated_asset = await store.get(
        media_id=generated["media_id"],
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert generated_asset is not None
    assert generated_asset.metadata["source"] == "image_generation"
    assert generated_asset.source_inbound_event_id == inbound_event_id
    stored_content = await asyncio.to_thread(Path(generated_asset.storage_key).read_bytes)
    assert stored_content == b"generated-image"


async def test_generate_image_respects_image_generation_quota(tmp_path: Path) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    generator = FakeGenerator()
    quota = FakeQuotaService(allowed=False)
    toolsets = build_image_generation_toolsets(
        AppSettings(environment="test"),
        generator=generator,
        media_asset_store=FakeMediaAssetStore(),
        media_store=PersistentFileMediaStore(base_dir=tmp_path),
        quota_service=quota,
    )
    tool = cast(Any, toolsets[0]).tools["generateImage"]
    ctx = cast(
        Any,
        FakeRunContext(deps={"user_id": user_id, "conversation_id": conversation_id}),
    )

    result = await tool.function(ctx, prompt="Нарисуй дом", source_media_ids=[])

    assert result["success"] is False
    assert result["error_code"] == "quota_exceeded"
    assert result["data"]["quota_metric"] == "image_generation"
    assert generator.requests == []
