from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

from agent_service.config import AppSettings
from agent_service.image_analysis import ImageAnalysisRequest, ImageAnalysisResult
from agent_service.image_analysis.toolsets import build_image_analysis_toolsets
from agent_service.media import MediaAsset, MediaAssetType


@dataclass(slots=True)
class FakeAnalyzer:
    requests: list[ImageAnalysisRequest] = field(default_factory=list)

    async def analyze(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        self.requests.append(request)
        return ImageAnalysisResult(
            analysis="На изображении таблица.",
            provider="test",
            model="vision-test",
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
class FakeRunContext:
    deps: dict[str, Any]


async def test_analyze_image_tool_checks_user_and_conversation_ownership() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    other_user_id = uuid4()
    store = FakeMediaAssetStore()
    await store.create(
        asset=MediaAsset(
            media_id="owned",
            user_id=user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.IMAGE,
            storage_key="/tmp/owned.jpg",
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
            storage_key="/tmp/other.jpg",
            content_type="image/jpeg",
            source_channel="telegram",
        )
    )
    analyzer = FakeAnalyzer()
    toolsets = build_image_analysis_toolsets(
        AppSettings(environment="test", image_analysis_model="openai/gpt-4.1-mini"),
        analyzer=analyzer,
        media_asset_store=store,
    )
    tool = cast(Any, toolsets[0]).tools["analyzeImage"]
    ctx = cast(
        Any,
        FakeRunContext(deps={"user_id": user_id, "conversation_id": conversation_id}),
    )

    denied = await tool.function(ctx, prompt="Что тут?", media_ids=["other"])
    allowed = await tool.function(ctx, prompt="Что тут?", media_ids=["owned"])

    assert denied == {
        "success": False,
        "error_code": "image_not_found_or_not_accessible",
    }
    assert allowed["success"] is True
    assert allowed["data"]["analysis"] == "На изображении таблица."
    assert len(analyzer.requests) == 1
    assert analyzer.requests[0].assets[0].media_id == "owned"
