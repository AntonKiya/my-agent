from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agent_service.config import AppSettings
from agent_service.document_reading.toolsets import build_file_reading_toolsets
from agent_service.media import MediaAsset, MediaAssetType


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


async def test_file_reading_toolset_includes_usage_policy_instructions() -> None:
    toolsets = build_file_reading_toolsets(
        AppSettings(environment="test"),
        media_asset_store=FakeMediaAssetStore(),
    )
    instructions = await cast(Any, toolsets[0]).get_instructions(
        RunContext(deps={}, model=TestModel(), usage=RunUsage())
    )

    assert instructions is not None
    assert len(instructions) == 1
    content = instructions[0].content
    assert "previously attached file/document" in content
    assert "call readFileContent before answering" in content
    assert "Do not infer file contents from filename" in content


async def test_read_file_content_reads_owned_txt_file(tmp_path: Path) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    path = tmp_path / "notes.txt"
    path.write_text("hello from file", encoding="utf-8")
    store = FakeMediaAssetStore()
    await store.create(
        asset=MediaAsset(
            media_id="owned-file",
            user_id=user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.DOCUMENT,
            storage_key=str(path),
            content_type="text/plain",
            size_bytes=path.stat().st_size,
            source_channel="telegram",
            metadata={"original_filename": "notes.txt"},
        )
    )
    toolsets = build_file_reading_toolsets(
        AppSettings(environment="test"),
        media_asset_store=store,
    )
    tool = cast(Any, toolsets[0]).tools["readFileContent"]
    ctx = cast(
        Any,
        FakeRunContext(deps={"user_id": user_id, "conversation_id": conversation_id}),
    )

    result = await tool.function(ctx, media_ids=["owned-file"])

    assert result["success"] is True
    assert result["data"]["files"] == [
        {
            "media_id": "owned-file",
            "filename": "notes.txt",
            "content_type": "text/plain",
            "content": "hello from file",
            "truncated": False,
            "size_bytes": path.stat().st_size,
            "char_count": len("hello from file"),
        }
    ]


async def test_read_file_content_checks_ownership_and_media_type(tmp_path: Path) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    other_user_id = uuid4()
    path = tmp_path / "notes.md"
    path.write_text("# Owned", encoding="utf-8")
    store = FakeMediaAssetStore()
    await store.create(
        asset=MediaAsset(
            media_id="other-file",
            user_id=other_user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.DOCUMENT,
            storage_key=str(path),
            content_type="text/markdown",
            source_channel="telegram",
            metadata={"original_filename": "notes.md"},
        )
    )
    await store.create(
        asset=MediaAsset(
            media_id="image-id",
            user_id=user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.IMAGE,
            storage_key=str(tmp_path / "image.jpg"),
            content_type="image/jpeg",
            source_channel="telegram",
        )
    )
    toolsets = build_file_reading_toolsets(
        AppSettings(environment="test"),
        media_asset_store=store,
    )
    tool = cast(Any, toolsets[0]).tools["readFileContent"]
    ctx = cast(
        Any,
        FakeRunContext(deps={"user_id": user_id, "conversation_id": conversation_id}),
    )

    denied_other = await tool.function(ctx, media_ids=["other-file"])
    denied_image = await tool.function(ctx, media_ids=["image-id"])

    assert denied_other == {
        "success": False,
        "error_code": "file_not_found_or_not_accessible",
    }
    assert denied_image == {
        "success": False,
        "error_code": "file_not_found_or_not_accessible",
    }


async def test_read_file_content_truncates_large_content(tmp_path: Path) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    path = tmp_path / "notes.md"
    path.write_text("abcdef", encoding="utf-8")
    store = FakeMediaAssetStore()
    await store.create(
        asset=MediaAsset(
            media_id="owned-file",
            user_id=user_id,
            conversation_id=conversation_id,
            media_type=MediaAssetType.DOCUMENT,
            storage_key=str(path),
            content_type="text/markdown",
            size_bytes=path.stat().st_size,
            source_channel="telegram",
            metadata={"original_filename": "notes.md"},
        )
    )
    toolsets = build_file_reading_toolsets(
        AppSettings(environment="test", document_max_extracted_chars=3),
        media_asset_store=store,
    )
    tool = cast(Any, toolsets[0]).tools["readFileContent"]
    ctx = cast(
        Any,
        FakeRunContext(deps={"user_id": user_id, "conversation_id": conversation_id}),
    )

    result = await tool.function(ctx, media_ids=["owned-file"])

    assert result["success"] is True
    assert result["data"]["files"][0]["content"] == "abc"
    assert result["data"]["files"][0]["truncated"] is True
