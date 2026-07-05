from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ToolResultReference:
    selection_id: str
    provider: str
    source_tool_name: str
    user_id: UUID
    conversation_id: UUID
    item_kind: str
    item_index: int
    label: str | None
    display_snapshot: dict[str, Any]
    ref_payload: dict[str, Any]
    expires_at: datetime
    created_at: datetime

    def is_expired(self, *, now: datetime | None = None) -> bool:
        current_time = now or datetime.now(UTC)
        return self.expires_at <= current_time
