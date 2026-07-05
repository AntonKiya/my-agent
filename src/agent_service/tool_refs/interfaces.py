from typing import Protocol
from uuid import UUID

from agent_service.tool_refs.models import ToolResultReference


class ToolResultReferenceStore(Protocol):
    async def create(self, *, reference: ToolResultReference) -> ToolResultReference:
        """Persist a backend-only reference for a compact tool result item."""
        ...

    async def get(
        self,
        *,
        selection_id: str,
        user_id: UUID,
        conversation_id: UUID,
        provider: str | None = None,
    ) -> ToolResultReference | None:
        """Load a non-expired reference visible only to the owning conversation."""
        ...
