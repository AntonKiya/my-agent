from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from agent_service.tool_refs.interfaces import ToolResultReferenceStore
from agent_service.tool_refs.models import ToolResultReference


class InMemoryToolResultReferenceStore(ToolResultReferenceStore):
    def __init__(self, *, now_provider: Callable[[], datetime] | None = None) -> None:
        self._references: dict[str, ToolResultReference] = {}
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    async def create(self, *, reference: ToolResultReference) -> ToolResultReference:
        self._references[reference.selection_id] = reference
        return reference

    async def get(
        self,
        *,
        selection_id: str,
        user_id: UUID,
        conversation_id: UUID,
        provider: str | None = None,
    ) -> ToolResultReference | None:
        reference = self._references.get(selection_id)
        if reference is None:
            return None
        if reference.user_id != user_id or reference.conversation_id != conversation_id:
            return None
        if provider is not None and reference.provider != provider:
            return None
        if reference.is_expired(now=self._now_provider()):
            self._references.pop(selection_id, None)
            return None
        return reference
