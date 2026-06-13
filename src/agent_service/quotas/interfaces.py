from typing import Protocol

from agent_service.quotas.models import QuotaReservationRequest, QuotaReservationResult


class QuotaService(Protocol):
    async def reserve(self, request: QuotaReservationRequest) -> QuotaReservationResult:
        """Atomically reserve one unit from a usage quota."""
        ...
