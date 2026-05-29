from typing import Protocol, runtime_checkable

from agent_service.agents.models import AgentRequest, AgentResponse


@runtime_checkable
class AgentBoundary(Protocol):
    async def run(self, request: AgentRequest) -> AgentResponse:
        """Run the agent against a channel-agnostic request and prepared context."""
        ...
