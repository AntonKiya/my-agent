from datetime import UTC
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agent_service.agents import (
    AgentBoundary,
    AgentContext,
    AgentContextMessage,
    AgentContextRole,
    AgentRequest,
    AgentResponse,
    AgentToolInfo,
    AgentToolStatus,
    AgentUsage,
    PydanticAIRunContext,
)
from agent_service.channels import Attachment, AttachmentType


class EchoAgentBoundary:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(text=f"echo: {request.text}", trace_id=request.trace_id)


def test_agent_request_contains_only_channel_agnostic_payload() -> None:
    request = AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        text="hello",
        pydantic_ai=PydanticAIRunContext(
            user_prompt="hello",
            message_history=[ModelRequest(parts=[UserPromptPart(content="previous")])],
            conversation_id="conversation-1",
            instructions="summary: user prefers concise answers",
        ),
        context=AgentContext(
            system_prompt_parts=["summary: user prefers concise answers"],
            recent_messages=[
                AgentContextMessage(
                    role=AgentContextRole.ASSISTANT,
                    text="previous answer",
                )
            ],
        ),
        metadata={"idempotency_key": "telegram:12345:42"},
        trace_id="trace-1",
    )

    assert request.channel == "telegram"
    assert request.text == "hello"
    assert request.context.system_prompt_parts == ["summary: user prefers concise answers"]
    assert request.context.recent_messages[0].role is AgentContextRole.ASSISTANT
    assert request.pydantic_ai is not None
    assert request.pydantic_ai.user_prompt == "hello"
    assert request.pydantic_ai.conversation_id == "conversation-1"
    assert request.trace_id == "trace-1"
    assert "external_chat_id" not in request.model_dump()
    assert "raw_update" not in request.model_dump()


def test_agent_request_accepts_attachments_for_future_non_text_messages() -> None:
    request = AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        attachments=[
            Attachment(
                attachment_type=AttachmentType.DOCUMENT,
                external_id="file-1",
                content_type="application/pdf",
            )
        ],
    )

    assert request.text is None
    assert request.attachments[0].external_id == "file-1"


def test_agent_contract_defaults_are_isolated() -> None:
    first = AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        text="first",
    )
    second = AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        text="second",
    )

    first.context.system_prompt_parts.append("summary")
    first.metadata["key"] = "value"

    assert second.context.system_prompt_parts == []
    assert second.metadata == {}


def test_agent_request_rejects_empty_payload_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            inbound_event_id=uuid4(),
            channel="telegram",
        )

    with pytest.raises(ValidationError):
        AgentRequest.model_validate(
            {
                "user_id": str(uuid4()),
                "conversation_id": str(uuid4()),
                "inbound_event_id": str(uuid4()),
                "channel": "telegram",
                "text": "hello",
                "external_chat_id": "12345",
            }
        )


def test_agent_response_records_usage_tools_and_trace() -> None:
    new_messages = [ModelResponse(parts=[TextPart(content="answer")])]
    response = AgentResponse(
        text="answer",
        usage=AgentUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        tool_info=[
            AgentToolInfo(
                tool_name="search",
                status=AgentToolStatus.SUCCEEDED,
                call_id="call-1",
                metadata={"result_count": 3},
            )
        ],
        pydantic_ai_new_messages=new_messages,
        trace_id="trace-1",
    )

    assert response.text == "answer"
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.tool_info is not None
    assert response.tool_info[0].tool_name == "search"
    assert response.pydantic_ai_new_messages == new_messages
    assert response.trace_id == "trace-1"


def test_agent_context_message_has_utc_created_at() -> None:
    message = AgentContextMessage(role=AgentContextRole.USER, text="hello")

    assert message.created_at.tzinfo is UTC


async def test_agent_boundary_protocol_accepts_async_implementation() -> None:
    boundary = EchoAgentBoundary()
    request = AgentRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        inbound_event_id=uuid4(),
        channel="telegram",
        text="hello",
        trace_id="trace-1",
    )

    response = await boundary.run(request)

    assert isinstance(boundary, AgentBoundary)
    assert boundary.requests == [request]
    assert response.text == "echo: hello"
    assert response.trace_id == "trace-1"
