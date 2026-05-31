from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import asyncpg

from agent_service.channels import InboundEvent, InboundEventStatus
from agent_service.inbound import PostgresInboundIdempotencyStore
from agent_service.inbound.postgres import (
    INSERT_INBOUND_EVENT_PROCESSING_SQL,
    RELEASE_INBOUND_EVENT_CLAIM_SQL,
    UPDATE_INBOUND_EVENT_STATUS_SQL,
)


@dataclass(slots=True)
class FakeConnection:
    fetch_results: list[Mapping[str, object] | None] = field(default_factory=list)
    execute_errors: list[BaseException | None] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetch_calls.append((query, args))
        if not self.fetch_results:
            return None
        return self.fetch_results.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        if self.execute_errors:
            error = self.execute_errors.pop(0)
            if error is not None:
                raise error
        return "OK"


@dataclass(slots=True)
class FakeAcquire:
    connection: FakeConnection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


@dataclass(slots=True)
class FakePool:
    connection: FakeConnection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


def inbound_event() -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        external_update_id="100",
        idempotency_key="telegram:12345:42",
        user_id=uuid4(),
        text="hello",
    )


async def test_postgres_inbound_idempotency_store_claims_new_event() -> None:
    connection = FakeConnection()
    store = PostgresInboundIdempotencyStore(FakePool(connection))
    event = inbound_event()

    claim = await store.claim(event)

    assert claim.claimed
    assert claim.event_id == event.event_id
    assert connection.execute_calls[0][0] == INSERT_INBOUND_EVENT_PROCESSING_SQL
    assert connection.execute_calls[0][1][0:9] == (
        event.event_id,
        "telegram",
        "telegram:12345:42",
        "100",
        "12345",
        "42",
        event.user_id,
        "queued",
        event.trace_id,
    )


async def test_postgres_inbound_idempotency_store_returns_duplicate_claim() -> None:
    existing_event_id = uuid4()
    connection = FakeConnection(
        fetch_results=[{"event_id": existing_event_id, "status": "completed"}],
        execute_errors=[asyncpg.UniqueViolationError("duplicate inbound event")],
    )
    store = PostgresInboundIdempotencyStore(FakePool(connection))
    event = inbound_event()

    claim = await store.claim(event)

    assert not claim.claimed
    assert claim.event_id == event.event_id
    assert claim.existing_event_id == existing_event_id
    assert claim.existing_status is InboundEventStatus.COMPLETED
    assert len(connection.fetch_calls) == 1
    assert len(connection.execute_calls) == 2


async def test_postgres_inbound_idempotency_store_releases_unpublished_claim() -> None:
    connection = FakeConnection()
    store = PostgresInboundIdempotencyStore(FakePool(connection))
    event_id = uuid4()

    await store.release_claim(event_id=event_id)

    assert connection.execute_calls == [(RELEASE_INBOUND_EVENT_CLAIM_SQL, (event_id,))]


async def test_postgres_inbound_idempotency_store_marks_processing_status() -> None:
    connection = FakeConnection()
    store = PostgresInboundIdempotencyStore(FakePool(connection))
    event_id = uuid4()

    await store.mark_status(
        event_id=event_id,
        status=InboundEventStatus.PROCESSING,
    )

    assert connection.execute_calls[0][0] == UPDATE_INBOUND_EVENT_STATUS_SQL
    assert connection.execute_calls[0][1][0:3] == (
        event_id,
        "processing",
        None,
    )
