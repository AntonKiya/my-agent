import os
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Protocol, cast

import asyncpg
import pytest

from agent_service.database import migrate_database
from agent_service.users import (
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    PostgresPool,
    PostgresUserStore,
)

pytestmark = pytest.mark.integration


class AsyncClosable(Protocol):
    async def close(self) -> None:
        ...


def postgres_dsn() -> str:
    dsn = os.environ.get("AGENT_SERVICE_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.skip("AGENT_SERVICE_TEST_POSTGRES_DSN is not configured")
        raise RuntimeError("pytest.skip did not stop execution")
    return dsn


async def clean_user_tables(dsn: str) -> None:
    connection = await asyncpg.connect(dsn=dsn)
    try:
        await connection.execute("TRUNCATE channel_identities, users RESTART IDENTITY CASCADE")
    finally:
        await connection.close()


async def test_postgres_migrations_and_user_store_roundtrip() -> None:
    dsn = postgres_dsn()
    await migrate_database(dsn=dsn)
    await clean_user_tables(dsn)
    pool_object = await cast(
        Awaitable[object],
        asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2),
    )
    pool = cast(PostgresPool, pool_object)
    try:
        store = PostgresUserStore(pool)
        observed = ObservedChannelIdentity(
            channel="telegram",
            external_user_id="67890",
            external_chat_id="12345",
            username="handle",
            metadata={"first_name": "Anton"},
            observed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        )

        created = await store.get_or_create_active_user_with_identity(identity=observed)
        loaded = await store.get_by_channel_identity(
            lookup=ChannelIdentityLookup(
                channel="telegram",
                external_user_id="67890",
            )
        )

        assert loaded is not None
        assert loaded.user.id == created.user.id
        assert loaded.identity.external_user_id == "67890"
        assert loaded.identity.username == "handle"
        assert loaded.identity.metadata == {"first_name": "Anton"}
    finally:
        await cast(AsyncClosable, pool_object).close()
