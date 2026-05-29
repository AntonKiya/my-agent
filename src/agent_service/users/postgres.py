import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

import asyncpg

from agent_service.users.errors import UserResolutionError
from agent_service.users.interfaces import UserStore
from agent_service.users.models import (
    ChannelIdentity,
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    User,
    UserMetadata,
    UserStatus,
    UserWithIdentity,
)

USER_WITH_IDENTITY_SELECT = """
SELECT
    users.id AS user_id,
    users.status AS user_status,
    users.metadata AS user_metadata,
    users.created_at AS user_created_at,
    users.updated_at AS user_updated_at,
    channel_identities.id AS identity_id,
    channel_identities.user_id AS identity_user_id,
    channel_identities.channel AS identity_channel,
    channel_identities.external_user_id AS identity_external_user_id,
    channel_identities.external_chat_id AS identity_external_chat_id,
    channel_identities.username AS identity_username,
    channel_identities.metadata AS identity_metadata,
    channel_identities.created_at AS identity_created_at,
    channel_identities.updated_at AS identity_updated_at,
    channel_identities.last_seen_at AS identity_last_seen_at
FROM channel_identities
JOIN users ON users.id = channel_identities.user_id
WHERE channel_identities.channel = $1
  AND channel_identities.external_user_id = $2
"""

INSERT_USER_SQL = """
INSERT INTO users (
    id,
    status,
    metadata,
    created_at,
    updated_at
) VALUES ($1, $2, $3::jsonb, $4, $5)
"""

INSERT_IDENTITY_SQL = """
INSERT INTO channel_identities (
    id,
    user_id,
    channel,
    external_user_id,
    external_chat_id,
    username,
    metadata,
    created_at,
    updated_at,
    last_seen_at
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
"""

UPDATE_IDENTITY_SEEN_SQL = """
UPDATE channel_identities
SET
    external_chat_id = $3,
    username = $4,
    metadata = $5::jsonb,
    updated_at = $6,
    last_seen_at = $6
WHERE id = $1
  AND user_id = $2
"""


class PostgresConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """Execute a SQL command against Postgres."""
        ...

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Open a transaction on this connection."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresUserStore(UserStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def get_by_channel_identity(
        self,
        *,
        lookup: ChannelIdentityLookup,
    ) -> UserWithIdentity | None:
        async with self._pool.acquire() as connection:
            return await self._fetch_user_with_identity(connection, lookup)

    async def get_or_create_active_user_with_identity(
        self,
        *,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        lookup = identity.lookup()
        async with self._pool.acquire() as connection:
            try:
                async with connection.transaction():
                    existing = await self._fetch_user_with_identity(connection, lookup)
                    if existing is not None:
                        return await self._update_identity_seen_on_connection(
                            connection=connection,
                            user=existing,
                            identity=identity,
                        )
                    return await self._create_active_user_with_identity_on_connection(
                        connection=connection,
                        identity=identity,
                    )
            except asyncpg.UniqueViolationError:
                existing = await self._fetch_user_with_identity(connection, lookup)
                if existing is None:
                    raise
                return await self._update_identity_seen_on_connection(
                    connection=connection,
                    user=existing,
                    identity=identity,
                )

    async def update_identity_seen(
        self,
        *,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                return await self._update_identity_seen_on_connection(
                    connection=connection,
                    user=user,
                    identity=identity,
                )

    async def _fetch_user_with_identity(
        self,
        connection: PostgresConnection,
        lookup: ChannelIdentityLookup,
    ) -> UserWithIdentity | None:
        row = await connection.fetchrow(
            USER_WITH_IDENTITY_SELECT,
            lookup.channel,
            lookup.external_user_id,
        )
        if row is None:
            return None
        return _user_with_identity_from_row(row)

    async def _create_active_user_with_identity_on_connection(
        self,
        *,
        connection: PostgresConnection,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        now = identity.observed_at
        user = User(
            id=uuid4(),
            status=UserStatus.ACTIVE,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        channel_identity = ChannelIdentity(
            id=uuid4(),
            user_id=user.id,
            channel=identity.channel,
            external_user_id=identity.external_user_id,
            external_chat_id=identity.external_chat_id,
            username=identity.username,
            metadata=dict(identity.metadata),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
        )

        await connection.execute(
            INSERT_USER_SQL,
            user.id,
            user.status.value,
            _jsonb(user.metadata),
            user.created_at,
            user.updated_at,
        )
        await connection.execute(
            INSERT_IDENTITY_SQL,
            channel_identity.id,
            channel_identity.user_id,
            channel_identity.channel,
            channel_identity.external_user_id,
            channel_identity.external_chat_id,
            channel_identity.username,
            _jsonb(channel_identity.metadata),
            channel_identity.created_at,
            channel_identity.updated_at,
            channel_identity.last_seen_at,
        )
        return UserWithIdentity(user=user, identity=channel_identity)

    async def _update_identity_seen_on_connection(
        self,
        *,
        connection: PostgresConnection,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        if user.identity.channel != identity.channel:
            raise UserResolutionError("Observed identity channel does not match stored identity")
        if user.identity.external_user_id != identity.external_user_id:
            raise UserResolutionError("Observed external_user_id does not match stored identity")

        await connection.execute(
            UPDATE_IDENTITY_SEEN_SQL,
            user.identity.id,
            user.user.id,
            identity.external_chat_id,
            identity.username,
            _jsonb(identity.metadata),
            identity.observed_at,
        )
        return UserWithIdentity(
            user=user.user,
            identity=ChannelIdentity(
                id=user.identity.id,
                user_id=user.identity.user_id,
                channel=user.identity.channel,
                external_user_id=user.identity.external_user_id,
                external_chat_id=identity.external_chat_id,
                username=identity.username,
                metadata=dict(identity.metadata),
                created_at=user.identity.created_at,
                updated_at=identity.observed_at,
                last_seen_at=identity.observed_at,
            ),
        )


def _user_with_identity_from_row(row: Mapping[str, object]) -> UserWithIdentity:
    user = User(
        id=_uuid(row["user_id"]),
        status=UserStatus(str(row["user_status"])),
        metadata=_metadata(row["user_metadata"]),
        created_at=_datetime(row["user_created_at"]),
        updated_at=_datetime(row["user_updated_at"]),
    )
    identity = ChannelIdentity(
        id=_uuid(row["identity_id"]),
        user_id=_uuid(row["identity_user_id"]),
        channel=str(row["identity_channel"]),
        external_user_id=str(row["identity_external_user_id"]),
        external_chat_id=_optional_str(row["identity_external_chat_id"]),
        username=_optional_str(row["identity_username"]),
        metadata=_metadata(row["identity_metadata"]),
        created_at=_datetime(row["identity_created_at"]),
        updated_at=_datetime(row["identity_updated_at"]),
        last_seen_at=_datetime(row["identity_last_seen_at"]),
    )
    return UserWithIdentity(user=user, identity=identity)


def _jsonb(metadata: UserMetadata) -> str:
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def _metadata(value: object) -> UserMetadata:
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise UserResolutionError("Postgres metadata value must decode to an object")
        return cast(UserMetadata, decoded)
    if isinstance(value, dict):
        return cast(UserMetadata, dict(value))
    raise UserResolutionError("Postgres metadata value must be a JSON object")


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise UserResolutionError("Postgres UUID value has unexpected type")


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise UserResolutionError("Postgres datetime value has unexpected type")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
