from agent_service.users import (
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    User,
    UserStore,
    UserWithIdentity,
)


class FakeUserStore:
    async def get_by_channel_identity(
        self,
        *,
        lookup: ChannelIdentityLookup,
    ) -> UserWithIdentity | None:
        return None

    async def get_or_create_active_user_with_identity(
        self,
        *,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        raise NotImplementedError

    async def update_identity_seen(
        self,
        *,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        return user


async def test_user_store_protocol_describes_identity_storage_boundary() -> None:
    store: UserStore = FakeUserStore()

    result = await store.get_by_channel_identity(
        lookup=ChannelIdentityLookup(
            channel="telegram",
            external_user_id="67890",
        ),
    )

    assert isinstance(store, UserStore)
    assert result is None


def test_user_type_is_exported_from_domain_package() -> None:
    user = User()

    assert isinstance(user, User)
