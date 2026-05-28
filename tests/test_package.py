import asyncio

import agent_service


def test_package_exposes_main() -> None:
    assert callable(agent_service.main)


async def test_async_test_runtime_is_configured() -> None:
    await asyncio.sleep(0)
