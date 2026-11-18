"""Standalone route construction and network resource ownership."""

import asyncio
import os
import unittest
from unittest.mock import patch

from agent0_runtime import Credential, FailoverRoute, RouteTarget, build_auto_route


class ClosingClient:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class RuntimeRoutingTests(unittest.TestCase):
    def test_build_auto_route_uses_runtime_providers(self):
        credential = Credential("sk-test", "https://api.deepseek.com", "env")
        with patch.dict(os.environ, {}, clear=True):
            route = build_auto_route({"deepseek": credential})
        self.assertTrue(route.targets)
        self.assertTrue(all(target.provider_id == "deepseek" for target in route.targets))
        self.assertEqual(route.current.provider_id, "deepseek")
        asyncio.run(route.close())

    def test_route_close_deduplicates_shared_clients(self):
        client = ClosingClient()
        credential = Credential("sk-test", "https://example.invalid", "test")
        route = FailoverRoute(
            (
                RouteTarget("one", "model-a", credential, client),
                RouteTarget("two", "model-b", credential, client),
            )
        )
        asyncio.run(route.close())
        self.assertEqual(client.closed, 1)


if __name__ == "__main__":
    unittest.main()
