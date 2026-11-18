"""Standalone managed providers and host-selected credential storage."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent0_runtime import FileCredentialStore, ManagedProvider, make_openai_client


class ManagedProviderTests(unittest.TestCase):
    def test_login_resolve_active_and_logout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config" / "auth.json"
            store = FileCredentialStore(path, source="host")
            provider = ManagedProvider(
                "test",
                (),
                ("TEST_API_KEY",),
                "https://provider.example/v1",
                "Test key",
                "test-model",
                credential_store=store,
            )

            credential = asyncio.run(
                provider.login(_args=[], prompt_secret=lambda _prompt: "sk-secret")
            )
            self.assertEqual(credential.source, "login")
            self.assertEqual(store.active_provider(), "test")
            resolved = provider.resolve()
            self.assertEqual(resolved.api_key, "sk-secret")
            self.assertEqual(resolved.source, "host")

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["test"]["type"], "api_key")
            self.assertEqual(data["active"], "test")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertTrue(provider.logout())
            self.assertEqual(store.active_provider(), "")
            self.assertIsNone(provider.resolve())

    def test_dynamic_store_path_and_client_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "one.json"
            store = FileCredentialStore(lambda: selected)
            store.save({"active": "deepseek"})
            self.assertEqual(store.active_provider(), "deepseek")
            selected = Path(tmp) / "two.json"
            self.assertEqual(store.load(), {})

        client = make_openai_client(
            "sk-test",
            "https://example.invalid/v1",
            application="host-app",
            version="1.2.3",
        )
        self.assertEqual(client.default_headers["User-Agent"], "host-app/1.2.3")
        asyncio.run(client.close())


if __name__ == "__main__":
    unittest.main()
