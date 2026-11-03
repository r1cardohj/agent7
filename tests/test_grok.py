import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent7.commands import resolve_model_ref
from agent7.providers import default_model_for, get_provider, login_hints
from agent7.session import resolve_context_limit


def fake_session(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**{"models_cache": None, "provider_id": None, **kwargs})


class GrokProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {"AGENT7_HOME": str(Path(self.tmp.name) / "config")}
        self.env = patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_grok_is_an_alias_for_xai(self) -> None:
        self.assertEqual(get_provider("grok").id, "xai")
        self.assertEqual(get_provider("XAI").id, "xai")
        self.assertIn("/login grok", login_hints())

    def test_grok_key_resolves_from_env(self) -> None:
        provider = get_provider("grok")
        for env in ("GROK_API_KEY", "XAI_API_KEY"):
            with self.subTest(env=env):
                with patch.dict(os.environ, {env: "k-123"}):
                    cred = provider.resolve()
                self.assertEqual(cred.api_key, "k-123")
                self.assertEqual(cred.base_url, "https://api.x.ai/v1")
                self.assertEqual(cred.source, "env")

    def test_deepseek_key_does_not_authenticate_xai(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-123"}):
            self.assertIsNone(get_provider("grok").resolve())

    def test_base_url_can_be_overridden(self) -> None:
        env = {"GROK_API_KEY": "k", "AGENT7_BASE_URL": "http://local:1/v1"}
        with patch.dict(os.environ, env):
            self.assertEqual(get_provider("grok").resolve().base_url, "http://local:1/v1")

    def test_default_model_when_models_list_is_unavailable(self) -> None:
        self.assertEqual(default_model_for("xai"), "grok-4")
        self.assertEqual(default_model_for("grok"), "grok-4")
        self.assertEqual(default_model_for("deepseek"), "deepseek-chat")
        self.assertEqual(default_model_for(None), "deepseek-chat")

    def test_bare_grok_model_resolves_to_xai(self) -> None:
        self.assertEqual(
            resolve_model_ref(fake_session(), "grok-4"),
            ("xai", "grok-4", "xai/grok-4"),
        )
        cache = ["xai/grok-4-fast"]
        self.assertEqual(
            resolve_model_ref(fake_session(models_cache=cache), "grok-4-fast"),
            ("xai", "grok-4-fast", "xai/grok-4-fast"),
        )

    def test_grok_context_limit(self) -> None:
        self.assertEqual(resolve_context_limit("grok-4"), 256_000)
        self.assertEqual(resolve_context_limit("grok-4-fast"), 256_000)
        self.assertEqual(resolve_context_limit("grok-3"), 131_072)


if __name__ == "__main__":
    unittest.main()
