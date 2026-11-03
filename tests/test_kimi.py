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


class KimiProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {"AGENT7_HOME": str(Path(self.tmp.name) / "config")}
        self.env = patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_kimi_is_registered(self) -> None:
        self.assertEqual(get_provider("kimi").id, "kimi")
        self.assertEqual(get_provider("KIMI").id, "kimi")
        self.assertIn("/login kimi", login_hints())

    def test_kimi_key_resolves_from_env(self) -> None:
        with patch.dict(os.environ, {"KIMI_API_KEY": "sk-kimi"}):
            cred = get_provider("kimi").resolve()
        self.assertEqual(cred.api_key, "sk-kimi")
        self.assertEqual(cred.base_url, "https://api.kimi.com/coding/v1")
        self.assertEqual(cred.source, "env")

    def test_other_provider_keys_do_not_authenticate_kimi(self) -> None:
        env = {"DEEPSEEK_API_KEY": "sk-123", "XAI_API_KEY": "xai-123"}
        with patch.dict(os.environ, env):
            self.assertIsNone(get_provider("kimi").resolve())

    def test_base_url_can_be_overridden(self) -> None:
        env = {"KIMI_API_KEY": "k", "AGENT7_BASE_URL": "http://local:1/v1"}
        with patch.dict(os.environ, env):
            self.assertEqual(get_provider("kimi").resolve().base_url, "http://local:1/v1")

    def test_default_model_when_models_list_is_unavailable(self) -> None:
        self.assertEqual(default_model_for("kimi"), "kimi-for-coding")
        self.assertEqual(default_model_for(None), "deepseek-chat")

    def test_bare_kimi_model_resolves_to_kimi(self) -> None:
        self.assertEqual(
            resolve_model_ref(fake_session(), "kimi-for-coding"),
            ("kimi", "kimi-for-coding", "kimi/kimi-for-coding"),
        )
        cache = ["kimi/kimi-for-coding-highspeed"]
        self.assertEqual(
            resolve_model_ref(fake_session(models_cache=cache), "kimi-for-coding-highspeed"),
            ("kimi", "kimi-for-coding-highspeed", "kimi/kimi-for-coding-highspeed"),
        )

    def test_kimi_context_limit(self) -> None:
        self.assertEqual(resolve_context_limit("kimi-for-coding"), 262_144)
        self.assertEqual(resolve_context_limit("kimi-for-coding-highspeed"), 262_144)


if __name__ == "__main__":
    unittest.main()
