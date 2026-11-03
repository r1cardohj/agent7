import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.session import (
    Session,
    fmt_limit,
    fmt_tokens,
    format_context_prompt,
    format_usage,
    resolve_context_limit,
)


def make_session(limit: int | None, prompt_tokens: int | None = None) -> Session:
    s = Session(model="m", cwd=Path("."), client=None)
    s.context_limit = limit
    s.prompt_tokens = prompt_tokens
    return s


class FmtLimitTests(unittest.TestCase):
    def test_limits_use_the_vendor_unit(self) -> None:
        cases = {
            1_048_576: "1M",      # deepseek-flash, sold as 1M
            1_000_000: "1M",      # AGENT7_CONTEXT_LIMIT=1000000
            2_097_152: "2M",
            262_144: "256k",      # kimi-for-coding
            256_000: "256k",      # grok-4
            131_072: "128k",      # grok-3, sold as 128k
            128_000: "128k",
        }
        for n, want in cases.items():
            with self.subTest(n=n):
                self.assertEqual(fmt_limit(n), want)

    def test_no_1049k(self) -> None:
        self.assertNotIn("1049", fmt_limit(1_048_576))

    def test_odd_limits_still_render(self) -> None:
        self.assertEqual(fmt_limit(1_500_000), "1.5M")
        self.assertEqual(fmt_limit(200_000), "200k")
        self.assertEqual(fmt_limit(4_868), "4,868")

    def test_token_counts_keep_decimal_units(self) -> None:
        # real usage is a count, not a label — 1,049k of 1M is honest here
        self.assertEqual(fmt_tokens(1_048_576), "1049k")


class ContextPromptTests(unittest.TestCase):
    def test_prompt_shows_the_vendor_label(self) -> None:
        self.assertEqual(format_context_prompt(make_session(1_048_576, 4_868)), "4,868/1M")
        self.assertEqual(format_context_prompt(make_session(262_144, 4_868)), "4,868/256k")

    def test_prompt_without_usage_or_limit(self) -> None:
        self.assertEqual(format_context_prompt(make_session(None, 4_868)), "4,868")
        self.assertEqual(format_context_prompt(make_session(1_048_576, None)), "?/1M")

    def test_status_line(self) -> None:
        self.assertIn("context_limit=1M", format_usage(make_session(1_048_576, 4_868)))
        self.assertIn("context_limit=256k", format_usage(make_session(262_144, None)))


class ResolveContextLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_known_models(self) -> None:
        self.assertEqual(resolve_context_limit("deepseek-flash"), 1_048_576)
        self.assertEqual(resolve_context_limit("kimi-for-coding"), 262_144)
        self.assertEqual(resolve_context_limit("grok-4"), 256_000)

    def test_limit_named_in_the_model(self) -> None:
        self.assertEqual(resolve_context_limit("qwen3-256k"), 256_000)
        self.assertEqual(resolve_context_limit("qwen3-1m"), 1_000_000)

    def test_env_override_wins(self) -> None:
        with patch.dict(os.environ, {"AGENT7_CONTEXT_LIMIT": "1_000_000"}):
            self.assertEqual(resolve_context_limit("deepseek-flash"), 1_000_000)
            self.assertEqual(fmt_limit(resolve_context_limit("deepseek-flash")), "1M")


if __name__ == "__main__":
    unittest.main()
