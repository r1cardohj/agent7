"""Standalone DeepSeek search integration with injectable host credentials."""

import unittest
from unittest.mock import patch

from agent0_runtime import Credential, DeepSeekSearchProvider
from agent0_runtime.web_search import search, skill_path


class RuntimeSearchTests(unittest.TestCase):
    def test_search_posts_ranked_query(self):
        seen = {}

        def post(url, payload, api_key, timeout):
            seen.update(url=url, payload=payload, api_key=api_key, timeout=timeout)
            return {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "summary": "A ranked result",
                        "rerank_score": 2.0,
                    }
                ]
            }

        with patch("agent0_runtime.web_search._post", side_effect=post):
            text = search(
                "runtime",
                count=3,
                api_key="sk-test",
                base_url="https://api.deepseek.com/v1",
            )

        self.assertEqual(seen["url"], "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(seen["payload"]["count"], 3)
        self.assertIn("Result", text)
        self.assertIn("A ranked result", text)

    def test_provider_uses_host_credentials_and_original_question(self):
        seen = {}
        credential = Credential("sk-host", "https://host.example", "host")

        def post(_url, payload, api_key, _timeout):
            seen.update(payload=payload, api_key=api_key)
            return {"results": []}

        provider = DeepSeekSearchProvider(lambda: credential)
        with patch("agent0_runtime.web_search._post", side_effect=post):
            provider.search("query", user_question="original question")

        self.assertEqual(seen["api_key"], "sk-host")
        self.assertEqual(seen["payload"]["ori_question"], "original question")

    def test_search_runbook_is_packaged(self):
        self.assertTrue(skill_path().is_file())
        self.assertIn("search", skill_path().read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
