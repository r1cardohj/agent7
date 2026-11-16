"""Tests for the trafilatura-backed fetch tool."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.agent import TOOLS, dispatch_tool, tool_summary
from agent7.web_fetch import (
    DEFAULT_FETCH_CHARS,
    MAX_FETCH_CHARS,
    fetch,
    tool_fetch,
)


HTML = """<!doctype html>
<html><head><title>Agent Fetch</title></head><body>
<article>
<h1>Getting started</h1>
<p>Use fetch to turn web pages into model-readable Markdown text.</p>
<table><tr><td>Tool</td><td>Purpose</td></tr></table>
<pre><code>agent7 fetch https://example.com</code></pre>
</article>
</body></html>
"""


def _tool_schema(name: str) -> dict:
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool["function"]
    raise AssertionError(f"{name} tool missing from TOOLS")


class TrafilaturaFetchTests(unittest.TestCase):
    def test_extracts_markdown_and_strips_url_fragment(self) -> None:
        with patch("trafilatura.fetch_url", return_value=HTML) as download:
            result = fetch("https://example.com/docs#section")

        self.assertEqual(download.call_args.args, ("https://example.com/docs",))
        self.assertTrue(result.startswith("---\n"))
        self.assertIn("title: Getting started", result)
        self.assertIn("Getting started", result)
        self.assertIn("model-readable Markdown text", result)
        self.assertIn("agent7 fetch https://example.com", result)

    def test_uses_recall_fallback_when_precision_extraction_is_empty(self) -> None:
        calls: list[dict] = []

        def extract(downloaded: str, **kwargs: object) -> str | None:
            calls.append(kwargs)
            return "fallback text" if kwargs.get("favor_recall") else None

        with patch("trafilatura.fetch_url", return_value=HTML), \
             patch("trafilatura.extract", side_effect=extract):
            result = fetch("https://example.com/sparse")

        self.assertEqual(result, "fallback text")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("favor_recall", calls[0])
        self.assertIs(calls[1]["favor_recall"], True)

    def test_max_chars_truncates_output(self) -> None:
        with patch("trafilatura.fetch_url", return_value=HTML), \
             patch("trafilatura.extract", return_value="x" * 100):
            result = fetch("https://example.com/long", max_chars=10)

        self.assertEqual(result, "x" * 10 + "\n... [truncated 90 chars]")

    def test_limits_are_available_and_fetch_output_is_capped(self) -> None:
        self.assertEqual(DEFAULT_FETCH_CHARS, 12_000)
        self.assertEqual(MAX_FETCH_CHARS, 50_000)
        with patch("trafilatura.fetch_url", return_value=HTML), \
             patch("trafilatura.extract", return_value="y" * (MAX_FETCH_CHARS + 1)):
            result = fetch("https://example.com/long", max_chars=MAX_FETCH_CHARS)

        self.assertIn("truncated 1 chars", result)
        self.assertLess(len(result), MAX_FETCH_CHARS + 100)

    def test_fetch_failure_reports_error(self) -> None:
        with patch("trafilatura.fetch_url", return_value=None):
            self.assertEqual(fetch("https://example.com/missing"), "error: could not fetch URL")

    def test_unextractable_page_reports_error(self) -> None:
        with patch("trafilatura.fetch_url", return_value="<html></html>"), \
             patch("trafilatura.extract", return_value=None):
            self.assertEqual(
                fetch("https://example.com/empty"),
                "error: no readable text extracted",
            )

    def test_only_http_urls_are_accepted(self) -> None:
        for url in ("file:///etc/passwd", "ftp://example.com", "example.com", ""):
            with self.subTest(url=url):
                self.assertTrue(fetch(url).startswith("error: url must"))


class FetchToolWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fetch_schema_and_summary(self) -> None:
        schema = _tool_schema("fetch")
        props = schema["parameters"]["properties"]
        self.assertEqual(schema["parameters"]["required"], ["url"])
        self.assertEqual(props["max_chars"]["maximum"], MAX_FETCH_CHARS)
        self.assertEqual(tool_summary("fetch", {"url": "https://example.com/docs"}), "https://example.com/docs")

    def test_dispatch_runs_fetch_tool(self) -> None:
        with patch("trafilatura.fetch_url", return_value=HTML):
            result = dispatch_tool(
                self.cwd, "fetch", {"url": "https://example.com/docs"}
            )

        self.assertIn("title: Getting started", result)

    def test_handler_accepts_max_chars(self) -> None:
        with patch("trafilatura.fetch_url", return_value=HTML), \
             patch("trafilatura.extract", return_value="z" * 100):
            result = tool_fetch(self.cwd, "https://example.com/docs", max_chars=5)

        self.assertTrue(result.startswith("zzzzz"))
        self.assertIn("truncated 95 chars", result)


if __name__ == "__main__":
    unittest.main()
