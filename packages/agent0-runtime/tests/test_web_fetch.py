"""Tests for the proxy-aware fetch tool and trafilatura HTML extraction."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent0_runtime.web_fetch import (
    DEFAULT_FETCH_CHARS,
    MAX_FETCH_CHARS,
    _download,
    _Downloaded,
    _FetchError,
    _proxy_for,
    fetch,
    tool_fetch,
)
from urllib3 import HTTPHeaderDict

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


def _downloaded(
    body: str | bytes = HTML,
    media_type: str = "text/html",
    charset: str | None = "utf-8",
) -> _Downloaded:
    return _Downloaded(body.encode() if isinstance(body, str) else body, media_type, charset)


def _http_response(
    body: bytes = b"hello",
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> Mock:
    response = Mock()
    response.status = status
    response.headers = HTTPHeaderDict({"Content-Type": content_type})
    response.stream.return_value = iter([body])
    return response


class DownloadTests(unittest.TestCase):
    def test_download_uses_selected_proxy_and_bounded_timeouts(self) -> None:
        manager = Mock()
        manager.request.return_value = _http_response()
        with (
            patch("agent0_runtime.web_fetch._proxy_for", return_value="http://127.0.0.1:7890"),
            patch("agent0_runtime.web_fetch._pool", return_value=manager) as pool,
        ):
            result = _download("https://example.com/docs")

        pool.assert_called_once_with("http://127.0.0.1:7890")
        self.assertEqual(result, _Downloaded(b"hello", "text/plain", "utf-8"))
        request = manager.request.call_args
        self.assertEqual(request.args, ("GET", "https://example.com/docs"))
        self.assertFalse(request.kwargs["preload_content"])
        self.assertEqual(request.kwargs["timeout"].connect_timeout, 8)
        self.assertEqual(request.kwargs["timeout"].read_timeout, 20)

    def test_https_proxy_and_no_proxy_are_respected(self) -> None:
        proxies = {"http": "http://localhost:8000", "https": "http://localhost:7890"}
        with (
            patch("agent0_runtime.web_fetch.getproxies", return_value=proxies),
            patch("agent0_runtime.web_fetch.proxy_bypass", return_value=False),
        ):
            self.assertEqual(_proxy_for("https://example.com"), "http://localhost:7890")
        with (
            patch("agent0_runtime.web_fetch.getproxies", return_value=proxies),
            patch("agent0_runtime.web_fetch.proxy_bypass", return_value=True),
        ):
            self.assertIsNone(_proxy_for("https://example.com"))

    def test_http_status_is_reported_and_connection_is_released(self) -> None:
        response = _http_response(status=503)
        manager = Mock()
        manager.request.return_value = response
        with (
            patch("agent0_runtime.web_fetch._proxy_for", return_value=None),
            patch("agent0_runtime.web_fetch._pool", return_value=manager),
            self.assertRaisesRegex(_FetchError, "HTTP 503"),
        ):
            _download("https://example.com/unavailable")
        response.release_conn.assert_called_once_with()

    def test_oversized_response_is_rejected_and_connection_is_released(self) -> None:
        response = _http_response(b"12345")
        manager = Mock()
        manager.request.return_value = response
        with (
            patch("agent0_runtime.web_fetch.MAX_DOWNLOAD_BYTES", 4),
            patch("agent0_runtime.web_fetch._proxy_for", return_value=None),
            patch("agent0_runtime.web_fetch._pool", return_value=manager),
            self.assertRaisesRegex(_FetchError, "exceeds 4 bytes"),
        ):
            _download("https://example.com/large")
        response.release_conn.assert_called_once_with()


class TrafilaturaFetchTests(unittest.TestCase):
    def test_extracts_markdown_and_strips_url_fragment(self) -> None:
        with patch("agent0_runtime.web_fetch._download", return_value=_downloaded()) as download:
            result = fetch("https://example.com/docs#section")

        self.assertEqual(download.call_args.args, ("https://example.com/docs",))
        self.assertTrue(result.startswith("---\n"))
        self.assertIn("title: Getting started", result)
        self.assertIn("Getting started", result)
        self.assertIn("model-readable Markdown text", result)
        self.assertIn("agent7 fetch https://example.com", result)

    def test_uses_recall_fallback_when_precision_extraction_is_empty(self) -> None:
        calls: list[dict] = []

        def extract(_downloaded: bytes, **kwargs: object) -> str | None:
            calls.append(kwargs)
            return "fallback text" if kwargs.get("favor_recall") else None

        with (
            patch("agent0_runtime.web_fetch._download", return_value=_downloaded()),
            patch("trafilatura.extract", side_effect=extract),
        ):
            result = fetch("https://example.com/sparse")

        self.assertEqual(result, "fallback text")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("favor_recall", calls[0])
        self.assertIs(calls[1]["favor_recall"], True)

    def test_plain_text_and_json_bypass_html_extraction(self) -> None:
        cases = (
            ("# Raw README\n\nKeep *Markdown* unchanged.\n", "text/plain"),
            ('{"name": "agent7", "version": "0.3.1"}', "application/json"),
        )
        for body, media_type in cases:
            with (
                self.subTest(media_type=media_type),
                patch(
                    "agent0_runtime.web_fetch._download",
                    return_value=_downloaded(body, media_type),
                ),
                patch("trafilatura.extract") as extract,
            ):
                self.assertEqual(fetch("https://example.com/raw"), body)
                extract.assert_not_called()

    def test_html_extraction_failure_falls_back_to_clean_text(self) -> None:
        page = """<html><head><style>hidden style</style></head><body>
        <p>Readable fallback &amp; useful text.</p>
        <script>hidden script</script></body></html>"""
        with (
            patch("agent0_runtime.web_fetch._download", return_value=_downloaded(page)),
            patch("trafilatura.extract", return_value=None),
        ):
            result = fetch("https://example.com/fallback")

        self.assertEqual(result, "Readable fallback & useful text.")
        self.assertNotIn("hidden", result)

    def test_max_chars_truncates_output(self) -> None:
        with (
            patch("agent0_runtime.web_fetch._download", return_value=_downloaded()),
            patch("trafilatura.extract", return_value="x" * 100),
        ):
            result = fetch("https://example.com/long", max_chars=10)

        self.assertEqual(result, "x" * 10 + "\n... [truncated 90 chars]")

    def test_limits_are_available_and_fetch_output_is_capped(self) -> None:
        self.assertEqual(DEFAULT_FETCH_CHARS, 12_000)
        self.assertEqual(MAX_FETCH_CHARS, 50_000)
        with (
            patch("agent0_runtime.web_fetch._download", return_value=_downloaded()),
            patch("trafilatura.extract", return_value="y" * (MAX_FETCH_CHARS + 1)),
        ):
            result = fetch("https://example.com/long", max_chars=MAX_FETCH_CHARS)

        self.assertIn("truncated 1 chars", result)
        self.assertLess(len(result), MAX_FETCH_CHARS + 100)

    def test_fetch_failure_reports_error(self) -> None:
        with patch(
            "agent0_runtime.web_fetch._download",
            side_effect=_FetchError("could not fetch URL"),
        ):
            self.assertEqual(fetch("https://example.com/missing"), "error: could not fetch URL")

    def test_unextractable_page_reports_error(self) -> None:
        with (
            patch(
                "agent0_runtime.web_fetch._download",
                return_value=_downloaded("<html></html>"),
            ),
            patch("trafilatura.extract", return_value=None),
        ):
            self.assertEqual(
                fetch("https://example.com/empty"),
                "error: no readable text extracted",
            )

    def test_binary_content_is_rejected(self) -> None:
        with patch(
            "agent0_runtime.web_fetch._download",
            return_value=_downloaded(b"PDF", "application/pdf"),
        ):
            self.assertEqual(
                fetch("https://example.com/file"),
                "error: unsupported content type: application/pdf",
            )

    def test_only_http_urls_are_accepted(self) -> None:
        for url in ("file:///etc/passwd", "ftp://example.com", "example.com", ""):
            with self.subTest(url=url):
                self.assertTrue(fetch(url).startswith("error: url must"))


class FetchToolTests(unittest.TestCase):
    def test_handler_accepts_max_chars(self) -> None:
        with (
            patch("agent0_runtime.web_fetch._download", return_value=_downloaded()),
            patch("trafilatura.extract", return_value="z" * 100),
        ):
            result = tool_fetch(Path("."), "https://example.com/docs", max_chars=5)

        self.assertTrue(result.startswith("zzzzz"))
        self.assertIn("truncated 95 chars", result)


if __name__ == "__main__":
    unittest.main()
