"""Tests for the DeepSeek Search API backend behind the search tool."""

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from agent7.agent import TOOLS, _last_user_question, dispatch_tool, tool_summary
from agent7.providers import Credential
from agent7.web_search import (
    DEFAULT_COUNT,
    GOOD_SCORE,
    MAX_CONTENT_CHARS,
    MAX_QUERY_CHARS,
    MAX_RESULTS_CHARS,
    MAX_SUMMARY_CHARS,
    _endpoint,
    _HttpError,
    search,
    skill_path,
)


def _tool_schema(name: str) -> dict:
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool["function"]
    raise AssertionError(f"{name} tool missing from TOOLS")


def _result(**overrides: object) -> dict:
    page = {
        "rerank_score": 2.4,
        "title": "DeepSeek V4 模型发布",
        "url": "https://example.com/deepseek-v4",
        "snippet": "DeepSeek V4 已上线。",
        "summary": "DeepSeek 发布了 V4 模型，改进了工具调用与长上下文能力。",
        "content": "",
        "site_name": "Example News",
        "site_icon": "https://example.com/favicon.ico",
        "date": "2026-06-01T00:00:00+00:00",
        "authority": 3,
    }
    page.update(overrides)
    return page


def _body(*results: dict) -> dict:
    return {"results": list(results)}


class _StubProvider:
    def __init__(self, credential: Credential | None) -> None:
        self._credential = credential

    def resolve(self) -> Credential | None:
        return self._credential


@contextmanager
def _with_key():
    credential = Credential("sk-test", "https://api.deepseek.com", "env")
    provider = _StubProvider(credential)
    with (
        patch("agent7.web_search.get_provider", return_value=provider),
        patch("agent7.providers.get_provider", return_value=provider),
    ):
        yield


class SearchRequestTests(unittest.TestCase):
    def test_posts_query_with_count_and_returns_ranked_text(self) -> None:
        seen: dict = {}

        def post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
            seen.update(url=url, payload=payload, api_key=api_key, timeout=timeout)
            return _body(_result())

        with _with_key(), patch("agent7.web_search._post", side_effect=post):
            text = search("DeepSeek V4 发布", count=5)

        self.assertEqual(seen["url"], "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(seen["api_key"], "sk-test")
        self.assertEqual(seen["payload"]["query"], "DeepSeek V4 发布")
        self.assertEqual(seen["payload"]["count"], 5)
        self.assertNotIn("need_content", seen["payload"])  # summaries only by default
        self.assertNotIn("ori_question", seen["payload"])
        self.assertIn("1. DeepSeek V4 模型发布", text)
        self.assertIn("score 2.4", text)
        self.assertIn("authority 3", text)
        self.assertIn("2026-06-01", text)
        self.assertIn("Example News", text)
        self.assertIn("https://example.com/deepseek-v4", text)
        self.assertIn("改进了工具调用与长上下文能力", text)

    def test_default_count_is_used_when_omitted(self) -> None:
        seen: dict = {}

        def post(_url: str, payload: dict, _api_key: str, _timeout: float) -> dict:
            seen.update(payload)
            return _body(_result())

        with _with_key(), patch("agent7.web_search._post", side_effect=post):
            search("rust async runtime")

        self.assertEqual(seen["count"], DEFAULT_COUNT)

    def test_low_scoring_results_are_hidden_with_a_note(self) -> None:
        body = _body(
            _result(title="Good"),
            _result(title="Noise", rerank_score=0.4, url="https://example.com/noise"),
        )
        with _with_key(), patch("agent7.web_search._post", return_value=body):
            text = search("q")

        self.assertIn("Good", text)
        self.assertNotIn("Noise", text)
        self.assertIn("1 lower-scoring results hidden", text)

    def test_all_low_scoring_results_still_show_a_few(self) -> None:
        body = _body(*[_result(title=f"p{i}", rerank_score=0.1) for i in range(6)])
        with _with_key(), patch("agent7.web_search._post", return_value=body):
            text = search("q")

        self.assertIn(f"no result scored >= {GOOD_SCORE}", text)
        self.assertIn("1. p0", text)
        self.assertIn("2. p1", text)
        self.assertNotIn("4. p3", text)

    def test_empty_results_say_so_and_invite_a_broader_query(self) -> None:
        with _with_key(), patch("agent7.web_search._post", return_value={"results": []}):
            text = search("nothing here site:example.com")

        self.assertIn("no results for", text)
        self.assertIn("broaden the query", text)

    def test_ori_question_is_forwarded_and_capped(self) -> None:
        seen: dict = {}

        def post(_url: str, payload: dict, _api_key: str, _timeout: float) -> dict:
            seen.update(payload)
            return _body(_result())

        with _with_key(), patch("agent7.web_search._post", side_effect=post):
            search("q", ori_question="x" * 300)

        self.assertEqual(len(seen["ori_question"]), MAX_QUERY_CHARS)

    def test_blank_ori_question_is_dropped(self) -> None:
        seen: dict = {}

        def post(_url: str, payload: dict, _api_key: str, _timeout: float) -> dict:
            seen.update(payload)
            return _body(_result())

        with _with_key(), patch("agent7.web_search._post", side_effect=post):
            search("q", ori_question="   ")

        self.assertNotIn("ori_question", seen)

    def test_need_content_adds_page_text(self) -> None:
        body = _body(_result(content="z" * (MAX_CONTENT_CHARS + 5)))
        with _with_key(), patch("agent7.web_search._post", return_value=body):
            text = search("q", need_content=True)

        self.assertIn("content truncated 5 chars", text)
        self.assertLess(len(text), MAX_RESULTS_CHARS + 200)

    def test_long_summaries_are_capped_per_result(self) -> None:
        body = _body(_result(summary="s" * (MAX_SUMMARY_CHARS + 7)))
        with _with_key(), patch("agent7.web_search._post", return_value=body):
            text = search("q")

        self.assertIn("summary truncated 7 chars", text)
        self.assertIn("fetch the URL", text)

    def test_output_is_capped(self) -> None:
        body = _body(*[_result(summary="s" * 2_000, title=f"p{i}") for i in range(10)])
        with _with_key(), patch("agent7.web_search._post", return_value=body):
            text = search("q")

        self.assertIn("...[truncated", text)
        self.assertLess(len(text), MAX_RESULTS_CHARS + 200)

    def test_query_and_count_are_validated_before_any_request(self) -> None:
        with _with_key(), patch("agent7.web_search._post") as post:
            self.assertIn("non-empty", search("   "))
            self.assertIn(f"at most {MAX_QUERY_CHARS}", search("q" * (MAX_QUERY_CHARS + 1)))
            self.assertIn("between 1 and 50", search("q", count=0))
            self.assertIn("between 1 and 50", search("q", count=51))
            self.assertIn("must be an integer", search("q", count="many"))

        post.assert_not_called()

    def test_rate_limit_is_retried_before_succeeding(self) -> None:
        failures = [_HttpError(429, "Too Many Requests")]
        calls = {"n": 0}

        def post(_url: str, _payload: dict, _api_key: str, _timeout: float) -> dict:
            calls["n"] += 1
            if failures:
                raise failures.pop()
            return _body(_result())

        with (
            _with_key(),
            patch("agent7.web_search._post", side_effect=post),
            patch("agent7.web_search.time.sleep") as sleep,
        ):
            text = search("q")

        self.assertEqual(calls["n"], 2)  # one retry, then success
        self.assertEqual(sleep.call_count, 1)
        self.assertIn("DeepSeek V4", text)

    def test_exhausted_retries_report_the_rate_limit(self) -> None:
        with (
            _with_key(),
            patch("agent7.web_search._post", side_effect=_HttpError(503, "upstream")),
            patch("agent7.web_search.time.sleep"),
        ):
            text = search("q")

        self.assertIn("503", text)
        self.assertIn("upstream", text)

    def test_payment_required_is_not_retried(self) -> None:
        calls = {"n": 0}

        def post(_url: str, _payload: dict, _api_key: str, _timeout: float) -> dict:
            calls["n"] += 1
            raise _HttpError(402, "Insufficient Balance")

        with (
            _with_key(),
            patch("agent7.web_search._post", side_effect=post),
            patch("agent7.web_search.time.sleep") as sleep,
        ):
            text = search("q")

        self.assertEqual(calls["n"], 1)
        sleep.assert_not_called()
        self.assertIn("402", text)
        self.assertIn("top up", text)

    def test_missing_key_explains_how_to_log_in(self) -> None:
        with (
            patch("agent7.web_search.get_provider", return_value=_StubProvider(None)),
            patch("agent7.web_search._post") as post,
        ):
            text = search("q")

        post.assert_not_called()
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertIn("/login deepseek", text)

    def test_endpoint_tolerates_a_versioned_base_url(self) -> None:
        self.assertEqual(
            _endpoint("https://api.deepseek.com/"), "https://api.deepseek.com/v1/tools/search"
        )
        self.assertEqual(
            _endpoint("https://api.deepseek.com/v1"), "https://api.deepseek.com/v1/tools/search"
        )


class HttpPostTests(unittest.TestCase):
    def test_error_body_message_is_surfaced(self) -> None:
        payload = json.dumps({"error": {"message": "'count' must be between 1 and 50"}}).encode()
        failure = urllib.error.HTTPError(
            "https://api.deepseek.com/v1/tools/search", 400, "Bad Request", {}, io.BytesIO(payload)
        )

        with patch("urllib.request.urlopen", side_effect=failure):
            text = search("q", api_key="sk-test", base_url="https://api.deepseek.com")

        self.assertIn("400", text)
        self.assertIn("'count' must be between 1 and 50", text)

    def test_transport_failures_are_retried_once_then_reported(self) -> None:
        with (
            patch(
                "urllib.request.urlopen", side_effect=urllib.error.URLError("dns failure")
            ) as open_url,
            patch("agent7.web_search.time.sleep"),
        ):
            text = search("q", api_key="sk-test", base_url="https://api.deepseek.com")

        self.assertEqual(open_url.call_count, 3)  # one attempt plus two retries
        self.assertIn("search request failed", text)
        self.assertIn("dns failure", text)

    def test_successful_response_is_parsed(self) -> None:
        body = json.dumps(_body(_result())).encode()

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return body

        with patch("urllib.request.urlopen", return_value=_Response()):
            text = search("DeepSeek V4", api_key="sk-test", base_url="https://api.deepseek.com")

        self.assertIn("1. DeepSeek V4 模型发布", text)


class SearchToolWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_schema_requires_query_and_bounds_count(self) -> None:
        schema = _tool_schema("search")
        props = schema["parameters"]["properties"]
        self.assertEqual(schema["parameters"]["required"], ["query"])
        self.assertEqual(props["query"]["maxLength"], MAX_QUERY_CHARS)
        self.assertEqual(props["count"]["minimum"], 1)
        self.assertEqual(props["count"]["maximum"], 50)
        self.assertIn("site:", schema["description"])
        self.assertIn(str(skill_path()), schema["description"])

    def test_bundled_skill_is_shipped_readable(self) -> None:
        text = skill_path().read_text(encoding="utf-8")
        self.assertTrue(skill_path().is_file())
        self.assertIn("search", text)
        self.assertIn("after:", text)
        self.assertIn("fetch", text)

    def test_tool_summary_shows_the_query(self) -> None:
        self.assertEqual(tool_summary("search", {"query": "grpc deadline"}), "grpc deadline")
        self.assertEqual(
            tool_summary("search", {"query": "grpc deadline", "count": 5}),
            "grpc deadline  (count=5)",
        )

    def test_dispatch_forwards_the_user_question(self) -> None:
        seen: dict = {}

        def post(_url: str, payload: dict, _api_key: str, _timeout: float) -> dict:
            seen.update(payload)
            return _body(_result())

        with _with_key(), patch("agent7.web_search._post", side_effect=post):
            result = dispatch_tool(
                self.cwd,
                "search",
                {"query": "latest deepseek release"},
                user_question="what did DeepSeek ship this week?",
            )

        self.assertEqual(seen["ori_question"], "what did DeepSeek ship this week?")
        self.assertIn("DeepSeek V4", result)

    def test_last_user_question_wins_over_tool_traffic(self) -> None:
        messages = [
            {"role": "user", "content": "first ask"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "output"},
            {"role": "user", "content": "latest ask"},
            {"role": "assistant", "content": "thinking"},
        ]
        self.assertEqual(_last_user_question(messages), "latest ask")
        self.assertIsNone(_last_user_question([{"role": "assistant", "content": "hi"}]))
        self.assertIsNone(_last_user_question([{"role": "user", "content": "   "}]))

    def test_dispatch_works_without_a_user_question(self) -> None:
        with _with_key(), patch("agent7.web_search._post", return_value=_body(_result())):
            result = dispatch_tool(self.cwd, "search", {"query": "q"})

        self.assertIn("DeepSeek V4", result)


if __name__ == "__main__":
    unittest.main()
