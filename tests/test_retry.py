"""Transient failures are retried; a context overflow compacts and retries."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import agent7.agent as A
from agent7.agent import estimate_tokens, run_agent


class _Status(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        self.status_code = status
        super().__init__(message or f"status {status}")


def _chunk(*, content: str | None = None, usage: dict[str, int] | None = None) -> Any:
    delta = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, model_extra=None
    )
    choices = [SimpleNamespace(delta=delta)] if content is not None else []
    return SimpleNamespace(choices=choices, usage=SimpleNamespace(**usage) if usage else None)


class _Stream:
    def __init__(self, chunks: list[Any], fail: BaseException | None = None) -> None:
        self._chunks, self._fail = chunks, fail

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk
            if self._fail is not None:
                raise self._fail

        return gen()

    async def close(self) -> None:
        pass


class _Client:
    """Script items: an exception to raise, or chunks to stream (optionally failing at the end)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> _Stream:
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        chunks, fail = item if isinstance(item, tuple) else (item, None)
        return _Stream(chunks, fail)


class RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        patcher = patch.object(A, "RETRY_SLEEPS", (0, 0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_agent(self, script: list[Any], **kwargs: Any) -> tuple[str, _Client]:
        client = _Client(script)
        kwargs.setdefault("on_tool", lambda *a: None)
        answer = asyncio.run(
            run_agent(
                "go", cwd=self.cwd, client=client, model="test-model",
                messages=[], **kwargs,
            )
        )
        return answer, client

    def test_rate_limit_is_retried(self) -> None:
        answer, client = self.run_agent([_Status(429), [_chunk(content="ok")]])

        self.assertEqual(answer, "ok")
        self.assertEqual(client.calls, 2)

    def test_connection_error_is_retried(self) -> None:
        class APIConnectionError(Exception):
            pass

        answer, client = self.run_agent([APIConnectionError("boom"), [_chunk(content="ok")]])

        self.assertEqual(answer, "ok")
        self.assertEqual(client.calls, 2)

    def test_retry_budget_is_finite(self) -> None:
        with self.assertRaises(_Status):
            self.run_agent([_Status(503)])

    def test_non_retryable_error_is_raised(self) -> None:
        with self.assertRaises(_Status) as caught:
            self.run_agent([_Status(400, "invalid request: bad tool schema")])

        self.assertEqual(caught.exception.status_code, 400)

    def test_stream_that_already_printed_is_not_replayed(self) -> None:
        seen: list[str] = []
        with self.assertRaises(_Status):
            self.run_agent(
                [([_chunk(content="partial")], _Status(503))],
                on_delta=seen.append,
            )

        self.assertEqual(seen, ["partial"])

    def test_stream_options_is_dropped_only_for_bad_param_errors(self) -> None:
        bad = _Status(400, "Unrecognized request argument supplied: stream_options")
        answer, client = self.run_agent([bad, [_chunk(content="ok")]])

        self.assertEqual(answer, "ok")
        self.assertEqual(client.calls, 2)


class CompactionFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_agent(
        self, script: list[Any], seed: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> tuple[str, list[int]]:
        client = _Client(script)
        compacted: list[int] = []
        kwargs.setdefault("on_tool", lambda *a: None)
        answer = asyncio.run(
            run_agent(
                "go", cwd=self.cwd, client=client, model="test-model",
                messages=list(seed or []), on_compact=compacted.append, **kwargs,
            )
        )
        return answer, compacted

    @staticmethod
    def history(results: int) -> list[dict[str, Any]]:
        """Old tool output big enough for `compact_messages` to shrink."""
        return [
            {"role": "user", "content": "earlier turn"},
            *[
                {"role": "tool", "tool_call_id": f"c{i}", "content": "z" * 3_000}
                for i in range(results)
            ],
        ]

    def test_missing_usage_still_triggers_the_compaction_check(self) -> None:
        """No usage chunk at all: the 75% check must fall back to an estimate."""
        calls: list[list[dict[str, Any]]] = []
        real = A.compact_messages

        def spy(messages: list[dict[str, Any]], **kwargs: Any):
            calls.append(messages)
            return real(messages, **kwargs)

        client = _Client([[_chunk(content="a")]])
        with patch.object(A, "compact_messages", side_effect=spy):
            asyncio.run(
                run_agent(
                    "go", cwd=self.cwd, client=client, model="test-model",
                    messages=[], context_limit=1, on_tool=lambda *a: None,
                )
            )

        self.assertTrue(calls)

    def test_context_overflow_compacts_and_retries_the_turn(self) -> None:
        overflow = _Status(400, "This model's maximum context length is 8192 tokens")

        answer, compacted = self.run_agent(
            [overflow, [_chunk(content="done")]], seed=self.history(3)
        )

        self.assertEqual(answer, "done")
        self.assertEqual(compacted, [3])  # every old tool result was shrunk, then the turn retried


class EstimateTokensTests(unittest.TestCase):
    def test_counts_content_and_tool_arguments(self) -> None:
        messages = [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "read", "arguments": json.dumps({"path": "y" * 400})}}
            ]},
        ]

        self.assertGreaterEqual(estimate_tokens(messages), 200)

    def test_empty_history_is_non_zero(self) -> None:
        self.assertEqual(estimate_tokens([]), 1)


if __name__ == "__main__":
    unittest.main()
