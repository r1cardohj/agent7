"""End-to-end run_agent test: the preview callback is wired through the tool loop."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent7.agent import run_agent


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: Any = None,
) -> Any:
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=None, model_extra=None
    )
    choices = [SimpleNamespace(delta=delta)] if (content is not None or tool_calls) else []
    return SimpleNamespace(choices=choices, usage=usage)


def _tool_call_chunk(name: str, arguments: dict[str, Any]) -> Any:
    return _chunk(
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="call_1",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )
        ]
    )


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def close(self) -> None:
        pass


class _FakeCompletions:
    def __init__(self, turns: list[list[Any]]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeStream:
        self.requests.append(kwargs)
        return _FakeStream(self._turns.pop(0))


class _FakeClient:
    def __init__(self, turns: list[list[Any]]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(turns))


class RunAgentPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.target = self.cwd / "app.py"
        self.target.write_text("value = 'old'\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, turns, **kwargs):
        client = _FakeClient(turns)
        messages: list[dict[str, Any]] = []
        kwargs.setdefault("on_tool", lambda *a: None)  # keep stderr quiet
        answer = asyncio.run(
            run_agent(
                "change it",
                cwd=self.cwd,
                client=client,
                model="test-model",
                messages=messages,
                **kwargs,
            )
        )
        return answer, messages, client

    def test_edit_preview_reaches_callback_and_file_is_written(self) -> None:
        previews: list[str] = []
        seen_tools: list[tuple[int, str, dict[str, Any], str]] = []
        turns = [
            [
                _tool_call_chunk(
                    "edit",
                    {
                        "path": "app.py",
                        "edits": [{"old_string": "'old'", "new_string": "'new'"}],
                    },
                )
            ],
            [_chunk(content="done")],
        ]

        answer, messages, client = self._run(
            turns, on_preview=previews.append, on_tool=lambda *a: seen_tools.append(a)
        )

        self.assertEqual(answer, "done")
        self.assertEqual(self.target.read_text(), "value = 'new'\n")
        self.assertEqual(previews, [f"edit  {self.target.resolve()}  +1 -1"])
        self.assertEqual(len(seen_tools), 1)
        self.assertEqual(seen_tools[0][1], "edit")
        self.assertTrue(seen_tools[0][3].startswith("edited "))
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_preview_is_optional(self) -> None:
        turns = [
            [_tool_call_chunk("write", {"path": "new.txt", "content": "hi\n"})],
            [_chunk(content="done")],
        ]

        answer, _, _ = self._run(turns)

        self.assertEqual(answer, "done")
        self.assertEqual((self.cwd / "new.txt").read_text(), "hi\n")

    def test_tool_error_does_not_preview(self) -> None:
        previews: list[str] = []
        turns = [
            [
                _tool_call_chunk(
                    "edit",
                    {
                        "path": "app.py",
                        "edits": [{"old_string": "missing", "new_string": "x"}],
                    },
                )
            ],
            [_chunk(content="gave up")],
        ]

        answer, messages, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "gave up")
        self.assertEqual(previews, [])
        self.assertEqual(self.target.read_text(), "value = 'old'\n")
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("occurs 0 times", tool_msgs[0]["content"])

    def test_whole_file_write_previews_with_zero_deletions(self) -> None:
        previews: list[str] = []
        turns = [
            [_tool_call_chunk("write", {"path": "fresh.txt", "content": "one\ntwo\n"})],
            [_chunk(content="ok")],
        ]

        answer, _, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "ok")
        self.assertEqual((self.cwd / "fresh.txt").read_text(), "one\ntwo\n")
        self.assertEqual(previews, [f"write  {(self.cwd / 'fresh.txt').resolve()}  +2 -0"])

    def test_noop_write_reports_no_changes_without_preview(self) -> None:
        previews: list[str] = []
        turns = [
            [_tool_call_chunk("write", {"path": "app.py", "content": "value = 'old'\n"})],
            [_chunk(content="nothing to do")],
        ]

        answer, messages, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "nothing to do")
        self.assertEqual(previews, [])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertTrue(tool_msgs[0]["content"].startswith("no changes:"))


if __name__ == "__main__":
    unittest.main()
