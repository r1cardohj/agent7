"""A finished tool call must never be the last thing you see: empty/capped runs get a note."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

import agent7.repl as repl
from agent7.repl import NO_TEXT_NOTICE, TRUNCATED_NOTICE, reply_notice


def _chunk(
    content: str | None = None, tool_calls: list[Any] | None = None, finish_reason: str | None = None
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, model_extra=None)
    choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)] if (
        content is not None or tool_calls or finish_reason
    ) else []
    return SimpleNamespace(choices=choices, usage=None)


def _bash(cmd: str) -> Any:
    call = SimpleNamespace(
        index=0, id="call_1", function=SimpleNamespace(name="bash", arguments=json.dumps({"command": cmd}))
    )
    return _chunk(tool_calls=[call])


class _Stream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()

    async def close(self) -> None:
        pass


class _Client:
    """Serves the queued turns; once they run out it keeps returning empty replies."""

    def __init__(self, turns: list[list[Any]]) -> None:
        self._turns = list(turns)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> _Stream:
        return _Stream(self._turns.pop(0) if self._turns else [_chunk(content="")])


class ReplyNoticeTests(unittest.TestCase):
    def test_text_answer_needs_no_note(self) -> None:
        self.assertIsNone(reply_notice("all done"))

    def test_empty_answer_gets_the_no_text_note(self) -> None:
        self.assertEqual(reply_notice(""), NO_TEXT_NOTICE)

    def test_whitespace_answer_gets_the_no_text_note(self) -> None:
        self.assertEqual(reply_notice("\n  \n"), NO_TEXT_NOTICE)

    def test_turn_cap_is_reported_with_recovery_guidance(self) -> None:
        notice = reply_notice("(stopped after 30 turns)")
        self.assertIsNotNone(notice)
        assert notice is not None
        self.assertIn("(stopped after 30 turns)", notice)
        self.assertIn("task may be unfinished", notice)
        self.assertIn("session is preserved", notice)
        self.assertIn('agent7 -r "continue"', notice)
        self.assertIn('agent7 --max-turns 500 -r "continue"', notice)

    def test_output_limit_gets_its_own_note(self) -> None:
        self.assertEqual(reply_notice("", "length"), TRUNCATED_NOTICE)

    def test_other_finish_reasons_keep_the_generic_note(self) -> None:
        self.assertEqual(reply_notice("", "stop"), NO_TEXT_NOTICE)


class RunOnceSilenceTests(unittest.TestCase):
    """`_run_once` used to swallow both cases: you saw the tool lines, then nothing."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.orig = repl.err, repl.console
        self.stream = io.StringIO()
        repl.err = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        repl.console = Console(file=self.stream, width=120, color_system=None, force_terminal=False)

    def tearDown(self) -> None:
        repl.err, repl.console = self.orig
        self.tmp.cleanup()

    def out(self) -> str:
        return self.stream.getvalue()

    def run_once(self, turns: list[list[Any]], **kwargs: Any) -> str:
        session = SimpleNamespace(
            model="test-model", cwd=self.cwd, client=_Client(turns),
            max_turns=kwargs.get("max_turns", 30), messages=[],
            context_limit=None, prompt_tokens=None, update_usage=lambda u: None,
        )
        return asyncio.run(repl.run_turn(session, "go"))

    def test_empty_reply_after_tool_call_says_so(self) -> None:
        answer = self.run_once([[_bash("echo hi")], [_chunk(content="")]])

        self.assertEqual(answer, "")
        self.assertIn("✓ bash", self.out())
        self.assertIn("no text in the reply", self.out())

    def test_whitespace_reply_after_tool_call_says_so(self) -> None:
        self.run_once([[_bash("echo hi")], [_chunk(content="\n \n")]])
        self.assertIn("no text in the reply", self.out())

    def test_truncated_reply_says_so(self) -> None:
        self.run_once([[_bash("echo hi")], [_chunk(content="", finish_reason="length")]])
        self.assertIn("output token limit", self.out())

    def test_turn_cap_is_printed_with_recovery_guidance(self) -> None:
        self.run_once([[_bash("echo 1")], [_bash("echo 2")], [_bash("echo 3")]], max_turns=2)
        self.assertIn("(stopped after 2 turns)", self.out())
        self.assertIn("task may be unfinished", self.out())
        self.assertIn("session is preserved", self.out())
        self.assertIn('agent7 -r "continue"', self.out())
        self.assertIn('agent7 --max-turns 500 -r "continue"', self.out())

    def test_normal_answer_prints_no_note(self) -> None:
        self.run_once([[_bash("echo hi")], [_chunk(content="done")]])
        self.assertIn("done", self.out())
        self.assertNotIn("no text in the reply", self.out())
        self.assertNotIn("stopped after", self.out())


if __name__ == "__main__":
    unittest.main()
