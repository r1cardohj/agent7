"""Turn budget: a sane default, per-run semantics, and retries that cost nothing."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from rich.console import Console

import agent7.__main__ as cli
from agent7.__main__ import _parse_args, resolve_max_turns
from agent7.agent import DEFAULT_MAX_TURNS
from agent7.session import Session


def _turn(content: str = "", **extra: Any) -> list[Any]:
    delta = SimpleNamespace(
        content=content or None, tool_calls=None, reasoning_content=None, model_extra=None
    )
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None, **extra)]


def _bash(cmd: str) -> list[Any]:
    call = SimpleNamespace(
        index=0, id=f"call_{cmd}", function=SimpleNamespace(name="bash", arguments=json.dumps({"command": cmd}))
    )
    delta = SimpleNamespace(content=None, tool_calls=[call], reasoning_content=None, model_extra=None)
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)]


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
    def __init__(self, turns: list[list[Any]]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> _Stream:
        self.requests.append(kwargs)
        return _Stream(self._turns.pop(0) if len(self._turns) > 1 else self._turns[0])


class TurnBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_agent(self, turns: list[list[Any]], **kwargs: Any) -> tuple[str, _Client]:
        client = _Client(turns)
        answer = asyncio.run(
            cli.run_agent(
                "go", cwd=self.cwd, client=client, model="test-model", messages=[],
                on_tool=lambda *a: None, on_compact=lambda n: None, **kwargs,
            )
        )
        return answer, client

    def test_default_budget_is_high_enough_for_real_tasks(self) -> None:
        self.assertGreaterEqual(DEFAULT_MAX_TURNS, 100)
        self.assertEqual(Session(model="m", cwd=self.cwd, client=None).max_turns, DEFAULT_MAX_TURNS)

    def test_default_budget_is_used_when_no_flag_is_given(self) -> None:
        answer, client = self.run_agent([_turn("done")])

        self.assertEqual(answer, "done")
        self.assertEqual(len(client.requests), 1)

    def test_long_tool_chain_is_not_cut_short(self) -> None:
        """31 tool rounds used to be impossible; they are ordinary work now."""
        answer, client = self.run_agent(
            [_bash(f"echo {i}") for i in range(31)] + [_turn("done")]
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(client.requests), 32)

    def test_explicit_budget_still_caps_the_loop(self) -> None:
        answer, client = self.run_agent(
            [_bash(f"echo {i}") for i in range(5)], max_turns=3
        )

        self.assertEqual(answer, "(stopped after 3 turns)")
        self.assertEqual(len(client.requests), 3)

    def test_compaction_retry_does_not_spend_a_turn(self) -> None:
        """A 400 overflow retry is a retry, not progress: the budget must survive it."""

        class Overflow(Exception):
            status_code = 400

        big = [{"role": "user", "content": "earlier"}]
        big += [{"role": "tool", "tool_call_id": f"c{i}", "content": "z" * 3_000} for i in range(3)]

        client = _Client([_turn("done")])
        calls = {"n": 0}
        original = client.create

        async def flaky(**kwargs: Any) -> _Stream:
            calls["n"] += 1
            if calls["n"] == 1:
                raise Overflow("maximum context length is 8192 tokens")
            return await original(**kwargs)

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=flaky))
        answer = asyncio.run(
            cli.run_agent(
                "go", cwd=self.cwd, client=client, model="test-model", messages=list(big),
                max_turns=1, on_tool=lambda *a: None, on_compact=lambda n: None,
            )
        )

        self.assertEqual(answer, "done")  # budget of 1 survived the retry
        self.assertEqual(calls["n"], 2)

    def test_persistent_overflow_raises_instead_of_looping(self) -> None:
        class Overflow(Exception):
            status_code = 400

        def create(**kwargs: Any) -> None:
            raise Overflow("maximum context length exceeded")

        client = _Client([_turn("done")])
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        big = [{"role": "tool", "tool_call_id": "c", "content": "z" * 9_000}]

        with self.assertRaises(Overflow):
            asyncio.run(
                cli.run_agent(
                    "go", cwd=self.cwd, client=client, model="test-model", messages=big,
                    max_turns=5, on_tool=lambda *a: None, on_compact=lambda n: None,
                )
            )


class ResolveMaxTurnsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("AGENT7_MAX_TURNS", None)

    def tearDown(self) -> None:
        self.env.stop()

    def test_default(self) -> None:
        self.assertEqual(resolve_max_turns(_parse_args([])), DEFAULT_MAX_TURNS)

    def test_env_var(self) -> None:
        os.environ["AGENT7_MAX_TURNS"] = "500"

        self.assertEqual(resolve_max_turns(_parse_args([])), 500)

    def test_flag_beats_env(self) -> None:
        os.environ["AGENT7_MAX_TURNS"] = "500"

        self.assertEqual(resolve_max_turns(_parse_args(["--max-turns", "7"])), 7)

    def test_bad_env_value_is_ignored(self) -> None:
        os.environ["AGENT7_MAX_TURNS"] = "lots"
        stream = io.StringIO()
        with patch.object(cli, "err", Console(file=stream, width=200, color_system=None)):
            self.assertEqual(resolve_max_turns(_parse_args([])), DEFAULT_MAX_TURNS)
        self.assertIn("ignoring bad AGENT7_MAX_TURNS", stream.getvalue())

    def test_zero_and_negative_are_rejected_on_the_command_line(self) -> None:
        for raw in ("0", "-3"):
            with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()):
                _parse_args(["--max-turns", raw])
            self.assertEqual(caught.exception.code, 2)

    def test_help_mentions_the_default(self) -> None:
        stream = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stream), \
             patch("sys.stdout", stream):
            _parse_args(["--help"])
        self.assertIn(str(DEFAULT_MAX_TURNS), stream.getvalue())


class ResumeBudgetTests(unittest.TestCase):
    """Sessions written by an older build stored max_turns=30; resuming must not honour it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop("AGENT7_MAX_TURNS", None)

        old = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "old"}],
        )
        cli.save(old, created_at="2025-01-01T00:00:00+00:00")

    def resumed_max_turns(self, argv: list[str]) -> int:
        seen: list[int] = []

        async def fake_run(session: Session, task: str, **kwargs: Any) -> str:
            seen.append(session.max_turns)
            return "done"

        provider = SimpleNamespace(id="deepseek", make_client=lambda cred: _FakeClient())
        with patch.object(cli, "resolve_all", return_value={"deepseek": object()}), \
             patch.object(cli, "resolve_default", return_value=(provider, SimpleNamespace(source="t"))), \
             patch.object(cli, "fetch_models", new=_async_empty), \
             patch.object(cli, "switch_model", new=_async_ok), \
             patch.object(cli, "_run_once", new=fake_run):
            asyncio.run(cli.async_main(["--cwd", str(self.cwd), "-r", "continue", *argv]))
        return seen[0]

    def test_stale_snapshot_value_is_ignored(self) -> None:
        self.assertEqual(self.resumed_max_turns([]), DEFAULT_MAX_TURNS)

    def test_flag_still_wins_on_resume(self) -> None:
        self.assertEqual(self.resumed_max_turns(["--max-turns", "9"]), 9)


class _FakeClient:
    async def close(self) -> None:
        pass


async def _async_empty(*args: Any, **kwargs: Any) -> list[str]:
    return []

async def _async_ok(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(message="ok")


if __name__ == "__main__":
    unittest.main()
