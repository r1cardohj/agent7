import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rich.console import Console

from agent7.__main__ import _parse_args, async_main, print_history
from agent7.agent import SYSTEM
from agent7.store import load, save


class SessionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_resume_is_a_flag_and_session_is_just_a_task_word(self) -> None:
        self.assertFalse(_parse_args([]).resume)
        self.assertTrue(_parse_args(["-r"]).resume)
        self.assertEqual(_parse_args(["-r", "继续"]).task, ["继续"])
        self.assertEqual(_parse_args(["session", "docs"]).task, ["session", "docs"])

    def test_new_repl_overwrites_the_workspace_snapshot(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        with patch("agent7.__main__.resolve_all", return_value={}), \
             patch("agent7.__main__.resolve_default", return_value=None), \
             patch("agent7.__main__.AsyncOpenAI") as client, \
             patch("agent7.__main__.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(["--cwd", str(self.cwd)]))
        snapshots = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(snapshots), 1)
        data = load(self.cwd)
        self.assertEqual(data["messages"], [{"role": "system", "content": SYSTEM}])

    def test_ctrl_c_discards_input_and_returns_to_the_prompt(self) -> None:
        with patch("agent7.__main__.resolve_all", return_value={}), \
             patch("agent7.__main__.resolve_default", return_value=None), \
             patch("agent7.__main__.AsyncOpenAI") as client, \
             patch("agent7.__main__.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = [KeyboardInterrupt, EOFError]
            asyncio.run(async_main(["--cwd", str(self.cwd)]))
        self.assertEqual(prompt.return_value.prompt_async.call_count, 2)

    def test_new_session_reuses_last_model_with_fresh_context(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="kimi/kimi-for-coding", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        client = SimpleNamespace(close=AsyncMock())
        provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
        credential = SimpleNamespace(source="test")
        switch = AsyncMock(return_value=SimpleNamespace(message="ok"))
        with patch("agent7.__main__.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.__main__.resolve_default", return_value=(provider, credential)), \
             patch("agent7.__main__.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.__main__.switch_model", new=switch), \
             patch("agent7.__main__.PromptSession") as prompt:
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(["--cwd", str(self.cwd)]))
        switch.assert_awaited_once()
        self.assertEqual(switch.await_args.args[1], "kimi/kimi-for-coding")
        self.assertEqual(load(self.cwd)["messages"], [{"role": "system", "content": SYSTEM}])

    def test_history_shows_conversation_text_without_internal_messages(self) -> None:
        stream = io.StringIO()
        with patch("agent7.__main__.console", new=Console(file=stream, width=120, color_system=None)):
            print_history([
                {"role": "system", "content": "hidden system"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "tool", "content": "hidden tool result"},
                {"role": "assistant", "content": "", "tool_calls": []},
            ])
        text = stream.getvalue()
        self.assertIn("previous conversation", text)
        self.assertIn("old question", text)
        self.assertIn("old answer", text)
        self.assertNotIn("hidden system", text)
        self.assertNotIn("hidden tool result", text)

    def test_resume_prints_saved_history_before_one_shot(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        client = SimpleNamespace(close=AsyncMock())
        provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
        credential = SimpleNamespace(source="test")
        with patch("agent7.__main__.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.__main__.resolve_default", return_value=(provider, credential)), \
             patch("agent7.__main__.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.__main__.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))), \
             patch("agent7.__main__.print_history") as history:
            asyncio.run(async_main(["--cwd", str(self.cwd), "-r", "继续"]))
        history.assert_called_once_with(state.messages)

    def test_new_session_does_not_print_previous_history(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        with patch("agent7.__main__.resolve_all", return_value={}), \
             patch("agent7.__main__.resolve_default", return_value=None), \
             patch("agent7.__main__.AsyncOpenAI") as client, \
             patch("agent7.__main__.PromptSession") as prompt, \
             patch("agent7.__main__.print_history") as history:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(["--cwd", str(self.cwd)]))
        history.assert_not_called()

    def test_resume_passes_saved_context_to_agent(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        seen = []
        async def fake_run(session, task, **kwargs):
            seen.append((list(session.messages), task))
            kwargs["on_saved"]()
            return "done"
        client = SimpleNamespace(close=AsyncMock())
        provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
        credential = SimpleNamespace(source="test")
        with patch("agent7.__main__.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.__main__.resolve_default", return_value=(provider, credential)), \
             patch("agent7.__main__.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.__main__.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))), \
             patch("agent7.__main__._run_once", new=fake_run):
            asyncio.run(async_main(["--cwd", str(self.cwd), "-r", "继续"]))
        self.assertEqual(seen[0][0][1:], [{"role": "user", "content": "old"}])
        self.assertEqual(seen[0][1], "继续")
        snapshots = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(load(self.cwd)["messages"][1]["content"], "old")


if __name__ == "__main__":
    unittest.main()
