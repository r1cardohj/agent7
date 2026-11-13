import asyncio
import io
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_toolkit.keys import Keys
from rich.console import Console

from agent7.agent import SYSTEM, Interrupted
from agent7.cli import Options, async_main, main
from agent7.repl import print_history, repl_key_bindings, run_turn
from agent7.store import load, save


class SessionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")})
        self.env.start()
        self.prefetch_patch = patch("agent7.cli.prefetch_models")
        self.prefetch = self.prefetch_patch.start()
        self.addCleanup(self.prefetch_patch.stop)

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def _binding(self, key: Keys):
        return repl_key_bindings().get_bindings_for_keys((key,))[-1]

    def test_enter_accepts_multiline_input(self) -> None:
        event = MagicMock()
        self._binding(Keys.ControlM).handler(event)
        event.current_buffer.validate_and_handle.assert_called_once_with()

    def test_ctrl_e_returns_from_editor_without_submitting(self) -> None:
        event = MagicMock()
        self._binding(Keys.ControlE).handler(event)
        event.current_buffer.open_in_editor.assert_called_once_with(validate_and_handle=False)
        event.current_buffer.validate_and_handle.assert_not_called()

    def test_resume_is_a_flag_and_session_is_just_a_task_word(self) -> None:
        self.assertFalse(Options.parse([]).resume)
        self.assertTrue(Options.parse(["-r"]).resume)
        self.assertEqual(Options.parse(["-r", "继续"]).task, "继续")
        self.assertEqual(Options.parse(["session", "docs"]).task, "session docs")
        self.assertEqual(Options.parse(["--max-turns", "9"]).max_turns, 9)

    def test_empty_new_repl_preserves_the_workspace_snapshot(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        path = save(state, created_at="2025-01-01T00:00:00+00:00")
        original = path.read_bytes()
        with patch("agent7.cli.resolve_all", return_value={}), \
             patch("agent7.cli.resolve_default", return_value=None), \
             patch("agent7.cli.AsyncOpenAI") as client, \
             patch("agent7.repl.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(load(self.cwd)["messages"], state.messages)

    def test_empty_new_repl_does_not_create_a_snapshot(self) -> None:
        with patch("agent7.cli.resolve_all", return_value={}), \
             patch("agent7.cli.resolve_default", return_value=None), \
             patch("agent7.cli.AsyncOpenAI") as client, \
             patch("agent7.repl.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
        self.assertFalse((self.root / "state" / "agent7" / "sessions").exists())

    def test_ctrl_c_discards_input_and_returns_to_the_prompt(self) -> None:
        with patch("agent7.cli.resolve_all", return_value={}), \
             patch("agent7.cli.resolve_default", return_value=None), \
             patch("agent7.cli.AsyncOpenAI") as client, \
             patch("agent7.repl.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = [KeyboardInterrupt, EOFError]
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
        self.assertEqual(prompt.return_value.prompt_async.call_count, 2)

    def test_repeated_ctrl_c_cancels_turn_without_escalating(self) -> None:
        started = asyncio.Event()
        saved = []

        async def stuck_agent(session, task):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                try:  # mimic a cancelled network stream's asynchronous cleanup
                    await asyncio.Event().wait()
                except asyncio.CancelledError as e:
                    raise Interrupted from e
                raise Interrupted

        async def exercise() -> None:
            before = signal.getsignal(signal.SIGINT)
            state = SimpleNamespace()
            with patch("agent7.repl._run_agent_once", new=stuck_agent):
                task = asyncio.create_task(run_turn(state, "go", on_saved=lambda: saved.append(True)))
                await started.wait()
                os.kill(os.getpid(), signal.SIGINT)
                await asyncio.sleep(0)
                os.kill(os.getpid(), signal.SIGINT)
                with self.assertRaises(Interrupted):
                    await task
            self.assertIs(signal.getsignal(signal.SIGINT), before)

        asyncio.run(exercise())
        self.assertEqual(saved, [True])

    def test_one_shot_repeated_ctrl_c_is_handled(self) -> None:
        started = asyncio.Event()

        async def stuck_agent(session, task):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.Event().wait()

        async def exercise() -> None:
            async def interrupt_twice() -> None:
                await started.wait()
                os.kill(os.getpid(), signal.SIGINT)
                await asyncio.sleep(0)
                os.kill(os.getpid(), signal.SIGINT)

            client = SimpleNamespace(close=AsyncMock())
            provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
            credential = SimpleNamespace(source="test")
            with patch("agent7.cli.resolve_all", return_value={"deepseek": credential}), \
                 patch("agent7.cli.resolve_default", return_value=(provider, credential)), \
                 patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])), \
                 patch("agent7.cli.switch_model", new=AsyncMock(
                     return_value=SimpleNamespace(message="ok")
                 )), \
                 patch("agent7.repl._run_agent_once", new=stuck_agent):
                sender = asyncio.create_task(interrupt_twice())
                try:
                    await async_main(Options.parse(["--cwd", str(self.cwd), "go"]))
                finally:
                    await sender

        stream = io.StringIO()
        with patch("agent7.cli.err", new=Console(file=stream, color_system=None)):
            asyncio.run(exercise())
        self.assertIn("# interrupted", stream.getvalue())

    def test_top_level_ctrl_c_is_a_notice_not_a_traceback(self) -> None:
        def interrupt(coro):
            coro.close()
            raise KeyboardInterrupt
        with patch("agent7.cli.asyncio.run", side_effect=interrupt), \
             patch("agent7.cli.err") as err:
            # click reports a clean exit once the command returns.
            with self.assertRaises(SystemExit):
                main([])
        err.print.assert_called_once_with("\n[dim]# interrupted[/dim]")

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
        with patch("agent7.cli.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.cli.resolve_default", return_value=(provider, credential)), \
             patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.cli.switch_model", new=switch), \
             patch("agent7.repl.PromptSession") as prompt:
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
        switch.assert_awaited_once()
        self.assertEqual(switch.await_args.args[1], "kimi/kimi-for-coding")
        self.assertEqual(load(self.cwd)["messages"], state.messages)

    def test_new_repl_replaces_the_snapshot_after_an_agent_task(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")

        async def fake_run(session, task, **kwargs):
            session.messages.extend([
                {"role": "user", "content": task},
                {"role": "assistant", "content": "new answer"},
            ])
            kwargs["on_saved"]()
            return "new answer"

        inputs = iter(["new task", EOFError()])

        async def fake_prompt(*args, **kwargs):
            value = next(inputs)
            if isinstance(value, BaseException):
                raise value
            return value

        client = SimpleNamespace(close=AsyncMock())
        provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
        credential = SimpleNamespace(source="test")
        with patch("agent7.cli.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.cli.resolve_default", return_value=(provider, credential)), \
             patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.cli.switch_model", new=AsyncMock(
                 return_value=SimpleNamespace(message="ok")
             )), \
             patch("agent7.repl.PromptSession") as prompt, \
             patch("agent7.repl.run_turn", new=fake_run):
            prompt.return_value.prompt_async.side_effect = fake_prompt
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
        self.assertEqual(load(self.cwd)["messages"][1:], [
            {"role": "user", "content": "new task"},
            {"role": "assistant", "content": "new answer"},
        ])

    def test_history_shows_conversation_text_without_internal_messages(self) -> None:
        stream = io.StringIO()
        with patch("agent7.repl.console", new=Console(file=stream, width=120, color_system=None)):
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
        with patch("agent7.cli.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.cli.resolve_default", return_value=(provider, credential)), \
             patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.cli.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))), \
             patch("agent7.cli.print_history") as history:
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd), "-r", "继续"])))
        history.assert_called_once_with(state.messages)

    def test_new_session_does_not_print_previous_history(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        save(state, created_at="2025-01-01T00:00:00+00:00")
        with patch("agent7.cli.resolve_all", return_value={}), \
             patch("agent7.cli.resolve_default", return_value=None), \
             patch("agent7.cli.AsyncOpenAI") as client, \
             patch("agent7.repl.PromptSession") as prompt, \
             patch("agent7.cli.print_history") as history:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd)])))
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
        with patch("agent7.cli.resolve_all", return_value={"deepseek": credential}), \
             patch("agent7.cli.resolve_default", return_value=(provider, credential)), \
             patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])), \
             patch("agent7.cli.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))), \
             patch("agent7.cli.run_turn", new=fake_run):
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd), "-r", "继续"])))
        self.assertEqual(seen[0][0][1:], [{"role": "user", "content": "old"}])
        self.assertEqual(seen[0][1], "继续")
        snapshots = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(load(self.cwd)["messages"][1]["content"], "old")


if __name__ == "__main__":
    unittest.main()
