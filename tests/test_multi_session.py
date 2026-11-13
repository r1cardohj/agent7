"""Multiple isolated sessions, their picker, and REPL switching."""

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from agent7.agent import SYSTEM
from agent7.cli import Options, async_main
from agent7.commands import dispatch
from agent7.repl import pick_session, session_choices
from agent7.session import Session
from agent7.store import load, new_session_id, save


def snapshot_state(cwd: Path, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        cwd=cwd, model_ref="deepseek/test", max_turns=30,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"answer to {text}"},
        ],
    )


class SessionPickerTests(unittest.TestCase):
    def test_rows_show_names_and_return_the_id_not_the_row(self) -> None:
        snapshots = [
            {"id": "sess-old", "name": "old task"},
            {"id": "sess-current", "name": "readable current task"},
        ]
        items, ids = session_choices(snapshots, "sess-current")

        self.assertIn("readable current task", items[0])
        self.assertIn("sess-current", items[0])
        with patch("agent7.repl.Picker") as picker:
            picker.return_value.pick_async = AsyncMock(return_value=items[0])
            picked = asyncio.run(pick_session(snapshots, current="sess-current"))
        self.assertEqual(picked, "sess-current")
        self.assertEqual(ids[items[1]], "sess-old")

    def test_cancelling_the_picker_returns_none(self) -> None:
        with patch("agent7.repl.Picker") as picker:
            picker.return_value.pick_async = AsyncMock(return_value=None)
            picked = asyncio.run(pick_session([{"id": "sess-a", "name": "a"}]))
        self.assertIsNone(picked)


class SessionSwitchTests(unittest.TestCase):
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

    def test_session_command_switches_identity_history_and_save_target(self) -> None:
        first = snapshot_state(self.cwd, "first task")
        second = snapshot_state(self.cwd, "second task")
        save(first, created_at="2025-01-01T00:00:00+00:00")
        save(second, created_at="2025-01-02T00:00:00+00:00")
        state = Session(
            model="test", model_ref="deepseek/test", cwd=self.cwd, client=None,
            messages=list(first.messages), id=first.id, name=first.name,
            created_at=first.created_at, persisted=True, prompt_tokens=999,
        )

        with patch("agent7.repl.pick_session", new=AsyncMock(return_value=second.id)), \
             patch("agent7.repl.print_history") as history:
            result = asyncio.run(dispatch(state, "/session"))

        self.assertIsNotNone(result)
        self.assertEqual(state.id, second.id)
        self.assertEqual(state.name, "second task")
        self.assertEqual(state.created_at, "2025-01-02T00:00:00+00:00")
        self.assertEqual(state.messages[1]["content"], "second task")
        self.assertIsNone(state.prompt_tokens)
        history.assert_called_once_with(second.messages)

        state.messages.append({"role": "user", "content": "follow up"})
        save(state, created_at=state.created_at)
        self.assertEqual(load(self.cwd, first.id)["messages"], first.messages)
        self.assertEqual(load(self.cwd, second.id)["messages"][-1]["content"], "follow up")

    def test_session_command_accepts_an_id_without_opening_the_picker(self) -> None:
        first, second = snapshot_state(self.cwd, "first"), snapshot_state(self.cwd, "second")
        save(first, created_at="2025-01-01T00:00:00+00:00")
        save(second, created_at="2025-01-02T00:00:00+00:00")
        state = Session(
            model="test", model_ref="deepseek/test", cwd=self.cwd, client=None,
            messages=list(first.messages), id=first.id, persisted=True,
        )
        with patch("agent7.repl.pick_session", new=AsyncMock()) as picker, \
             patch("agent7.repl.print_history"):
            result = asyncio.run(dispatch(state, f"/session {second.id}"))
        picker.assert_not_awaited()
        self.assertIn("second", result.message)
        self.assertEqual(state.id, second.id)


class ResumeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")})
        self.env.start()
        self.saved = snapshot_state(self.cwd, "readable session name")
        save(self.saved, created_at="2025-01-01T00:00:00+00:00")
        self.client = SimpleNamespace(close=AsyncMock())
        self.provider = SimpleNamespace(id="deepseek", make_client=lambda cred: self.client)
        self.credential = SimpleNamespace(source="test")

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def patches(self, picker=None, run=None, prompt=None, output=None):
        """Common startup doubles; each test adds only its UI double."""
        return [
            patch("agent7.cli.resolve_all", return_value={"deepseek": self.credential}),
            patch("agent7.cli.resolve_default", return_value=(self.provider, self.credential)),
            patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])),
            patch("agent7.cli.prefetch_models"),
            patch("agent7.cli.can_pick_models", return_value=True),
            patch("agent7.cli.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))),
            patch("agent7.cli.pick_session", new=picker or AsyncMock()),
            patch("agent7.cli.run_turn", new=run or AsyncMock(return_value="done")),
            patch("agent7.cli.print_history"),
        ]

    def test_bare_resume_picks_and_direct_resume_uses_the_id(self) -> None:
        picker = AsyncMock(return_value=self.saved.id)
        run = AsyncMock(return_value="done")
        patches = self.patches(picker=picker, run=run)
        patches.append(patch("agent7.cli.repl_loop", new=AsyncMock()))
        for item in patches:
            item.start()
        try:
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd), "-r"])))
        finally:
            for item in reversed(patches):
                item.stop()
        picker.assert_awaited_once()
        self.assertEqual(picker.await_args.args[0][0]["name"], "readable session name")

        picker.reset_mock()
        patches = self.patches(picker=picker, run=run)
        for item in patches:
            item.start()
        try:
            asyncio.run(async_main(Options.parse([
                "--cwd", str(self.cwd), "-r", self.saved.id, "continue",
            ])))
        finally:
            for item in reversed(patches):
                item.stop()
        picker.assert_not_awaited()
        self.assertEqual(run.await_args.args[1], "continue")

    def test_resume_option_distinguishes_ids_from_legacy_task_text(self) -> None:
        session_id = new_session_id()
        direct = Options.parse(["-r", session_id, "continue"])
        self.assertEqual(direct.resume, session_id)
        self.assertEqual(direct.task, "continue")
        legacy = Options.parse(["-r", "continue"])
        self.assertIs(legacy.resume, True)
        self.assertEqual(legacy.task, "continue")
        self.assertIs(Options.parse(["-r"]).resume, True)

    def test_ctrl_d_prints_a_working_resume_command(self) -> None:
        stream = io.StringIO()
        output = Console(file=stream, width=120, color_system=None)
        patches = self.patches()
        prompt = MagicMock()
        prompt.prompt_async.side_effect = EOFError
        patches.extend([
            patch("agent7.cli.err", new=output),
            patch("agent7.repl.err", new=output),
            patch("agent7.repl.PromptSession", return_value=prompt),
        ])
        for item in patches:
            item.start()
        try:
            asyncio.run(async_main(Options.parse([
                "--cwd", str(self.cwd), "-r", self.saved.id,
            ])))
        finally:
            for item in reversed(patches):
                item.stop()
        self.assertIn(f"resume this session: agent7 -r {self.saved.id}", stream.getvalue())
        self.assertEqual(load(self.cwd, self.saved.id)["id"], self.saved.id)


if __name__ == "__main__":
    unittest.main()
