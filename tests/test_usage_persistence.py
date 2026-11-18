"""Usage travels with a saved conversation: /status and the gauge after -r."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent7.cli import Options, async_main
from agent7.config import SYSTEM
from agent7.file_sessions import session_usage
from agent7.session import Session, format_usage
from agent7.store import load, resume_messages, save


class UsageSnapshotTests(unittest.TestCase):
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

    def _session(self) -> Session:
        session = Session(model="test", cwd=self.cwd, client=None, context_limit=100_000)
        session.messages[:] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "old question"},
        ]
        session.update_usage({"prompt_tokens": 4_868, "completion_tokens": 210})
        session.record_run_usage(
            {"prompt_tokens": 9_000, "completion_tokens": 500, "total_tokens": 9_500}, rounds=2
        )
        return session

    def test_save_stores_the_context_and_the_spend(self) -> None:
        save(self._session())

        usage = load(self.cwd)["usage"]

        self.assertEqual(usage["last"]["prompt_tokens"], 4_868)
        self.assertEqual(usage["session"]["total_tokens"], 9_500)
        self.assertEqual(usage["session"]["rounds"], 2)

    def test_resume_restores_the_saved_numbers(self) -> None:
        save(self._session())
        resumed = Session(model="test", cwd=self.cwd, client=None, context_limit=100_000)

        resumed.restore_usage(load(self.cwd)["usage"])

        self.assertEqual(resumed.prompt_tokens, 4_868)
        self.assertEqual(resumed.spent_tokens, 9_500)
        self.assertFalse(resumed.context_estimated)
        text = format_usage(resumed)
        self.assertIn("context=4,868/100k", text)
        self.assertIn("session=9,500", text)

    def test_a_snapshot_without_usage_falls_back_to_an_estimate(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd,
            model_ref="deepseek/test",
            max_turns=30,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "x" * 400},
            ],
        )
        save(state)  # no prompt_tokens/totals on the object at all
        snapshot = load(self.cwd)
        resumed = Session(model="test", cwd=self.cwd, client=None, context_limit=100_000)
        resumed.messages[:] = resume_messages(snapshot)

        resumed.restore_usage(snapshot.get("usage"))

        self.assertIsNone(resumed.prompt_tokens)
        self.assertTrue(resumed.context_estimated)
        self.assertGreater(resumed.context_tokens, 0)
        self.assertIsNone(resumed.spent_tokens)
        self.assertIn("session=unknown", format_usage(resumed))

    def test_malformed_usage_blocks_are_ignored(self) -> None:
        blocks = (None, [], "nope", {"last": "x", "session": 5}, {"session": {"total_tokens": "9"}})
        for block in blocks:
            with self.subTest(block=block):
                session = Session(model="test", cwd=self.cwd, client=None)
                session.messages[:] = [{"role": "user", "content": "hi"}]
                session.update_usage({"prompt_tokens": 99})

                session.restore_usage(block)

                self.assertIsNone(session.prompt_tokens)
                self.assertIsNone(session.spent_tokens)

    def test_session_usage_survives_a_lightweight_state(self) -> None:
        state = SimpleNamespace(cwd=self.cwd, model_ref="m", max_turns=30, messages=[])

        usage = session_usage(state)

        self.assertEqual(usage["last"]["prompt_tokens"], None)
        self.assertEqual(usage["session"]["rounds"], 0)


class ResumeCliUsageTests(unittest.TestCase):
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

    def test_resume_passes_the_saved_usage_into_the_turn(self) -> None:
        session = Session(model="test", cwd=self.cwd, client=None, context_limit=100_000)
        session.messages[:] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "old"},
        ]
        session.update_usage({"prompt_tokens": 4_868})
        session.record_run_usage({"prompt_tokens": 9_000, "total_tokens": 9_500}, rounds=2)
        save(session, created_at="2025-01-01T00:00:00+00:00")
        seen = []

        async def fake_run(state, _task, **kwargs):
            seen.append((state.prompt_tokens, state.spent_tokens, state.session_tokens.rounds))
            kwargs["on_saved"]()
            return "done"

        client = SimpleNamespace(close=AsyncMock())
        provider = SimpleNamespace(id="deepseek", make_client=lambda _cred: client)
        credential = SimpleNamespace(source="test")
        with (
            patch("agent7.cli.resolve_all", return_value={"deepseek": credential}),
            patch("agent7.cli.resolve_default", return_value=(provider, credential)),
            patch("agent7.cli.fetch_models", new=AsyncMock(return_value=[])),
            patch(
                "agent7.cli.switch_model", new=AsyncMock(return_value=SimpleNamespace(message="ok"))
            ),
            patch("agent7.cli.run_turn", new=fake_run),
        ):
            asyncio.run(async_main(Options.parse(["--cwd", str(self.cwd), "-r", "继续"])))

        self.assertEqual(seen, [(4_868, 9_500, 2)])


if __name__ == "__main__":
    unittest.main()
