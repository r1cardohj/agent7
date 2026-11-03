import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent7.__main__ import _parse_args, async_main
from agent7.agent import DEEPSEEK_RL_SYSTEM, SYSTEM
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
        # default model is deepseek-chat → frozen RL persona
        self.assertEqual(data["messages"], [{"role": "system", "content": DEEPSEEK_RL_SYSTEM}])

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
