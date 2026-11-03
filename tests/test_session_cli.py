import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent7.__main__ import _parse_args, async_main
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

    def test_resume_defaults_to_latest_and_accepts_an_id(self) -> None:
        self.assertIsNone(_parse_args([]).resume)
        self.assertEqual(_parse_args(["-r"]).resume, "latest")
        self.assertEqual(_parse_args(["-r", "abcdef123456"]).resume, "abcdef123456")
        self.assertEqual(_parse_args(["session", "-l"]).session_list, True)
        self.assertEqual(_parse_args(["--cwd", "/tmp", "session", "-l"]).session_list, True)
        self.assertEqual(_parse_args(["session", "修复认证"]).session_name, "修复认证")
        self.assertEqual(_parse_args(["fix", "session", "docs"]).task, ["fix", "session", "docs"])

    def test_list_works_without_credentials(self) -> None:
        state = type("State", (), {
            "cwd": self.cwd, "model_ref": "deepseek/test", "max_turns": 30,
            "messages": [{"role": "system", "content": SYSTEM}],
        })()
        path = save("abcdef123456", state, name="修复认证", created_at="2025-01-01T00:00:00+00:00")
        with patch("agent7.__main__.print") as output:
            import asyncio
            asyncio.run(async_main(["--cwd", str(self.cwd), "session", "-l"]))
        rows = [call.args[0] for call in output.call_args_list]
        self.assertEqual(rows[0], "ID\tNAME\tPATH")
        self.assertEqual(rows[1], f"abcdef123456\t修复认证\t{path}")

    def test_named_session_enters_repl_and_saves_name(self) -> None:
        import asyncio
        with patch("agent7.__main__.resolve_all", return_value={}), \
             patch("agent7.__main__.resolve_default", return_value=None), \
             patch("agent7.__main__.AsyncOpenAI") as client, \
             patch("agent7.__main__.PromptSession") as prompt:
            client.return_value.close = AsyncMock()
            prompt.return_value.prompt_async.side_effect = EOFError
            asyncio.run(async_main(["--cwd", str(self.cwd), "session", "修复认证"]))
        snapshots = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(load(snapshots[0].stem)["name"], "修复认证")
    def test_resume_passes_saved_context_to_agent_and_keeps_identity(self) -> None:
        state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=30,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "old"}],
        )
        path = save("abcdef123456", state, name="旧会话", created_at="2025-01-01T00:00:00+00:00")
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
            asyncio.run(async_main(["--cwd", str(self.cwd), "-r", "abcdef123456", "继续"]))
        self.assertEqual(seen[0][0][1:], [{"role": "user", "content": "old"}])
        self.assertEqual(seen[0][1], "继续")
        self.assertEqual(len(list(path.parent.glob("*.json"))), 1)
        self.assertEqual(load("abcdef123456")["name"], "旧会话")


if __name__ == "__main__":
    unittest.main()
