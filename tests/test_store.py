import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent7.agent import SYSTEM
from agent7.store import list_sessions, load, resume, resume_messages, save


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")})
        self.env.start()
        self.state = SimpleNamespace(
            cwd=self.cwd, model_ref="deepseek/test", max_turns=12,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "你好"}],
        )

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def save(self, session_id: str, **kwargs) -> Path:
        return save(session_id, self.state, created_at="2025-01-01T00:00:00+00:00", **kwargs)

    def test_round_trip_has_no_credentials_and_private_permissions(self) -> None:
        path = self.save("abcdef123456", name="修复认证")
        data = load("abcdef123456", cwd=self.cwd)
        self.assertEqual(data["name"], "修复认证")
        self.assertEqual(data["messages"][1]["content"], "你好")
        self.assertEqual(data["path"], path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertNotIn("credential", path.read_text())
        self.assertNotIn("api_key", path.read_text())

    def test_latest_is_scoped_to_workspace_and_sorted_by_update_time(self) -> None:
        self.save("aaaaaaaaaaaa", name="old")
        self.save("bbbbbbbbbbbb", name="new")
        paths = sorted((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        for path in paths:
            data = json.loads(path.read_text())
            data["updated_at"] = "2025-01-0" + ("2" if data["id"] == "bbbbbbbbbbbb" else "1") + "T00:00:00+00:00"
            path.write_text(json.dumps(data))
        other = SimpleNamespace(**vars(self.state))
        other.cwd = self.root / "other"
        save("cccccccccccc", other, created_at="2025-01-01T00:00:00+00:00")
        self.assertEqual(resume(self.cwd)["id"], "bbbbbbbbbbbb")
        self.assertEqual([s["id"] for s in list_sessions(self.cwd)], ["bbbbbbbbbbbb", "aaaaaaaaaaaa"])
        with self.assertRaisesRegex(ValueError, "another workspace"):
            load("cccccccccccc", cwd=self.cwd)

    def test_missing_corrupt_and_bad_id_fail_without_overwriting(self) -> None:
        with self.assertRaisesRegex(ValueError, "no sessions"):
            resume(self.cwd)
        with self.assertRaisesRegex(ValueError, "invalid session id"):
            load("../auth", cwd=self.cwd)
        path = self.save("abcdef123456")
        path.write_text("{broken")
        with self.assertRaisesRegex(ValueError, "cannot read"):
            resume(self.cwd, "abcdef123456")
        self.assertEqual(path.read_text(), "{broken")

    def test_resume_refreshes_system_and_preserves_completed_tools(self) -> None:
        self.state.messages = [
            {"role": "system", "content": "old system"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "done", "type": "function"},
            ]},
            {"role": "tool", "tool_call_id": "done", "content": "ok"},
        ]
        messages = resume_messages({"messages": self.state.messages}, SYSTEM)
        self.assertEqual(messages, [{"role": "system", "content": SYSTEM}, *self.state.messages[1:]])

    def test_resume_closes_orphaned_tool_calls(self) -> None:
        self.state.messages = [
            {"role": "system", "content": "old system"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "pending", "type": "function"},
            ]},
        ]
        messages = resume_messages({"messages": self.state.messages}, SYSTEM)
        self.assertEqual(messages[0]["content"], SYSTEM)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "pending")
        self.assertEqual(len([m for m in messages if m.get("tool_call_id") == "pending"]), 1)

    def test_resume_repairs_partially_answered_final_tool_group(self) -> None:
        self.state.messages = [
            {"role": "system", "content": "old system"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "done", "type": "function"},
                {"id": "pending", "type": "function"},
            ]},
            {"role": "tool", "tool_call_id": "done", "content": "ok"},
        ]
        messages = resume_messages({"messages": self.state.messages}, SYSTEM)
        self.assertEqual([m.get("tool_call_id") for m in messages[2:]], ["done", "pending"])
        self.assertIn("interrupted", messages[-1]["content"])

    def test_resume_omits_orphan_tool_result_without_its_assistant_call(self) -> None:
        self.state.messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "lost", "content": "ok"},
        ]
        messages = resume_messages({"messages": self.state.messages}, SYSTEM)
        self.assertEqual(messages, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "go"},
        ])

    def test_atomic_save_leaves_no_temporary_file(self) -> None:
        path = self.save("abcdef123456")
        self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
