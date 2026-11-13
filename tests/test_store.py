import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent7.agent import SYSTEM
from agent7.store import (
    VERSION,
    list_sessions,
    load,
    resume,
    resume_messages,
    save,
    session_path,
    workspace_key,
)


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

    def save(self, state=None, **kwargs) -> Path:
        return save(state or self.state, created_at="2025-01-01T00:00:00+00:00", **kwargs)

    def test_round_trip_has_no_credentials_and_private_permissions(self) -> None:
        path = self.save()
        data = load(self.cwd)
        self.assertEqual(data["messages"][1]["content"], "你好")
        self.assertEqual(data["path"], path)
        self.assertEqual(path, session_path(self.cwd))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        text = path.read_text()
        self.assertNotIn("credential", text)
        self.assertNotIn("api_key", text)
        self.assertIn('"id"', text)
        self.assertEqual(data["name"], "你好")

    def test_one_snapshot_per_workspace_and_other_cwds_stay_separate(self) -> None:
        self.save()
        self.state.messages.append({"role": "assistant", "content": "好"})
        self.save()
        files = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(load(self.cwd)["messages"][-1]["content"], "好")

        other_cwd = self.root / "other"
        other_cwd.mkdir()
        other = SimpleNamespace(**vars(self.state))
        other.cwd = other_cwd
        other.id = ""  # A different session must not reuse the original identity.
        self.save(other)
        files = list((self.root / "state" / "agent7" / "sessions").glob("*.json"))
        self.assertEqual(len(files), 2)
        self.assertEqual(Path(resume(self.cwd)["cwd"]), self.cwd.resolve())
        self.assertEqual(Path(resume(other_cwd)["cwd"]), other_cwd.resolve())
        self.assertNotEqual(workspace_key(self.cwd), workspace_key(other_cwd))

    def test_same_workspace_sessions_have_distinct_files_and_names(self) -> None:
        other = SimpleNamespace(**vars(self.state))
        other.id = ""
        other.name = ""
        other.messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "  修复   多会话\n存储问题  "},
        ]
        first_path = self.save()
        second_path = self.save(other)

        self.assertNotEqual(self.state.id, other.id)
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(load(self.cwd, self.state.id)["messages"], self.state.messages)
        self.assertEqual(load(self.cwd, other.id)["name"], "修复 多会话 存储问题")
        self.assertEqual(len(list_sessions(self.cwd)), 2)
        self.assertEqual(len(list(session_path(self.cwd).parent.glob("*.json"))), 2)

    def test_name_is_compact_and_limited_to_a_readable_length(self) -> None:
        self.state.messages[1]["content"] = "  first   line\n" + "x" * 100
        data = load(self.cwd, self.save().stem)
        self.assertTrue(data["name"].startswith("first line"))
        self.assertEqual(len(data["name"]), 60)
        self.assertTrue(data["name"].endswith("…"))

    def test_missing_corrupt_fail_without_overwriting(self) -> None:
        with self.assertRaisesRegex(ValueError, "no session"):
            resume(self.cwd)
        path = self.save()
        path.write_text("{broken")
        with self.assertRaisesRegex(ValueError, "cannot read"):
            resume(self.cwd)
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
        path = self.save()
        self.assertEqual(list(path.parent.iterdir()), [path])

    def _write_snapshot(self, version: int) -> Path:
        path = session_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": version,
            "cwd": str(self.cwd.resolve()),
            "model_ref": "deepseek/test",
            "max_turns": 12,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "messages": [{"role": "user", "content": "old"}],
        }), encoding="utf-8")
        return path

    def test_save_writes_responses_snapshot_version(self) -> None:
        path = self.save()
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], VERSION)
        self.assertEqual(VERSION, 2)

    def test_version_1_snapshot_still_loads(self) -> None:
        self._write_snapshot(1)
        self.assertEqual(load(self.cwd)["messages"][0]["content"], "old")

    def test_legacy_snapshot_gets_a_stable_id_and_migrates_on_save(self) -> None:
        legacy_path = self._write_snapshot(2)
        first = load(self.cwd)
        self.assertEqual(first["id"], load(self.cwd)["id"])
        self.assertEqual(first["name"], "old")
        self.assertEqual(resume(self.cwd, first["id"])["path"], legacy_path)

        migrated = SimpleNamespace(**first)
        migrated.cwd = self.cwd
        path = save(migrated, created_at=first["created_at"])
        self.assertNotEqual(path, legacy_path)
        self.assertFalse(legacy_path.exists())
        self.assertEqual(load(self.cwd, first["id"])["path"], path)

    def test_unknown_snapshot_version_is_refused(self) -> None:
        self._write_snapshot(3)
        with self.assertRaisesRegex(ValueError, "unsupported session format"):
            load(self.cwd)


if __name__ == "__main__":
    unittest.main()
