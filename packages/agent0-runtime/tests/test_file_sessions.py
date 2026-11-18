"""Standalone workspace-scoped JSON session storage."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent0_runtime import FileSessionStore


class FileSessionStoreTests(unittest.TestCase):
    def test_save_list_resume_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            cwd = Path(tmp) / "workspace"
            cwd.mkdir()
            store = FileSessionStore(root)
            session = SimpleNamespace(
                id="",
                cwd=cwd,
                model="deepseek-chat",
                model_ref="deepseek/deepseek-chat",
                max_turns=20,
                messages=[{"role": "user", "content": "hello runtime"}],
            )

            path = store.save(session)
            self.assertTrue(path.is_file())
            self.assertTrue(session.persisted)
            self.assertEqual(session.name, "hello runtime")
            loaded = store.load(cwd, session.id)
            self.assertEqual(loaded["messages"], session.messages)
            self.assertEqual(store.list(cwd)[0]["id"], session.id)

            trashed = store.delete(cwd, session.id)
            self.assertTrue(trashed.is_file())
            self.assertFalse(path.exists())

            # Recreating an ID must not overwrite its earlier trash entry.
            store.save(session)
            second = store.delete(cwd, session.id)
            self.assertEqual(second.name, f"{session.id}.2.json")
            self.assertEqual(store.list(cwd), [])

    def test_resume_refreshes_system_and_repairs_pending_tool(self):
        snapshot = {
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call-1", "function": {"name": "echo", "arguments": "{}"}}
                    ],
                },
            ]
        }
        messages = FileSessionStore.resume_messages(snapshot, "new")
        self.assertEqual(messages[0]["content"], "new")
        self.assertEqual(messages[-1]["tool_call_id"], "call-1")
        self.assertIn("interrupted", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
