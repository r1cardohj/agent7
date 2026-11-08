"""End-to-end run_agent test: the preview callback is wired through the tool loop."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent7.agent import run_agent
from responses_fakes import FakeClient, call_turn, text_turn


class RunAgentPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.target = self.cwd / "app.py"
        self.target.write_text("value = 'old'\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, turns, **kwargs):
        client = FakeClient(turns)
        messages: list[dict[str, Any]] = []
        kwargs.setdefault("on_tool", lambda *a: None)  # keep stderr quiet
        answer = asyncio.run(
            run_agent(
                "change it",
                cwd=self.cwd,
                client=client,
                model="test-model",
                messages=messages,
                **kwargs,
            )
        )
        return answer, messages, client

    def test_edit_preview_reaches_callback_and_file_is_written(self) -> None:
        previews: list[str] = []
        seen_tools: list[tuple[int, str, dict[str, Any], str]] = []
        turns = [
            call_turn(("edit", {
                "path": "app.py",
                "edits": [{"old_string": "'old'", "new_string": "'new'"}],
            })),
            text_turn("done"),
        ]

        answer, messages, client = self._run(
            turns, on_preview=previews.append, on_tool=lambda *a: seen_tools.append(a)
        )

        self.assertEqual(answer, "done")
        self.assertEqual(self.target.read_text(), "value = 'new'\n")
        self.assertEqual(previews, [f"edit  {self.target.resolve()}  +1 -1"])
        self.assertEqual(len(seen_tools), 1)
        self.assertEqual(seen_tools[0][1], "edit")
        self.assertTrue(seen_tools[0][3].startswith("edited "))
        self.assertEqual(len(client.responses.requests), 2)

    def test_tool_start_event_precedes_execution_and_result(self) -> None:
        events: list[tuple[object, ...]] = []
        turns = [
            call_turn(("bash", {"command": "printf done"})),
            text_turn("done"),
        ]

        self._run(
            turns,
            on_tool_start=lambda turn, name, args: events.append(("start", turn, name)),
            on_tool=lambda turn, name, args, result: events.append(("finish", turn, name)),
        )

        self.assertEqual(events, [("start", 1, "bash"), ("finish", 1, "bash")])

    def test_each_model_round_including_the_one_after_edit_has_a_start_event(self) -> None:
        events: list[object] = []
        turns = [
            call_turn(("edit", {
                "path": "app.py",
                "edits": [{"old_string": "'old'", "new_string": "'new'"}],
            })),
            text_turn("done"),
        ]

        self._run(
            turns,
            on_model_start=lambda turn: events.append(("model", turn)),
            on_preview=lambda stats: events.append("edit finished"),
        )

        self.assertEqual(events, [("model", 1), "edit finished", ("model", 2)])

    def test_preview_is_optional(self) -> None:
        turns = [
            call_turn(("write", {"path": "new.txt", "content": "hi\n"})),
            text_turn("done"),
        ]

        answer, _, _ = self._run(turns)

        self.assertEqual(answer, "done")
        self.assertEqual((self.cwd / "new.txt").read_text(), "hi\n")

    def test_tool_error_does_not_preview(self) -> None:
        previews: list[str] = []
        turns = [
            call_turn(("edit", {
                "path": "app.py",
                "edits": [{"old_string": "missing", "new_string": "x"}],
            })),
            text_turn("gave up"),
        ]

        answer, messages, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "gave up")
        self.assertEqual(previews, [])
        self.assertEqual(self.target.read_text(), "value = 'old'\n")
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("occurs 0 times", tool_msgs[0]["content"])

    def test_whole_file_write_previews_with_zero_deletions(self) -> None:
        previews: list[str] = []
        turns = [
            call_turn(("write", {"path": "fresh.txt", "content": "one\ntwo\n"})),
            text_turn("ok"),
        ]

        answer, _, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "ok")
        self.assertEqual((self.cwd / "fresh.txt").read_text(), "one\ntwo\n")
        self.assertEqual(previews, [f"write  {(self.cwd / 'fresh.txt').resolve()}  +2 -0"])

    def test_noop_write_reports_no_changes_without_preview(self) -> None:
        previews: list[str] = []
        turns = [
            call_turn(("write", {"path": "app.py", "content": "value = 'old'\n"})),
            text_turn("nothing to do"),
        ]

        answer, messages, _ = self._run(turns, on_preview=previews.append)

        self.assertEqual(answer, "nothing to do")
        self.assertEqual(previews, [])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertTrue(tool_msgs[0]["content"].startswith("no changes:"))


if __name__ == "__main__":
    unittest.main()
