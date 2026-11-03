import tempfile
import unittest
from pathlib import Path

from agent7.agent import (
    DEFAULT_READ_LINES,
    MAX_BASH_CHARS,
    dispatch_tool,
    tool_edit,
    tool_write,
)


class WriteToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_new_file_previews_diff_and_writes(self) -> None:
        previews: list[str] = []
        result = tool_write(
            self.cwd,
            "nested/hello.py",
            "print('hi')\n",
            on_preview=previews.append,
        )

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual((self.cwd / "nested/hello.py").read_text(), "print('hi')\n")
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0], f"write  {self.cwd / 'nested/hello.py'}  +1 -0")

    def test_noop_does_not_preview_or_write(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("same\n")
        previews: list[str] = []

        result = tool_write(self.cwd, "hello.txt", "same\n", on_preview=previews.append)

        self.assertTrue(result.startswith("no changes:"))
        self.assertEqual(previews, [])


class EditToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_single_span_replaces(self) -> None:
        path = self.cwd / "hello.py"
        path.write_text("def hello():\n    return 'old'\n")
        previews: list[str] = []

        result = dispatch_tool(
            self.cwd,
            "edit",
            {
                "path": "hello.py",
                "edits": [{"old_string": "return 'old'", "new_string": "return 'new'"}],
            },
            on_preview=previews.append,
        )

        self.assertTrue(result.startswith("edited "))
        self.assertEqual(path.read_text(), "def hello():\n    return 'new'\n")
        self.assertEqual(previews, [f"edit  {path}  +1 -1"])

    def test_multi_hunk_against_original(self) -> None:
        path = self.cwd / "f.py"
        path.write_text("a = 1\nb = 2\nc = 3\n")
        result = tool_edit(
            self.cwd,
            "f.py",
            [
                {"old_string": "a = 1", "new_string": "a = 10"},
                {"old_string": "c = 3", "new_string": "c = 30"},
            ],
        )
        self.assertEqual(result, f"edited 2 spans in {path}")
        self.assertEqual(path.read_text(), "a = 10\nb = 2\nc = 30\n")

    def test_old_string_must_be_unique(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("same\nsame\n")
        previews: list[str] = []

        result = tool_edit(
            self.cwd,
            "hello.txt",
            [{"old_string": "same", "new_string": "different"}],
            on_preview=previews.append,
        )

        self.assertIn("occurs 2 times", result)
        self.assertEqual(previews, [])
        self.assertEqual(path.read_text(), "same\nsame\n")

    def test_requires_existing_file(self) -> None:
        result = tool_edit(
            self.cwd, "missing.txt", [{"old_string": "old", "new_string": "new"}]
        )

        self.assertTrue(result.startswith("error: not a file:"))
        self.assertFalse((self.cwd / "missing.txt").exists())

    def test_empty_old_string_is_rejected(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("content\n")

        result = tool_edit(
            self.cwd, "hello.txt", [{"old_string": "", "new_string": "new"}]
        )

        self.assertIn("old_string must be a non-empty string", result)
        self.assertEqual(path.read_text(), "content\n")

    def test_overlapping_edits_are_rejected(self) -> None:
        path = self.cwd / "f.txt"
        path.write_text("abcdef\n")
        result = tool_edit(
            self.cwd,
            "f.txt",
            [
                {"old_string": "abc", "new_string": "XXX"},
                {"old_string": "bcd", "new_string": "YYY"},
            ],
        )
        self.assertIn("overlap", result)
        self.assertEqual(path.read_text(), "abcdef\n")

    def test_empty_edits_rejected(self) -> None:
        path = self.cwd / "f.txt"
        path.write_text("x\n")
        self.assertIn("non-empty", tool_edit(self.cwd, "f.txt", []))


class ReadToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_read_caps_lines_and_hints_offset(self) -> None:
        path = self.cwd / "big.txt"
        path.write_text("\n".join(f"line{i}" for i in range(1, 251)) + "\n")

        result = dispatch_tool(self.cwd, "read", {"path": "big.txt"})

        self.assertIn("1|line1", result)
        self.assertIn(f"{DEFAULT_READ_LINES}|line{DEFAULT_READ_LINES}", result)
        self.assertNotIn(f"{DEFAULT_READ_LINES + 1}|", result)
        self.assertIn(f"lines 1-{DEFAULT_READ_LINES} of 250", result)
        self.assertIn(f"continue with offset={DEFAULT_READ_LINES + 1}", result)

    def test_offset_continues_reading(self) -> None:
        path = self.cwd / "big.txt"
        path.write_text("\n".join(f"line{i}" for i in range(1, 251)) + "\n")

        result = dispatch_tool(self.cwd, "read", {"path": "big.txt", "offset": 201})

        self.assertIn("201|line201", result)
        self.assertIn("250|line250", result)
        self.assertNotIn("continue with offset", result)

    def test_small_file_has_no_pagination_hint(self) -> None:
        (self.cwd / "small.txt").write_text("a\nb\n")

        result = dispatch_tool(self.cwd, "read", {"path": "small.txt"})

        self.assertEqual(result, "1|a\n2|b")

    def test_explicit_limit_is_respected(self) -> None:
        (self.cwd / "small.txt").write_text("a\nb\nc\nd\n")

        result = dispatch_tool(self.cwd, "read", {"path": "small.txt", "limit": 2})

        self.assertIn("1|a\n2|b", result)
        self.assertNotIn("3|c", result)
        self.assertIn("lines 1-2 of 4", result)

    def test_empty_file(self) -> None:
        (self.cwd / "empty.txt").write_text("")

        self.assertEqual(dispatch_tool(self.cwd, "read", {"path": "empty.txt"}), "(empty)")


class OutputBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bash_output_is_truncated(self) -> None:
        result = dispatch_tool(
            self.cwd,
            "bash",
            {"command": f"python -c \"print('x' * {MAX_BASH_CHARS * 2})\""},
        )

        self.assertIn("truncated", result)
        self.assertLess(len(result), MAX_BASH_CHARS + 200)

    def test_grep_tool_is_gone(self) -> None:
        self.assertEqual(dispatch_tool(self.cwd, "grep", {"pattern": "x"}), "error: unknown tool grep")


if __name__ == "__main__":
    unittest.main()
