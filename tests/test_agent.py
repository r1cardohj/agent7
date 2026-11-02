import tempfile
import unittest
from pathlib import Path

from agent7.agent import (
    DEFAULT_READ_LINES,
    MAX_BASH_CHARS,
    MAX_GREP_LINES,
    dispatch_tool,
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
        self.assertEqual(previews[0], f"{self.cwd / 'nested/hello.py'}  +1 -0")

    def test_old_string_replaces_one_unique_span(self) -> None:
        path = self.cwd / "hello.py"
        path.write_text("def hello():\n    return 'old'\n")
        previews: list[str] = []

        result = dispatch_tool(
            self.cwd,
            "write",
            {
                "path": "hello.py",
                "old_string": "return 'old'",
                "content": "return 'new'",
            },
            on_preview=previews.append,
        )

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(path.read_text(), "def hello():\n    return 'new'\n")
        self.assertEqual(previews, [f"{path}  +1 -1"])

    def test_old_string_must_be_unique(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("same\nsame\n")
        previews: list[str] = []

        result = tool_write(
            self.cwd, "hello.txt", "different", old_string="same", on_preview=previews.append
        )

        self.assertIn("occurs 2 times", result)
        self.assertEqual(previews, [])
        self.assertEqual(path.read_text(), "same\nsame\n")

    def test_old_string_requires_existing_file(self) -> None:
        result = tool_write(self.cwd, "missing.txt", "new", old_string="old")

        self.assertTrue(result.startswith("error: not a file:"))
        self.assertFalse((self.cwd / "missing.txt").exists())

    def test_noop_does_not_preview_or_write(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("same\n")
        previews: list[str] = []

        result = tool_write(self.cwd, "hello.txt", "same\n", on_preview=previews.append)

        self.assertTrue(result.startswith("no changes:"))
        self.assertEqual(previews, [])

    def test_empty_old_string_is_rejected(self) -> None:
        path = self.cwd / "hello.txt"
        path.write_text("content\n")

        result = tool_write(self.cwd, "hello.txt", "new", old_string="")

        self.assertIn("old_string must not be empty", result)
        self.assertEqual(path.read_text(), "content\n")

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

    def test_grep_caps_matches(self) -> None:
        sub = self.cwd / "src"
        sub.mkdir()
        (sub / "many.txt").write_text("hit\n" * (MAX_GREP_LINES + 55))

        result = dispatch_tool(self.cwd, "grep", {"pattern": "hit", "path": "src"})

        self.assertIn("more]", result)
        self.assertLessEqual(len(result.splitlines()), MAX_GREP_LINES + 1)


if __name__ == "__main__":
    unittest.main()
