import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_failed_write_leaves_the_old_file_intact(self) -> None:
        path = self.cwd / "keep.txt"
        path.write_text("original\n")

        with patch("agent7.agent.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                tool_write(self.cwd, "keep.txt", "half a rewrite")

        self.assertEqual(path.read_text(), "original\n")
        self.assertEqual([p.name for p in self.cwd.iterdir()], ["keep.txt"])

    def test_write_follows_a_symlink_instead_of_replacing_it(self) -> None:
        real = self.cwd / "real.txt"
        real.write_text("old\n")
        link = self.cwd / "link.txt"
        link.symlink_to(real)

        tool_write(self.cwd, "link.txt", "new\n")

        self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_text(), "new\n")


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

    def test_one_enormous_line_cannot_blow_up_memory(self) -> None:
        path = self.cwd / "min.js"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("y" * 6_000_000)  # a single 6 MB line, no newline

        tracemalloc.start()
        try:
            result = dispatch_tool(self.cwd, "read", {"path": "min.js"})
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertTrue(result.startswith("1|"))
        self.assertIn("truncated", result)
        self.assertLess(peak, 4_000_000)

    def test_offset_past_eof_says_so(self) -> None:
        (self.cwd / "small.txt").write_text("a\nb\n")

        result = dispatch_tool(self.cwd, "read", {"path": "small.txt", "offset": 99})

        self.assertIn("(empty)", result)
        self.assertIn("lines 99-2 of 2", result)

    def test_binary_file_is_reported(self) -> None:
        (self.cwd / "blob.bin").write_bytes(b"\xff\xfe\x00binary")

        self.assertTrue(
            dispatch_tool(self.cwd, "read", {"path": "blob.bin"}).startswith("error:")
        )

    def test_large_file_is_streamed_not_buffered(self) -> None:
        path = self.cwd / "big.log"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(200_000):
                fh.write(f"line{i} " + "x" * 40 + "\n")  # ~9 MB

        tracemalloc.start()
        try:
            result = dispatch_tool(self.cwd, "read", {"path": "big.log"})
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertIn("1|line0", result)
        self.assertIn("of 200000", result)
        self.assertIn("continue with offset=201", result)
        self.assertLess(peak, 4_000_000)


class OutputBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bash_output_is_truncated(self) -> None:
        py = sys.executable.replace("'", "'\\''")
        result = dispatch_tool(
            self.cwd,
            "bash",
            {"command": f"'{py}' -c \"print('x' * {MAX_BASH_CHARS * 2})\""},
        )

        self.assertIn("truncated", result)
        self.assertLess(len(result), MAX_BASH_CHARS + 200)

    def test_large_bash_output_is_capped_in_memory(self) -> None:
        py = sys.executable.replace("'", "'\\''")
        tracemalloc.start()
        try:
            result = dispatch_tool(
                self.cwd,
                "bash",
                {"command": f"'{py}' -c \"print('y' * {8_000_000})\""},
            )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertIn("truncated", result)
        self.assertLess(len(result), MAX_BASH_CHARS + 200)
        self.assertLess(peak, 4_000_000)

    def test_grep_tool_is_gone(self) -> None:
        self.assertEqual(dispatch_tool(self.cwd, "grep", {"pattern": "x"}), "error: unknown tool grep")


if __name__ == "__main__":
    unittest.main()
