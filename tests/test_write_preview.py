"""Tests for the write tool's diff preview / old_string behaviour."""

import tempfile
import unittest
from pathlib import Path

from agent7.agent import (
    MAX_DIFF_LINES,
    TOOLS,
    _write_diff,
    dispatch_tool,
    tool_summary,
    tool_write,
)


def _write_tool_schema() -> dict:
    for tool in TOOLS:
        if tool["function"]["name"] == "write":
            return tool["function"]
    raise AssertionError("write tool missing from TOOLS")


class DiffRenderingTests(unittest.TestCase):
    def test_overwrite_shows_removed_and_added_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "cfg.txt"
            path.write_text("alpha\nbeta\ngamma\n")
            previews: list[str] = []

            tool_write(cwd, "cfg.txt", "alpha\nBETA\ngamma\n", on_preview=previews.append)

            self.assertEqual(len(previews), 1)
            diff = previews[0]
            self.assertIn(f"{path}  +1 -1", diff)
            self.assertIn("@@", diff)
            self.assertIn("-beta", diff)
            self.assertIn("+BETA", diff)
            self.assertNotIn("--- ", diff)
            self.assertNotIn("+++ ", diff)
            self.assertEqual(path.read_text(), "alpha\nBETA\ngamma\n")

    def test_diff_keeps_one_line_of_context(self) -> None:
        before = "1\n2\n3\n4\n5\n6\n7\n8\n9\n"
        after = before.replace("5", "five")

        diff = _write_diff(Path("f.txt"), before, after)

        self.assertIn(" 4", diff)
        self.assertIn(" 6", diff)
        self.assertNotIn(" 3", diff)
        self.assertNotIn(" 7", diff)

    def test_preview_is_truncated_to_max_diff_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            before = "".join(f"old line {i}\n" for i in range(2000))
            after = "".join(f"new line {i}\n" for i in range(2000))
            path = cwd / "big.txt"
            path.write_text(before)
            previews: list[str] = []

            tool_write(cwd, "big.txt", after, on_preview=previews.append)

            self.assertEqual(len(previews), 1)
            body_lines = [
                ln
                for ln in previews[0].splitlines()[1:]
                if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
            ]
            self.assertLessEqual(len(body_lines), MAX_DIFF_LINES)
            self.assertIn("...[", previews[0])
            self.assertIn(f"+{2000} -{2000}", previews[0].splitlines()[0])
            self.assertEqual(path.read_text(), after)

    def test_new_file_preview_is_compact(self) -> None:
        diff = _write_diff(Path("new.txt"), "", "a\nb\nc\nd\ne\n", max_lines=3)

        self.assertTrue(diff.startswith("new.txt  +5 -0"))
        self.assertIn("+a", diff)
        self.assertIn("+c", diff)
        self.assertNotIn("+d", diff)
        self.assertIn("...[2 more changed lines]", diff)


class WriteResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_works_without_preview_callback(self) -> None:
        result = tool_write(self.cwd, "plain.txt", "hello\n")

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual((self.cwd / "plain.txt").read_text(), "hello\n")

    def test_byte_count_is_utf8_length(self) -> None:
        result = tool_write(self.cwd, "u.txt", "héllo\n")

        self.assertIn("wrote 7 bytes", result)

    def test_old_string_not_found_is_rejected(self) -> None:
        path = self.cwd / "f.txt"
        path.write_text("abc\n")
        previews: list[str] = []

        result = tool_write(self.cwd, "f.txt", "x", old_string="zzz", on_preview=previews.append)

        self.assertIn("occurs 0 times", result)
        self.assertEqual(previews, [])
        self.assertEqual(path.read_text(), "abc\n")

    def test_old_string_replaces_only_the_matched_span(self) -> None:
        path = self.cwd / "f.txt"
        path.write_text("head\nkeep me\ntail keep me\n")

        # the span appears twice -> rejected, nothing written
        result = tool_write(self.cwd, "f.txt", "mine", old_string="keep me")

        self.assertIn("occurs 2 times", result)
        self.assertEqual(path.read_text(), "head\nkeep me\ntail keep me\n")

        # a unique span is replaced in place, siblings untouched
        result = tool_write(self.cwd, "f.txt", "mine", old_string="tail keep me")

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(path.read_text(), "head\nkeep me\nmine\n")

    def test_multiline_old_string(self) -> None:
        path = self.cwd / "f.py"
        path.write_text("def f():\n    a = 1\n    b = 2\n    return a + b\n")

        result = tool_write(
            self.cwd,
            "f.py",
            "    a = 10\n    b = 20",
            old_string="    a = 1\n    b = 2",
        )

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(path.read_text(), "def f():\n    a = 10\n    b = 20\n    return a + b\n")


class WriteToolSchemaTests(unittest.TestCase):
    def test_schema_advertises_optional_old_string(self) -> None:
        schema = _write_tool_schema()

        self.assertIn("old_string", schema["parameters"]["properties"])
        self.assertEqual(sorted(schema["parameters"]["required"]), ["content", "path"])
        self.assertIn("replace one unique text span", schema["description"])

    def test_summary_distinguishes_write_and_replace(self) -> None:
        write_summary = tool_summary("write", {"path": "a.py", "content": "12345"})
        replace_summary = tool_summary(
            "write", {"path": "a.py", "content": "12345", "old_string": "1"}
        )

        self.assertEqual(write_summary, "a.py  (write, 5 chars)")
        self.assertEqual(replace_summary, "a.py  (replace, 5 chars)")


class DispatchWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dispatch_forwards_preview_for_write(self) -> None:
        (self.cwd / "a.txt").write_text("old\n")
        previews: list[str] = []

        result = dispatch_tool(
            self.cwd,
            "write",
            {"path": "a.txt", "old_string": "old", "content": "new"},
            on_preview=previews.append,
        )

        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(len(previews), 1)
        self.assertIn("+new", previews[0])
        self.assertEqual((self.cwd / "a.txt").read_text(), "new\n")

    def test_other_tools_still_dispatch_with_preview_set(self) -> None:
        (self.cwd / "a.txt").write_text("hi\n")
        previews: list[str] = []

        result = dispatch_tool(self.cwd, "read", {"path": "a.txt"}, on_preview=previews.append)

        self.assertEqual(result, "1|hi")
        self.assertEqual(previews, [])

    def test_unknown_tool_still_errors(self) -> None:
        self.assertEqual(dispatch_tool(self.cwd, "nope", {}), "error: unknown tool nope")

    def test_bad_write_args_are_reported(self) -> None:
        result = dispatch_tool(
            self.cwd, "write", {"path": "a.txt", "content": "x", "old_sting": "typo"}
        )

        self.assertTrue(result.startswith("error: bad args for write:"))
        self.assertFalse((self.cwd / "a.txt").exists())

    def test_unreadable_target_reports_error(self) -> None:
        (self.cwd / "bin.dat").write_bytes(b"\xff\xfe\x00binary")
        previews: list[str] = []

        result = dispatch_tool(
            self.cwd,
            "write",
            {"path": "bin.dat", "content": "text", "old_string": ""},
            on_preview=previews.append,
        )

        self.assertTrue(result.startswith("error: "))
        self.assertEqual(previews, [])
        self.assertEqual((self.cwd / "bin.dat").read_bytes(), b"\xff\xfe\x00binary")


if __name__ == "__main__":
    unittest.main()
