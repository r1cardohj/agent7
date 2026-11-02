"""Tests for write preview (+/− stats) and old_string behaviour."""

import tempfile
import unittest
from pathlib import Path

from agent7.agent import TOOLS, _write_stats, dispatch_tool, tool_summary, tool_write


def _write_tool_schema() -> dict:
    for tool in TOOLS:
        if tool["function"]["name"] == "write":
            return tool["function"]
    raise AssertionError("write tool missing from TOOLS")


class DiffRenderingTests(unittest.TestCase):
    def test_overwrite_shows_add_del_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "cfg.txt"
            path.write_text("alpha\nbeta\ngamma\n")
            previews: list[str] = []
            tool_write(cwd, "cfg.txt", "alpha\nBETA\ngamma\n", on_preview=previews.append)
            self.assertEqual(previews, [f"{path}  +1 -1"])
            self.assertEqual(path.read_text(), "alpha\nBETA\ngamma\n")

    def test_stats_only_no_hunk_body(self) -> None:
        before = "1\n2\n3\n4\n5\n"
        after = before.replace("3", "three")
        stats = _write_stats(Path("f.txt"), before, after)
        self.assertEqual(stats, "f.txt  +1 -1")
        self.assertNotIn("@@", stats)
        self.assertNotIn("+three", stats)

    def test_new_file_counts_adds(self) -> None:
        stats = _write_stats(Path("new.txt"), "", "a\nb\nc\n")
        self.assertEqual(stats, "new.txt  +3 -0")


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
        self.assertIn("wrote 7 bytes", tool_write(self.cwd, "u.txt", "héllo\n"))

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
        result = tool_write(self.cwd, "f.txt", "mine", old_string="keep me")
        self.assertIn("occurs 2 times", result)
        result = tool_write(self.cwd, "f.txt", "mine", old_string="tail keep me")
        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(path.read_text(), "head\nkeep me\nmine\n")

    def test_multiline_old_string(self) -> None:
        path = self.cwd / "f.py"
        path.write_text("def f():\n    a = 1\n    b = 2\n    return a + b\n")
        result = tool_write(
            self.cwd, "f.py", "    a = 10\n    b = 20", old_string="    a = 1\n    b = 2",
        )
        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(path.read_text(), "def f():\n    a = 10\n    b = 20\n    return a + b\n")


class WriteToolSchemaTests(unittest.TestCase):
    def test_schema_advertises_optional_old_string(self) -> None:
        schema = _write_tool_schema()
        self.assertIn("old_string", schema["parameters"]["properties"])
        self.assertEqual(sorted(schema["parameters"]["required"]), ["content", "path"])

    def test_summary_distinguishes_write_and_replace(self) -> None:
        self.assertEqual(tool_summary("write", {"path": "a.py", "content": "12345"}), "a.py  (write, 5 chars)")
        self.assertEqual(
            tool_summary("write", {"path": "a.py", "content": "12345", "old_string": "1"}),
            "a.py  (replace, 5 chars)",
        )


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
            self.cwd, "write", {"path": "a.txt", "old_string": "old", "content": "new"},
            on_preview=previews.append,
        )
        self.assertTrue(result.startswith("wrote "))
        self.assertEqual(previews, [f"{self.cwd / 'a.txt'}  +1 -1"])
        self.assertEqual((self.cwd / "a.txt").read_text(), "new\n")

    def test_other_tools_still_dispatch_with_preview_set(self) -> None:
        (self.cwd / "a.txt").write_text("hi\n")
        previews: list[str] = []
        self.assertEqual(
            dispatch_tool(self.cwd, "read", {"path": "a.txt"}, on_preview=previews.append), "1|hi"
        )
        self.assertEqual(previews, [])

    def test_unknown_tool_still_errors(self) -> None:
        self.assertEqual(dispatch_tool(self.cwd, "nope", {}), "error: unknown tool nope")

    def test_bad_write_args_are_reported(self) -> None:
        result = dispatch_tool(self.cwd, "write", {"path": "a.txt", "content": "x", "old_sting": "typo"})
        self.assertTrue(result.startswith("error: bad args for write:"))
        self.assertFalse((self.cwd / "a.txt").exists())

    def test_unreadable_target_reports_error(self) -> None:
        (self.cwd / "bin.dat").write_bytes(b"\xff\xfe\x00binary")
        previews: list[str] = []
        result = dispatch_tool(
            self.cwd, "write", {"path": "bin.dat", "content": "text", "old_string": ""},
            on_preview=previews.append,
        )
        self.assertTrue(result.startswith("error: "))
        self.assertEqual(previews, [])


if __name__ == "__main__":
    unittest.main()
