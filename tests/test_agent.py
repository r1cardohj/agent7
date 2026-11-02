import tempfile
import unittest
from pathlib import Path

from agent7.agent import dispatch_tool, tool_write


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
        self.assertTrue(previews[0].startswith(f"{self.cwd / 'nested/hello.py'}  +1 -0"))
        self.assertIn("@@", previews[0])
        self.assertIn("+print('hi')", previews[0])
        # --- / +++ file headers are skipped; the summary line names the file
        self.assertNotIn("--- ", previews[0])
        self.assertNotIn("+++ ", previews[0])

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
        self.assertEqual(len(previews), 1)
        self.assertIn("-    return 'old'", previews[0])
        self.assertIn("+    return 'new'", previews[0])

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


if __name__ == "__main__":
    unittest.main()
