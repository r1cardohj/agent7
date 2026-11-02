"""Tests for ReplyPrinter.on_preview (stderr rendering of the write diff)."""

import io
import unittest

from rich.console import Console

import agent7.__main__ as cli


class ReplyPrinterPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err = cli.err
        self.stream = io.StringIO()
        cli.err = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        self.rp = cli.ReplyPrinter()

    def tearDown(self) -> None:
        cli.err = self.orig_err

    def out(self) -> str:
        return self.stream.getvalue()

    def test_prints_header_and_diff_verbatim(self) -> None:
        diff = "a.py  +1 -1\n@@ -1 +1 @@\n-[a] {b}\n+[c] {d}\n"

        self.rp.on_preview(diff)

        text = self.out()
        self.assertIn("# write", text)
        self.assertIn("a.py", text)
        self.assertIn("-[a] {b}", text)
        self.assertIn("+[c] {d}", text)

    def test_header_shows_add_and_del_counts(self) -> None:
        self.rp.on_preview("cfg.txt  +2 -3\n@@ ...\n")

        text = self.out()
        self.assertIn("# write", text)
        self.assertIn("cfg.txt", text)
        self.assertIn("+2", text)
        self.assertIn("-3", text)

    def test_empty_diff_prints_nothing(self) -> None:
        self.rp.on_preview("")

        self.assertEqual(self.out(), "")

    def test_closes_open_thinking_block_first(self) -> None:
        self.rp.on_thinking("pondering")
        self.assertTrue(self.rp._thinking_open)

        self.rp.on_preview("x.txt  +1 -0\n@@ ...\n")

        text = self.out()
        self.assertIn("# write", text)
        # thinking is a status-line spinner, not dumped into the preview stream
        self.assertNotIn("pondering", text)
        self.assertFalse(self.rp._thinking_open)

    def test_preview_does_not_emit_markdown_answer(self) -> None:
        self.rp.on_delta("some streamed answer text")

        self.rp.on_preview("x.txt  +1 -1\n@@ ...\n")

        self.assertEqual(self.rp.buf, ["some streamed answer text"])
        # the buffered reply is only rendered on __exit__, never inside the preview
        self.assertNotIn("some streamed answer text", self.out())


if __name__ == "__main__":
    unittest.main()
