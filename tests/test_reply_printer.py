"""Tests for ReplyPrinter.on_preview (write +/− stats line)."""

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

    def test_prints_stats_header(self) -> None:
        self.rp.on_preview("write  a.py  +1 -1")
        text = self.out()
        self.assertIn("# write", text)
        self.assertIn("a.py", text)
        self.assertIn("+1", text)
        self.assertIn("-1", text)

    def test_prints_edit_kind(self) -> None:
        self.rp.on_preview("edit  a.py  +2 -1")
        text = self.out()
        self.assertIn("# edit", text)
        self.assertIn("a.py", text)
        self.assertIn("+2", text)

    def test_ignores_extra_lines(self) -> None:
        self.rp.on_preview("write  cfg.txt  +2 -3\n@@ ignored\n+body")
        text = self.out()
        self.assertIn("+2", text)
        self.assertNotIn("@@", text)
        self.assertNotIn("+body", text)

    def test_empty_diff_prints_nothing(self) -> None:
        self.rp.on_preview("")
        self.assertEqual(self.out(), "")

    def test_closes_open_thinking_block_first(self) -> None:
        self.rp.on_thinking("pondering")
        self.assertTrue(self.rp._thinking_open)
        self.rp.on_preview("write  x.txt  +1 -0")
        self.assertIn("# write", self.out())
        self.assertNotIn("pondering", self.out())
        self.assertFalse(self.rp._thinking_open)

    def test_whitespace_only_answer_prints_nothing(self) -> None:
        with self.rp as rp:
            rp.on_delta("\n  \n")
        self.assertEqual(self.out(), "")

    def test_preview_does_not_emit_markdown_answer(self) -> None:
        self.rp.on_delta("some streamed answer text")
        self.rp.on_preview("write  x.txt  +1 -1")
        self.assertEqual(self.rp.buf, ["some streamed answer text"])
        self.assertNotIn("some streamed answer text", self.out())


if __name__ == "__main__":
    unittest.main()
