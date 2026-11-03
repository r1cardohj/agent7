"""Tests for ReplyPrinter.on_preview (write +/− stats line) and thinking output."""

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

    def test_streams_thinking_text_under_a_header(self) -> None:
        self.rp.on_thinking("looking at ")
        self.rp.on_thinking("agent.py\nand tests")
        text = self.out()
        self.assertIn("thinking", text)
        self.assertIn("  │ looking at agent.py", text)
        self.assertIn("  │ and tests", text)
        self.assertTrue(self.rp._thinking_open)

    def test_blank_line_skips_the_gutter(self) -> None:
        self.rp.on_thinking("a\n\nb")
        self.assertIn("  │ a\n\n  │ b", self.out())

    def test_closes_open_thinking_block_first(self) -> None:
        self.rp.on_thinking("pondering")
        self.assertTrue(self.rp._thinking_open)
        self.rp.on_preview("write  x.txt  +1 -0")
        self.assertIn("pondering", self.out())
        self.assertIn("# write", self.out())
        self.assertLess(self.out().index("pondering"), self.out().index("# write"))
        self.assertFalse(self.rp._thinking_open)

    def test_thinking_is_not_part_of_the_answer(self) -> None:
        with self.rp as rp:
            rp.on_thinking("private reasoning")
        self.assertEqual(rp.buf, [])
        self.assertIn("private reasoning", self.out())

    def test_tool_call_ends_the_thinking_block(self) -> None:
        self.rp.on_thinking("plan")
        self.rp.on_answer_to_thinking()
        self.assertFalse(self.rp._thinking_open)
        self.assertTrue(self.out().endswith("plan\n"))
        self.rp.on_thinking("next")
        self.assertIn("  │ next", self.out())

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
