"""Tests for incremental Markdown rendering of streamed answers."""

import io
import unittest
from unittest.mock import patch

from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segments

import agent7.mdstream as mdstream
from agent7.mdstream import MarkdownStream, _stable_prefix_len


def make_console(width: int = 40) -> Console:
    return Console(file=io.StringIO(), width=width, force_terminal=False)


def plain(lines) -> list[str]:
    return ["".join(segment.text for segment in line) for line in lines]


class StablePrefixTests(unittest.TestCase):
    def test_blank_line_outside_fence_is_a_boundary(self) -> None:
        self.assertEqual(_stable_prefix_len("hello\n\nwor"), 7)

    def test_no_blank_line_means_no_stable_prefix(self) -> None:
        self.assertEqual(_stable_prefix_len("hello world"), 0)

    def test_blank_line_inside_fence_is_not_a_boundary(self) -> None:
        self.assertEqual(_stable_prefix_len("```py\nx = 1\n\ny = 2"), 0)

    def test_closed_fence_allows_a_boundary(self) -> None:
        text = "```py\nx = 1\n```\n\nnext"
        self.assertEqual(_stable_prefix_len(text), len("```py\nx = 1\n```\n\n"))

    def test_longer_closing_fence_closes(self) -> None:
        text = "````\ncode\n`````\n\nnext"
        self.assertEqual(_stable_prefix_len(text), len("````\ncode\n`````\n\n"))

    def test_shorter_closing_fence_does_not_close(self) -> None:
        self.assertEqual(_stable_prefix_len("````\ncode\n```\n\nnext"), 0)

    def test_inline_double_backticks_are_not_a_fence(self) -> None:
        text = "``inline``\n\nnext"
        self.assertEqual(_stable_prefix_len(text), len("``inline``\n\n"))

    def test_indented_fence_is_just_indented_code(self) -> None:
        text = "    ```\n    code\n\nnext"
        self.assertEqual(_stable_prefix_len(text), len("    ```\n    code\n\n"))

    def test_trailing_blank_line_can_be_fully_consumed(self) -> None:
        self.assertEqual(_stable_prefix_len("hello\n\n"), 7)


class MarkdownStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.console = make_console()
        self.stream = MarkdownStream(self.console)

    def test_tail_only_before_any_boundary(self) -> None:
        group = self.stream.render("hello")
        self.assertEqual(len(group.renderables), 1)
        self.assertEqual(group.renderables[0].markup, "hello")

    def test_completed_block_becomes_pre_rendered_segments(self) -> None:
        self.stream.render("hello\n\nwor")
        group = self.stream.render("hello\n\nworld")
        self.assertIsInstance(group.renderables[0], Segments)
        self.assertEqual(group.renderables[-1].markup, "world")

    def test_stable_is_not_reparsed_when_the_tail_grows(self) -> None:
        with patch.object(mdstream, "Markdown", wraps=Markdown) as md:
            self.stream.render("done\n\npart")
            self.stream.render("done\n\npartial")
            self.stream.render("done\n\npartially")
        stable_parses = [c for c in md.call_args_list if c.args[0] == "done\n\n"]
        self.assertEqual(len(stable_parses), 1)

    def test_new_block_completion_parses_only_the_new_chunk(self) -> None:
        with patch.object(mdstream, "Markdown", wraps=Markdown) as md:
            self.stream.render("one\n\ntwo")
            self.stream.render("one\n\ntwo\n\nthree")
        chunk_parses = [c for c in md.call_args_list if c.args[0] == "two\n\n"]
        self.assertEqual(len(chunk_parses), 1)
        self.assertFalse(any(c.args[0] == "one\n\ntwo\n\n" for c in md.call_args_list))

    def test_resize_rebuilds_the_stable_render(self) -> None:
        self.stream.render("one\n\ntwo")
        stable_before = self.stream._stable
        self.console._width = 20  # simulate a resize, as the tool-status tests do
        with patch.object(mdstream, "Markdown", wraps=Markdown) as md:
            self.stream.render("one\n\ntwo")
        self.assertIsNot(self.stream._stable, stable_before)
        self.assertTrue(any(c.args[0] == "one\n\n" for c in md.call_args_list))

    def test_incremental_render_matches_a_full_markdown_render(self) -> None:
        texts = (
            "# Title\n\npara **bold** text\n\n```py\ncode = 1\n\nmore = 2\n```\n\n- a\n- b\n\ntail",
            "# Title\n\npara **bold** text\n\n- a\n- b\n\n",
            "# T\n\n1. a\n\n2. b\n\nend",
        )
        for text in texts:
            with self.subTest(text=text):
                # A stream is append-only; each text gets a fresh one.
                group = MarkdownStream(self.console).render(text)
                expected = self.console.render_lines(Markdown(text), pad=False)
                actual = self.console.render_lines(group, pad=False)
                self.assertEqual(plain(actual), plain(expected))

    def test_unclosed_fence_stays_in_the_reparsed_tail(self) -> None:
        group = self.stream.render("```py\nx = 1")
        self.assertEqual(len(group.renderables), 1)
        self.assertEqual(group.renderables[0].markup, "```py\nx = 1")


if __name__ == "__main__":
    unittest.main()
