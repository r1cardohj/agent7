"""Tests for application-wide Markdown rendering defaults."""

import io
import unittest

from rich.console import Console

from agent7.markdown import Markdown


class MarkdownTests(unittest.TestCase):
    def test_wrapped_code_has_highlighting_without_a_painted_background(self) -> None:
        console = Console(
            file=io.StringIO(), width=20,
            force_terminal=True, color_system="truecolor",
        )
        markdown = Markdown(
            '```python\ndef greet(name):\n    print(f"Hello, {name}! This line wraps.")\n```'
        )

        lines = console.render_lines(markdown, pad=False)
        segments = [segment for line in lines for segment in line]

        self.assertTrue(any(segment.style and segment.style.color for segment in segments))
        self.assertTrue(all(
            not segment.style or segment.style.bgcolor is None
            for segment in segments
        ))


if __name__ == "__main__":
    unittest.main()
