"""Application-wide Markdown rendering defaults."""

from rich.markdown import Markdown as RichMarkdown


class Markdown(RichMarkdown):
    """Render code with terminal-native colors and no painted background."""

    def __init__(self, markup: str) -> None:
        # Rich's default monokai theme paints padded code rows #272822. Terminal
        # reflow after a resize can interleave those rows with default-background
        # rows, producing dark/brown stripes. ANSI colors retain syntax highlighting
        # without taking ownership of the terminal's background.
        super().__init__(markup, code_theme="ansi_dark")
