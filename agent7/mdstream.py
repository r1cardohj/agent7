"""Incremental Markdown rendering for streamed answers.

A growing answer is split at blank lines that sit outside fenced code blocks.
The stable prefix is parsed and pre-rendered once per completed block; only
the in-progress tail is re-parsed on each repaint, so frame cost stays bounded
by the block being typed instead of the whole reply.
"""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.segment import Segment, Segments

_FENCE_CHARS = ("`", "~")
# Rich separates block elements with one blank line; the pre-rendered stable
# prefix ends at a block boundary, so the tail needs that separator restored.
_BLOCK_SEPARATOR = Segments([Segment.line(), Segment.line()])


def _stable_prefix_len(text: str) -> int:
    """Length of the prefix ending at a blank line outside a code fence.

    A blank line inside a fenced block is ordinary content, so splitting
    there would freeze half a code block.  Indented code may contain blank
    lines too; splitting it merely renders two adjacent code blocks, which
    is visually identical.
    """
    boundary = 0
    in_fence = False
    fence_char = ""
    fence_len = 0
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if in_fence:
            run = len(stripped) - len(stripped.lstrip(fence_char))
            if indent <= 3 and run >= fence_len and not stripped[run:].strip():
                in_fence = False
        elif indent <= 3 and stripped[:1] in _FENCE_CHARS:
            run = len(stripped) - len(stripped.lstrip(stripped[0]))
            # A backtick fence's info string may not contain another backtick
            # (``foo`` inline code is not a fence).
            if run >= 3 and not (stripped[0] == "`" and "`" in stripped[run:]):
                in_fence = True
                fence_char = stripped[0]
                fence_len = run
        if not in_fence and not stripped and end < len(text):
            boundary = end + 1
        pos = end + 1
    return boundary


class MarkdownStream:
    """Render one growing Markdown answer without re-parsing it per frame."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._width = console.width
        self._consumed = 0
        self._flat: list[Segment] = []
        self._stable: Segments | None = None

    def render(self, text: str) -> Group:
        if self._width != self._console.width:
            # Pre-rendered segments no longer reflow after a resize; rebuild
            # the cache from the full text at the new width.
            self._width = self._console.width
            self._consumed = 0
            self._flat = []
            self._stable = None
        split = _stable_prefix_len(text)
        if split > self._consumed:
            self._consume(text[self._consumed:split])
            self._consumed = split
        tail = text[self._consumed:]
        parts: list[RenderableType] = []
        if self._stable is not None:
            parts.append(self._stable)
        if tail:
            if parts:
                parts.append(_BLOCK_SEPARATOR)
            parts.append(Markdown(tail))
        return Group(*parts)

    def _consume(self, chunk: str) -> None:
        """Parse and pre-render one newly completed chunk (once per block)."""
        lines = self._console.render_lines(Markdown(chunk), pad=False)
        chunk_flat: list[Segment] = []
        for line in lines:
            if chunk_flat:
                chunk_flat.append(Segment.line())
            chunk_flat.extend(line)
        if not chunk_flat:
            return  # e.g. a run of blank lines: nothing to add
        if self._flat:
            # End the previous chunk's last row, then leave one blank row
            # between block elements, matching a full-document render.
            self._flat.extend([Segment.line(), Segment.line()])
        self._flat.extend(chunk_flat)
        self._stable = Segments(self._flat)
