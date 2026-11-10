"""A small prompt-toolkit pager for terminals with limited system pagers."""

from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.pager import Pager


def _scroll(event: KeyPressEvent, amount: int) -> None:
    window = event.app.layout.current_window
    info = window.render_info
    if info is None:
        return
    last = max(0, info.content_height - info.window_height)
    window.vertical_scroll = max(0, min(last, window.vertical_scroll + amount))


def _scroll_page(event: KeyPressEvent, direction: int) -> None:
    info = event.app.layout.current_window.render_info
    if info is not None:
        _scroll(event, direction * max(1, info.window_height - 1))


class PromptToolkitPager(Pager):
    """Page ANSI-formatted content without relying on Windows' `more.com`."""

    def show(self, content: str) -> None:
        # Rich has already wrapped the content to the terminal width. Keeping
        # one logical line per row also makes vertical_scroll unambiguous.
        window = Window(
            content=FormattedTextControl(
                ANSI(content),
                focusable=True,
                show_cursor=False,
                # Window keeps the cursor row visible, even when the cursor is
                # hidden. Track the scroll row so rendering cannot snap back
                # to the first line after every key press.
                get_cursor_position=lambda: Point(x=0, y=window.vertical_scroll),
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )
        bindings = KeyBindings()

        @bindings.add("down")
        @bindings.add("enter")
        def _down(event: KeyPressEvent) -> None:
            _scroll(event, 1)

        @bindings.add("up")
        def _up(event: KeyPressEvent) -> None:
            _scroll(event, -1)

        @bindings.add("pagedown")
        @bindings.add(" ")
        def _next_page(event: KeyPressEvent) -> None:
            _scroll_page(event, 1)

        @bindings.add("pageup")
        def _previous_page(event: KeyPressEvent) -> None:
            _scroll_page(event, -1)

        @bindings.add("q")
        @bindings.add("Q")
        @bindings.add("c-c")
        def _quit(event: KeyPressEvent) -> None:
            event.app.exit()

        # ReplyPrinter is synchronous but runs inside the agent's event loop;
        # a separate thread gives the pager its own loop instead of nesting one.
        Application(
            layout=Layout(window),
            key_bindings=bindings,
            full_screen=True,
        ).run(in_thread=True)
