"""A small, dependable pager for terminals with limited system pagers."""

from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.pager import Pager

_STYLE = Style.from_dict(
    {
        "pager.position": "reverse",
        "pager.hint": "reverse dim",
    }
)

_HINT = " up/down Enter Space PgUp/PgDn Home/End q quit "


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


def _scroll_half_page(event: KeyPressEvent, direction: int) -> None:
    info = event.app.layout.current_window.render_info
    if info is not None:
        _scroll(event, direction * max(1, info.window_height // 2))


def _scroll_to_end(event: KeyPressEvent, bottom: bool) -> None:
    info = event.app.layout.current_window.render_info
    if info is not None:
        _scroll(event, info.content_height if bottom else -info.content_height)


def _status(window: Window) -> FormattedText:
    """Report which lines are visible, so the pager never feels stuck."""
    info = window.render_info
    if info is None:
        return FormattedText([("class:pager.hint", _HINT)])
    total = max(1, info.content_height)
    first = min(total, window.vertical_scroll + 1)
    last = min(total, window.vertical_scroll + max(1, info.window_height))
    return FormattedText(
        [
            ("class:pager.position", f" {first}-{last}/{total} "),
            ("class:pager.hint", _HINT),
        ]
    )


def _bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("down")
    @bindings.add("enter")
    @bindings.add("j")
    @bindings.add("c-n")
    def _down(event: KeyPressEvent) -> None:
        _scroll(event, 1)

    @bindings.add("up")
    @bindings.add("k")
    @bindings.add("c-p")
    def _up(event: KeyPressEvent) -> None:
        _scroll(event, -1)

    @bindings.add("pagedown")
    @bindings.add(" ")
    @bindings.add("f")
    @bindings.add("c-f")
    def _next_page(event: KeyPressEvent) -> None:
        _scroll_page(event, 1)

    @bindings.add("pageup")
    @bindings.add("b")
    @bindings.add("c-b")
    def _previous_page(event: KeyPressEvent) -> None:
        _scroll_page(event, -1)

    @bindings.add("c-d")
    def _half_page_down(event: KeyPressEvent) -> None:
        _scroll_half_page(event, 1)

    @bindings.add("c-u")
    def _half_page_up(event: KeyPressEvent) -> None:
        _scroll_half_page(event, -1)

    @bindings.add("home")
    @bindings.add("g")
    def _first_line(event: KeyPressEvent) -> None:
        _scroll_to_end(event, bottom=False)

    @bindings.add("end")
    @bindings.add("G")
    def _last_line(event: KeyPressEvent) -> None:
        _scroll_to_end(event, bottom=True)

    # Escape and Ctrl-C are what people reach for when a pager will not move.
    @bindings.add("q")
    @bindings.add("Q")
    @bindings.add("escape")
    @bindings.add("c-c")
    def _quit(event: KeyPressEvent) -> None:
        event.app.exit()

    return bindings


class PromptToolkitPager(Pager):
    """Page ANSI-formatted content without relying on Windows' `more.com`."""

    def show(self, content: str) -> None:
        try:
            self._run(content)
        except Exception:
            # A pager that cannot start must not cost the user the reply, and
            # it must not escape: Rich's PagerContext skips its own cleanup
            # when a pager raises, which would leave the console capturing and
            # mute every later print. The caller prints the answer itself as
            # soon as this returns.
            return

    def _run(self, content: str) -> None:
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
        status = Window(
            FormattedTextControl(lambda: _status(window)),
            height=1,
            wrap_lines=False,
        )

        # ReplyPrinter is synchronous but runs inside the agent's event loop;
        # a separate thread gives the pager its own loop instead of nesting one.
        Application(
            layout=Layout(HSplit([window, status])),
            key_bindings=_bindings(),
            style=_STYLE,
            full_screen=True,
            # Turns the mouse wheel into scroll-up/scroll-down, which is how
            # most people try to page through a long answer.
            mouse_support=True,
        ).run(in_thread=True)
