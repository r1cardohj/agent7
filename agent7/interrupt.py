"""Cancel running turns with Escape or Ctrl-C without losing typeahead."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from typing import TypeVar

from prompt_toolkit.application import Application
from prompt_toolkit.input.base import Input
from prompt_toolkit.input.typeahead import get_typeahead, store_typeahead
from prompt_toolkit.key_binding import KeyBindings, KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

_T = TypeVar("_T")

async def run_with_interrupt_keys(work: Awaitable[_T], *, input_obj: Input) -> _T:
    """Run work while an invisible input layer watches for cancellation keys.

    The layer shares prompt-toolkit's Input with the next prompt. Every other
    key is put back in typeahead, so typing during a turn is not swallowed.
    """
    task = asyncio.ensure_future(work)
    pending: list[KeyPress] = []
    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def cancel(event: KeyPressEvent) -> None:
        task.cancel()
        event.app.exit()

    @bindings.add(Keys.Any)
    def preserve(event: KeyPressEvent) -> None:
        pending.extend(event.key_sequence)

    app: Application | None = None

    async def listen() -> None:
        nonlocal app
        app = Application(
            input=input_obj,
            output=DummyOutput(),
            key_bindings=bindings,
        )
        await app.run_async(set_exception_handler=False, handle_sigint=False)

    listener = asyncio.create_task(listen())
    try:
        return await task
    finally:
        if app is not None and app.is_running and not app.is_done:
            app.exit()
        elif not listener.done():
            listener.cancel()
        # An input failure disables these extra keys; it must not replace the
        # agent's result or its original exception.
        with suppress(asyncio.CancelledError, Exception):
            await listener
        queued = get_typeahead(input_obj)
        store_typeahead(input_obj, pending + queued)
