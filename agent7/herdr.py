"""herdr-core status reporting through OSC 7501.

This is deliberately only a lifecycle observer. It neither gives the model
herdr-specific tools nor tries to infer state from terminal output.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import TextIO

from .hooks import LifecycleEvent

_STATES = {
    "session_started": "idle",
    "turn_started": "working",
    "turn_succeeded": "done",
    "turn_interrupted": "idle",
    "turn_blocked": "blocked",
    "turn_failed": "error",
    "session_ended": "clear",
}
_BLOCK_KINDS = {"question", "permission", "auth"}
_VALID_STATES = {"idle", "working", "done", "blocked", "error", "clear"}


def status_sequence(state: str, *, kind: str | None = None) -> str:
    """Encode one root Program Status Protocol report."""
    if state not in _VALID_STATES:
        raise ValueError(f"unknown status state: {state!r}")
    fields = [f"state={state}"]
    if kind is not None:
        if kind not in _BLOCK_KINDS:
            raise ValueError(f"unknown blocked kind: {kind!r}")
        fields.append(f"kind={kind}")
    fields.append("app=agent7")
    return f"\x1b]7501;{':'.join(fields)}\x1b\\"


class HerdrReporter:
    """Write lifecycle status to the containing herdr-core terminal."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self._env = os.environ if env is None else env
        self._stream = stream
        self.enabled = bool(self._env.get("HC_TERMINAL_ID")) and self._env.get(
            "AGENT7_NO_HERDR"
        ) != "1"

    def __call__(self, event: LifecycleEvent) -> None:
        if not self.enabled:
            return
        state = _STATES[event.kind]
        kind = None
        if event.kind == "turn_blocked":
            kind = event.reason if event.reason in _BLOCK_KINDS else "question"
        self._write(status_sequence(state, kind=kind))

    def _write(self, sequence: str) -> None:
        stream = self._stream
        if stream is None:
            # Keep protocol bytes out of redirected answer output. Both streams
            # are normally the same PTY inside herdr-core.
            stream = next(
                (s for s in (sys.stderr, sys.stdout) if getattr(s, "isatty", lambda: False)()),
                None,
            )
        if stream is None or not getattr(stream, "isatty", lambda: False)():
            return
        stream.write(sequence)
        stream.flush()
