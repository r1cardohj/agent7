"""herdr-core status reporting through OSC 7501.

This is deliberately only a lifecycle observer. It neither gives the model
herdr-specific tools nor tries to infer state from terminal output.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
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

_HerdrWhich = Callable[[str], str | None]

_BACKGROUND_TASKS_POINTER = """\

Herdr background runs: `hc` is available in bash. Use plain bash for fast
commands. Before your first `hc` command for a long-running test, build, or
service in each task, read and follow this bundled skill exactly:
`{skill_path}`
Do not use hc before reading it. If the file cannot be read, use ordinary
bash with a suitable timeout instead of guessing the protocol.
"""


def herdr_skill_path() -> Path:
    """The packaged, on-demand Herdr runbook read by the model's read tool."""
    return Path(__file__).resolve().parent / "skills" / "herdr" / "SKILL.md"


def herdr_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when running inside a Herdr terminal and integration is not disabled."""
    env = os.environ if env is None else env
    return bool(env.get("HC_TERMINAL_ID")) and env.get("AGENT7_NO_HERDR") != "1"


def background_tasks_available(
    env: Mapping[str, str] | None = None,
    *,
    which: _HerdrWhich = shutil.which,
) -> bool:
    """Herdr background runs are usable only with both the terminal and its CLI."""
    return herdr_enabled(env) and which("hc") is not None


def background_tasks_instructions(
    env: Mapping[str, str] | None = None,
    *,
    which: _HerdrWhich = shutil.which,
) -> str:
    """The model-facing instructions for managed background tasks, or empty text.

    The fallback outside Herdr is the ordinary synchronous bash tool; there is
    deliberately no untracked local `nohup` fallback to keep command lifecycle
    and output semantics identical across platforms.
    """
    if not background_tasks_available(env, which=which):
        return ""
    path = herdr_skill_path()
    if not path.is_file():
        return ""
    return _BACKGROUND_TASKS_POINTER.format(skill_path=path)


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
        self.enabled = herdr_enabled(self._env)

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
