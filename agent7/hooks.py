"""Compatibility imports for the runtime lifecycle API.

Canonical definitions live in ``agent0_runtime.lifecycle`` so hosts do not need to
import the terminal or agent7 session/storage implementation.
"""

from agent0_runtime.lifecycle import (
    LifecycleContext,
    LifecycleEvent,
    LifecycleHooks,
    LifecycleKind,
    LifecycleListener,
    LifecycleListenerError,
    emit_lifecycle,
    run_with_lifecycle,
)

__all__ = [
    "LifecycleContext",
    "LifecycleEvent",
    "LifecycleHooks",
    "LifecycleKind",
    "LifecycleListener",
    "LifecycleListenerError",
    "emit_lifecycle",
    "run_with_lifecycle",
]
