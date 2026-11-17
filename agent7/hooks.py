"""Compatibility imports for the runtime lifecycle API.

Canonical definitions live in ``agent7.core.lifecycle`` so hosts do not need to
import the terminal or agent7 session/storage implementation.
"""

from .core.lifecycle import (
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
