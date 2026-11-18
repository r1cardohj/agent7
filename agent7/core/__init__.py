"""Compatibility aliases for the extracted :mod:`agent0_runtime` package."""

import importlib
import sys

import agent0_runtime as _runtime

__all__ = _runtime.__all__
globals().update({name: getattr(_runtime, name) for name in __all__})

_SUBMODULES = (
    "backend",
    "builtin_tools",
    "capabilities",
    "client",
    "credentials",
    "events",
    "file_sessions",
    "filesystem",
    "history",
    "images",
    "lifecycle",
    "loop",
    "managed",
    "openai_backend",
    "plugins",
    "providers",
    "responses",
    "routing",
    "routing_backend",
    "search",
    "session",
    "shell",
    "tools",
    "types",
    "web_fetch",
    "web_search",
)
for _name in _SUBMODULES:
    _module = importlib.import_module(f"agent0_runtime.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    # Keep normal package semantics for mock.patch and reflective importers.
    globals()[_name] = _module
