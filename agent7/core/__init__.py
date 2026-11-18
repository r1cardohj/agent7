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
    "events",
    "filesystem",
    "history",
    "images",
    "lifecycle",
    "loop",
    "openai_backend",
    "plugins",
    "providers",
    "responses",
    "search",
    "session",
    "shell",
    "tools",
    "types",
    "web_fetch",
)
for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"agent0_runtime.{_name}")
