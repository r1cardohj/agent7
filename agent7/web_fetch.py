"""Compatibility imports for the provider-neutral web fetch implementation."""

from agent0_runtime.web_fetch import (
    DEFAULT_FETCH_CHARS,
    MAX_DOWNLOAD_BYTES,
    MAX_FETCH_CHARS,
    _download,
    _Downloaded,
    _FetchError,
    _proxy_for,
    fetch,
    tool_fetch,
)

__all__ = [
    "DEFAULT_FETCH_CHARS",
    "MAX_DOWNLOAD_BYTES",
    "MAX_FETCH_CHARS",
    "_Downloaded",
    "_FetchError",
    "_download",
    "_proxy_for",
    "fetch",
    "tool_fetch",
]
