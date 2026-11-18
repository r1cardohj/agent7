"""Compatibility module for :mod:`agent0_runtime.web_search`."""

import sys

from agent0_runtime import web_search as _implementation

sys.modules[__name__] = _implementation
