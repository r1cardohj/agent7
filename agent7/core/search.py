"""Search provider port for the generic `search` tool."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol


class SearchProvider(Protocol):
    """Application-provided search backend.

    Providers own credentials, service selection, and result formatting. The
    tool only validates arguments and gives providers the original user wording.
    """

    def search(
        self,
        query: str,
        count: int | None = None,
        *,
        user_question: str | None = None,
    ) -> str | Awaitable[str]: ...
