"""Fetch web pages as model-readable text with trafilatura."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_FETCH_CHARS = 12_000
MAX_FETCH_CHARS = 50_000


def _normalized_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("url must start with http:// or https://")
    return urlunsplit(parsed._replace(fragment=""))


def _limit(max_chars: int | None) -> int:
    if max_chars is None:
        return DEFAULT_FETCH_CHARS
    limit = int(max_chars)
    if limit < 1:
        raise ValueError("max_chars must be at least 1")
    return min(limit, MAX_FETCH_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def fetch(url: str, max_chars: int | None = None) -> str:
    """Download and extract a web page, returning a bounded Markdown result."""
    try:
        url = _normalized_url(url)
        limit = _limit(max_chars)
    except (TypeError, ValueError) as e:
        return f"error: {e}"

    try:
        from trafilatura import extract, fetch_url
    except ImportError:
        return "error: trafilatura is not installed"

    downloaded = fetch_url(url)
    if downloaded is None:
        return "error: could not fetch URL"

    options = {
        "url": url,
        "output_format": "markdown",
        "with_metadata": True,
        "include_comments": False,
        "include_tables": True,
        "include_links": True,
        "include_images": False,
        "deduplicate": True,
    }
    text = extract(downloaded, **options)
    if not text:
        # Trafilatura's default precision setting can miss sparse documentation
        # pages; ask it to relax the extraction before declaring failure.
        text = extract(downloaded, **options, favor_recall=True)
    if not text:
        return "error: no readable text extracted"
    return _truncate(text, limit)


def tool_fetch(cwd: Path, url: str, max_chars: int | None = None) -> str:
    """fetch tool handler: cwd is unused because fetching has no workspace path."""
    return fetch(url, max_chars)
