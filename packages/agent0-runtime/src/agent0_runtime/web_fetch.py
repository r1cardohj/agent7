"""Proxy-aware downloads with trafilatura for HTML extraction only."""

from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import getproxies, proxy_bypass

DEFAULT_FETCH_CHARS = 12_000
MAX_FETCH_CHARS = 50_000
MAX_DOWNLOAD_BYTES = 20_000_000
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 20

_USER_AGENT = "Mozilla/5.0 (compatible; agent0-runtime)"
_RETRY_STATUS = (429, 500, 502, 503, 504)
_TEXT_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/toml",
        "application/x-yaml",
        "application/xml",
        "application/yaml",
    }
)


class _FetchError(Exception):
    """A download failure with a model-readable message."""


@dataclass(frozen=True)
class _Downloaded:
    data: bytes
    media_type: str
    charset: str | None = None


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


def _proxy_for(url: str) -> str | None:
    """Resolve lowercase/uppercase environment proxies and no_proxy rules."""
    parsed = urlsplit(url)
    if parsed.hostname and proxy_bypass(parsed.hostname):
        return None
    proxies = getproxies()
    return proxies.get(parsed.scheme.lower()) or proxies.get("all")


def _http_stack() -> Any:
    """Load optional HTTP dependencies only when the fetch tool is used."""

    try:
        import certifi
        import urllib3
    except ImportError as e:
        raise RuntimeError("the fetch tool requires the web extra (urllib3 + certifi)") from e
    return urllib3, certifi


@lru_cache(maxsize=8)
def _pool(proxy_url: str | None) -> Any:
    # Reuse connections within a host without keeping proxy credentials elsewhere.
    urllib3, certifi = _http_stack()
    options = {"ca_certs": certifi.where(), "cert_reqs": "CERT_REQUIRED", "num_pools": 4}
    if proxy_url:
        return urllib3.ProxyManager(proxy_url, **options)
    return urllib3.PoolManager(**options)


def _download(url: str) -> _Downloaded:
    urllib3, _certifi = _http_stack()
    retries = urllib3.Retry(
        total=None,
        connect=1,
        read=1,
        status=1,
        redirect=2,
        backoff_factor=0.5,
        status_forcelist=_RETRY_STATUS,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=False,
        raise_on_status=False,
    )
    response = None
    try:
        manager = _pool(_proxy_for(url))
        response = manager.request(
            "GET",
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=urllib3.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT),
            preload_content=False,
            retries=retries,
        )
        if not 200 <= response.status < 300:
            raise _FetchError(f"HTTP {response.status}")

        body = bytearray()
        for chunk in response.stream(2**16):
            body.extend(chunk)
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise _FetchError(f"response exceeds {MAX_DOWNLOAD_BYTES} bytes")

        content_type = response.headers.get("Content-Type", "")
        header = Message()
        header["content-type"] = content_type
        media_type = header.get_content_type().lower() if content_type else ""
        return _Downloaded(bytes(body), media_type, header.get_content_charset())
    except _FetchError:
        raise
    except (OSError, ValueError, urllib3.exceptions.HTTPError) as e:
        raise _FetchError("could not fetch URL") from e
    finally:
        if response is not None:
            response.release_conn()


def _decode(downloaded: _Downloaded) -> str:
    if downloaded.charset:
        try:
            return downloaded.data.decode(downloaded.charset)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return downloaded.data.decode("utf-8")
    except UnicodeDecodeError:
        # Reuse trafilatura's encoding detection for legacy/non-UTF-8 pages.
        from trafilatura.utils import decode_file

        return decode_file(downloaded.data)


def _is_html(downloaded: _Downloaded) -> bool:
    if downloaded.media_type in ("text/html", "application/xhtml+xml"):
        return True
    if downloaded.media_type:
        return False
    beginning = downloaded.data[:4096].lstrip().lower()
    return beginning.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def _is_text(downloaded: _Downloaded) -> bool:
    media_type = downloaded.media_type
    if not media_type:
        return b"\0" not in downloaded.data[:4096]
    return (
        media_type.startswith("text/")
        or media_type in _TEXT_TYPES
        or media_type.endswith(("+json", "+xml"))
    )


class _PlainTextParser(HTMLParser):
    """Last-resort HTML text extraction without script/style contents."""

    _HIDDEN = frozenset({"script", "style", "noscript", "svg"})
    _BREAKS = frozenset(
        {
            "article",
            "br",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "p",
            "pre",
            "section",
            "table",
            "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._HIDDEN:
            self.hidden += 1
        elif not self.hidden and tag in self._BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in self._BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _plain_text(downloaded: _Downloaded) -> str:
    parser = _PlainTextParser()
    parser.feed(_decode(downloaded))
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def _extract_html(downloaded: _Downloaded, url: str) -> str | None:
    from trafilatura import extract

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
    text = extract(downloaded.data, **options)
    if not text:
        # Relax precision before falling back to bounded, cleaned plain text.
        text = extract(downloaded.data, **options, favor_recall=True)
    return text


def fetch(url: str, max_chars: int | None = None) -> str:
    """Download a page and return bounded Markdown or readable plain text."""
    try:
        url = _normalized_url(url)
        limit = _limit(max_chars)
        downloaded = _download(url)
    except (TypeError, ValueError, _FetchError) as e:
        return f"error: {e}"

    if _is_html(downloaded):
        try:
            text = _extract_html(downloaded, url)
        except ImportError:
            return "error: trafilatura is not installed"
        if not text:
            text = _plain_text(downloaded)
    elif _is_text(downloaded):
        # Raw Markdown, source files, and JSON need no HTML extraction pass.
        text = _decode(downloaded)
    else:
        return f"error: unsupported content type: {downloaded.media_type or 'unknown'}"

    if not text or not text.strip():
        return "error: no readable text extracted"
    return _truncate(text, limit)


def tool_fetch(_cwd: Path | None, url: str, max_chars: int | None = None) -> str:
    """fetch tool handler: cwd is unused because fetching has no workspace path."""
    return fetch(url, max_chars)
