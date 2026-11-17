"""Web search through DeepSeek's Search API.

`fetch` answers "what does this page say"; `search` answers "which pages say
anything about this". The API returns ranked results with a long `summary`
(600-800 chars) that the model can read directly, so `need_content` stays off:
opening a page is what `fetch` is for.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .providers import get_provider

DEFAULT_COUNT = 8
MIN_COUNT, MAX_COUNT = 1, 50
MAX_QUERY_CHARS = 128  # the API rejects longer queries outright
MAX_SUMMARY_CHARS = 2_000  # the API promises 600-800, but serves whole pages sometimes
MAX_CONTENT_CHARS = 4_000
MAX_RESULTS_CHARS = 12_000
GOOD_SCORE = 1.5  # rerank_score >= 1.5 means "relevant" per the API docs
FALLBACK_RESULTS = 3  # never hide everything when no result clears GOOD_SCORE
DEFAULT_TIMEOUT = 30.0
RETRY_SLEEPS = (0.5, 1.5)
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
SEARCH_PATH = "/v1/tools/search"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class _HttpError(Exception):
    """One failed Search request: `status` 0 means the request never landed."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail or f"HTTP {status}")


def skill_path() -> Path:
    """The packaged, on-demand syntax and triage runbook for this tool."""
    return Path(__file__).resolve().parent / "skills" / "websearch" / "SKILL.md"


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized_query(query: Any) -> str:
    text = _clean(query)
    if not text:
        raise ValueError("query must be a non-empty string")
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(
            f"query is {len(text)} chars; the API accepts at most {MAX_QUERY_CHARS} — shorten it"
        )
    return text


def _limited_count(count: Any) -> int:
    if count is None:
        return DEFAULT_COUNT
    try:
        limit = int(count)
    except (TypeError, ValueError) as e:
        raise ValueError(f"count must be an integer, got {count!r}") from e
    if not MIN_COUNT <= limit <= MAX_COUNT:
        raise ValueError(f"count must be between {MIN_COUNT} and {MAX_COUNT}, got {limit}")
    return limit


def _truncated_question(question: Any) -> str | None:
    """`ori_question` sharpens intent; the API caps it at 128 chars like `query`."""
    text = _clean(question)
    return text[:MAX_QUERY_CHARS] or None


def _endpoint(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):  # providers may already point at the versioned root
        base = base[:-3]
    return base + SEARCH_PATH


def _credentials(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    if api_key or base_url:
        if not api_key:
            raise ValueError("api_key is required when base_url is set")
        return api_key, base_url or DEFAULT_BASE_URL
    provider = get_provider("deepseek")
    credential = provider.resolve() if provider else None
    if credential is None:
        raise ValueError(
            "no DeepSeek API key for web search — run `/login deepseek` or export DEEPSEEK_API_KEY"
        )
    return credential.api_key, credential.base_url


def _post(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    """One JSON POST. Raises _HttpError for anything that is not a 200 with JSON."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            error = json.loads(e.read().decode("utf-8", "replace")).get("error") or {}
            detail = _clean(error.get("message"))
        except (ValueError, OSError, AttributeError):
            pass
        raise _HttpError(e.code, detail) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise _HttpError(0, str(e)) from e
    try:
        return json.loads(body or "{}")
    except ValueError as e:
        raise _HttpError(0, f"unreadable response: {body[:200]}") from e


def _retryable(error: _HttpError) -> bool:
    # A dead connection is worth one more try; 402 and other 4xx are not.
    return error.status == 0 or error.status in RETRY_STATUS


def _failed(error: _HttpError) -> str:
    detail = f": {error.detail}" if error.detail else ""
    if error.status == 0:
        return f"error: search request failed{detail}"
    if error.status == 401:
        return "error: DeepSeek API key rejected for web search (401); check /login deepseek"
    if error.status == 402:
        return "error: DeepSeek balance is empty (402); top up before searching again"
    if error.status == 429:
        return "error: search rate limited (429); wait a moment before searching again"
    if error.status >= 500:
        return f"error: search service unavailable ({error.status}){detail} — try again later"
    return f"error: search rejected ({error.status}){detail}"


def _score(result: dict[str, Any]) -> float:
    value = result.get("rerank_score")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _indent(text: str) -> str:
    return "\n".join(f"   {line}" for line in text.splitlines())


def _format_result(index: int, result: dict[str, Any], *, with_content: bool) -> str:
    meta = [f"score {_score(result):.1f}"]
    authority = result.get("authority")
    if isinstance(authority, int) and authority >= 0:
        meta.append(f"authority {authority}")
    if date := _clean(result.get("date"))[:10]:
        meta.append(date)
    if site := _clean(result.get("site_name")):
        meta.append(site)
    title = _clean(result.get("title")) or "(untitled)"
    head = f"{index}. {title}  ·  " + "  ·  ".join(meta)
    lines = [head, f"   {_clean(result.get('url'))}"]
    if body := _clean(result.get("summary")) or _clean(result.get("snippet")):
        if len(body) > MAX_SUMMARY_CHARS:
            cut = len(body) - MAX_SUMMARY_CHARS
            body = (
                body[:MAX_SUMMARY_CHARS]
                + f"\n...[summary truncated {cut} chars; fetch the URL for the rest]"
            )
        lines.append(_indent(body))
    if with_content and (content := _clean(result.get("content"))):
        if len(content) > MAX_CONTENT_CHARS:
            cut = len(content) - MAX_CONTENT_CHARS
            content = content[:MAX_CONTENT_CHARS] + f"\n...[content truncated {cut} chars]"
        lines += ["   ---", _indent(content)]
    return "\n".join(lines)


def _format(query: str, results: list[Any], *, with_content: bool) -> str:
    usable = [r for r in results if isinstance(r, dict)]
    if not usable:
        return f"no results for {query!r}"
    kept = [r for r in usable if _score(r) >= GOOD_SCORE]
    notes: list[str] = []
    if not kept:
        kept = usable[:FALLBACK_RESULTS]
        notes.append(f"(no result scored >= {GOOD_SCORE}; showing the top {len(kept)})")
    elif len(kept) < len(usable):
        notes.append(f"({len(usable) - len(kept)} lower-scoring results hidden)")
    blocks = [_format_result(i, r, with_content=with_content) for i, r in enumerate(kept, 1)]
    text = "\n\n".join(blocks)
    if len(text) > MAX_RESULTS_CHARS:
        text = text[:MAX_RESULTS_CHARS] + f"\n...[truncated {len(text) - MAX_RESULTS_CHARS} chars]"
    return text + ("\n" + "\n".join(notes) if notes else "")


def search(
    query: str,
    count: int | None = None,
    need_content: bool = False,
    *,
    ori_question: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Search the web and return ranked, model-readable results as text."""
    try:
        text = _normalized_query(query)
        limit = _limited_count(count)
        key, base = _credentials(api_key, base_url)
    except (TypeError, ValueError) as e:
        return f"error: {e}"

    payload: dict[str, Any] = {"query": text, "count": limit}
    if need_content:
        payload["need_content"] = True
    if question := _truncated_question(ori_question):
        payload["ori_question"] = question

    url = _endpoint(base)
    for attempt in range(len(RETRY_SLEEPS) + 1):
        try:
            body = _post(url, payload, key, timeout)
            break
        except _HttpError as e:
            if attempt < len(RETRY_SLEEPS) and _retryable(e):
                time.sleep(RETRY_SLEEPS[attempt])
                continue
            return _failed(e)

    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or not results:
        return (
            f"no results for {text!r} — the search found nothing; "
            "broaden the query or drop restrictive operators"
        )
    return _format(text, results, with_content=need_content)


def tool_search(
    _cwd: Path,
    query: str,
    count: int | None = None,
    *,
    user_question: str | None = None,
) -> str:
    """search tool handler: cwd is unused because searching has no workspace path."""
    return search(query, count, ori_question=user_question)
