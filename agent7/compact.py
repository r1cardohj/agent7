"""Simple context compaction: shrink old tool / assistant payloads."""

from __future__ import annotations

from typing import Any

THRESHOLD = 0.75
KEEP_USER_TURNS = 3
KEEP_TOOL_RESULTS = 4
TOOL_MAX_CHARS = 2_000
ASSISTANT_MAX_CHARS = 4_000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[compacted {len(text) - limit} chars]"


def should_compact(
    prompt_tokens: int | None,
    context_limit: int | None,
    *,
    threshold: float = THRESHOLD,
) -> bool:
    if prompt_tokens is None or context_limit is None or context_limit <= 0:
        return False
    return prompt_tokens >= int(context_limit * threshold)


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_user_turns: int = KEEP_USER_TURNS,
    keep_tool_results: int = KEEP_TOOL_RESULTS,
    tool_max_chars: int = TOOL_MAX_CHARS,
    assistant_max_chars: int = ASSISTANT_MAX_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    """
    Shrink old tool results and prose without dropping any messages.

    In a short history, tool results get their own recency boundary: a one-shot
    task can run many tool turns without another user message, so keep the last
    `keep_tool_results` intact and compact older ones. Once there are
    `keep_user_turns` user messages, use that turn boundary for all content.
    """
    if not messages:
        return messages, 0

    head: list[dict[str, Any]] = []
    rest = messages
    if messages[0].get("role") == "system":
        head = [messages[0]]
        rest = messages[1:]

    user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if len(user_idxs) >= keep_user_turns:
        keep_from = user_idxs[-keep_user_turns]
    else:
        keep_from = 0  # short history: prose is still potentially relevant

    tool_idxs = [i for i, m in enumerate(rest) if m.get("role") == "tool"]
    old_tool_before = keep_from or len(rest)
    if keep_from == 0 and keep_tool_results >= 0 and tool_idxs:
        old_tool_before = tool_idxs[max(0, len(tool_idxs) - keep_tool_results)]

    out = list(head)
    changed = 0
    for i, m in enumerate(rest):
        role = m.get("role")
        content = m.get("content")
        if (
            role == "tool"
            and i < old_tool_before
            and isinstance(content, str)
            and len(content) > tool_max_chars
        ):
            out.append({**m, "content": _truncate(content, tool_max_chars)})
            changed += 1
        elif (
            role == "assistant"
            and i < keep_from
            and isinstance(content, str)
            and len(content) > assistant_max_chars
            and not m.get("tool_calls")
        ):
            out.append({**m, "content": _truncate(content, assistant_max_chars)})
            changed += 1
        else:
            out.append(m)
    return out, changed
