"""Generic chat-history estimation and compaction policies."""

from __future__ import annotations

from typing import Any

from .responses import IMAGES_KEY, RAW_ITEMS_KEY, discard_raw_output
from .types import TOOL_ATTACHMENTS_KEY, Recovery

COMPACT_THRESHOLD, KEEP_USER_TURNS, KEEP_TOOL_RESULTS = 0.75, 3, 4
TOOL_MAX_CHARS, ASSISTANT_MAX_CHARS = 2_000, 4_000
OVERFLOW_RETRIES = 2
IMAGE_TOKEN_ESTIMATE = 1000


def _truncate(s: str, limit: int, *, tag: str = "truncated") -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[{tag} {len(s) - limit} chars]"


def should_compact(
    prompt_tokens: int | None, context_limit: int | None, *, threshold: float = COMPACT_THRESHOLD
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
    if not messages:
        return messages, 0
    head, rest = ([], messages)
    if messages[0].get("role") == "system":
        head, rest = [messages[0]], messages[1:]
    user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    keep_from = user_idxs[-keep_user_turns] if len(user_idxs) >= keep_user_turns else 0
    tool_idxs = [i for i, m in enumerate(rest) if m.get("role") == "tool"]
    old_tool_before = keep_from or len(rest)
    if keep_from == 0 and tool_idxs:
        old_tool_before = (
            len(rest)
            if keep_tool_results <= 0
            else tool_idxs[max(0, len(tool_idxs) - keep_tool_results)]
        )
    out, changed = list(head), 0
    for i, m in enumerate(rest):
        role, content = m.get("role"), m.get("content")
        if role == "tool" and i < old_tool_before:
            updated = m
            image_blocks = m.get(IMAGES_KEY) or m.get(TOOL_ATTACHMENTS_KEY)
            if image_blocks:
                updated = {
                    **updated,
                    "content": (content if isinstance(content, str) else "")
                    + "\n[image omitted: compacted]",
                }
                updated.pop(IMAGES_KEY, None)
                updated.pop(TOOL_ATTACHMENTS_KEY, None)
                content = updated["content"]
            if isinstance(content, str) and len(content) > tool_max_chars:
                updated = {
                    **updated,
                    "content": _truncate(content, tool_max_chars, tag="compacted"),
                }
            if updated is not m:
                changed += 1
            out.append(updated)
        elif (
            role == "assistant"
            and i < keep_from
            and isinstance(content, str)
            and len(content) > assistant_max_chars
            and not m.get("tool_calls")
        ):
            out.append({**m, "content": _truncate(content, assistant_max_chars, tag="compacted")})
            changed += 1
        else:
            out.append(m)
    return out, changed


def without_orphan_tool_results(messages: list[dict[str, Any]]) -> None:
    """Drop tool results whose assistant call is no longer in history."""

    calls = {
        call.get("id")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    messages[:] = [
        message
        for message in messages
        if message.get("role") != "tool" or message.get("tool_call_id") in calls
    ]


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough size for providers that never report usage: ~4 chars per token."""

    chars = images = 0
    for message in messages:
        chars += len(str(message.get("content") or ""))
        images += len(message.get(IMAGES_KEY) or [])
        images += len(message.get(TOOL_ATTACHMENTS_KEY) or [])
        for call in message.get("tool_calls") or []:
            chars += len(str((call.get("function") or {}).get("arguments") or ""))
    return chars // 4 + images * IMAGE_TOKEN_ESTIMATE + 1


class ResponsesHistoryPolicy:
    """OpenAI-compatible Responses compaction and safe-retry policy."""

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        return estimate_tokens(messages)

    def should_compact(self, prompt_tokens: int | None, context_limit: int | None) -> bool:
        return should_compact(prompt_tokens, context_limit)

    def compact(
        self, messages: list[dict[str, Any]], *, aggressive: bool = False
    ) -> tuple[list[dict[str, Any]], int]:
        if aggressive:
            new_messages, changed = compact_messages(
                messages, keep_user_turns=1, keep_tool_results=1
            )
        else:
            new_messages, changed = compact_messages(messages)
        if changed:
            discard_raw_output(new_messages)
            without_orphan_tool_results(new_messages)
        return new_messages, changed

    def recover(
        self,
        messages: list[dict[str, Any]],
        error: Exception,
        *,
        attempt: int,
    ) -> Recovery | None:
        from .openai_backend import is_context_overflow, is_tool_history_error

        if is_tool_history_error(error) and any(RAW_ITEMS_KEY in message for message in messages):
            discard_raw_output(messages)
            return Recovery(messages)
        if attempt > OVERFLOW_RETRIES or not is_context_overflow(error):
            return None
        new_messages, changed = self.compact(messages, aggressive=True)
        return Recovery(new_messages, changed) if changed else None
