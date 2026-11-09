"""OpenAI Responses protocol helpers.

The session stores ordinary conversation/tool messages. Provider-specific output
items live in assistant messages' private ``_responses`` state: they are required
by some Responses implementations on the next request, but are never presented
as chat messages to the UI or to another provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RAW_ITEMS_KEY = "_responses"

@dataclass
class ResponseTurn:
    """One completed model round, independent of the provider SDK's types."""

    text: str = ""
    output: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [item for item in self.output if item.get("type") == "function_call"]

def plain_item(value: Any) -> dict[str, Any]:
    """Convert an SDK output item/event to a JSON-safe dict, dropping nulls."""
    if isinstance(value, dict):
        data = value
    elif hasattr(value, "model_dump"):
        data = value.model_dump()
    elif hasattr(value, "__dict__"):
        data = vars(value)
    else:
        data = dict(value)
    return {k: plain_value(v) for k, v in data.items() if v is not None}

def plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: plain_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [plain_value(v) for v in value]
    if hasattr(value, "model_dump"):
        return plain_value(value.model_dump())
    return value

def raw_output(message: dict[str, Any], provider_id: str | None) -> list[dict[str, Any]] | None:
    """Return raw items only on the provider that produced them."""
    state = message.get(RAW_ITEMS_KEY)
    if not isinstance(state, dict) or state.get("provider") != provider_id:
        return None
    items = state.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return None
    return items or None


def request_input(
    messages: list[dict[str, Any]], provider_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Project chat-shaped history to Responses input and optional instructions.

    Raw output items replace their assistant message only for the provider that
    produced them. This keeps compaction and provider switching stateless.
    """
    instructions = None
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions = message.get("content") or None
        elif role in ("user", "assistant"):
            raw = raw_output(message, provider_id) if role == "assistant" else None
            if raw is not None:
                out.extend(raw)
                continue
            out.append({"role": role, "content": message.get("content") or ""})
            for call in message.get("tool_calls") or []:
                out.append(legacy_function_call(call))
        elif role == "tool":
            out.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": message.get("content") or "",
            })
    return out, instructions

def legacy_function_call(call: dict[str, Any]) -> dict[str, Any]:
    """Read a version-1 snapshot without inventing unavailable reasoning state."""
    function = call.get("function") or {}
    return {
        "type": "function_call",
        "call_id": call.get("id", ""),
        "name": function.get("name", ""),
        "arguments": function.get("arguments") or "{}",
    }

def portable_history(messages: list[dict[str, Any]], provider_id: str | None) -> bool:
    """Whether history contains assistant output without this provider's raw items.

    DeepSeek rejects thinking-mode requests containing old assistant turns whose
    reasoning is unavailable (legacy snapshots, compaction, provider switches).
    """
    return any(
        message.get("role") == "assistant" and raw_output(message, provider_id) is None
        for message in messages
    )


def discard_raw_output(messages: list[dict[str, Any]]) -> None:
    """Use portable text after compaction makes raw output items stale."""
    for message in messages:
        message.pop(RAW_ITEMS_KEY, None)


def response_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat function definitions have one extra ``function`` wrapper."""
    converted = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append({
            "type": "function",
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        })
    return converted

def _first_int(get: Any, *keys: str) -> int | None:
    for key in keys:
        value = get(key)
        if value is not None:
            return int(value)
    return None

def response_usage(response: Any) -> dict[str, int] | None:
    """Map Responses or Chat Completions usage onto the UI's token names.

    Responses reports input_tokens/output_tokens. Some compatible endpoints
    still report prompt_tokens/completion_tokens. Prefer the Responses names.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else (
        lambda key, default=None: getattr(usage, key, default)
    )
    prompt = _first_int(get, "input_tokens", "prompt_tokens")
    completion = _first_int(get, "output_tokens", "completion_tokens")
    total = _first_int(get, "total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    out = {
        key: value
        for key, value in (
            ("prompt_tokens", prompt),
            ("completion_tokens", completion),
            ("total_tokens", total),
        )
        if value is not None
    }
    return out or None

def response_status(response: Any) -> tuple[str | None, str | None]:
    """Return (UI finish reason, incomplete reason).

    Responses reports status/incomplete_details. Compatible endpoints may still
    report Chat Completions finish_reason. Known cases become stop/length so
    the truncation notice does not depend on which name arrived.
    """
    get = response.get if isinstance(response, dict) else (
        lambda key, default=None: getattr(response, key, default)
    )
    status = get("status")
    details = get("incomplete_details") or {}
    reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
    finish = get("finish_reason")
    if reason == "max_output_tokens" or finish == "length":
        return "length", reason
    if status == "completed" or finish == "stop":
        return "stop", reason
    if isinstance(finish, str) and finish:
        return finish, reason
    return status if isinstance(status, str) else None, reason

def output_text(items: list[dict[str, Any]]) -> str:
    """Fallback for providers that send output items without text deltas."""
    parts = []
    for item in items:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)

def reasoning_text(item: dict[str, Any]) -> str:
    parts = [part.get("text", "") for part in item.get("content") or []]
    parts += [part.get("text", "") for part in item.get("summary") or []]
    return "".join(part for part in parts if isinstance(part, str))

def public_call(call: dict[str, Any]) -> dict[str, Any]:
    """Project a Responses function call to the existing agent loop's shape."""
    return {
        "id": call.get("call_id") or call.get("id") or "",
        "type": "function",
        "function": {
            "name": call.get("name") or "",
            "arguments": call.get("arguments") or "{}",
        },
    }


def raw_state(provider_id: str | None, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"provider": provider_id, "items": items}
