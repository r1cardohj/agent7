"""Small Responses API fakes shared by agent-loop tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def ns(**kwargs: Any) -> Any:
    return SimpleNamespace(**kwargs)


def response(output: list[dict[str, Any]], *, status: str = "completed",
             usage: dict[str, int] | None = None, reason: str | None = None) -> Any:
    counts = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    counts.update(usage or {})
    return ns(
        output=output, status=status,
        incomplete_details=ns(reason=reason) if reason else None,
        usage=ns(**counts), error=None,
    )


def completed(output: list[dict[str, Any]], **kwargs: Any) -> Any:
    return ns(type="response.completed", response=response(output, **kwargs))


def text_turn(content: str = "", *, finish_reason: str | None = None,
              reasoning: str | None = None) -> list[Any]:
    output: list[dict[str, Any]] = []
    events: list[Any] = []
    if reasoning:
        item = {"id": "rs_1", "type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": reasoning},
        ]}
        output.append(item)
        events.append(ns(type="response.reasoning_text.delta", delta=reasoning))
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    output.append(message)
    events.append(ns(type="response.output_text.delta", delta=content))
    events.append(ns(type="response.output_item.done", item=ns(**message)))
    kwargs: dict[str, Any] = {}
    if finish_reason == "length":
        kwargs.update(status="incomplete", reason="max_output_tokens")
    events.append(completed(output, **kwargs))
    return events


def call_turn(calls: list[tuple[str, dict[str, Any]]] | tuple[str, dict[str, Any]],
              *, reasoning: str | None = None) -> list[Any]:
    if isinstance(calls, tuple):
        calls = [calls]
    output: list[dict[str, Any]] = []
    events: list[Any] = []
    if reasoning:
        item = {"id": "rs_1", "type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": reasoning},
        ]}
        output.append(item)
        events.append(ns(type="response.reasoning_text.delta", delta=reasoning))
        events.append(ns(type="response.output_item.done", item=ns(**item)))
    for index, (name, arguments) in enumerate(calls):
        item = {
            "id": f"fc_{index}", "type": "function_call", "call_id": f"call_{index}",
            "name": name, "arguments": json.dumps(arguments), "status": "completed",
        }
        output.append(item)
        events.append(ns(type="response.output_item.done", item=ns(**item)))
    events.append(completed(output))
    return events


def usage_turn(content: str = "ok", usage: dict[str, int] | None = None) -> list[Any]:
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    return [
        ns(type="response.output_text.delta", delta=content),
        ns(type="response.output_item.done", item=ns(**message)),
        completed([message], usage=usage),
    ]


class FakeStream:
    def __init__(self, events: list[Any], fail: BaseException | None = None) -> None:
        self.events, self.fail = events, fail

    def __aiter__(self):
        async def gen():
            for event in self.events:
                yield event
            if self.fail:
                raise self.fail
        return gen()

    async def close(self) -> None:
        pass


class FakeResponses:
    """Serve scripted events/errors and record Responses request bodies."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.calls = 0

    async def create(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        events, fail = item if isinstance(item, tuple) else (item, None)
        return FakeStream(events, fail)


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.responses = FakeResponses(script)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.responses.requests

    @property
    def calls(self) -> int:
        return self.responses.calls
