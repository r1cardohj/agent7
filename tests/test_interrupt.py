"""Ctrl-C inside a tool batch must leave a history the next request can replay."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent7.agent import Interrupted, dispatch_tool, run_agent


def _chunk(*, content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=None, model_extra=None
    )
    choices = [SimpleNamespace(delta=delta)] if (content is not None or tool_calls) else []
    return SimpleNamespace(choices=choices, usage=None)


def _calls(n: int) -> list[Any]:
    return [
        SimpleNamespace(
            index=i,
            id=f"call_{i}",
            function=SimpleNamespace(name="bash", arguments=json.dumps({"command": f"echo {i}"})),
        )
        for i in range(n)
    ]


class _Stream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def close(self) -> None:
        pass


class _Client:
    def __init__(self, turns: list[list[Any]]) -> None:
        self._turns = list(turns)
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> _Stream:
        return _Stream(self._turns.pop(0))


class InterruptedToolBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_every_tool_call_gets_an_answer(self) -> None:
        messages: list[dict[str, Any]] = []
        seen: list[str] = []

        def flaky(cwd: Path, name: str, args: dict[str, Any], **kwargs: Any) -> str:
            seen.append(name)
            if len(seen) == 2:  # Ctrl-C lands on the second tool of the batch
                raise KeyboardInterrupt
            return dispatch_tool(cwd, name, args, **kwargs)

        client = _Client([[_chunk(tool_calls=_calls(3))]])
        with patch("agent7.agent.dispatch_tool", side_effect=flaky):
            with self.assertRaises(Interrupted):
                asyncio.run(
                    run_agent(
                        "go", cwd=self.cwd, client=client, model="test-model",
                        messages=messages, on_tool=lambda *a: None,
                    )
                )

        answered = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
        self.assertEqual(answered, ["call_0", "call_1", "call_2"])


if __name__ == "__main__":
    unittest.main()
