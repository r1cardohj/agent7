"""Ctrl-C inside a tool batch must leave a history the next request can replay."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from agent7.agent import Interrupted, _stream_once, dispatch_tool, run_agent
from responses_fakes import FakeClient, call_turn


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

        client = FakeClient([call_turn([
            ("bash", {"command": f"echo {i}"}) for i in range(3)
        ])])
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

    def test_close_that_ignores_cancellation_cannot_delay_interrupt(self) -> None:
        class StubbornStream:
            def __aiter__(self):
                async def gen():
                    await asyncio.Event().wait()
                    yield None
                return gen()

            async def close(self) -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await asyncio.Event().wait()

        stream = StubbornStream()
        client = SimpleNamespace(responses=SimpleNamespace(
            create=AsyncMock(return_value=stream)
        ))

        async def interrupt_stream() -> None:
            task = asyncio.create_task(_stream_once(
                client, {}, on_delta=None, on_thinking=None,
                on_answer_to_thinking=None, on_finish=None,
            ))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            await task

        with patch("agent7.agent.STREAM_CLOSE_TIMEOUT", 0.01):
            with self.assertRaises(Interrupted):
                asyncio.run(asyncio.wait_for(interrupt_stream(), timeout=0.1))

    def test_interrupted_stream_close_has_a_timeout(self) -> None:
        class StuckStream:
            closed = False

            def __aiter__(self):
                async def gen():
                    await asyncio.Event().wait()
                    yield None
                return gen()

            async def close(self) -> None:
                self.closed = True
                await asyncio.Event().wait()

        stream = StuckStream()
        client = SimpleNamespace(responses=SimpleNamespace(
            create=AsyncMock(return_value=stream)
        ))

        async def interrupt_stream() -> None:
            task = asyncio.create_task(_stream_once(
                client, {}, on_delta=None, on_thinking=None,
                on_answer_to_thinking=None, on_finish=None,
            ))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            await task

        with patch("agent7.agent.STREAM_CLOSE_TIMEOUT", 0.01):
            with self.assertRaises(Interrupted):
                asyncio.run(asyncio.wait_for(interrupt_stream(), timeout=1))
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
