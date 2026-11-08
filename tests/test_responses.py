"""Responses wire format, raw-output replay, and streaming edge cases."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent7.agent import TOOLS, run_agent
from agent7.responses import (
    RAW_ITEMS_KEY,
    output_text,
    portable_history,
    request_input,
    response_tools,
    response_usage,
)
from responses_fakes import FakeClient, call_turn, completed, ns, text_turn


class ResponsesRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_agent(self, events: list[list[Any]], **kwargs: Any):
        client = FakeClient(events)
        messages: list[dict[str, Any]] = []
        answer = asyncio.run(run_agent(
            "go", cwd=self.cwd, client=client, model="model", messages=messages,
            on_tool=lambda *a: None, **kwargs,
        ))
        return answer, messages, client

    def test_uses_responses_input_and_flat_function_tools(self) -> None:
        answer, _, client = self.run_agent([text_turn("done")])

        self.assertEqual(answer, "done")
        request = client.requests[0]
        self.assertTrue(request["stream"])
        self.assertEqual(request["input"], [{"role": "user", "content": "go"}])
        self.assertNotIn("messages", request)
        self.assertNotIn("stream_options", request)
        self.assertNotIn("previous_response_id", request)
        self.assertEqual(request["tools"], response_tools(TOOLS))
        self.assertTrue(all(tool["type"] == "function" for tool in request["tools"]))
        self.assertNotIn("web_search", json.dumps(request["tools"]))

    def test_system_message_becomes_instructions(self) -> None:
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        inputs, instructions = request_input(messages)

        self.assertEqual(instructions, "rules")
        self.assertEqual(inputs, [{"role": "user", "content": "go"}])

    def test_raw_output_is_replayed_with_the_function_result(self) -> None:
        answer, messages, client = self.run_agent(
            [call_turn(("bash", {"command": "printf ok"}), reasoning="inspect"), text_turn("done")],
            provider_id="deepseek",
        )

        self.assertEqual(answer, "done")
        raw = messages[1][RAW_ITEMS_KEY]
        self.assertEqual(raw["provider"], "deepseek")
        self.assertEqual([item["type"] for item in raw["items"]], ["reasoning", "function_call"])
        inputs = client.requests[1]["input"]
        self.assertEqual([item.get("type") for item in inputs], [
            None, "reasoning", "function_call", "function_call_output",
        ])
        self.assertEqual(inputs[-1], {
            "type": "function_call_output", "call_id": "call_0", "output": "exit=0\nstdout:\nok",
        })

    def test_raw_output_does_not_cross_providers(self) -> None:
        _, messages, _ = self.run_agent(
            [call_turn(("bash", {"command": "printf ok"}), reasoning="private"), text_turn("done")],
            provider_id="kimi",
        )
        inputs, _ = request_input(messages, "xai")

        self.assertNotIn("private", json.dumps(inputs))
        self.assertNotIn("encrypted_content", json.dumps(inputs))
        types = [item.get("type") for item in inputs]
        self.assertIn("function_call", types)
        self.assertNotIn("reasoning", types)

    def test_deepseek_disables_thinking_only_for_portable_history(self) -> None:
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
        self.assertTrue(portable_history(messages, "deepseek"))
        client = FakeClient([text_turn("new")])
        asyncio.run(run_agent(
            "continue", cwd=self.cwd, client=client, model="deepseek-flash",
            messages=messages, provider_id="deepseek", on_tool=lambda *a: None,
        ))
        self.assertEqual(client.requests[0]["reasoning"], {"effort": "none"})

        _, fresh, _ = self.run_agent([text_turn("answer")], provider_id="deepseek")
        self.assertFalse(portable_history(fresh, "deepseek"))

    def test_kimi_does_not_receive_deepseek_fallback(self) -> None:
        client = FakeClient([text_turn("new")])
        asyncio.run(run_agent(
            "continue", cwd=self.cwd, client=client, model="k3",
            messages=[{"role": "assistant", "content": "old"}], provider_id="kimi",
            on_tool=lambda *a: None,
        ))
        self.assertNotIn("reasoning", client.requests[0])

    def test_parallel_calls_get_distinct_outputs(self) -> None:
        answer, messages, client = self.run_agent([
            call_turn([("bash", {"command": "printf one"}), ("bash", {"command": "printf two"})]),
            text_turn("done"),
        ])
        results = [message for message in messages if message.get("role") == "tool"]
        self.assertEqual(answer, "done")
        self.assertEqual([message["tool_call_id"] for message in results], ["call_0", "call_1"])
        outputs = [
            item for item in client.requests[1]["input"]
            if item.get("type") == "function_call_output"
        ]
        self.assertIn("one", outputs[0]["output"])
        self.assertIn("two", outputs[1]["output"])

    def test_compaction_discards_stale_raw_output(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "go"},
            *(
                item
                for i in range(5)
                for item in (
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                            "name": "bash", "arguments": "{}",
                        }}],
                        RAW_ITEMS_KEY: {"provider": "deepseek", "items": [
                            {"type": "reasoning", "content": [
                                {"type": "reasoning_text", "text": f"old {i}"},
                            ]},
                        ]},
                    } if i == 0 else {"role": "assistant", "content": ""},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 3_000},
                )
            )
        ]
        client = FakeClient([text_turn("done")])
        asyncio.run(run_agent(
            "continue", cwd=self.cwd, client=client, model="deepseek-flash",
            messages=messages, provider_id="deepseek", context_limit=1,
            on_tool=lambda *a: None,
        ))
        self.assertNotIn(RAW_ITEMS_KEY, messages[2])
        self.assertEqual(client.requests[0]["reasoning"], {"effort": "none"})
        self.assertNotIn("reasoning_text", json.dumps(client.requests[0]["input"]))

    def test_json_snapshot_can_hold_raw_output(self) -> None:
        _, messages, _ = self.run_agent([text_turn("answer")], provider_id="kimi")
        restored = json.loads(json.dumps(messages))
        inputs, _ = request_input(restored, "kimi")
        self.assertEqual(inputs[-1]["type"], "message")
        self.assertEqual(output_text(inputs[-1:]), "answer")


class ResponsesStreamTests(unittest.TestCase):
    def test_final_response_output_supplies_missing_item_done_events(self) -> None:
        message = {
            "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "from final", "annotations": []}],
        }
        client = FakeClient([[completed([message])]])
        messages: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as td:
            answer = asyncio.run(run_agent(
                "go", cwd=Path(td), client=client, model="model", messages=messages,
                on_tool=lambda *a: None,
            ))
        self.assertEqual(answer, "from final")
        self.assertEqual(messages[-1][RAW_ITEMS_KEY]["items"], [message])

    def test_final_function_call_without_item_done_still_executes(self) -> None:
        call = {
            "id": "fc_1", "type": "function_call", "call_id": "call_1",
            "name": "bash", "arguments": '{"command": "printf final"}',
        }
        client = FakeClient([[completed([call])], text_turn("done")])
        with tempfile.TemporaryDirectory() as td:
            answer = asyncio.run(run_agent(
                "go", cwd=Path(td), client=client, model="model", messages=[],
                on_tool=lambda *a: None,
            ))
        self.assertEqual(answer, "done")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[1]["input"][-1]["type"], "function_call_output")

    def test_failed_stream_does_not_return_an_empty_success(self) -> None:
        failure = ns(
            type="response.failed",
            response=ns(output=[], status="failed", incomplete_details=None,
                        usage=None, error=ns(message="provider failed")),
        )
        client = FakeClient([[failure]])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                asyncio.run(run_agent(
                    "go", cwd=Path(td), client=client, model="model", messages=[],
                    on_tool=lambda *a: None,
                ))

    def test_reasoning_deltas_stream_but_never_become_answer_text(self) -> None:
        thinking: list[str] = []
        client = FakeClient([text_turn("answer", reasoning="consider")])
        messages: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as td:
            answer = asyncio.run(run_agent(
                "go", cwd=Path(td), client=client, model="model", messages=messages,
                on_tool=lambda *a: None, on_thinking=thinking.append,
            ))
        self.assertEqual(answer, "answer")
        self.assertEqual(thinking, ["consider"])
        self.assertEqual(messages[-1]["content"], "answer")

    def test_usage_mapping(self) -> None:
        response = ns(usage=ns(input_tokens=10, output_tokens=4, total_tokens=14))
        self.assertEqual(response_usage(response), {
            "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
        })

    def test_output_text_extracts_only_message_text(self) -> None:
        self.assertEqual(output_text([
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hidden"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "visible"}]},
        ]), "visible")


if __name__ == "__main__":
    unittest.main()
