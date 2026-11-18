"""CLI assembly integrates runtime tools, events, plugins, and host clients."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent0_runtime import RunStatus, Tool, ToolRegistry
from responses_fakes import FakeClient, call_turn, text_turn

from agent7.agent import create_tool_registry, run_agent_result


class ClosingFakeClient(FakeClient):
    def __init__(self, script):
        super().__init__(script)
        self.closed = 0

    async def close(self):
        self.closed += 1


class PluginTool:
    api_version = 1
    name = "integration-tool"

    def register(self, context):
        context.tools.register(
            Tool.define(
                "plugin_echo",
                "Echo plugin text.",
                {"text": {"type": "string"}},
                lambda _context, args: f"plugin:{args['text']}",
                required=["text"],
            )
        )


class RuntimeIntegrationTests(unittest.TestCase):
    def test_plugin_tool_callbacks_and_structured_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient([call_turn(("plugin_echo", {"text": "hello"})), text_turn("done")])
            events, deltas, tools = [], [], []
            messages = []
            result = asyncio.run(
                run_agent_result(
                    "go",
                    cwd=Path(tmp),
                    client=client,
                    model="model",
                    messages=messages,
                    registry=create_tool_registry(),
                    plugins=[PluginTool()],
                    on_event=events.append,
                    on_delta=deltas.append,
                    on_tool=lambda *args: tools.append(args),
                    run_id="run-gui",
                )
            )

            self.assertEqual(result.status, RunStatus.SUCCEEDED)
            self.assertEqual(result.run_id, "run-gui")
            self.assertEqual(deltas, ["done"])
            self.assertEqual(tools[0][3], "plugin:hello")
            self.assertIn("plugin_echo", json.dumps(client.requests[0]["tools"]))
            self.assertIn("plugin:hello", json.dumps(client.requests[1]["input"]))
            self.assertIn("tool_started", [event.kind for event in events])
            self.assertIn("turn_succeeded", [event.kind for event in events])
            self.assertEqual({event.run_id for event in events}, {"run-gui"})
            self.assertEqual([event.seq for event in events], list(range(1, len(events) + 1)))
            json.dumps([event.to_dict() for event in events])

    def test_host_owned_client_is_not_closed(self):
        client = ClosingFakeClient([text_turn("ok")])
        result = asyncio.run(
            run_agent_result(
                "hello",
                client=client,
                model="model",
                messages=[],
                registry=ToolRegistry(),
            )
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(client.closed, 0)


if __name__ == "__main__":
    unittest.main()
