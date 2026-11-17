"""Headless runtime contracts, independent of Responses and the terminal."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent7.core import (
    TOOL_ATTACHMENTS_KEY,
    Agent,
    AgentInterrupted,
    Attachment,
    BuiltinToolsConfig,
    CancellationToken,
    EventBus,
    LifecycleHooks,
    MemorySessionStore,
    ModelTurn,
    PluginError,
    RunContext,
    RunStatus,
    RuntimeSession,
    Shell,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    create_builtin_registry,
)


class FakeBackend:
    model = "fake-model"
    provider_id = "fake"
    context_limit = None

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def complete(self, request, events):
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        if item.text:
            await events.emit("text_delta", turn=request.turn, text=item.text)
        await events.emit("model_finished", turn=request.turn, finish_reason=item.finish_reason)
        return item


def echo_tool(name="echo", handler=None):
    return Tool.define(
        name,
        "Return the supplied text.",
        {"text": {"type": "string"}},
        handler or (lambda _context, args: args["text"]),
        required=["text"],
    )


class CoreLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_custom_tool_events_and_typed_result(self):
        backend = FakeBackend(
            [
                ModelTurn(calls=[ToolCall("call-1", "echo", '{"text":"hello"}')]),
                ModelTurn(
                    text="done",
                    finish_reason="stop",
                    usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                    state={"_wire": {"private": True}, "role": "cannot-override"},
                ),
            ]
        )
        registry = ToolRegistry((echo_tool(),))
        events = []
        agent = Agent(backend, tools=registry)
        agent.events.subscribe(events.append)
        messages = []

        result = asyncio.run(agent.run("go", cwd=self.cwd, messages=messages))

        self.assertEqual(result.status, RunStatus.SUCCEEDED)
        self.assertEqual(result.text, "done")
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.legacy_text, "done")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[2]["content"], "hello")
        self.assertEqual(messages[-1]["_wire"], {"private": True})
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual([tool["function"]["name"] for tool in backend.requests[0].tools], ["echo"])
        kinds = [event.kind for event in events]
        self.assertEqual(
            kinds,
            [
                "turn_started",
                "model_started",
                "model_finished",
                "tool_started",
                "tool_finished",
                "model_started",
                "text_delta",
                "model_finished",
                "usage",
                "turn_succeeded",
            ],
        )
        self.assertEqual([event.seq for event in events], list(range(1, len(events) + 1)))
        self.assertEqual({event.run_id for event in events}, {result.run_id})
        self.assertEqual(events[3].turn, 1)
        self.assertEqual(events[4].payload["result"], "hello")
        # Event payloads are transport-safe; no SDK/UI objects leak into them.
        json.dumps([event.to_dict() for event in events])

    def test_tool_result_attachments_are_provider_neutral(self):
        attachment = Attachment(data="QUJD", mime_type="image/png", kind="image")

        def attach_image(_context, _args):
            return ToolResult("image result", attachments=(attachment,))

        registry = ToolRegistry(
            (
                Tool.define(
                    "attach",
                    "Return an attachment.",
                    {},
                    attach_image,
                ),
            )
        )
        backend = FakeBackend(
            [ModelTurn(calls=[ToolCall("call-1", "attach", "{}")]), ModelTurn(text="done")]
        )
        messages = []

        result = asyncio.run(
            Agent(backend, tools=registry).run("go", cwd=self.cwd, messages=messages)
        )

        self.assertEqual(result.text, "done")
        tool_message = next(message for message in messages if message.get("role") == "tool")
        self.assertEqual(
            tool_message[TOOL_ATTACHMENTS_KEY],
            [{"kind": "image", "mime_type": "image/png", "data": "QUJD"}],
        )

    def test_turn_limit_is_a_typed_blocked_result(self):
        backend = FakeBackend([ModelTurn(calls=[ToolCall("call-1", "echo", '{"text":"again"}')])])
        agent = Agent(backend, tools=ToolRegistry((echo_tool(),)))

        result = asyncio.run(agent.run("go", cwd=self.cwd, messages=[], max_turns=1))

        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.stop_reason, "max_turns")
        self.assertEqual(result.text, "")
        self.assertEqual(result.legacy_text, "(stopped after 1 turns)")

    def test_cooperative_cancellation_answers_every_pending_call(self):
        token = CancellationToken()

        def cancel_inside_tool(_context, _args):
            token.cancel("host stopped")
            return "finished side effect"

        backend = FakeBackend(
            [
                ModelTurn(
                    calls=[
                        ToolCall("call-1", "echo", "{}"),
                        ToolCall("call-2", "echo", "{}"),
                    ]
                )
            ]
        )
        agent = Agent(backend, tools=ToolRegistry((echo_tool(handler=cancel_inside_tool),)))
        events = []
        agent.events.subscribe(events.append)
        messages = []

        with self.assertRaises(AgentInterrupted) as caught:
            asyncio.run(agent.run("go", cwd=self.cwd, messages=messages, cancellation=token))

        self.assertEqual(caught.exception.reason, "host stopped")
        self.assertTrue(caught.exception.run_id.startswith("run-"))
        answers = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in answers], ["call-1", "call-2"])
        self.assertTrue(all(m["content"] == "error: interrupted" for m in answers))
        self.assertEqual(events[-1].kind, "turn_interrupted")
        self.assertNotIn("tool_finished", [event.kind for event in events])

    def test_lifecycle_uses_real_session_and_matching_run_id(self):
        backend = FakeBackend([ModelTurn(text="done")])
        hooks = LifecycleHooks()
        lifecycle = []
        hooks.subscribe(lifecycle.append)
        session = SimpleNamespace(
            id="session-gui",
            cwd=self.cwd,
            model="fake-model",
            model_ref="fake/fake-model",
            hooks=hooks,
        )
        agent = Agent(backend, lifecycle=hooks)

        result = asyncio.run(agent.run("go", cwd=self.cwd, messages=[], session=session))

        self.assertEqual([event.kind for event in lifecycle], ["turn_started", "turn_succeeded"])
        self.assertEqual({event.session_id for event in lifecycle}, {"session-gui"})
        self.assertEqual({event.run_id for event in lifecycle}, {result.run_id})
        self.assertTrue(all(event.workspace_id for event in lifecycle))


class EventBusTests(unittest.TestCase):
    def test_async_observers_fault_isolation_and_unsubscribe(self):
        bus = EventBus()
        seen = []

        def broken(_event):
            raise RuntimeError("observer failed")

        async def observe(event):
            await asyncio.sleep(0)
            seen.append(event.kind)

        unsubscribe_broken = bus.subscribe(broken)
        unsubscribe = bus.subscribe(observe, kinds=["model_started"])
        emitter = bus.for_run("run-test")

        async def exercise():
            await emitter.emit("model_started", turn=1)
            unsubscribe_broken()
            await emitter.emit("text_delta", text="ignored by filtered observer")
            unsubscribe()
            await emitter.emit("model_started", turn=2)

        asyncio.run(exercise())

        self.assertEqual(seen, ["model_started"])
        self.assertEqual(len(bus.errors), 1)
        self.assertEqual(str(bus.errors[0].error), "observer failed")
        self.assertEqual(bus.errors[0].event.run_id, "run-test")


class BuiltinToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_generic_registry_runs_without_agent7_services(self):
        class LocalSearch:
            def __init__(self):
                self.seen = []

            def search(self, query, count=None, *, user_question=None):
                self.seen.append((query, count, user_question))
                return f"local result for {query}"

        provider = LocalSearch()
        registry = create_builtin_registry(
            BuiltinToolsConfig(shell=Shell("bash", ("/bin/bash", "-lc")), search_provider=provider)
        )
        target = self.cwd / "note.txt"
        target.write_text("hello\n", encoding="utf-8")
        backend = FakeBackend(
            [
                ModelTurn(calls=[ToolCall("call-1", "read", '{"path":"note.txt"}')]),
                ModelTurn(calls=[ToolCall("call-2", "search", '{"query":"agent"}')]),
                ModelTurn(text="done"),
            ]
        )
        messages = []

        result = asyncio.run(
            Agent(backend, tools=registry).run("go", cwd=self.cwd, messages=messages)
        )

        self.assertEqual(result.text, "done")
        self.assertEqual(messages[2]["content"], "1|hello")
        self.assertEqual(messages[4]["content"], "local result for agent")
        self.assertEqual(provider.seen, [("agent", None, "go")])
        self.assertEqual(registry.names(), ("read", "write", "edit", "bash", "fetch", "search"))


class GenericSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_runtime_session_runs_without_a_workspace(self):
        session = RuntimeSession(id="assistant-1", metadata={"model": "assistant"})
        store = MemorySessionStore()
        registry = create_builtin_registry(
            BuiltinToolsConfig(shell=Shell("bash", ("/bin/bash", "-lc")))
        )
        backend = FakeBackend(
            [
                ModelTurn(calls=[ToolCall("call-1", "read", '{"path":"any.txt"}')]),
                ModelTurn(text="assistant answer", finish_reason="stop"),
            ]
        )
        context = RunContext(session=session, run_id="run-assistant")

        result = asyncio.run(Agent(backend, tools=registry).run_in_context(context, "hello there"))

        self.assertEqual(result.text, "assistant answer")
        self.assertEqual(result.session_id, "assistant-1")
        self.assertEqual(result.run_id, "run-assistant")
        tool_message = next(m for m in session.messages if m.get("role") == "tool")
        self.assertEqual(tool_message["content"], "error: no workspace is configured")

        asyncio.run(store.save(session))
        loaded = asyncio.run(store.load("assistant-1"))
        self.assertEqual(loaded.id, "assistant-1")
        self.assertEqual(loaded.messages, session.messages)
        loaded.messages.append({"role": "user", "content": "mutation"})
        self.assertNotEqual(loaded.messages, session.messages)

    def test_runtime_session_lifecycle_needs_no_cwd_or_model_attribute(self):
        session = RuntimeSession(id="assistant-2")
        hooks = LifecycleHooks()
        events = []
        hooks.subscribe(events.append)
        backend = FakeBackend([ModelTurn(text="done")])
        agent = Agent(backend, lifecycle=hooks)

        result = asyncio.run(agent.run("hello", session=session))

        self.assertEqual(result.text, "done")
        observed = [
            (event.kind, event.session_id, event.cwd, event.model, event.run_id) for event in events
        ]
        self.assertEqual(
            observed,
            [
                ("turn_started", "assistant-2", None, "", result.run_id),
                ("turn_succeeded", "assistant-2", None, "", result.run_id),
            ],
        )


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    class DemoPlugin:
        api_version = 1
        name = "demo"

        def __init__(self):
            self.observed = []
            self.closed = 0

        def register(self, context):
            context.tools.register(echo_tool("plugin_echo"))
            context.backends.register("demo-backend", lambda: "backend-instance")
            context.events.subscribe(self.observed.append, kinds=["turn_succeeded"])

        async def close(self):
            self.closed += 1

    def test_programmatic_plugin_registers_tools_backend_and_observer(self):
        plugin = self.DemoPlugin()
        backend = FakeBackend(
            [
                ModelTurn(calls=[ToolCall("call-1", "plugin_echo", '{"text":"from plugin"}')]),
                ModelTurn(text="done"),
            ]
        )
        agent = Agent(backend, plugins=[plugin])

        async def exercise():
            result = await agent.run("go", cwd=self.cwd, messages=[])
            await agent.close()
            return result

        result = asyncio.run(exercise())

        self.assertEqual(result.text, "done")
        self.assertEqual(agent.plugins.loaded, ("demo",))
        self.assertEqual(agent.backends.create("demo-backend"), "backend-instance")
        self.assertEqual(len(plugin.observed), 1)
        self.assertEqual(plugin.observed[0].run_id, result.run_id)
        self.assertEqual(plugin.closed, 1)

    def test_discovery_does_not_import_until_explicitly_enabled(self):
        loaded = []
        plugin = self.DemoPlugin()

        class Point:
            name = "demo"
            value = "demo:Plugin"
            dist = SimpleNamespace(name="demo-dist")

            def load(self):
                loaded.append(self.name)
                return plugin

        point = Point()

        class Points:
            def select(self, *, group):
                return [point] if group == "agent0.plugins" else []

        agent = Agent(FakeBackend([ModelTurn(text="done")]))
        with patch("agent7.core.plugins.metadata.entry_points", return_value=Points()):
            discovered = agent.plugins.discover()
            self.assertEqual(loaded, [])
            self.assertEqual(discovered[0].distribution, "demo-dist")
            self.assertEqual(agent.plugins.load_discovered(["demo"]), ("demo",))
            self.assertEqual(loaded, ["demo"])
            with self.assertRaisesRegex(PluginError, "not installed"):
                agent.plugins.load_discovered(["missing"])

    def test_incompatible_plugin_is_rejected(self):
        class OldPlugin:
            api_version = 0

            def register(self, _context):
                pass

        with self.assertRaisesRegex(PluginError, "expected 1"):
            Agent(FakeBackend([ModelTurn(text="done")]), plugins=[OldPlugin()])


if __name__ == "__main__":
    unittest.main()
