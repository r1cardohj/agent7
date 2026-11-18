"""A second consumer style: an assistant that uses none of agent7's coding pieces."""

import asyncio
import os
import unittest
from unittest.mock import patch

from agent7.core import (
    Agent,
    MemorySessionStore,
    ModelTurn,
    OpenAIResponsesBackend,
    RuntimeSession,
    Tool,
    ToolCall,
    ToolRegistry,
    get_provider,
    resolve_all_env,
)


class ScriptedAssistantBackend:
    """The generic backend protocol requires only complete()."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def complete(self, request, events):
        self.requests.append(request)
        item = self.script.pop(0)
        if item.text:
            await events.emit("text_delta", turn=request.turn, text=item.text)
        await events.emit("model_finished", turn=request.turn, finish_reason="stop")
        return item


class AssistantRuntimeTests(unittest.TestCase):
    def test_memory_assistant_uses_a_generic_session_and_custom_tool(self):
        memory = {}

        def remember(_context, args):
            memory[args["key"]] = args["value"]
            return f"remembered {args['key']}"

        registry = ToolRegistry(
            (
                Tool.define(
                    "remember",
                    "Store a value in assistant memory.",
                    {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    remember,
                    required=["key", "value"],
                ),
            )
        )
        session = RuntimeSession(
            id="assistant-memory",
            metadata={"persona": "general assistant"},
        )
        store = MemorySessionStore()
        backend = ScriptedAssistantBackend(
            [
                ModelTurn(calls=[ToolCall("call-1", "remember", '{"key":"user","value":"Hou"}')]),
                ModelTurn(text="I remembered that."),
                ModelTurn(text="Continuing with your remembered name."),
            ]
        )
        agent = Agent(backend, tools=registry)

        first = asyncio.run(agent.run("remember my name is Hou", session=session))
        second = asyncio.run(agent.run("continue", session=session))

        self.assertEqual(first.text, "I remembered that.")
        self.assertEqual(second.text, "Continuing with your remembered name.")
        self.assertEqual(memory, {"user": "Hou"})
        self.assertEqual(session.metadata["persona"], "general assistant")
        self.assertEqual(
            [tool["function"]["name"] for tool in backend.requests[0].tools], ["remember"]
        )
        # The session is portable application state: an app can choose its own store.
        asyncio.run(store.save(session))
        restored = asyncio.run(store.load("assistant-memory"))
        self.assertGreater(len(restored.messages), 4)
        self.assertIn({"role": "user", "content": "remember my name is Hou"}, restored.messages)

    def test_provider_registry_is_available_without_agent7_auth(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            deepseek = get_provider("ds")
            self.assertEqual(deepseek.id, "deepseek")
            self.assertEqual(deepseek.default_model, "deepseek-chat")
            resolved = resolve_all_env()
            self.assertEqual(set(resolved), {"deepseek"})
            self.assertEqual(resolved["deepseek"].source, "env")
        kimi = get_provider("kimi")
        self.assertEqual(kimi.base_url, "https://api.kimi.com/coding/v1")

    def test_plain_responses_backend_has_no_route_policy(self):
        backend = OpenAIResponsesBackend(object(), "model")
        self.assertIsNone(backend.route)
        self.assertEqual(backend.model, "model")
        self.assertEqual(backend.provider_id, None)
        self.assertEqual(backend.context_limit, None)


if __name__ == "__main__":
    unittest.main()
