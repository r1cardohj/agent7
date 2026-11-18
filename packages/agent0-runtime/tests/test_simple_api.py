"""The small public API: default construction, decorators, and resource ownership."""

import asyncio
import os
import unittest
from unittest.mock import patch

from agent0_runtime import (
    Agent,
    ModelTurn,
    OpenAIResponsesBackend,
    ResponsesHistoryPolicy,
    Tool,
    ToolRegistry,
    tool,
)
from responses_fakes import FakeClient, call_turn, text_turn


@tool("reflected_add")
def reflected_add(a: float, b: float) -> str:
    """Add two numbers through reflection."""
    return str(a + b)


class ClosingFakeClient(FakeClient):
    def __init__(self, script):
        super().__init__(script)
        self.closed = 0

    async def close(self):
        self.closed += 1


class DecoratorTests(unittest.TestCase):
    def test_bare_decorator_builds_schema_and_calls_plain_function(self):
        @tool
        def add(a: int, b: int = 1) -> str:
            """Add two integers."""
            return str(a + b)

        tool_definition = add.__agent0_tool__

        self.assertEqual(add(20, 2), "22")
        self.assertIsInstance(tool_definition, Tool)
        self.assertEqual(tool_definition.name, "add")
        self.assertEqual(
            tool_definition.schema["function"]["parameters"],
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a"],
            },
        )
        self.assertEqual(tool_definition.handler(None, {"a": 20}), "21")

    def test_positional_decorator_registers_an_alias(self):
        @tool("sum_two")
        def add(a: int, b: int) -> str:
            """Add two integers."""
            return str(a + b)

        self.assertEqual(add.__agent0_tool__.name, "sum_two")
        self.assertEqual(add.__agent0_tool__.handler(None, {"a": 1, "b": 2}), "3")

    def test_configured_and_async_decorators(self):
        @tool(name="say", description="Say something.")
        async def echo(text: str) -> str:
            return text

        self.assertEqual(echo.__agent0_tool__.name, "say")
        self.assertEqual(echo.__agent0_tool__.schema["function"]["description"], "Say something.")
        self.assertEqual(
            asyncio.run(echo.__agent0_tool__.handler(None, {"text": "hello"})), "hello"
        )

    def test_collection_and_optional_annotations(self):
        @tool
        def summarize(words: list[str], limit: int | None = None) -> str:
            return ",".join(words[:limit])

        parameters = summarize.__agent0_tool__.schema["function"]["parameters"]
        self.assertEqual(parameters["properties"]["words"]["items"], {"type": "string"})
        self.assertIn("anyOf", parameters["properties"]["limit"])
        self.assertEqual(parameters["required"], ["words"])

    def test_variadic_functions_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "named parameters"):

            @tool
            def invalid(*args: str) -> str:
                return ",".join(args)


class DefaultAgentTests(unittest.TestCase):
    def test_minimal_agent_uses_defaults_and_closes_network_client(self):
        client = ClosingFakeClient(
            [call_turn(("reflected_add", {"a": 20, "b": 22})), text_turn("42")]
        )

        async def exercise():
            async with Agent.from_env(tools=["reflected_add"]) as agent:
                result = await agent.run("Use add to calculate 20 + 22.")
                self.assertEqual(result.text, "42")
                self.assertIsInstance(agent.history, ResponsesHistoryPolicy)
                self.assertEqual(agent.backend.model, "deepseek-chat")
                self.assertEqual(agent.backend.provider_id, "deepseek")
                self.assertEqual(agent.tools.names(), ("reflected_add",))

            self.assertEqual(client.closed, 1)
            await agent.close()
            self.assertEqual(client.closed, 1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await agent.run("again")

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            asyncio.run(exercise())

    def test_string_tools_resolve_from_the_calling_scope(self):
        @tool("weather")
        def local_weather(city: str) -> str:
            """Report local weather."""
            return f"sunny in {city}"

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True),
            patch(
                "openai.AsyncOpenAI", return_value=ClosingFakeClient([text_turn("ok")])
            ) as factory,
        ):
            agent = Agent.from_env(tools=["weather"])
            self.assertEqual(agent.tools.names(), ("weather",))
            asyncio.run(agent.close())

        @tool("weather")
        def other_weather(city: str) -> str:
            return f"rainy in {city}"

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True),
            factory,
            self.assertRaisesRegex(ValueError, "ambiguous"),
        ):
            Agent.from_env(tools=["weather"])

        self.assertEqual(local_weather("Rome"), "sunny in Rome")
        self.assertEqual(factory.return_value.closed, 1)

    def test_default_provider_and_explicit_model_selection(self):
        kimi_client = ClosingFakeClient([text_turn("ok")])
        with (
            patch.dict(os.environ, {"KIMI_API_KEY": "sk-kimi"}, clear=True),
            patch("openai.AsyncOpenAI", return_value=kimi_client),
        ):
            agent = Agent.from_env()
            self.assertEqual(agent.backend.model, "kimi-for-coding")
            self.assertEqual(agent.backend.provider_id, "kimi")
            asyncio.run(agent.close())

        deepseek_client = ClosingFakeClient([text_turn("ok")])
        with (
            patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "sk-ds", "KIMI_API_KEY": "sk-kimi"},
                clear=True,
            ),
            patch("openai.AsyncOpenAI", return_value=deepseek_client),
        ):
            agent = Agent.from_env("deepseek/deepseek-flash")
            self.assertEqual(agent.backend.model, "deepseek-flash")
            self.assertEqual(agent.backend.provider_id, "deepseek")
            asyncio.run(agent.close())

    def test_runtime_environment_uses_neutral_names(self):
        client = ClosingFakeClient([text_turn("ok")])
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "sk-test",
                    "AGENT0_MODEL": "deepseek/deepseek-flash",
                    "AGENT0_BASE_URL": "https://runtime.example/v1",
                    "AGENT7_MODEL": "kimi/kimi-for-coding",
                },
                clear=True,
            ),
            patch("openai.AsyncOpenAI", return_value=client) as factory,
        ):
            agent = Agent.from_env()
            self.assertEqual(agent.backend.model, "deepseek-flash")
            factory.assert_called_once_with(
                api_key="sk-test", base_url="https://runtime.example/v1"
            )
            asyncio.run(agent.close())

    def test_configuration_errors_are_early_and_clear(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY or KIMI_API_KEY"),
        ):
            Agent.from_env()

        with (
            patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "sk-ds", "KIMI_API_KEY": "sk-kimi"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "provider/model"),
        ):
            Agent.from_env("deepseek-chat")

    def test_one_shot_custom_backend_needs_no_messages(self):
        class Backend:
            async def complete(self, _request, _events):
                return ModelTurn(text="done")

        result = asyncio.run(Agent(Backend()).run("hello"))
        self.assertEqual(result.text, "done")


class ResourceOwnershipTests(unittest.TestCase):
    def test_agent_closes_owned_backend_once(self):
        class Backend:
            def __init__(self):
                self.closed = 0

            async def complete(self, _request, _events):
                return ModelTurn(text="done")

            async def close(self):
                self.closed += 1

        async def exercise():
            backend = Backend()
            agent = Agent(backend)
            await agent.close()
            await agent.close()
            self.assertEqual(backend.closed, 1)

            shared = Backend()
            await Agent(shared, close_backend=False).close()
            self.assertEqual(shared.closed, 0)

        asyncio.run(exercise())

    def test_responses_backend_closes_its_client(self):
        client = ClosingFakeClient([text_turn("ok")])
        backend = OpenAIResponsesBackend(client, "model")
        asyncio.run(backend.close())
        self.assertEqual(client.closed, 1)

    def test_registry_still_accepts_decorated_tools_directly(self):
        @tool
        def ping() -> str:
            return "pong"

        self.assertEqual(ToolRegistry([ping]).names(), ("ping",))


if __name__ == "__main__":
    unittest.main()
