"""The host's base prompt reaches the model through run(system=...)."""

import asyncio
import unittest

from responses_fakes import FakeClient, text_turn

from agent7.agent import run_agent_result
from agent7.config import SYSTEM, system_prompt
from agent7.herdr import background_tasks_instructions


def fragment_plugin(text: str):
    def register(_self, context) -> None:
        context.prompts.add(text)

    return type("FragmentPlugin", (), {"api_version": 1, "register": register})()


def run(client, messages, **kwargs):
    return asyncio.run(
        run_agent_result("go", client=client, model="model", messages=messages, **kwargs)
    )


class PromptWiringTests(unittest.TestCase):
    def test_the_host_prompt_is_written_and_sent_as_instructions(self) -> None:
        client = FakeClient([text_turn("done")])
        messages = []

        run(client, messages, system="BASE")

        self.assertEqual(messages[0], {"role": "system", "content": "BASE"})
        self.assertEqual(client.requests[0]["instructions"], "BASE")
        self.assertNotIn("system", {item.get("role") for item in client.requests[0]["input"]})

    def test_plugin_fragments_follow_the_host_prompt(self) -> None:
        client = FakeClient([text_turn("done")])
        messages = []

        run(client, messages, system="BASE", plugins=[fragment_plugin("FRAGMENT")])

        self.assertEqual(messages[0]["content"], "BASE\n\nFRAGMENT")
        self.assertEqual(client.requests[0]["instructions"], "BASE\n\nFRAGMENT")

    def test_a_stable_prompt_is_not_rewritten_on_later_turns(self) -> None:
        client = FakeClient([text_turn("done")])
        messages = []

        run(client, messages, system="BASE")
        first = messages[0]
        run(client, messages, system="BASE")

        self.assertIs(messages[0], first, "same dict, so the prefix stays cacheable")

    def test_the_default_prompt_is_agent7s_base(self) -> None:
        prompt = system_prompt()

        self.assertTrue(prompt.startswith(SYSTEM))
        # Environment notes may be empty (no background runner), so assert the
        # composition rather than a length that depends on the machine.
        self.assertEqual(prompt, SYSTEM + background_tasks_instructions())

    def test_without_a_prompt_the_runtime_leaves_history_alone(self) -> None:
        client = FakeClient([text_turn("done")])
        messages = []

        run(client, messages)

        self.assertEqual(messages[0]["role"], "user")
        self.assertNotIn("instructions", client.requests[0])


if __name__ == "__main__":
    unittest.main()
