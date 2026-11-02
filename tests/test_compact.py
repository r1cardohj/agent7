import unittest

from agent7.compact import KEEP_TOOL_RESULTS, compact_messages, should_compact


def tool_history(tool_count: int, *, chars: int = 3_000) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "fix the project"},
    ]
    for i in range(tool_count):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"call_{i}", "type": "function"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"result {i}: " + "x" * chars,
                },
            ]
        )
    return messages


class SingleTaskCompactionTests(unittest.TestCase):
    def test_old_tool_results_compact_without_another_user_turn(self) -> None:
        messages = tool_history(KEEP_TOOL_RESULTS + 2)

        compacted, changed = compact_messages(messages)

        self.assertEqual(changed, 2)
        tools = [m for m in compacted if m["role"] == "tool"]
        for result in tools[:2]:
            self.assertLess(len(result["content"]), 3_000)
            self.assertIn("...[compacted", result["content"])
        self.assertEqual(
            [m["content"] for m in tools[-KEEP_TOOL_RESULTS:]],
            [m["content"] for m in messages if m["role"] == "tool"][-KEEP_TOOL_RESULTS:],
        )

    def test_compaction_preserves_protocol_and_does_not_mutate_history(self) -> None:
        messages = tool_history(KEEP_TOOL_RESULTS + 1)
        original = messages[3]["content"]

        compacted, changed = compact_messages(messages)

        self.assertEqual(changed, 1)
        self.assertEqual(len(compacted), len(messages))
        self.assertEqual(
            [m["role"] for m in compacted], [m["role"] for m in messages]
        )
        self.assertEqual(compacted[3]["tool_call_id"], "call_0")
        expected = messages[3]["content"][:2_000] + "\n...[compacted 1010 chars]"
        self.assertEqual(compacted[3]["content"], expected)
        self.assertEqual(messages[3]["content"], original)
        self.assertEqual(compacted[0], messages[0])
        self.assertEqual(compacted[1], messages[1])

    def test_short_single_task_history_is_left_intact(self) -> None:
        messages = tool_history(KEEP_TOOL_RESULTS)

        compacted, changed = compact_messages(messages)

        self.assertEqual(changed, 0)
        self.assertEqual(compacted, messages)


class MultiTurnCompactionTests(unittest.TestCase):
    def test_recent_user_turns_keep_their_tool_results_intact(self) -> None:
        messages: list[dict] = [{"role": "system", "content": "system"}]
        for i in range(5):
            messages.extend(
                [
                    {"role": "user", "content": f"request {i}"},
                    {
                        "role": "assistant",
                        "content": "x" * 5_000 if i == 0 else "short answer",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": "x" * 3_000,
                    },
                ]
            )

        compacted, changed = compact_messages(messages)

        self.assertEqual(changed, 3)
        tools = [m for m in compacted if m["role"] == "tool"]
        self.assertTrue(all(len(m["content"]) < 3_000 for m in tools[:2]))
        self.assertTrue(all(len(m["content"]) == 3_000 for m in tools[2:]))
        self.assertLess(len(compacted[2]["content"]), 5_000)
        self.assertEqual(
            [m["content"] for m in compacted if m["role"] == "user"],
            [f"request {i}" for i in range(5)],
        )


class CompactionThresholdTests(unittest.TestCase):
    def test_existing_threshold_behavior_is_unchanged(self) -> None:
        self.assertFalse(should_compact(749, 1_000))
        self.assertTrue(should_compact(750, 1_000))
        self.assertFalse(should_compact(None, 1_000))


if __name__ == "__main__":
    unittest.main()
