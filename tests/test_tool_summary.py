"""Terminal summaries for tool progress rendered by agent7."""

import unittest

from agent7.config import tool_summary


class ToolSummaryTests(unittest.TestCase):
    def test_summary_for_write_and_edit(self) -> None:
        self.assertEqual(
            tool_summary("write", {"path": "a.py", "content": "12345"}),
            "a.py  (5 chars)",
        )
        self.assertEqual(
            tool_summary(
                "edit", {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]}
            ),
            "a.py  (1 edit)",
        )
        self.assertEqual(
            tool_summary("edit", {"path": "a.py", "edits": [{}, {}]}),
            "a.py  (2 edits)",
        )

    def test_summary_for_fetch_and_search(self) -> None:
        self.assertEqual(
            tool_summary("fetch", {"url": "https://example.com/docs"}),
            "https://example.com/docs",
        )
        self.assertEqual(tool_summary("search", {"query": "grpc deadline"}), "grpc deadline")
        self.assertEqual(
            tool_summary("search", {"query": "grpc deadline", "count": 5}),
            "grpc deadline  (count=5)",
        )

    def test_summary_for_read_and_bash(self) -> None:
        self.assertEqual(tool_summary("read", {"path": "a.py"}), "a.py")
        self.assertEqual(
            tool_summary("read", {"path": "a.py", "offset": 10, "limit": 5}),
            "a.py  (offset=10 limit=5)",
        )
        self.assertEqual(tool_summary("bash", {"command": "ls -la"}), "ls -la")


if __name__ == "__main__":
    unittest.main()
