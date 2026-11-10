"""Tests for ReplyPrinter.on_preview (write +/− stats line) and thinking output."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.keys import Keys
from rich.console import Console

import agent7.repl as repl
from agent7.pager import PromptToolkitPager


class ReplyPrinterPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err, self.orig_console = repl.err, repl.console
        self.stream = io.StringIO()
        repl.err = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        repl.console = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        self.rp = repl.ReplyPrinter()

    def tearDown(self) -> None:
        repl.err, repl.console = self.orig_err, self.orig_console

    def out(self) -> str:
        return self.stream.getvalue()

    def test_prints_stats_header(self) -> None:
        self.rp.on_preview("write  a.py  +1 -1")
        text = self.out()
        self.assertIn("# write", text)
        self.assertIn("a.py", text)
        self.assertIn("+1", text)
        self.assertIn("-1", text)

    def test_prints_edit_kind(self) -> None:
        self.rp.on_preview("edit  a.py  +2 -1")
        text = self.out()
        self.assertIn("# edit", text)
        self.assertIn("a.py", text)
        self.assertIn("+2", text)

    def test_ignores_extra_lines(self) -> None:
        self.rp.on_preview("write  cfg.txt  +2 -3\n@@ ignored\n+body")
        text = self.out()
        self.assertIn("+2", text)
        self.assertNotIn("@@", text)
        self.assertNotIn("+body", text)

    def test_empty_diff_prints_nothing(self) -> None:
        self.rp.on_preview("")
        self.assertEqual(self.out(), "")

    def test_streams_thinking_text_under_a_header(self) -> None:
        self.rp.on_thinking("looking at ")
        self.rp.on_thinking("agent.py\nand tests")
        text = self.out()
        self.assertIn("thinking", text)
        self.assertIn("  │ looking at agent.py", text)
        self.assertIn("  │ and tests", text)
        self.assertTrue(self.rp._thinking_open)

    def test_blank_line_skips_the_gutter(self) -> None:
        self.rp.on_thinking("a\n\nb")
        self.assertIn("  │ a\n\n  │ b", self.out())

    def test_closes_open_thinking_block_first(self) -> None:
        self.rp.on_thinking("pondering")
        self.assertTrue(self.rp._thinking_open)
        self.rp.on_preview("write  x.txt  +1 -0")
        self.assertIn("pondering", self.out())
        self.assertIn("# write", self.out())
        self.assertLess(self.out().index("pondering"), self.out().index("# write"))
        self.assertFalse(self.rp._thinking_open)

    def test_thinking_is_not_part_of_the_answer(self) -> None:
        with self.rp as rp:
            rp.on_thinking("private reasoning")
        self.assertEqual(rp.buf, [])
        self.assertIn("private reasoning", self.out())

    def test_tool_call_ends_the_thinking_block(self) -> None:
        self.rp.on_thinking("plan")
        self.rp.on_answer_to_thinking()
        self.assertFalse(self.rp._thinking_open)
        self.assertTrue(self.out().endswith("plan\n"))
        self.rp.on_thinking("next")
        self.assertIn("  │ next", self.out())

    def test_whitespace_only_answer_prints_nothing(self) -> None:
        with self.rp as rp:
            rp.on_delta("\n  \n")
        self.assertEqual(self.out(), "")

    def test_preview_does_not_emit_markdown_answer(self) -> None:
        self.rp.on_delta("some streamed answer text")
        self.rp.on_preview("write  x.txt  +1 -1")
        self.assertEqual(self.rp.buf, ["some streamed answer text"])
        self.assertNotIn("some streamed answer text", self.out())


class ReplyPrinterThinkingViewportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err = repl.err
        self.stream = io.StringIO()
        repl.err = Console(
            file=self.stream, width=20, color_system=None, force_terminal=True
        )
        self.rp = repl.ReplyPrinter()

    def tearDown(self) -> None:
        repl.err = self.orig_err

    def test_only_the_latest_six_visual_rows_are_displayed(self) -> None:
        self.rp.on_thinking("\n".join(f"line {i}" for i in range(3)))
        self.rp.on_thinking("\n" + "\n".join(f"line {i}" for i in range(3, 9)))
        text = self.stream.getvalue()
        self.assertIn("\033[2F", text)
        for i in range(3, 9):
            self.assertIn(f"│ line {i}", text)
        self.assertEqual(self.rp._thinking_screen_rows, 6)
        # Earlier rows have already scrolled out of the viewport.
        latest = text.rsplit("\033[2F", 1)[-1]
        self.assertNotIn("│ line 0", latest)
        self.assertNotIn("│ line 2", latest)

    def test_incremental_deltas_share_one_viewport(self) -> None:
        self.rp.on_thinking("first ")
        self.rp.on_thinking("line\nsecond")
        latest = self.stream.getvalue().rsplit("\033[1F", 1)[-1]
        self.assertIn("│ first line", latest)
        self.assertIn("│ second", latest)
        self.assertEqual(self.rp._thinking_screen_rows, 2)

    def test_wrapping_and_blank_lines_count_toward_the_viewport(self) -> None:
        self.rp.on_thinking("x" * 45 + "\n\nlast")
        self.assertEqual(self.rp._thinking_screen_rows, 5)
        latest = self.stream.getvalue().rsplit("\033[4F", 1)[-1]
        self.assertIn("last", latest)
        self.assertEqual(latest.count("\033[E"), 4)

    def test_tool_shows_a_dot_until_it_finishes(self) -> None:
        self.rp.on_tool_start(1, "bash", {"command": "pwd"})
        running = self.stream.getvalue()
        self.assertTrue(running.endswith("  • bash  pwd"))
        self.assertNotIn("✓", running)
        self.assertTrue(self.rp._tool_running)

        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        text = self.stream.getvalue()
        self.assertIn("  • bash  pwd\r\033[K  ✓ bash  pwd\n", text)
        self.assertTrue(text.endswith("  ✓ bash  pwd\n"))
        self.assertFalse(self.rp._tool_running)

    def test_preview_clears_the_running_dot_before_its_own_output(self) -> None:
        self.rp.on_tool_start(1, "write", {"path": "x.txt", "content": "new"})
        self.rp.on_preview("write  x.txt  +1 -0")
        self.rp.on_tool(1, "write", {"path": "x.txt", "content": "new"}, "wrote x.txt")
        text = self.stream.getvalue()
        self.assertIn("• write x.txt  (3 chars)\r\033[K# write", text)
        self.assertIn("✓ write x.txt", text.rsplit("# write", 1)[-1])

    def test_interruption_clears_the_running_dot(self) -> None:
        with self.rp:
            self.rp.on_tool_start(1, "bash", {"command": "sleep 10"})
        self.assertFalse(self.rp._tool_running)
        self.assertTrue(self.stream.getvalue().endswith("\r\033[K"))

    def test_closing_erases_the_viewport_before_tool_output(self) -> None:
        self.rp.on_thinking("one\ntwo")
        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        text = self.stream.getvalue()
        self.assertIn("\033[1F\r\033[J", text)
        self.assertTrue(text.endswith("✓ bash  pwd\n"))
        self.assertFalse(self.rp._thinking_open)
        self.assertEqual(self.rp._thinking_screen_rows, 0)

    def test_model_wait_is_visible_until_thinking_starts(self) -> None:
        with patch.object(self.rp, "_paint", side_effect=lambda text: text):
            self.rp.on_model_start(2)
            self.assertTrue(self.rp._waiting)
            self.assertIn("waiting...", self.stream.getvalue())
            self.assertNotIn("round", self.stream.getvalue())
            self.rp.on_thinking("next step")
        self.assertFalse(self.rp._waiting)
        self.assertIn("\r\033[K\n  thinking", self.stream.getvalue())

    def test_compaction_clears_model_wait_before_printing(self) -> None:
        self.rp.on_model_start(1)
        self.rp.on_compact(3)
        self.assertFalse(self.rp._waiting)
        compacted = self.stream.getvalue().rsplit("\r\033[K", 1)[-1]
        self.assertNotIn("waiting", compacted)
        self.assertEqual(compacted.split(), ["#", "compacted", "3", "message(s)"])

    def test_interruption_clears_model_wait(self) -> None:
        with self.rp:
            self.rp.on_model_start(1)
        self.assertFalse(self.rp._waiting)
        self.assertTrue(self.stream.getvalue().endswith("\r\033[K"))

    def test_each_new_thinking_block_starts_below_previous_output(self) -> None:
        self.rp.on_thinking("old")
        self.rp.on_answer_to_thinking()
        self.rp.on_thinking("new")
        text = self.stream.getvalue()
        self.assertIn("\r\033[J\n", text)
        reopened = text.rsplit("\r\033[J\n", 1)[-1]
        self.assertIn("thinking", reopened)
        self.assertIn("│ new", reopened)


class PromptToolkitPagerTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch("agent7.pager.Application") as application:
            PromptToolkitPager().show("\033[31mfirst\033[0m\nsecond")
        self.application = application
        self.kwargs = application.call_args.kwargs
        self.window = self.kwargs["layout"].current_window
        self.bindings = self.kwargs["key_bindings"]

    def press(self, key: str, event: SimpleNamespace) -> None:
        self.bindings.get_bindings_for_keys((key,))[-1].handler(event)

    def event(self) -> SimpleNamespace:
        layout = SimpleNamespace(current_window=self.window)
        return SimpleNamespace(app=SimpleNamespace(layout=layout))

    def test_preserves_all_text_and_runs_outside_the_agent_event_loop(self) -> None:
        fragments = to_formatted_text(self.window.content.text)
        self.assertEqual("".join(part[1] for part in fragments), "first\nsecond")
        self.assertTrue(self.kwargs["full_screen"])
        self.application.return_value.run.assert_called_once_with(in_thread=True)

    def test_arrow_and_page_keys_scroll_in_both_directions(self) -> None:
        self.window.render_info = SimpleNamespace(content_height=20, window_height=5)
        event = self.event()
        self.press(Keys.Down, event)
        self.assertEqual(self.window.vertical_scroll, 1)
        self.assertEqual(self.window.content.get_cursor_position().y, 1)
        self.press(" ", event)
        self.assertEqual(self.window.vertical_scroll, 5)
        self.press(Keys.PageDown, event)
        self.assertEqual(self.window.vertical_scroll, 9)
        self.press(Keys.PageUp, event)
        self.assertEqual(self.window.vertical_scroll, 5)
        self.press(Keys.Up, event)
        self.assertEqual(self.window.vertical_scroll, 4)

    def test_scrolling_stops_at_the_actual_bottom(self) -> None:
        self.window.render_info = SimpleNamespace(content_height=20, window_height=5)
        self.window.vertical_scroll = 14
        event = self.event()
        self.press(Keys.Down, event)
        self.press(" ", event)
        self.assertEqual(self.window.vertical_scroll, 15)

    def test_q_exits(self) -> None:
        event = MagicMock()
        self.press("q", event)
        event.app.exit.assert_called_once_with()


class ReplyPrinterPagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err, self.orig_console = repl.err, repl.console
        self.stream = io.StringIO()
        repl.err = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        repl.console = Console(file=self.stream, width=80, height=5, force_terminal=True)

    def tearDown(self) -> None:
        repl.err, repl.console = self.orig_err, self.orig_console

    def test_short_interactive_answer_is_printed_without_a_pager(self) -> None:
        with patch("agent7.repl.sys.stdin.isatty", return_value=True), \
             patch("agent7.repl.console.pager") as get_pager:
            with repl.ReplyPrinter() as rp:
                rp.on_delta("**final answer**")
        get_pager.assert_not_called()
        self.assertIn("final answer", self.stream.getvalue())

    def test_long_interactive_answer_is_paged_then_left_in_terminal(self) -> None:
        answer = "\n\n".join(f"paragraph {i}" for i in range(4))
        with patch("agent7.repl.sys.stdin.isatty", return_value=True), \
             patch("agent7.repl.console.pager", wraps=repl.console.pager) as get_pager, \
             patch("rich.pager.SystemPager.show") as show:
            with repl.ReplyPrinter() as rp:
                rp.on_delta(answer)
        get_pager.assert_called_once_with(pager=None, styles=True)
        show.assert_called_once()
        self.assertIn("paragraph 3", show.call_args.args[0])
        self.assertIn("paragraph 3", self.stream.getvalue())

    def test_windows_long_answer_uses_the_builtin_pager(self) -> None:
        answer = "\n\n".join(f"paragraph {i}" for i in range(4))
        with patch("agent7.repl.sys.stdin.isatty", return_value=True), \
             patch("agent7.repl.sys.platform", "win32"), \
             patch("agent7.repl.console.pager", wraps=repl.console.pager) as get_pager, \
             patch("agent7.repl.PromptToolkitPager.show") as show:
            with repl.ReplyPrinter() as rp:
                rp.on_delta(answer)
        pager = get_pager.call_args.kwargs["pager"]
        self.assertIsInstance(pager, PromptToolkitPager)
        get_pager.assert_called_once_with(pager=pager, styles=True)
        self.assertIn("paragraph 3", show.call_args.args[0])

    def test_answer_fitting_exactly_one_screen_is_not_paged(self) -> None:
        answer = "\n\n".join(f"line {i}" for i in range(3))
        with patch("agent7.repl.sys.stdin.isatty", return_value=True), \
             patch("agent7.repl.console.pager") as get_pager:
            with repl.ReplyPrinter() as rp:
                rp.on_delta(answer)
        get_pager.assert_not_called()

    def test_wrapping_counts_toward_the_screen_height(self) -> None:
        with patch("agent7.repl.sys.stdin.isatty", return_value=True), \
             patch("agent7.repl.console.pager") as get_pager:
            with repl.ReplyPrinter() as rp:
                rp.on_delta(" ".join(["word"] * 160))
        get_pager.assert_called_once_with(pager=None, styles=True)

    def test_non_interactive_output_does_not_open_a_pager(self) -> None:
        repl.console = Console(file=self.stream, width=80, force_terminal=False)
        with patch("agent7.repl.console.pager") as get_pager:
            with repl.ReplyPrinter() as rp:
                rp.on_delta("final answer")
        get_pager.assert_not_called()
        self.assertIn("final answer", self.stream.getvalue())


if __name__ == "__main__":
    unittest.main()
