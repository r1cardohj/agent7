"""Tests for ReplyPrinter.on_preview (write +/− stats line) and thinking output."""

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.segment import Segments

import agent7.repl as repl


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


class ReplyPrinterStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err, self.orig_console = repl.err, repl.console
        self.stream = io.StringIO()
        repl.err = Console(file=self.stream, width=80, color_system=None, force_terminal=False)
        repl.console = Console(
            file=self.stream, width=80, height=20,
            color_system="standard", force_terminal=True,
        )

    def tearDown(self) -> None:
        repl.err, repl.console = self.orig_err, self.orig_console

    def test_live_refresh_reads_the_accumulated_markdown(self) -> None:
        with patch("agent7.repl.Live") as live_type:
            live = live_type.return_value
            rp = repl.ReplyPrinter()
            rp.on_delta("# Hel")

            render = live_type.call_args.kwargs["get_renderable"]
            group = render()
            self.assertEqual(len(group.renderables), 1)
            self.assertEqual(group.renderables[0].markup, "# Hel")
            self.assertIs(live_type.call_args.kwargs["console"], repl.console)
            self.assertEqual(
                live_type.call_args.kwargs["refresh_per_second"],
                repl._ANSWER_REFRESH_PER_SECOND,
            )
            self.assertEqual(live_type.call_args.kwargs["vertical_overflow"], "visible")
            live.start.assert_called_once_with(refresh=True)

            rp.on_delta("lo\n\n**world**")
            group = render()
            # "# Hello\n\n" is a completed block (pre-rendered segments);
            # only the in-progress tail stays a re-parsed Markdown object.
            self.assertIsInstance(group.renderables[0], Segments)
            self.assertEqual(group.renderables[-1].markup, "**world**")
            live.update.assert_not_called()  # parsing is coalesced into Live repaints
            self.assertFalse(rp._counting)

            rp.__exit__(None, None, None)

        live.stop.assert_called_once_with()
        self.assertFalse(live.transient)
        self.assertIsNone(rp._live)

    def test_real_live_output_is_visible_before_the_stream_ends(self) -> None:
        rp = repl.ReplyPrinter()
        rp.on_delta("**bold**")
        self.assertIn("bold", self.stream.getvalue())
        self.assertNotIn("**bold**", self.stream.getvalue())

        rp.on_delta(" and `code`")
        rp.__exit__(None, None, None)
        self.assertIn("code", self.stream.getvalue())

    def test_tool_call_discards_a_streamed_draft(self) -> None:
        draft, final = MagicMock(), MagicMock()
        with patch("agent7.repl.Live", side_effect=[draft, final]):
            rp = repl.ReplyPrinter()
            rp.on_delta("temporary draft")
            rp.on_answer_to_thinking()

            self.assertTrue(draft.transient)
            draft.stop.assert_called_once_with()
            self.assertEqual(rp.buf, [])
            self.assertIsNone(rp._live)

            rp.on_delta("final answer")
            rp.__exit__(None, None, None)

        self.assertFalse(final.transient)
        final.stop.assert_called_once_with()

    def test_live_start_failure_falls_back_to_the_final_answer(self) -> None:
        with patch("agent7.repl.Live", side_effect=RuntimeError("no live display")):
            with repl.ReplyPrinter() as rp:
                rp.on_delta("**fallback**")
        self.assertIn("fallback", self.stream.getvalue())
        self.assertNotIn("**fallback**", self.stream.getvalue())

    def test_non_terminal_output_never_starts_live(self) -> None:
        repl.console = Console(file=self.stream, width=80, force_terminal=False)
        with patch("agent7.repl.Live") as live_type:
            with repl.ReplyPrinter() as rp:
                rp.on_delta("plain answer")
        live_type.assert_not_called()
        self.assertIn("plain answer", self.stream.getvalue())


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
        self.assertIn("\033[5F", text)
        for i in range(3, 9):
            self.assertIn(f"│ line {i}", text)
        self.assertEqual(self.rp._thinking_screen_rows, 6)
        # Earlier rows have already scrolled out of the viewport.
        latest = text.rsplit("\033[5F", 1)[-1]
        self.assertNotIn("│ line 0", latest)
        self.assertNotIn("│ line 2", latest)

    def test_incremental_deltas_share_one_viewport(self) -> None:
        self.rp.on_thinking("first ")
        self.rp.on_thinking("line\nsecond")
        latest = self.stream.getvalue().rsplit("\033[5F", 1)[-1]
        self.assertIn("│ first line", latest)
        self.assertIn("│ second", latest)
        self.assertEqual(self.rp._thinking_screen_rows, 2)

    def test_wrapping_and_blank_lines_count_toward_the_viewport(self) -> None:
        self.rp.on_thinking("x" * 45 + "\n\nlast")
        self.assertEqual(self.rp._thinking_screen_rows, 5)
        latest = self.stream.getvalue().rsplit("\033[5F", 1)[-1]
        self.assertIn("last", latest)
        # Five content rows plus one blank row fill the reserved viewport.
        self.assertEqual(latest.count("\033[E"), 5)

    def test_tool_shows_a_dot_until_it_finishes(self) -> None:
        with patch("agent7.repl.time.monotonic", side_effect=[10.0, 12.4]):
            self.rp.on_tool_start(1, "bash", {"command": "pwd"})
            running = self.stream.getvalue()
            self.assertTrue(running.endswith("  • bash  pwd  0s"))
            self.assertNotIn("✓", running)
            self.assertTrue(self.rp._tool_running)

            self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        text = self.stream.getvalue()
        self.assertIn("  • bash  pwd  0s\r\033[K  ✓ bash  pwd  2s\n", text)
        self.assertTrue(text.endswith("  ✓ bash  pwd  2s\n"))
        self.assertFalse(self.rp._tool_running)

    def test_tool_elapsed_seconds_tick_while_it_is_running(self) -> None:
        async def check() -> None:
            now = [10.0]
            with patch("agent7.repl.time", SimpleNamespace(monotonic=lambda: now[0])), \
                 patch("agent7.repl._TOOL_TICK_INTERVAL", 0.005):
                self.rp.on_tool_start(1, "bash", {"command": "pwd"})
                self.assertTrue(self.stream.getvalue().endswith("• bash  pwd  0s"))

                now[0] = 11.0
                await asyncio.sleep(0.03)
                self.assertIn("• bash  pwd  1s", self.stream.getvalue())

                now[0] = 12.0
                await asyncio.sleep(0.03)
                self.assertIn("• bash  pwd  2s", self.stream.getvalue())
                self.assertNotIn("✓", self.stream.getvalue())
                self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")

        asyncio.run(check())
        self.assertTrue(self.stream.getvalue().endswith("✓ bash  pwd  2s\n"))
        self.assertFalse(self.rp._tool_running)

    def test_tool_tick_clears_rows_reflowed_by_a_terminal_resize(self) -> None:
        repl.err = Console(
            file=self.stream, width=120, color_system=None, force_terminal=True
        )
        args = {"command": "x" * 80}
        self.rp.on_tool_start(1, "bash", args)
        self.assertNotIn("\033[4F", self.stream.getvalue())

        # Rich normally reads the new size from the terminal. A fixed-width
        # console needs its test value changed explicitly to simulate a resize.
        repl.err._width = 20
        self.rp._draw_running_tool("bash", args, 1, self.rp._tool_serial)

        # The 94-cell status was reflowed from one row to five. Repainting
        # starts at its first row and erases the whole status, not just row five.
        self.assertIn("\033[4F\r\033[J", self.stream.getvalue())
        self.assertTrue(self.stream.getvalue().endswith("1s"))

    def test_tool_elapsed_time_does_not_replace_bash_exit_status(self) -> None:
        with patch("agent7.repl.time.monotonic", side_effect=[20.0, 23.0]):
            self.rp.on_tool_start(1, "bash", {"command": "pwd"})
            self.rp.on_tool(1, "bash", {"command": "pwd"}, "exit=0\nstdout:\n/tmp")
        text = self.stream.getvalue()
        self.assertIn("✓ bash  pwd  3s", text)
        self.assertIn("exit=0", text)

    def test_preview_clears_the_running_dot_before_its_own_output(self) -> None:
        self.rp.on_tool_start(1, "write", {"path": "x.txt", "content": "new"})
        self.rp.on_preview("write  x.txt  +1 -0")
        self.rp.on_tool(1, "write", {"path": "x.txt", "content": "new"}, "wrote x.txt")
        text = self.stream.getvalue()
        self.assertIn("• write x.txt  (3 chars)  0s\033[1F\r\033[J# write", text)
        self.assertIn("✓ write x.txt", text.rsplit("# write", 1)[-1])

    def test_interruption_clears_the_running_dot(self) -> None:
        with self.rp:
            self.rp.on_tool_start(1, "bash", {"command": "sleep 10"})
        self.assertFalse(self.rp._tool_running)
        self.assertIsNone(self.rp._tool_started_at)
        self.assertTrue(self.stream.getvalue().endswith("\033[1F\r\033[J"))

    def test_bottom_of_screen_uses_a_fixed_reserved_viewport(self) -> None:
        self.rp.on_thinking("one")
        # The region is reserved before the first token, while the cursor is
        # still at its first body row. Later growth cannot change the rewind.
        self.assertEqual(self.rp._thinking_height, 6)
        self.assertIn("\n" * 6 + "\033[5F\r\033[J", self.stream.getvalue())

        self.rp.on_thinking("\n" + "\n".join(f"line {i}" for i in range(12)))
        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        self.assertIn("\033[5F\r\033[J  ✓ bash  pwd", self.stream.getvalue())

    def test_short_terminal_reserves_a_smaller_viewport(self) -> None:
        repl.err._height = 6
        self.rp.on_thinking("one")
        self.assertEqual(self.rp._thinking_height, 2)
        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        self.assertIn("\033[1F\r\033[J  ✓ bash  pwd", self.stream.getvalue())

    def test_closing_erases_the_viewport_before_tool_output(self) -> None:
        self.rp.on_thinking("one\ntwo")
        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        text = self.stream.getvalue()
        self.assertIn("\033[5F\r\033[J", text)
        self.assertTrue(text.endswith("✓ bash  pwd\n"))
        self.assertFalse(self.rp._thinking_open)
        self.assertEqual(self.rp._thinking_screen_rows, 0)

    def test_thinking_rewind_recalculates_rows_after_a_terminal_resize(self) -> None:
        self.rp.on_thinking("x" * 32)  # two 20-cell rows at the initial width
        self.assertEqual(self.rp._thinking_screen_rows, 2)

        # Six reserved rows plus one reflowed row for each painted line mean
        # the cursor is eight rows from the top after the terminal narrows.
        repl.err._width = 10
        self.rp.on_tool(1, "bash", {"command": "pwd"}, "done")
        self.assertIn("\033[7F\r\033[J", self.stream.getvalue())
        self.assertTrue(self.stream.getvalue().endswith("pwd\n"))

    def test_notices_close_thinking_before_printing(self) -> None:
        selection = SimpleNamespace(intent="coding", scores={}, error=None)
        cases = (
            (lambda: self.rp.on_compact(2), "# compacted"),
            (lambda: self.rp.on_failover("old", "new", RuntimeError("boom")), "# route:"),
            (lambda: self.rp.on_auto_selected("new", selection), "# auto:"),
        )
        for notify, expected in cases:
            with self.subTest(expected=expected):
                self.stream.seek(0)
                self.stream.truncate()
                self.rp.on_thinking("one\ntwo")
                notify()
                self.assertFalse(self.rp._thinking_open)
                text = self.stream.getvalue()
                self.assertIn("\033[5F\r\033[J" + expected.split()[0], text)
                self.assertIn(expected, text)

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


class ReplyPrinterFinalAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_err, self.orig_console = repl.err, repl.console
        self.stream = io.StringIO()
        repl.err = Console(file=self.stream, width=120, color_system=None, force_terminal=False)
        repl.console = Console(file=self.stream, width=80, force_terminal=False)

    def tearDown(self) -> None:
        repl.err, repl.console = self.orig_err, self.orig_console

    def test_buffered_answer_is_rendered_as_markdown_once(self) -> None:
        with repl.ReplyPrinter(stream_markdown=False) as rp:
            rp.on_delta("**final answer**")
        text = self.stream.getvalue()
        self.assertIn("final answer", text)
        self.assertNotIn("**", text)
        self.assertEqual(text.count("final answer"), 1)

    def test_long_buffered_answer_prints_fully(self) -> None:
        answer = "\n\n".join(f"paragraph {i}" for i in range(40))
        with repl.ReplyPrinter(stream_markdown=False) as rp:
            rp.on_delta(answer)
        text = self.stream.getvalue()
        self.assertIn("paragraph 0", text)
        self.assertIn("paragraph 39", text)
        self.assertEqual(text.count("paragraph 39"), 1)

    def test_non_terminal_output_prints_the_answer(self) -> None:
        with repl.ReplyPrinter() as rp:
            rp.on_delta("final answer")
        self.assertIn("final answer", self.stream.getvalue())


if __name__ == "__main__":
    unittest.main()
