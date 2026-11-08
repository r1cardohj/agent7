"""Fuzzy (fzp) ranking of `@path`, slash commands and their arguments."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.styles.defaults import default_ui_style
from prompt_toolkit.styles.style import merge_styles

from agent7.repl import MATCH_STYLE, PROMPT_STYLE, ReplCompleter, path_completions


def menu_style(*styles: str):
    """Resolve a menu style the way the running application does."""
    return merge_styles([default_ui_style(), PROMPT_STYLE]).get_attrs_for_style_str(
        " ".join(styles)
    )


class MenuStyleTests(unittest.TestCase):
    """Color is for the match, background is for the selection — never both."""

    row = "class:completion-menu.completion.current"

    def test_a_match_paints_no_background(self) -> None:
        attrs = PROMPT_STYLE.get_attrs_for_style_str(MATCH_STYLE)

        self.assertFalse(attrs.bgcolor)
        self.assertTrue(attrs.bold)

    def test_the_selected_row_has_a_background(self) -> None:
        attrs = menu_style(self.row)

        self.assertTrue(attrs.bgcolor and attrs.bgcolor != "default")
        self.assertFalse(attrs.reverse)  # reverse would turn a green match into a block

    def test_a_match_inside_the_selected_row_keeps_its_background(self) -> None:
        row, matched = menu_style(self.row), menu_style(self.row, MATCH_STYLE)

        self.assertEqual(matched.bgcolor, row.bgcolor)
        self.assertEqual(matched.color, "ansibrightgreen")
        self.assertTrue(matched.bold)

    def test_unselected_rows_stay_on_the_terminal_background(self) -> None:
        attrs = menu_style("class:completion-menu.completion")

        self.assertEqual(attrs.bgcolor, "default")


class HighlightTests(unittest.TestCase):
    def test_matched_characters_carry_the_match_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "main.py").write_text("x", encoding="utf-8")

            (completion,) = path_completions(cwd, "@mn")

        styled = "".join(text for style, text in completion.display if style == MATCH_STYLE)
        self.assertEqual(styled, "mn")
        self.assertEqual(completion.display_text, "main.py")


class PathFuzzyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        for name in ("main.py", "m_i_n.py", "README.md", "module"):
            (self.cwd / name).write_text("x", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def texts(self, text: str) -> list[str]:
        return [c.text for c in path_completions(self.cwd, text)]

    def test_subsequence_not_just_prefix(self) -> None:
        self.assertIn("main.py", self.texts("@mn"))

    def test_best_scoring_match_comes_first(self) -> None:
        for name in ("app.py", "a_p_p.py", "my_app.py"):
            (self.cwd / name).write_text("x", encoding="utf-8")

        # Consecutive, word-start hits beat scattered ones (see fzp's README).
        self.assertEqual(self.texts("@app")[0], "app.py")

    def test_missing_characters_yield_nothing(self) -> None:
        self.assertEqual(self.texts("@zzz"), [])

    def test_empty_fragment_keeps_everything(self) -> None:
        self.assertEqual(len(self.texts("@")), 4)


class SlashCommandFuzzyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.session = SimpleNamespace(
            cwd=Path(self.tmp.name),
            models_cache=[
                "anthropic/claude-sonnet-4",
                "deepseek/deepseek-chat",
                "openai/gpt-4o",
            ],
        )
        self.completer = ReplCompleter(self.session)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def texts(self, text: str) -> list[str]:
        return [c.text for c in self.completer.get_completions(Document(text), CompleteEvent())]

    def test_command_by_subsequence(self) -> None:
        self.assertEqual(self.texts("/mdl"), ["model"])

    def test_command_prefix_still_works(self) -> None:
        self.assertEqual(self.texts("/mod"), ["model"])

    def test_bare_slash_lists_every_command(self) -> None:
        self.assertGreaterEqual(len(self.texts("/")), 5)

    def test_command_display_highlights_after_the_slash(self) -> None:
        (completion,) = self.completer.get_completions(Document("/mdl"), CompleteEvent())
        self.assertEqual(completion.display_text, "/model")
        self.assertEqual(completion.start_position, -3)

    def test_model_argument_matches_the_bare_id(self) -> None:
        self.assertEqual(self.texts("/model sonnet"), ["anthropic/claude-sonnet-4"])

    def test_model_argument_accepts_a_provider_prefix(self) -> None:
        self.assertIn("deepseek/deepseek-chat", self.texts("/model deepseek/ch"))

    def test_model_argument_ranking_is_scored(self) -> None:
        self.assertEqual(self.texts("/model gpt")[0], "openai/gpt-4o")

    def test_provider_list_is_case_insensitive(self) -> None:
        got = self.texts("/login KIM")
        self.assertIn("kimi", got)


if __name__ == "__main__":
    unittest.main()
