"""Tests for `@path` completion in the REPL prompt."""

import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent7.agent import _resolve
from agent7.__main__ import ReplCompleter, at_path_prefix, path_completions


class AtPathPrefixTests(unittest.TestCase):
    def test_bare_at(self) -> None:
        self.assertEqual(at_path_prefix("read @"), "")

    def test_partial_path(self) -> None:
        self.assertEqual(at_path_prefix("look at @agent7/ma"), "agent7/ma")

    def test_after_cjk_text_without_space(self) -> None:
        self.assertEqual(at_path_prefix("看下@agent7/"), "agent7/")

    def test_ignores_email(self) -> None:
        self.assertIsNone(at_path_prefix("mail me@example.com"))

    def test_ignores_completed_word(self) -> None:
        self.assertIsNone(at_path_prefix("@agent7 done"))

    def test_keeps_escaped_space(self) -> None:
        self.assertEqual(at_path_prefix("@my\\ fi"), "my\\ fi")


class PathCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        (self.cwd / "agent7").mkdir()
        (self.cwd / "agent7" / "main.py").write_text("x", encoding="utf-8")
        (self.cwd / "agent7" / "agent.py").write_text("x", encoding="utf-8")
        (self.cwd / "docs" / "guides").mkdir(parents=True)
        (self.cwd / "docs" / "guides" / "setup.md").write_text("x", encoding="utf-8")
        (self.cwd / ".git").mkdir()
        (self.cwd / ".git" / "config").write_text("x", encoding="utf-8")
        (self.cwd / "README.md").write_text("x", encoding="utf-8")
        (self.cwd / ".env").write_text("x", encoding="utf-8")
        (self.cwd / "my file.txt").write_text("x", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def inserts(self, text: str) -> dict[str, str]:
        """text inserted after '@' -> display shown in the menu."""
        return {c.text: c.display_text for c in path_completions(self.cwd, text)}

    def test_lists_cwd_after_bare_at(self) -> None:
        got = self.inserts("@")
        self.assertEqual(set(got), {
            "agent7/", "agent7/agent.py", "agent7/main.py", "docs/", "docs/guides/",
            "docs/guides/setup.md", "README.md", "my\\ file.txt",
        })
        self.assertEqual(got["agent7/"], "agent7/")
        self.assertEqual(got["my\\ file.txt"], "my file.txt")

    def test_directories_sort_first(self) -> None:
        order = [c.text for c in path_completions(self.cwd, "@")]
        self.assertEqual(order[0], "agent7/")

    def test_nested_paths_are_displayed_with_their_directory(self) -> None:
        self.assertEqual(self.inserts("@")["docs/guides/setup.md"], "docs/guides/setup.md")

    def test_hidden_files_need_dot_prefix(self) -> None:
        self.assertNotIn(".env", self.inserts("@"))
        self.assertNotIn(".git/config", self.inserts("@"))
        self.assertIn(".env", self.inserts("@."))

    def test_filters_by_prefix(self) -> None:
        self.assertEqual(set(self.inserts("@READ")), {"README.md"})

    def test_descends_into_directory(self) -> None:
        self.assertEqual(set(self.inserts("@agent7/")), {"agent7/agent.py", "agent7/main.py"})

    def test_start_position_replaces_only_the_fragment(self) -> None:
        text = "check @agent7/ma"
        (c,) = path_completions(self.cwd, text)
        self.assertEqual(c.start_position, -len("agent7/ma"))

    def test_unknown_directory_yields_nothing(self) -> None:
        self.assertEqual(self.inserts("@nope/x"), {})

    def test_absolute_path(self) -> None:
        (c,) = path_completions(self.cwd, f"@{self.cwd}/agent7/main")
        self.assertEqual(c.text, f"{self.cwd}/agent7/main.py")


class ReplCompleterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.session = SimpleNamespace(cwd=Path(self.tmp.name), models_cache=[])
        (self.session.cwd / "notes.md").write_text("x", encoding="utf-8")
        self.completer = ReplCompleter(self.session)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def texts(self, text: str) -> list[str]:
        return [c.text for c in self.completer.get_completions(Document(text), CompleteEvent())]

    def test_completes_paths(self) -> None:
        self.assertEqual(self.texts("read @not"), ["notes.md"])

    def test_slash_commands_still_work(self) -> None:
        self.assertIn("model", self.texts("/mod"))

    def test_slash_command_args_do_not_complete_paths(self) -> None:
        self.assertNotIn("notes.md", self.texts("/model @not"))


class ResolveAtPathTests(unittest.TestCase):
    """A path the prompt inserted as `@x` / `@my\\ file.txt` must still resolve."""

    cwd = Path("/tmp/project")

    def test_strips_at_prefix(self) -> None:
        self.assertEqual(_resolve(self.cwd, "@agent7/agent.py"), self.cwd / "agent7/agent.py")

    def test_unescapes_spaces(self) -> None:
        self.assertEqual(_resolve(self.cwd, "@my\\ file.txt"), self.cwd / "my file.txt")

    def test_plain_and_absolute_paths_unchanged(self) -> None:
        self.assertEqual(_resolve(self.cwd, "a.py"), self.cwd / "a.py")
        self.assertEqual(_resolve(self.cwd, "/etc/hosts"), Path("/etc/hosts"))


if __name__ == "__main__":
    unittest.main()
