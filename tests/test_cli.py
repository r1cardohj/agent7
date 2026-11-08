"""The click command line: flags, help/version output, and defaults."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from agent7 import __version__
from agent7.cli import Options, main


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_version(self) -> None:
        result = self.runner.invoke(main, ["--version"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), f"agent7 {__version__}")

    def test_help_documents_every_flag(self) -> None:
        result = self.runner.invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        for flag in ("--cwd", "--max-turns", "-r, --resume", "--version"):
            self.assertIn(flag, result.output)

    def test_zero_or_negative_turns_are_rejected(self) -> None:
        for raw in ("0", "-3"):
            result = self.runner.invoke(main, ["--max-turns", raw])

            self.assertEqual(result.exit_code, 2)
            self.assertIn("must be at least 1", result.output)

    def test_words_are_joined_into_one_task(self) -> None:
        with patch("agent7.cli.async_main", new=AsyncMock()) as run:
            result = self.runner.invoke(main, ["fix", "the", "bug"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(run.await_args.args[0], Options(task="fix the bug"))

    def test_no_task_opens_the_repl_in_the_given_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent7.cli.async_main", new=AsyncMock()) as run:
            result = self.runner.invoke(main, ["--cwd", tmp])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(run.await_args.args[0], Options(cwd=Path(tmp)))

    def test_parse_is_equivalent_to_the_command(self) -> None:
        self.assertEqual(
            Options.parse(["-r", "--max-turns", "5", "go", "on"]),
            Options(task="go on", max_turns=5, resume=True),
        )


if __name__ == "__main__":
    unittest.main()
