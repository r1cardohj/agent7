"""`-m/--model`: a named model for one run, or a fuzzy menu when given bare."""

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from rich.console import Console

from agent7 import cli, repl
from agent7.agent import SYSTEM
from agent7.cli import PICK_MODEL, Options, main
from agent7.store import load, save


class ModelOptionParseTests(unittest.TestCase):
    def test_a_named_model_is_part_of_the_run(self) -> None:
        self.assertEqual(
            Options.parse(["-m", "deepseek-flash", "fix", "it"]),
            Options(model="deepseek-flash", task="fix it"),
        )

    def test_a_bare_dash_m_asks_for_the_menu(self) -> None:
        self.assertEqual(Options.parse(["-m"]).model, PICK_MODEL)

    def test_no_option_leaves_the_model_alone(self) -> None:
        self.assertIsNone(Options.parse(["go"]).model)

    def test_a_bare_dash_m_can_still_be_followed_by_a_task(self) -> None:
        # `-m` takes the next word as its value, so a task needs a separator…
        self.assertEqual(
            Options.parse(["-m", "--", "fix", "it"]),
            Options(model=PICK_MODEL, task="fix it"),
        )
        # …or it goes first, where click keeps it as a positional.
        self.assertEqual(
            Options.parse(["fix", "it", "-m"]),
            Options(model=PICK_MODEL, task="fix it"),
        )

    def test_help_documents_the_option(self) -> None:
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("-m, --model REF", result.output)


class ModelMenuTests(unittest.TestCase):
    def test_the_current_model_is_listed_first_and_only_once(self) -> None:
        self.assertEqual(
            repl.model_choices(["a/one", "a/two"], current="a/two"),
            ["a/two", "a/one"],
        )

    def test_an_unknown_current_model_is_still_offered(self) -> None:
        self.assertEqual(repl.model_choices(["a/one"], current="a/gone"), ["a/gone", "a/one"])

    def test_an_empty_menu_never_opens_the_picker(self) -> None:
        with patch.object(repl, "Picker") as picker:
            self.assertIsNone(asyncio.run(repl.pick_model([])))
        picker.assert_not_called()

    def test_the_picker_returns_the_chosen_ref(self) -> None:
        with patch.object(repl, "Picker") as picker:
            picker.return_value.pick_async = AsyncMock(return_value="a/two")
            chosen = asyncio.run(repl.pick_model(["a/one", "a/two"], current="a/one"))

        self.assertEqual(chosen, "a/two")
        picker.assert_called_once_with(["a/one", "a/two"])


class ModelOptionRunTests(unittest.TestCase):
    """End to end through `async_main`, with the network and the agent stubbed."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cwd = root / "work"
        self.cwd.mkdir()
        self.state_home = root / "state"
        env = patch.dict(os.environ, {"XDG_STATE_HOME": str(self.state_home)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT7_MODEL", None)
        self.addCleanup(os.environ.pop, "AGENT7_MODEL", None)

    def snapshot_paths(self) -> list[Path]:
        root = self.state_home / "agent7" / "sessions"
        return list(root.glob("*.json")) if root.exists() else []

    def save_snapshot(self, model_ref: str = "kimi/kimi-for-coding") -> None:
        save(SimpleNamespace(
            cwd=self.cwd, model_ref=model_ref, max_turns=30,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "old"}],
        ), created_at="2025-01-01T00:00:00+00:00")

    def run_cli(
        self,
        argv: list[str],
        *,
        models: tuple[str, ...] = ("deepseek/deepseek-chat", "deepseek/deepseek-flash"),
        providers: tuple[str, ...] = ("deepseek",),
        pick: str | None = None,
        can_pick: bool = True,
        switch_message: str = "ok",
        env_model: str | None = None,
    ) -> None:
        """Run async_main; the mocks it used are left on `self.last`."""
        if env_model is not None:
            os.environ["AGENT7_MODEL"] = env_model
        credential = SimpleNamespace(source="test")
        provider = SimpleNamespace(
            id="deepseek", make_client=lambda cred: SimpleNamespace(close=AsyncMock())
        )
        switch = AsyncMock(return_value=SimpleNamespace(message=switch_message))
        run_turn = AsyncMock(return_value="done")
        picker = AsyncMock(return_value=pick)
        stream = io.StringIO()
        self.last = SimpleNamespace(switch=switch, run=run_turn, picker=picker,
                                    output=stream)
        with patch.object(cli, "resolve_all",
                          return_value={pid: credential for pid in providers}), \
             patch.object(cli, "resolve_default",
                          return_value=(provider, credential) if providers else None), \
             patch.object(cli, "fetch_models", new=AsyncMock(return_value=list(models))), \
             patch.object(cli, "switch_model", new=switch), \
             patch.object(cli, "run_turn", new=run_turn), \
             patch.object(cli, "pick_model", new=picker), \
             patch.object(cli, "can_pick_models", return_value=can_pick), \
             patch.object(cli, "err", new=Console(file=stream, width=120, color_system=None)):
            try:
                asyncio.run(cli.async_main(Options.parse(["--cwd", str(self.cwd), *argv])))
            finally:
                self.last.output = stream.getvalue()

    # ---------------------------------------------------------- named model

    def test_a_named_model_runs_the_task_on_it(self) -> None:
        self.run_cli(["-m", "deepseek-flash", "fix", "it"])

        self.last.switch.assert_awaited_once()
        self.assertEqual(self.last.switch.await_args.args[1], "deepseek-flash")
        self.assertEqual(self.last.run.await_args.args[1], "fix it")
        self.last.picker.assert_not_awaited()

    def test_dash_m_beats_the_environment_model(self) -> None:
        self.run_cli(["-m", "deepseek-flash", "go"], env_model="kimi/kimi-for-coding")

        self.assertEqual(self.last.switch.await_args.args[1], "deepseek-flash")

    def test_dash_m_beats_the_snapshot_model(self) -> None:
        self.save_snapshot()

        self.run_cli(["-m", "deepseek-flash", "go"])

        self.assertEqual(self.last.switch.await_args.args[1], "deepseek-flash")

    def test_an_unknown_explicit_model_stops_the_run(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(["-m", "nope", "go"],
                         switch_message="unknown model 'nope' — try /model")

        self.assertEqual(raised.exception.code, 1)
        self.last.run.assert_not_awaited()
        # No silent fallback to the first fetched model.
        self.assertEqual([c.args[1] for c in self.last.switch.await_args_list], ["nope"])
        self.assertIn("unknown model", self.last.output)

    def test_an_unknown_environment_model_still_falls_back(self) -> None:
        self.run_cli(["go"], env_model="nope",
                     switch_message="unknown model 'nope' — try /model")

        self.assertEqual(
            [c.args[1] for c in self.last.switch.await_args_list],
            ["nope", "deepseek/deepseek-chat"],
        )
        self.last.run.assert_awaited_once()

    # ------------------------------------------------------------ model menu

    def test_a_bare_dash_m_uses_the_picked_model(self) -> None:
        self.run_cli(["-m", "--", "go"], pick="deepseek/deepseek-flash")

        self.assertEqual(self.last.picker.await_args.args[0],
                         ["deepseek/deepseek-chat", "deepseek/deepseek-flash", "auto"])
        self.assertEqual(self.last.switch.await_args.args[1], "deepseek/deepseek-flash")
        self.assertEqual(self.last.run.await_args.args[1], "go")

    def test_the_menu_starts_from_the_workspace_model(self) -> None:
        self.save_snapshot()

        self.run_cli(["-m", "--", "go"], pick="deepseek/deepseek-flash")

        self.assertEqual(self.last.picker.await_args.kwargs["current"],
                         "kimi/kimi-for-coding")

    def test_the_menu_falls_back_to_provider_defaults(self) -> None:
        self.run_cli(["-m", "--", "go"], models=(), providers=("deepseek",),
                     pick="deepseek/deepseek-chat")

        self.assertEqual(self.last.picker.await_args.args[0], ["deepseek/deepseek-chat", "auto"])

    def test_cancelling_the_menu_runs_nothing(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(["-m", "--", "go"], pick=None)

        self.assertEqual(raised.exception.code, 1)
        self.last.switch.assert_not_awaited()
        self.last.run.assert_not_awaited()
        self.assertIn("cancelled", self.last.output)
        self.assertEqual(self.snapshot_paths(), [])

    def test_cancelling_the_menu_keeps_the_workspace_snapshot(self) -> None:
        self.save_snapshot()
        before = self.snapshot_paths()[0].read_bytes()

        with self.assertRaises(SystemExit):
            self.run_cli(["-m", "--", "go"], pick=None)

        self.assertEqual(self.snapshot_paths()[0].read_bytes(), before)

    def test_a_bare_dash_m_without_a_terminal_says_what_to_do(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(["-m", "--", "go"], can_pick=False)

        self.assertEqual(raised.exception.code, 1)
        self.last.picker.assert_not_awaited()
        self.assertIn("agent7 -m deepseek-flash", self.last.output)

    def test_a_bare_dash_m_without_models_says_to_log_in(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(["-m"], providers=(), models=(), pick=None)

        self.assertEqual(raised.exception.code, 1)
        self.last.picker.assert_not_awaited()
        self.assertIn("log in first", self.last.output)
        self.assertIn("/login deepseek", self.last.output)


class ChosenModelSessionTests(unittest.TestCase):
    """The chosen model is the session's model, so the snapshot keeps it."""

    def test_the_model_chosen_with_dash_m_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "work"
            cwd.mkdir()
            credential = SimpleNamespace(source="test", base_url="http://x")
            provider = SimpleNamespace(id="deepseek", make_client=lambda cred: SimpleNamespace(
                close=AsyncMock()))
            with patch.dict(os.environ, {"XDG_STATE_HOME": str(root / "state")}), \
                 patch.object(cli, "resolve_all", return_value={"deepseek": credential}), \
                 patch.object(cli, "resolve_default", return_value=(provider, credential)), \
                 patch.object(cli, "fetch_models",
                              new=AsyncMock(return_value=["deepseek/deepseek-chat"])), \
                 patch.object(cli, "run_turn", new=AsyncMock(return_value="done")):
                asyncio.run(cli.async_main(
                    Options.parse(["--cwd", str(cwd), "-m", "deepseek-flash", "go"])))
                model_ref = load(cwd)["model_ref"]

            self.assertEqual(model_ref, "deepseek/deepseek-flash")


if __name__ == "__main__":
    unittest.main()
