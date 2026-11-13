import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agent7.cli as cli
from agent7.agent import Interrupted
from agent7.cli import Options
from agent7.hooks import LifecycleEvent, LifecycleHooks, emit_lifecycle
from agent7.repl import run_turn
from agent7.session import Session
from agent7.store import workspace_key


def make_session(cwd: Path) -> Session:
    return Session(model="m", model_ref="test/m", cwd=cwd, client=None)


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def recording_hooks(self) -> tuple[LifecycleHooks, list[LifecycleEvent]]:
        events: list[LifecycleEvent] = []
        hooks = LifecycleHooks()
        hooks.subscribe(events.append)
        return hooks, events

    def test_event_metadata_and_fault_isolation(self) -> None:
        events: list[LifecycleEvent] = []
        seen: list[LifecycleEvent] = []

        def broken(_event: LifecycleEvent) -> None:
            raise RuntimeError("observer failed")

        hooks = LifecycleHooks()
        hooks.subscribe(broken)
        hooks.subscribe(events.append)
        hooks.subscribe(seen.append)
        session = make_session(self.cwd)
        session.hooks = hooks
        emit_lifecycle(session, "session_started")
        emit_lifecycle(session, "turn_blocked", reason="max_turns")

        self.assertEqual([event.kind for event in seen], ["session_started", "turn_blocked"])
        self.assertEqual([event.seq for event in seen], [1, 2])
        self.assertEqual(seen[0].session_id, workspace_key(self.cwd))
        self.assertEqual(seen[0].cwd, self.cwd.resolve())
        self.assertEqual(seen[0].model, "test/m")
        self.assertEqual(seen[1].reason, "max_turns")
        self.assertEqual(len(events), 2)

    def run_with_answer(self, answer: str) -> list[LifecycleEvent]:
        session = make_session(self.cwd)
        hooks, events = self.recording_hooks()
        session.hooks = hooks
        with patch("agent7.repl._run_agent_once", new=AsyncMock(return_value=answer)):
            self.assertEqual(asyncio.run(run_turn(session, "go")), answer)
        return events

    def test_success_emits_done_without_a_followup_idle(self) -> None:
        events = self.run_with_answer("finished")
        self.assertEqual([event.kind for event in events], ["turn_started", "turn_succeeded"])

    def test_turn_limit_emits_blocked(self) -> None:
        events = self.run_with_answer("(stopped after 2 turns)")
        self.assertEqual([event.kind for event in events], ["turn_started", "turn_blocked"])
        self.assertEqual(events[-1].reason, "max_turns")

    def test_interruption_and_failure_have_distinct_events(self) -> None:
        cases = [(Interrupted(), "turn_interrupted"), (RuntimeError("x"), "turn_failed")]
        for error, kind in cases:
            with self.subTest(kind=kind):
                session = make_session(self.cwd)
                hooks, events = self.recording_hooks()
                session.hooks = hooks
                with patch("agent7.repl._run_agent_once", new=AsyncMock(side_effect=error)):
                    with self.assertRaises(type(error)):
                        asyncio.run(run_turn(session, "go"))
                self.assertEqual([event.kind for event in events], ["turn_started", kind])
                if kind == "turn_failed":
                    self.assertEqual(events[-1].reason, "RuntimeError")


class CliLifecycleTests(unittest.TestCase):
    def test_cli_emits_session_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cwd = Path(root) / "work"
            cwd.mkdir()
            events: list[LifecycleEvent] = []
            systems: list[str] = []
            client = SimpleNamespace(close=AsyncMock())
            provider = SimpleNamespace(id="deepseek", make_client=lambda cred: client)
            credential = SimpleNamespace(source="test")

            def session_with_events(**kwargs) -> Session:
                systems.append(kwargs["messages"][0]["content"])
                session = Session(**kwargs)
                session.hooks.subscribe(events.append)
                return session

            async def fake_run(session: Session, task: str, **kwargs) -> str:
                emit_lifecycle(session, "turn_started")
                emit_lifecycle(session, "turn_succeeded")
                return "done"

            with patch.dict(os.environ, {"XDG_STATE_HOME": str(Path(root) / "state")}), \
                 patch.object(cli, "background_tasks_instructions",
                              return_value="\nHERDR-BACKGROUND") as instructions, \
                 patch.object(cli, "HerdrReporter", return_value=lambda event: None) as herdr, \
                 patch.object(cli, "Session", new=session_with_events), \
                 patch.object(cli, "resolve_all", return_value={"deepseek": credential}), \
                 patch.object(cli, "resolve_default", return_value=(provider, credential)), \
                 patch.object(cli, "fetch_models", new=AsyncMock(return_value=[])), \
                 patch.object(cli, "switch_model", new=AsyncMock(
                     return_value=SimpleNamespace(message="ok")
                 )), \
                 patch.object(cli, "run_turn", new=fake_run):
                asyncio.run(cli.async_main(Options.parse(["--cwd", str(cwd), "go"])))

            instructions.assert_called_once_with()
            herdr.assert_called_once_with()
            self.assertEqual(len(systems), 1)
            self.assertTrue(systems[0].endswith("HERDR-BACKGROUND"))
            self.assertEqual(
                [event.kind for event in events],
                ["session_started", "turn_started", "turn_succeeded", "session_ended"],
            )


if __name__ == "__main__":
    unittest.main()
