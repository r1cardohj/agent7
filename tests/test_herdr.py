import io
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.herdr import (
    HerdrReporter,
    background_tasks_available,
    background_tasks_instructions,
    herdr_enabled,
    herdr_skill_path,
    status_sequence,
)
from agent7.hooks import LifecycleEvent, LifecycleKind


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def make_event(kind: LifecycleKind, *, reason: str = "") -> LifecycleEvent:
    return LifecycleEvent(
        kind=kind, seq=1, session_id="s", cwd=Path("/work"), model="m", reason=reason
    )


class HerdrBackgroundTaskTests(unittest.TestCase):
    def test_runtime_enabled_requires_terminal_and_not_disabled(self) -> None:
        self.assertFalse(herdr_enabled({}))
        self.assertFalse(herdr_enabled({"HC_TERMINAL_ID": "", "AGENT7_NO_HERDR": "0"}))
        self.assertTrue(herdr_enabled({"HC_TERMINAL_ID": "t7"}))
        self.assertFalse(herdr_enabled({"HC_TERMINAL_ID": "t7", "AGENT7_NO_HERDR": "1"}))

    def test_background_tasks_require_terminal_and_hc_cli(self) -> None:
        def found(_: str) -> str:
            return "/usr/bin/hc"

        def missing(_: str) -> None:
            return None

        self.assertFalse(background_tasks_available({}, which=found))
        self.assertFalse(background_tasks_available({"HC_TERMINAL_ID": "t7"}, which=missing))
        self.assertFalse(background_tasks_available(
            {"HC_TERMINAL_ID": "t7", "AGENT7_NO_HERDR": "1"}, which=found
        ))
        self.assertTrue(background_tasks_available({"HC_TERMINAL_ID": "t7"}, which=found))

    def test_instructions_are_empty_without_the_herdr_cli(self) -> None:
        env = {"HC_TERMINAL_ID": "t7"}
        self.assertEqual(
            background_tasks_instructions(env, which=lambda _: None), ""
        )
        self.assertEqual(
            background_tasks_instructions({}, which=lambda _: "/usr/bin/hc"), ""
        )

    def test_instructions_point_to_the_packaged_skill(self) -> None:
        prompt = background_tasks_instructions(
            {"HC_TERMINAL_ID": "t7"}, which=lambda _: "/usr/bin/hc"
        )
        path = herdr_skill_path()
        self.assertTrue(path.is_file())
        self.assertIn(str(path), prompt)
        self.assertIn("read and follow this bundled skill exactly", prompt)
        self.assertIn("Do not use hc before reading it", prompt)
        self.assertIn("If the file cannot be read", prompt)
        self.assertNotIn("hc run --bg", prompt)
        self.assertLess(len(prompt), 600)

    def test_missing_skill_disables_the_pointer(self) -> None:
        with patch("agent7.herdr.herdr_skill_path", return_value=Path("/missing/SKILL.md")):
            self.assertEqual(
                background_tasks_instructions(
                    {"HC_TERMINAL_ID": "t7"}, which=lambda _: "/usr/bin/hc"
                ),
                "",
            )

    def test_skill_covers_launch_wait_output_and_fallback(self) -> None:
        skill = herdr_skill_path().read_text(encoding="utf-8")
        self.assertIn("hc run --bg --no-notify", skill)
        self.assertIn("about 60 seconds", skill)
        self.assertIn("git status", skill)
        self.assertIn("never wrap", skill)
        self.assertIn("one ordinary bash call", skill)
        self.assertIn("pass/fail-only", skill)
        self.assertIn("add `--rm`", skill)
        self.assertIn("skip both log", skill)
        self.assertIn("reading and cleanup", skill)
        self.assertIn("hc wait <terminal>", skill)
        self.assertIn("hc read --lines 100 <terminal>", skill)
        self.assertIn("never claim success before observing exit code 0", skill)
        self.assertIn("command-not-found", skill)
        self.assertIn("absolute path", skill)
        self.assertIn("same", skill)
        self.assertIn("terminal", skill)
        self.assertIn("again rather than rerunning", skill)
        self.assertIn("fall back", skill)
        self.assertIn("ordinary bash", skill)
        self.assertIn("Do not use hc for agents", skill)
        self.assertIn("--outlive-parent", skill)

    def test_service_instructions_cover_readiness_lifecycle_and_reuse(self) -> None:
        skill = herdr_skill_path().read_text(encoding="utf-8")
        self.assertIn("## Long-running service", skill)
        self.assertIn("stable, meaningful name", skill)
        self.assertIn("never pass `--rm`", skill)
        self.assertIn("hc get <name>", skill)
        self.assertIn("reuse that service", skill)
        self.assertIn("Never call `hc wait` without `--output`", skill)
        self.assertIn("--timeout 30 --output <ready-marker>", skill)
        self.assertIn("listening endpoint", skill)
        self.assertIn("never restart merely because the wait timed out", skill)
        self.assertIn("Leave the terminal running", skill)
        self.assertIn("explicit request", skill)
        self.assertIn("hc kill <terminal>", skill)
        self.assertIn("port reuse", skill)
        self.assertIn("never blindly relaunch", skill)


class HerdrTests(unittest.TestCase):
    def test_no_output_outside_herdr_or_when_disabled(self) -> None:
        for env in ({}, {"HC_TERMINAL_ID": "t7", "AGENT7_NO_HERDR": "1"}):
            with self.subTest(env=env):
                stream = TTY()
                HerdrReporter(env=env, stream=stream)(make_event("turn_started"))
                self.assertEqual(stream.getvalue(), "")

    def test_redirected_output_does_not_receive_escape_sequences(self) -> None:
        stream = io.StringIO()
        HerdrReporter(env={"HC_TERMINAL_ID": "t7"}, stream=stream)(
            make_event("turn_started")
        )
        self.assertEqual(stream.getvalue(), "")

    def test_lifecycle_maps_to_osc_7501(self) -> None:
        cases = [
            ("session_started", "idle"),
            ("turn_started", "working"),
            ("turn_succeeded", "done"),
            ("turn_interrupted", "idle"),
            ("turn_failed", "error"),
            ("session_ended", "clear"),
        ]
        for kind, state in cases:
            with self.subTest(kind=kind):
                stream = TTY()
                HerdrReporter(env={"HC_TERMINAL_ID": "t7"}, stream=stream)(make_event(kind))
                self.assertEqual(stream.getvalue(), f"\x1b]7501;state={state}:app=agent7\x1b\\")

        stream = TTY()
        HerdrReporter(env={"HC_TERMINAL_ID": "t7"}, stream=stream)(
            make_event("turn_blocked", reason="max_turns")
        )
        self.assertEqual(
            stream.getvalue(), "\x1b]7501;state=blocked:kind=question:app=agent7\x1b\\"
        )

    def test_protocol_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            status_sequence("finished")
        with self.assertRaises(ValueError):
            status_sequence("blocked", kind="unknown")


if __name__ == "__main__":
    unittest.main()
