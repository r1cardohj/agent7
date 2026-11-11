import io
import unittest
from pathlib import Path

from agent7.herdr import HerdrReporter, status_sequence
from agent7.hooks import LifecycleEvent, LifecycleKind


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def make_event(kind: LifecycleKind, *, reason: str = "") -> LifecycleEvent:
    return LifecycleEvent(
        kind=kind, seq=1, session_id="s", cwd=Path("/work"), model="m", reason=reason
    )


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
