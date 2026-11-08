import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.repl import context_tone, status_bar
from agent7.session import Session, context_pressure, fmt_cwd


def make_session(
    *,
    model: str = "k3.1",
    ref: str | None = "kimi/k3.1",
    limit: int | None = 262_144,
    tokens: int | None = None,
    cwd: Path = Path("/tmp/work"),
) -> Session:
    session = Session(model=model, cwd=cwd, client=None)
    session.model_ref = ref
    session.context_limit = limit
    session.prompt_tokens = tokens
    return session


def flat(segments: list[tuple[str, str]]) -> str:
    return "".join(text for _, text in segments)


class StatusBarTests(unittest.TestCase):
    def test_bar_shows_model_context_and_cwd(self) -> None:
        line = flat(status_bar(make_session(tokens=4_868), mode=("INSERT", "ansigreen"), width=80))
        self.assertIn("kimi/k3.1", line)
        self.assertIn("4,868/256k", line)
        self.assertIn(" 2%", line)
        self.assertIn("work", line)
        self.assertIn("INSERT", line)

    def test_bar_fills_exactly_one_line(self) -> None:
        for width in (40, 80, 120):
            with self.subTest(width=width):
                self.assertEqual(len(flat(status_bar(make_session(), mode=("NORMAL", "ansiyellow"), width=width))), width)

    def test_mode_sits_on_the_right(self) -> None:
        segments = status_bar(make_session(), mode=("NORMAL", "ansiyellow"), width=60)
        self.assertTrue(flat(segments).endswith(" NORMAL "))
        self.assertEqual(segments[-1], ("fg:ansiyellow", " NORMAL "))

    def test_narrow_terminal_drops_the_cwd_before_the_gauge(self) -> None:
        narrow = flat(status_bar(make_session(tokens=4_868), mode=("INSERT", "ansigreen"), width=30))
        self.assertNotIn("work", narrow)
        self.assertIn("4,868/256k", narrow)
        self.assertIn("INSERT", narrow)

    def test_model_falls_back_when_no_ref(self) -> None:
        line = flat(status_bar(make_session(ref=None), mode=("INSERT", "ansigreen"), width=80))
        self.assertIn("k3.1", line)


class ContextPressureTests(unittest.TestCase):
    def test_percentage_and_cap(self) -> None:
        self.assertEqual(context_pressure(make_session(limit=100_000, tokens=25_000)), 25)
        self.assertEqual(context_pressure(make_session(limit=100_000, tokens=250_000)), 100)

    def test_unknown_without_usage_or_limit(self) -> None:
        self.assertIsNone(context_pressure(make_session(limit=100_000)))
        self.assertIsNone(context_pressure(make_session(limit=None, tokens=10)))

    def test_tone_tracks_pressure(self) -> None:
        self.assertEqual(context_tone(None), "ansigreen")
        self.assertEqual(context_tone(0), "ansigreen")
        self.assertEqual(context_tone(59), "ansigreen")
        self.assertEqual(context_tone(60), "ansiyellow")
        self.assertEqual(context_tone(84), "ansiyellow")
        self.assertEqual(context_tone(85), "ansired")
        self.assertEqual(context_tone(100), "ansired")


class FmtCwdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        env = patch.dict(os.environ, {"HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_home_is_a_tilde(self) -> None:
        self.assertEqual(fmt_cwd(self.home), "~")
        self.assertEqual(fmt_cwd(self.home / "proj" / "agent7"), "~/proj/agent7")

    def test_outside_home_is_absolute(self) -> None:
        self.assertEqual(fmt_cwd(Path("/srv/app")), "/srv/app")

    def test_long_paths_keep_their_tail(self) -> None:
        deep = self.home / "a" / "bb" / "ccc" / "dddd" / "eeeee"
        shown = fmt_cwd(deep, limit=16)
        self.assertLessEqual(len(shown), 16)
        self.assertTrue(shown.startswith("…/"))
        self.assertTrue(shown.endswith("eeeee"))


if __name__ == "__main__":
    unittest.main()
