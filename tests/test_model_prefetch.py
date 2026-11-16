"""Model discovery stays responsive, shares one worker, and never delays exit."""

import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.commands import fetch_models, prefetch_models
from agent7.session import Session


class ModelPrefetchTests(unittest.TestCase):
    def test_model_command_joins_the_background_prefetch(self) -> None:
        async def exercise() -> None:
            session = Session(model="m", cwd=Path("."), client=None)
            calls = 0
            started = threading.Event()
            release = threading.Event()

            def discover(_session):
                nonlocal calls
                calls += 1
                started.set()
                self.assertTrue(release.wait(timeout=2))
                return ["deepseek/deepseek-flash"], {}

            with patch("agent7.commands._discover_models", new=discover):
                background = prefetch_models(session)
                first = asyncio.create_task(fetch_models(session))
                second = asyncio.create_task(fetch_models(session))
                for _ in range(200):
                    if started.is_set() and first and second:
                        break
                    await asyncio.sleep(0.01)
                self.assertIs(session._models_fetch, background)
                self.assertFalse(first.done())
                self.assertFalse(second.done())

                release.set()
                refs = await asyncio.gather(first, second)
                self.assertEqual(refs, [["deepseek/deepseek-flash"]] * 2)
                self.assertEqual(calls, 1)
                self.assertEqual(session.models_cache, ["deepseek/deepseek-flash"])
            await session.close()

        asyncio.run(exercise())

    def test_closing_the_session_never_waits_for_a_stuck_prefetch(self) -> None:
        async def exercise() -> None:
            session = Session(model="m", cwd=Path("."), client=None)
            started = threading.Event()
            release = threading.Event()

            def discover(_session):
                started.set()
                self.assertTrue(release.wait(timeout=2))
                return [], {}

            with patch("agent7.commands._discover_models", new=discover):
                task = prefetch_models(session)
                self.assertTrue(started.wait(timeout=1))
                await asyncio.wait_for(session.close(), timeout=1)
                self.assertFalse(task.done)
                release.set()
                self.assertTrue(task._done.wait(timeout=2))
                self.assertIsNone(session.models_cache)

        asyncio.run(exercise())

    def test_background_failure_surfaces_when_model_is_requested(self) -> None:
        async def exercise() -> None:
            session = Session(model="m", cwd=Path("."), client=None)

            def discover(_session):
                raise RuntimeError("network blocked")

            with patch("agent7.commands._discover_models", new=discover):
                prefetch_models(session)
                with self.assertRaisesRegex(RuntimeError, "network blocked"):
                    await fetch_models(session)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
