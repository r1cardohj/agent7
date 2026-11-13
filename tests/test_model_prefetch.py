"""Background model discovery shares one request and never outlives the session."""

import asyncio
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
            release = asyncio.Event()

            async def fetch(_session):
                nonlocal calls
                calls += 1
                await release.wait()
                return ["deepseek/deepseek-flash"]

            with patch("agent7.commands._fetch_models", new=fetch):
                background = prefetch_models(session)
                first = asyncio.create_task(fetch_models(session))
                second = asyncio.create_task(fetch_models(session))
                await asyncio.sleep(0)
                self.assertIs(session._models_fetch, background)
                self.assertFalse(first.done())
                self.assertFalse(second.done())

                release.set()
                refs = await asyncio.gather(first, second)
                self.assertEqual(refs, [["deepseek/deepseek-flash"]] * 2)
                self.assertEqual(calls, 1)
            await session.close()

        asyncio.run(exercise())

    def test_closing_the_session_cancels_an_unfinished_prefetch(self) -> None:
        async def exercise() -> None:
            session = Session(model="m", cwd=Path("."), client=None)
            started = asyncio.Event()
            cancelled = asyncio.Event()

            async def fetch(_session):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            with patch("agent7.commands._fetch_models", new=fetch):
                task = prefetch_models(session)
                await started.wait()
                await session.close()
                self.assertTrue(cancelled.is_set())
                self.assertTrue(task.done())

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
