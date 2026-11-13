"""Minimal asynchronous worker with a shutdown bug."""

import asyncio


class Worker:
    def __init__(self):
        self._closing = asyncio.Event()
        self._tick = asyncio.Event()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while not self._closing.is_set():
            await self._tick.wait()

    async def close(self):
        self._closing.set()
        await self._task
