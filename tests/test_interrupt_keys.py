import asyncio
import unittest

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.input.typeahead import get_typeahead

from agent7.interrupt import run_with_interrupt_keys


class InterruptKeyTests(unittest.TestCase):
    def assert_cancels(self, data: bytes) -> None:
        async def exercise() -> None:
            with create_pipe_input() as input_obj:
                started = asyncio.Event()
                cancelled = asyncio.Event()

                async def work() -> None:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise

                running = asyncio.create_task(
                    run_with_interrupt_keys(work(), input_obj=input_obj)
                )
                await started.wait()
                await asyncio.sleep(0.02)
                input_obj.send_text("next")
                input_obj.send_bytes(data)
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(running, timeout=2)
                self.assertTrue(cancelled.is_set())
                keys = get_typeahead(input_obj)
                self.assertEqual("".join(key.data for key in keys), "next")

        asyncio.run(exercise())

    def test_escape_cancels_and_preserves_typeahead(self) -> None:
        self.assert_cancels(b"\x1b")

    def test_ctrl_c_cancels_and_preserves_typeahead(self) -> None:
        self.assert_cancels(b"\x03")

    def test_normal_completion_closes_listener_and_keeps_typeahead(self) -> None:
        async def exercise() -> None:
            with create_pipe_input() as input_obj:
                async def work() -> str:
                    input_obj.send_text("later")
                    await asyncio.sleep(0.05)
                    return "done"

                result = await asyncio.wait_for(
                    run_with_interrupt_keys(work(), input_obj=input_obj), timeout=2
                )
                self.assertEqual(result, "done")
                keys = get_typeahead(input_obj)
                self.assertEqual("".join(key.data for key in keys), "later")

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
