"""Image handling: pi-style clipboard paths, `read` attachments, wire format."""

import asyncio
import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from responses_fakes import FakeClient, call_turn, text_turn

from agent7.agent import compact_messages, dispatch_tool, estimate_tokens, run_agent
from agent7.core import TOOL_ATTACHMENTS_KEY
from agent7.images import (
    ToolImageOutput,
    detect_mime,
    paste_image_file,
    process_image,
    supports_vision,
)
from agent7.repl import repl_key_bindings
from agent7.responses import IMAGES_KEY, request_input, tool_output


def png_bytes(width: int = 8, height: int = 8, color: tuple = (255, 0, 0)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


class ProcessImageTests(unittest.TestCase):
    def test_small_png_is_sent_unchanged(self) -> None:
        data = png_bytes()
        block, hints = process_image(data, "image/png")

        self.assertEqual(block.data, base64.b64encode(data).decode())
        self.assertEqual(block.mime_type, "image/png")
        self.assertEqual(hints, [])

    def test_oversized_image_is_resized_with_a_hint(self) -> None:
        block, hints = process_image(png_bytes(4000, 3000), "image/png", max_bytes=10_000_000)

        with Image.open(io.BytesIO(base64.b64decode(block.data))) as image:
            self.assertLessEqual(max(image.width, image.height), 2000)
        self.assertTrue(any("resized" in hint for hint in hints))

    def test_bmp_is_converted_to_a_supported_format(self) -> None:
        out = io.BytesIO()
        Image.new("RGB", (8, 8), (0, 255, 0)).save(out, format="BMP")

        block, hints = process_image(out.getvalue(), "image/bmp")

        self.assertIn(block.mime_type, ("image/png", "image/jpeg"))
        self.assertTrue(any("converted" in hint for hint in hints))

    def test_undecodable_bytes_are_omitted(self) -> None:
        self.assertIsNone(process_image(b"not an image", "image/png"))

    def test_supports_vision_uses_the_reviewed_table_then_heuristics(self) -> None:
        self.assertTrue(supports_vision("deepseek-flash"))
        self.assertTrue(supports_vision("deepseek/deepseek-flash"))
        self.assertTrue(supports_vision("kimi/kimi-for-coding"))
        self.assertTrue(supports_vision("deepseek-v4-pro"))
        self.assertFalse(supports_vision("text-embedding-3-large"))
        self.assertFalse(supports_vision("whisper-1"))
        self.assertTrue(supports_vision("some-unknown-model"))
        self.assertTrue(supports_vision(None))


class ReadImageToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        (self.cwd / "shot.png").write_bytes(png_bytes())

    def test_detect_mime_uses_suffix_and_magic_bytes(self) -> None:
        self.assertEqual(detect_mime(self.cwd / "shot.png"), "image/png")
        bare = self.cwd / "clipboard"
        bare.write_bytes(png_bytes())
        self.assertEqual(detect_mime(bare), "image/png")
        self.assertIsNone(detect_mime(self.cwd / "missing.txt"))

    def test_read_returns_text_plus_an_image_block(self) -> None:
        result = dispatch_tool(self.cwd, "read", {"path": "shot.png"}, model="kimi-for-coding")

        self.assertIsInstance(result, ToolImageOutput)
        self.assertIn("Read image file [image/png]", result.text)
        self.assertEqual(len(result.images), 1)
        self.assertNotIn(result.images[0].data, result.text)

    def test_read_omits_the_image_for_a_text_only_model(self) -> None:
        result = dispatch_tool(
            self.cwd, "read", {"path": "shot.png"}, model="text-embedding-3-large"
        )

        self.assertIsInstance(result, str)
        self.assertIn("does not support images", result)

    def test_read_keeps_the_image_for_deepseek_flash(self) -> None:
        result = dispatch_tool(self.cwd, "read", {"path": "shot.png"}, model="deepseek-flash")
        self.assertIsInstance(result, ToolImageOutput)
        self.assertEqual(len(result.images), 1)

    def test_read_still_reads_text(self) -> None:
        (self.cwd / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.assertIn("x = 1", dispatch_tool(self.cwd, "read", {"path": "a.py"}))

    def test_run_agent_puts_the_image_in_the_next_request(self) -> None:
        client = FakeClient([call_turn(("read", {"path": "shot.png"})), text_turn("done")])
        messages = []

        answer = asyncio.run(
            run_agent(
                "look",
                cwd=self.cwd,
                client=client,
                model="kimi-for-coding",
                messages=messages,
                on_tool=lambda *_a: None,
            )
        )

        self.assertEqual(answer, "done")
        tool_message = next(m for m in messages if m.get("role") == "tool")
        self.assertEqual(len(tool_message[TOOL_ATTACHMENTS_KEY]), 1)
        outputs = [
            i for i in client.requests[1]["input"] if i.get("type") == "function_call_output"
        ]
        parts = outputs[0]["output"]
        self.assertEqual(parts[0], {"type": "input_text", "text": tool_message["content"]})
        self.assertEqual(parts[1]["type"], "input_image")
        self.assertTrue(parts[1]["image_url"].startswith("data:image/png;base64,"))

    def test_request_drops_images_for_a_text_only_model(self) -> None:
        client = FakeClient([call_turn(("read", {"path": "shot.png"})), text_turn("done")])
        messages = []

        asyncio.run(
            run_agent(
                "look",
                cwd=self.cwd,
                client=client,
                model="text-embedding-3-large",
                messages=messages,
                on_tool=lambda *_a: None,
            )
        )

        outputs = [
            i for i in client.requests[1]["input"] if i.get("type") == "function_call_output"
        ]
        self.assertIsInstance(outputs[0]["output"], str)


class ImageHistoryTests(unittest.TestCase):
    def message(self) -> dict:
        return {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Read image file",
            IMAGES_KEY: [{"data": "QUJD", "mimeType": "image/png"}],
        }

    def test_tool_output_accepts_string_and_blocks(self) -> None:
        message = self.message()
        parts = tool_output(message)
        self.assertEqual(parts[1]["image_url"], "data:image/png;base64,QUJD")
        self.assertEqual(tool_output(message, vision=False), "Read image file")
        self.assertEqual(tool_output({"content": "plain"}), "plain")

    def test_request_input_keeps_other_tools_as_strings(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            self.message(),
            {"role": "tool", "tool_call_id": "call_2", "content": "text only"},
        ]
        inputs, _ = request_input(messages)
        self.assertEqual(inputs[0], {"role": "user", "content": "go"})
        self.assertIsInstance(inputs[1]["output"], list)
        self.assertEqual(inputs[2]["output"], "text only")

    def test_compaction_drops_old_images(self) -> None:
        old = self.message()
        messages = [old] + [{"role": "user", "content": f"turn {i}"} for i in range(4)]

        compacted, changed = compact_messages(messages, keep_user_turns=2)

        self.assertGreaterEqual(changed, 1)
        self.assertNotIn(IMAGES_KEY, compacted[0])
        self.assertIn("[image omitted: compacted]", compacted[0]["content"])

    def test_estimate_tokens_charges_an_image_not_its_base64(self) -> None:
        with_image = estimate_tokens([self.message()])
        without = estimate_tokens([{"role": "tool", "content": "Read image file"}])
        self.assertEqual(with_image - without, 1000)


class KeyBindingTests(unittest.TestCase):
    """Ctrl-V and Alt-V both paste from the clipboard; there is no third key."""

    def bindings(self) -> dict[tuple[str, ...], object]:
        return {
            tuple(str(k) for k in binding.keys): binding.handler
            for binding in repl_key_bindings().bindings
        }

    def test_ctrl_v_and_alt_v_share_the_paste_handler(self) -> None:
        handlers = self.bindings()
        self.assertEqual(handlers[("Keys.ControlV",)].__name__, "_paste")
        self.assertEqual(handlers[("Keys.Escape", "v")].__name__, "_paste")

    def test_paste_inserts_the_image_path_as_one_segment(self) -> None:
        inserted: list[str] = []
        buffer = SimpleNamespace(text="look at", insert_text=inserted.append)
        image = Path("/tmp/agent7-images/paste-abc.png")
        handler = self.bindings()[("Keys.ControlV",)]
        with (
            patch("agent7.repl.paste_image_file", return_value=image),
            patch(
                "agent7.repl.clipboard_text", side_effect=AssertionError("text must not be pasted")
            ),
        ):
            asyncio.run(handler(SimpleNamespace(current_buffer=buffer)))
        self.assertEqual(inserted, [f" @{image} "])

    def test_paste_falls_back_to_text_without_an_image(self) -> None:
        inserted: list[str] = []
        buffer = SimpleNamespace(text="", insert_text=inserted.append)
        handler = self.bindings()[("Keys.Escape", "v")]
        with (
            patch("agent7.repl.paste_image_file", return_value=None),
            patch("agent7.repl.clipboard_text", return_value="plain text"),
        ):
            asyncio.run(handler(SimpleNamespace(current_buffer=buffer)))
        self.assertEqual(inserted, ["plain text"])


class ClipboardTests(unittest.TestCase):
    def test_paste_image_file_writes_the_bytes_to_a_temp_path(self) -> None:
        data = png_bytes()
        with patch("agent7.images.read_clipboard_image", return_value=(data, "image/png")):
            path = paste_image_file()
        self.assertIsNotNone(path)
        self.assertTrue(path.name.startswith("paste-"))
        self.assertEqual(path.suffix, ".png")
        self.assertEqual(path.read_bytes(), data)
        path.unlink()

    def test_paste_image_file_returns_none_without_an_image(self) -> None:
        with patch("agent7.images.read_clipboard_image", return_value=None):
            self.assertIsNone(paste_image_file())


class WslClipboardTests(unittest.TestCase):
    def test_powershell_ok_with_crlf_is_accepted(self) -> None:
        """Write-Output ends in \r\n; comparing raw bytes to b"ok" always failed."""
        from agent7.images import _powershell_image

        with (
            patch("agent7.images._powershell", return_value="powershell.exe"),
            patch("agent7.images.shutil.which", return_value="/usr/bin/wslpath"),
            patch("agent7.images._run", side_effect=[b"C:\\Temp\\x.png", b"ok\r\n"]),
            patch("agent7.images.Path.read_bytes", return_value=b"PNGDATA"),
            patch("agent7.images.Path.unlink"),
        ):
            self.assertEqual(_powershell_image(), (b"PNGDATA", "image/png"))


class PasteKeyIntegrationTests(unittest.TestCase):
    """Drive the real prompt through a pipe: both keys must insert the path."""

    def drive(self, keys: str) -> str:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        async def run() -> str:
            with create_pipe_input() as inp:
                session = PromptSession(
                    input=inp,
                    output=DummyOutput(),
                    history=InMemoryHistory(),
                    editing_mode=EditingMode.VI,
                    multiline=True,
                    key_bindings=repl_key_bindings(),
                )
                task = asyncio.ensure_future(session.prompt_async("> "))
                await asyncio.sleep(0.1)
                for key in keys:
                    inp.send_text(key)
                    await asyncio.sleep(0.3)
                await asyncio.sleep(0.3)
                inp.send_text("\r")
                return await asyncio.wait_for(task, 5)

        return asyncio.run(run())

    def test_ctrl_v_and_alt_v_insert_the_image_path(self) -> None:
        image = Path("/tmp/agent7-images/fake.png")
        with (
            patch("agent7.repl.paste_image_file", return_value=image),
            patch("agent7.repl.clipboard_text", return_value=None),
        ):
            self.assertEqual(self.drive("\x16"), f"@{image} ")  # Ctrl-V
            self.assertEqual(self.drive("\x1bv"), f"@{image} ")  # Alt-V = ESC v


if __name__ == "__main__":
    unittest.main()
