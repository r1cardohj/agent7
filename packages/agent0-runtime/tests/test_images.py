"""Standalone image tools use runtime-owned model capabilities and Pillow."""

import base64
import io
import tempfile
import unittest
from pathlib import Path

from agent0_runtime.images import (
    PillowImageReader,
    ToolResult,
    detect_mime,
    process_image,
    supports_vision,
)
from PIL import Image


def png_bytes(width=8, height=8):
    output = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(output, format="PNG")
    return output.getvalue()


class RuntimeImageTests(unittest.TestCase):
    def test_small_png_is_unchanged(self):
        data = png_bytes()
        block, hints = process_image(data, "image/png")
        self.assertEqual(block.data, base64.b64encode(data).decode())
        self.assertEqual(hints, [])

    def test_large_image_is_resized(self):
        block, hints = process_image(png_bytes(4000, 3000), "image/png")
        with Image.open(io.BytesIO(base64.b64decode(block.data))) as image:
            self.assertLessEqual(max(image.width, image.height), 2000)
        self.assertTrue(any("resized" in hint for hint in hints))

    def test_reader_returns_provider_neutral_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.png"
            path.write_bytes(png_bytes())
            self.assertEqual(detect_mime(path), "image/png")
            result = PillowImageReader().read(path, "image/png")
            self.assertIsInstance(result, ToolResult)
            self.assertEqual(result.attachments[0].mime_type, "image/png")

    def test_reviewed_and_text_only_models(self):
        self.assertTrue(supports_vision("deepseek/deepseek-flash"))
        self.assertTrue(supports_vision("kimi-for-coding"))
        self.assertFalse(supports_vision("text-embedding-3-large"))
        self.assertFalse(supports_vision("whisper-1"))
        self.assertTrue(supports_vision("unknown-model"))


if __name__ == "__main__":
    unittest.main()
