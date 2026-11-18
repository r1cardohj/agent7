"""Image decoding, normalization, vision policy, and tool attachments."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

from .capabilities import MODEL_CAPABILITIES
from .filesystem import ImageReader
from .types import Attachment, ToolResult

SUPPORTED_MIME = ("image/png", "image/jpeg", "image/webp", "image/gif")
MIME_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
MIME_EXTENSION = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_BYTES = int(4.5 * 1024 * 1024)
IMAGE_TOKEN_ESTIMATE = 1000

_TEXT_ONLY = (
    "text-",
    "embedding",
    "embed",
    "whisper",
    "tts",
    "rerank",
    "bge-",
    "e5-",
    "moderation",
)


def supports_vision(model: str | None) -> bool:
    """Check reviewed capabilities first, then exclude known text-only families."""

    name = (model or "").lower()
    if not name:
        return True
    bare = name.rsplit("/", 1)[-1]
    for capability in MODEL_CAPABILITIES:
        if capability.model.lower() == bare:
            return capability.supports_vision
    return not any(hint in name for hint in _TEXT_ONLY)


def base_mime(mime_type: str) -> str:
    return (mime_type or "").split(";")[0].strip().lower()


def detect_mime(path: Path) -> str | None:
    """Use the suffix first, then magic bytes for suffix-less screenshots."""

    mime = MIME_SUFFIX.get(path.suffix.lower())
    if mime:
        return mime
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"BM"):
        return "image/bmp"
    return None


def _encode(image: Image.Image, mime_type: str, quality: int = 85) -> bytes:
    out = io.BytesIO()
    if mime_type == "image/jpeg":
        image.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def process_image(
    data: bytes,
    mime_type: str,
    *,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[Attachment, list[str]] | None:
    """Normalize an image to a supported format and fit inline size limits."""

    mime = base_mime(mime_type)
    hints: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.width, image.height
            if (
                mime in SUPPORTED_MIME
                and len(data) <= max_bytes
                and max(width, height) <= max_dimension
            ):
                # Keep original bytes so an animated GIF stays animated.
                return Attachment(base64.b64encode(data).decode(), mime), hints
            target = mime if mime in SUPPORTED_MIME else "image/png"
            if target != mime:
                hints.append(f"[Image converted from {mime or 'unknown'} to {target}.]")
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
                hints.append(
                    f"[Image resized from {width}x{height} to {image.width}x{image.height}.]"
                )
            candidates = [(_encode(image, target), target)]
            if target != "image/gif":
                candidates.append((_encode(image, "image/jpeg"), "image/jpeg"))
            payload, final_mime = min(candidates, key=lambda candidate: len(candidate[0]))
            while len(payload) > max_bytes and max(image.width, image.height) > 256:
                image = image.resize(
                    (max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)),
                    Image.Resampling.LANCZOS,
                )
                payload = _encode(image, "image/jpeg", quality=70)
                final_mime = "image/jpeg"
            if len(payload) > max_bytes:
                return None
            if final_mime != target:
                hints.append(f"[Image re-encoded as {final_mime} to fit the size limit.]")
            return Attachment(base64.b64encode(payload).decode(), final_mime), hints
    except (OSError, ValueError):
        return None


class PillowImageReader(ImageReader):
    """Default ImageReader backed by Pillow and provider-neutral attachments."""

    def read(self, path: Path, mime_type: str | None) -> ToolResult:
        mime = mime_type or detect_mime(path) or ""
        try:
            data = path.read_bytes()
        except OSError as e:
            return ToolResult(f"error: {type(e).__name__}: {e}")
        if not data:
            return ToolResult(f"error: empty file: {path}")
        processed = process_image(data, mime)
        if processed is None:
            return ToolResult(
                f"Read image file [{mime or 'unknown'}]\n"
                "[Image omitted: could not be converted to a supported inline image format.]"
            )
        attachment, hints = processed
        text = f"Read image file [{attachment.mime_type}]"
        if hints:
            text += "\n" + "\n".join(hints)
        return ToolResult(text, attachments=(attachment,))
