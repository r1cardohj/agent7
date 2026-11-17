"""Inline images, pi-style: clipboard → temp path → the `read` tool.

Nothing in the prompt line is parsed into an attachment. Pasting an image only
inserts the path of a temporary file; the model reads that file with `read`,
whose tool result carries the image. Encoding, resizing and the clipboard
backends live here so neither the agent loop nor the UI touches raw bytes.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

# Same inline formats and limits as pi: anything else is converted to PNG, and
# oversized images are resized/re-encoded instead of being rejected.
SUPPORTED_MIME = ("image/png", "image/jpeg", "image/webp", "image/gif")
MIME_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}
MIME_EXTENSION = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_BYTES = int(4.5 * 1024 * 1024)
IMAGE_TOKEN_ESTIMATE = 1000  # rough per-image cost for compaction pressure
CLIPBOARD_TIMEOUT = 10.0

# Known text-only model families. Reviewed vision models are looked up in the
# capability table first; an unknown model gets the image and the provider's own
# error, which beats silently dropping context.
_TEXT_ONLY = (
    "text-", "embedding", "embed", "whisper", "tts", "rerank",
    "bge-", "e5-", "moderation",
)


def supports_vision(model: str | None) -> bool:
    """Whether this model may receive inline images.

    Reviewed models answer from the routing capability table (DeepSeek Flash
    and the Kimi coding models are vision-capable); anything unlisted falls
    back to excluding only known text-only families.
    """
    name = (model or "").lower()
    if not name:
        return True
    bare = name.rsplit("/", 1)[-1]
    from .routing import MODEL_CAPABILITIES  # local: keeps images.py dependency-free
    for capability in MODEL_CAPABILITIES:
        if capability.model.lower() == bare:
            return capability.supports_vision
    return not any(hint in name for hint in _TEXT_ONLY)


@dataclass
class ImageBlock:
    """One inline image, base64-encoded, ready for a Responses input_image."""

    data: str
    mime_type: str


@dataclass
class ToolImageOutput:
    """A tool result that is text for the UI plus images for the model."""

    text: str
    images: list[ImageBlock] = field(default_factory=list)


def base_mime(mime_type: str) -> str:
    return (mime_type or "").split(";")[0].strip().lower()


def detect_mime(path: Path) -> str | None:
    """Suffix first, then magic bytes, so a suffix-less screenshot still works."""
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
    data: bytes, mime_type: str, *,
    max_dimension: int = MAX_IMAGE_DIMENSION, max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[ImageBlock, list[str]] | None:
    """Normalize to a supported format and fit the inline size limits.

    Returns the image plus human-readable hints (also shown to the model), or
    None when the bytes cannot be decoded at all.
    """
    mime = base_mime(mime_type)
    hints: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.width, image.height
            if mime in SUPPORTED_MIME and len(data) <= max_bytes and max(width, height) <= max_dimension:
                # Keep the original bytes: an animated GIF stays animated.
                return ImageBlock(base64.b64encode(data).decode(), mime), hints
            target = mime if mime in SUPPORTED_MIME else "image/png"
            if target != mime:
                hints.append(f"[Image converted from {mime or 'unknown'} to {target}.]")
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
                hints.append(f"[Image resized from {width}x{height} to {image.width}x{image.height}.]")
            candidates = [(_encode(image, target), target)]
            if target != "image/gif":
                candidates.append((_encode(image, "image/jpeg"), "image/jpeg"))
            payload, final_mime = min(candidates, key=lambda candidate: len(candidate[0]))
            # Shrink further until the encoded image fits the byte budget.
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
            return ImageBlock(base64.b64encode(payload).decode(), final_mime), hints
    except (OSError, ValueError):
        return None


def read_image(path: Path) -> ToolImageOutput:
    """Tool result for an image file, mirroring pi's `read` text + image blocks."""
    mime = detect_mime(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        return ToolImageOutput(f"error: {type(e).__name__}: {e}")
    if not data:
        return ToolImageOutput(f"error: empty file: {path}")
    processed = process_image(data, mime or "")
    if processed is None:
        return ToolImageOutput(
            f"Read image file [{mime or 'unknown'}]\n"
            "[Image omitted: could not be converted to a supported inline image format.]"
        )
    block, hints = processed
    text = f"Read image file [{block.mime_type}]"
    if hints:
        text += "\n" + "\n".join(hints)
    return ToolImageOutput(text, [block])


# ---------------------------------------------------------------- clipboard

def _run(argv: list[str], timeout: float = CLIPBOARD_TIMEOUT) -> bytes | None:
    """stdout of a clipboard helper; None means unavailable or failed."""
    try:
        done = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 and done.stdout else None


def _preferred_mime(types: list[str]) -> str | None:
    normalized = [base_mime(t) for t in types if t.strip()]
    for mime in SUPPORTED_MIME:
        if mime in normalized:
            return mime
    return next((t for t in normalized if t.startswith("image/")), None)


def _wl_paste() -> tuple[bytes, str] | None:
    if not shutil.which("wl-paste"):
        return None
    listing = _run(["wl-paste", "--list-types"], timeout=2.0)
    if listing is None:
        return None
    mime = _preferred_mime(listing.decode("utf-8", "replace").splitlines())
    if not mime:
        return None
    data = _run(["wl-paste", "--type", mime, "--no-newline"])
    return (data, mime) if data else None


def _xclip() -> tuple[bytes, str] | None:
    if not shutil.which("xclip"):
        return None
    listing = _run(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"], timeout=2.0)
    if listing is None:
        return None
    mime = _preferred_mime(listing.decode("utf-8", "replace").splitlines())
    if not mime:
        return None
    data = _run(["xclip", "-selection", "clipboard", "-t", mime, "-o"])
    return (data, mime) if data else None


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh") or shutil.which("powershell.exe")


def _powershell_image() -> tuple[bytes, str] | None:
    """Windows and WSL: PowerShell saves the clipboard image as PNG for us."""
    exe = _powershell()
    if not exe:
        return None
    target = Path(tempfile.gettempdir()) / f"agent7-clip-{uuid.uuid4().hex}.png"
    save_as = str(target)
    if exe.endswith("powershell.exe") and shutil.which("wslpath"):
        win = _run(["wslpath", "-w", save_as], timeout=2.0)
        if not win:
            return None
        save_as = win.decode().strip()
    script = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
        f"$p='{save_as.replace(chr(39), chr(39) * 2)}';"
        "$i=[System.Windows.Forms.Clipboard]::GetImage();"
        "if($i -ne $null){$i.Save($p,[System.Drawing.Imaging.ImageFormat]::Png);"
        "Write-Output 'ok'}else{Write-Output 'empty'}"
    )
    try:
        # Write-Output terminates with \r\n; compare the trimmed text, not raw bytes.
        result = _run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout=8.0)
        if result is None or result.decode("utf-8", "replace").strip() != "ok":
            return None
        data = target.read_bytes()
        return (data, "image/png") if data else None
    except OSError:
        return None
    finally:
        target.unlink(missing_ok=True)


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def read_clipboard_image() -> tuple[bytes, str] | None:
    """Raw clipboard image bytes and their mime type, or None when there is none."""
    if sys.platform == "win32":
        return _powershell_image()
    if sys.platform == "darwin":
        # macOS has no built-in clipboard-to-PNG CLI; `brew install pngpaste`.
        if shutil.which("pngpaste"):
            data = _run(["pngpaste", "-"])
            return (data, "image/png") if data else None
        return None
    image = None
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland" or _is_wsl():
        image = _wl_paste()
    if image is None:
        image = _xclip()
    if image is None and _is_wsl():
        image = _powershell_image()
    if image and base_mime(image[1]) not in SUPPORTED_MIME:
        # e.g. Windows DIB data served as BMP: convert before it reaches `read`.
        converted = process_image(image[0], image[1])
        if converted is None:
            return None
        block, _ = converted
        return base64.b64decode(block.data), block.mime_type
    return image


def paste_image_file() -> Path | None:
    """Save a clipboard image to a temp file and return the path to insert."""
    image = read_clipboard_image()
    if not image:
        return None
    data, mime = image
    ext = MIME_EXTENSION.get(base_mime(mime), "png")
    root = Path(tempfile.gettempdir()) / "agent7-images"
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"paste-{uuid.uuid4().hex}.{ext}"
        path.write_bytes(data)
    except OSError:
        return None
    return path


def clipboard_text() -> str | None:
    """Plain-text fallback when Ctrl-V finds no image."""
    if sys.platform == "win32":
        exe = _powershell()
        raw = _run([exe, "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"]) if exe else None
    elif sys.platform == "darwin":
        raw = _run(["pbpaste"])
    elif shutil.which("wl-paste") and (
        os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland"
    ):
        raw = _run(["wl-paste", "--no-newline"])
    elif shutil.which("xclip"):
        raw = _run(["xclip", "-selection", "clipboard", "-o"])
    else:
        return None
    if not raw:
        return None
    return raw.decode("utf-8", "replace").replace("\r\n", "\n")
