"""Terminal clipboard integration; image processing lives in agent0-runtime."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from agent0_runtime.images import (
    IMAGE_TOKEN_ESTIMATE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MIME_EXTENSION,
    MIME_SUFFIX,
    SUPPORTED_MIME,
    ImageBlock,
    ToolImageOutput,
    base_mime,
    detect_mime,
    process_image,
    read_image,
    supports_vision,
)

__all__ = [
    "IMAGE_TOKEN_ESTIMATE",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_DIMENSION",
    "MIME_EXTENSION",
    "MIME_SUFFIX",
    "SUPPORTED_MIME",
    "ImageBlock",
    "ToolImageOutput",
    "base_mime",
    "clipboard_text",
    "detect_mime",
    "paste_image_file",
    "process_image",
    "read_clipboard_image",
    "read_image",
    "supports_vision",
]

CLIPBOARD_TIMEOUT = 10.0


def _run(argv: list[str], timeout: float = CLIPBOARD_TIMEOUT) -> bytes | None:
    """Return clipboard helper stdout, or None when unavailable or failed."""

    try:
        done = subprocess.run(argv, check=False, capture_output=True, timeout=timeout)
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
    if not listing:
        return None
    mime = _preferred_mime(listing.decode("utf-8", "replace").splitlines())
    if not mime:
        return None
    data = _run(["xclip", "-selection", "clipboard", "-t", mime, "-o"])
    return (data, mime) if data else None


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh") or shutil.which("powershell.exe")


def _powershell_image() -> tuple[bytes, str] | None:
    """On Windows and WSL, PowerShell saves the clipboard image as PNG."""

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
    """Return raw clipboard image bytes and mime type, or None when empty."""

    if sys.platform == "win32":
        return _powershell_image()
    if sys.platform == "darwin":
        if shutil.which("pngpaste"):
            data = _run(["pngpaste", "-"])
            return (data, "image/png") if data else None
        return None
    image = None
    if (
        os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        or _is_wsl()
    ):
        image = _wl_paste()
    if image is None:
        image = _xclip()
    if image is None and _is_wsl():
        image = _powershell_image()
    if image and base_mime(image[1]) not in SUPPORTED_MIME:
        converted = process_image(image[0], image[1])
        if converted is None:
            return None
        block, _ = converted
        return base64.b64decode(block.data), block.mime_type
    return image


def paste_image_file() -> Path | None:
    """Save a clipboard image to a temp file and return its path."""

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
    """Return plain-text clipboard content when Ctrl-V finds no image."""

    if sys.platform == "win32":
        exe = _powershell()
        raw = (
            _run([exe, "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"])
            if exe
            else None
        )
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
