"""Provider-neutral filesystem tool implementations."""

from __future__ import annotations

import difflib
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from .types import ToolResult

MAX_READ_CHARS = 50_000
DEFAULT_READ_LINES = 200
IMAGE_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class ImageReader(Protocol):
    """Host-provided image processing, keeping optional Pillow out of the core."""

    def read(self, path: Path, mime_type: str | None) -> ToolResult: ...


class VisionSupport(Protocol):
    """Host policy for whether the active model can consume an attachment."""

    def __call__(self, model: str | None) -> bool: ...


def resolve_path(cwd: Path, path: str) -> Path:
    p = Path(path.removeprefix("@").replace("\\ ", " ")).expanduser()
    return p if p.is_absolute() else cwd / p


def detect_image_mime(path: Path) -> str | None:
    """Suffix first, then magic bytes, so a suffix-less screenshot still works."""

    mime = IMAGE_SUFFIX_MIME.get(path.suffix.lower())
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


def write_atomic(fp: Path, text: str) -> None:
    """Temp file + os.replace: an interrupted write can never truncate the file."""

    target = fp.resolve() if fp.is_symlink() else fp  # keep symlinks pointing where they did
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        with suppress(OSError):
            os.chmod(tmp, target.stat().st_mode & 0o7777)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def count_lines(fp: Path) -> int:
    """Total line count at C speed (bytes.count) so a 100 MB log stays fast."""

    total, tail = 0, b"\n"
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            total += chunk.count(b"\n")
            tail = chunk[-1:]
    return total if tail == b"\n" or total == 0 else total + 1


def _truncate_text(s: str, limit: int, *, tag: str = "truncated") -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[{tag} {len(s) - limit} chars]"


def read_file(
    cwd: Path | None,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    *,
    model: str | None = None,
    image_reader: ImageReader | None = None,
    supports_vision: VisionSupport | None = None,
) -> str | ToolResult:
    if cwd is None:
        return "error: no workspace is configured"
    fp = resolve_path(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    mime = detect_image_mime(fp)
    if mime:
        if supports_vision is not None and not supports_vision(model):
            return (
                f"Read image file [{mime}]\n"
                "[Current model does not support images. The image is omitted from this request.]"
            )
        if image_reader is None:
            return f"error: image reader is not available for {fp}"
        return image_reader.read(fp, mime)
    want = limit if limit is not None and limit > 0 else DEFAULT_READ_LINES
    start = max(0, (offset or 1) - 1) if offset is not None else 0
    total = count_lines(fp)
    window: list[str] = []
    kept = end = 0
    try:
        # Only the wanted window is decoded and kept, and no single line can be
        # unbounded: a 100 MB log costs a few hundred KB instead of 300 MB.
        with fp.open(encoding="utf-8") as fh:
            lineno = 0
            while line := fh.readline(MAX_READ_CHARS + 1):
                lineno += 1
                if lineno > start + want or kept >= MAX_READ_CHARS:
                    break
                if lineno <= start:
                    continue
                text = line.rstrip("\r\n")
                window.append(text)
                kept += len(text) + 1
                end = lineno
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    end = end or min(start, total)
    body = "\n".join(f"{i + start + 1}|{line}" for i, line in enumerate(window)) or "(empty)"
    body = _truncate_text(body, MAX_READ_CHARS)
    if start > 0 or end < total:
        hint = f"...[lines {start + 1}-{end} of {total}"
        if end < total:
            hint += f"; continue with offset={end + 1}"
        body = f"{body}\n{hint}]"
    return body


def _write_stats(fp: Path, before: str, after: str, *, kind: str = "write") -> str:
    a, b = before.splitlines(), after.splitlines()
    add = delete = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "insert":
            add += j2 - j1
        elif tag == "delete":
            delete += i2 - i1
        elif tag == "replace":
            delete += i2 - i1
            add += j2 - j1
    return f"{kind}  {fp}  +{add} -{delete}"


def write_file(
    cwd: Path | None,
    path: str,
    content: str,
    on_preview: Callable[[str], None] | None = None,
) -> str:
    if cwd is None:
        return "error: no workspace is configured"
    fp = resolve_path(cwd, path)
    try:
        before = fp.read_text(encoding="utf-8") if fp.exists() else ""
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    after = content
    if before == after and fp.is_file():
        return f"no changes: {fp}"
    if on_preview:
        on_preview(_write_stats(fp, before, after, kind="write"))
    write_atomic(fp, after)
    return f"wrote {len(after.encode('utf-8'))} bytes to {fp}"


def edit_file(
    cwd: Path | None,
    path: str,
    edits: list[dict[str, Any]],
    on_preview: Callable[[str], None] | None = None,
) -> str:
    if cwd is None:
        return "error: no workspace is configured"
    fp = resolve_path(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    try:
        before = fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    if not isinstance(edits, list) or not edits:
        return "error: edits must be a non-empty array"
    spans: list[tuple[int, int, str, int]] = []
    for i, ed in enumerate(edits):
        if not isinstance(ed, dict):
            return f"error: edits[{i}] must be an object"
        old, new = ed.get("old_string"), ed.get("new_string")
        if not isinstance(old, str) or not old:
            return f"error: edits[{i}].old_string must be a non-empty string"
        if not isinstance(new, str):
            return f"error: edits[{i}].new_string must be a string"
        n = before.count(old)
        if n != 1:
            return f"error: edits[{i}].old_string occurs {n} times in {fp}; expected exactly 1"
        start = before.index(old)
        spans.append((start, start + len(old), new, i))
    ordered = sorted(spans, key=lambda s: s[0])
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a[1] > b[0]:
            return (
                f"error: edits[{a[3]}] and edits[{b[3]}] overlap; "
                "merge nearby changes into one edit"
            )
    after = before
    for start, end, new, _ in sorted(spans, key=lambda s: s[0], reverse=True):
        after = after[:start] + new + after[end:]
    if before == after:
        return f"no changes: {fp}"
    if on_preview:
        on_preview(_write_stats(fp, before, after, kind="edit"))
    write_atomic(fp, after)
    n = len(edits)
    return f"edited {n} span{'s' if n != 1 else ''} in {fp}"
