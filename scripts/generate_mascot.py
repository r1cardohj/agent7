"""Regenerate the README mascot using only the Python standard library."""

from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent7.mascot import COLORS, PIXEL_ROWS  # noqa: E402

SIZE, SCALE, MARGIN = 320, 15, 10
BACKGROUND = "#151222"


def make_svg() -> str:
    paths = []
    for pixel, color in COLORS.items():
        runs = []
        for y, row in enumerate(PIXEL_ROWS):
            x = 0
            while x < len(row):
                if row[x] != pixel:
                    x += 1
                    continue
                end = x + 1
                while end < len(row) and row[end] == pixel:
                    end += 1
                runs.append(f"M{x} {y}h{end - x}v1H{x}z")
                x = end
        paths.append(f'    <path fill="{color}" d="{"".join(runs)}"/>')
    art = "\n".join(paths)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}" role="img" aria-labelledby="title desc">
  <title id="title">Seven, the agent7 pixel mascot</title>
  <desc id="desc">A lavender pixel robot with mint-green eyes, a seven-shaped antenna, and a glowing number seven on its chest.</desc>
  <rect width="{SIZE}" height="{SIZE}" fill="{BACKGROUND}"/>
  <g shape-rendering="crispEdges" transform="translate({MARGIN} {MARGIN}) scale({SCALE})">
{art}
  </g>
</svg>
'''


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def make_png() -> bytes:
    def rgba(value: str) -> bytes:
        return bytes.fromhex(value.removeprefix("#")) + b"\xff"

    scanlines = bytearray()
    for row in PIXEL_ROWS:
        pixels = bytearray(rgba(BACKGROUND) * MARGIN)
        for pixel in row:
            pixels.extend(rgba(COLORS.get(pixel, BACKGROUND)) * SCALE)
        pixels.extend(rgba(BACKGROUND) * MARGIN)
        scanlines.extend((b"\x00" + pixels) * SCALE)
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(bytes(scanlines), 9)
    ) + chunk(b"IEND", b"")


def main() -> None:
    assert len(PIXEL_ROWS) * SCALE + 2 * MARGIN == SIZE
    assert all(len(row) == len(PIXEL_ROWS[0]) for row in PIXEL_ROWS)
    (ROOT / "assets" / "mascot.svg").write_text(make_svg(), encoding="utf-8")
    (ROOT / "assets" / "mascot.png").write_bytes(make_png())


if __name__ == "__main__":
    main()
