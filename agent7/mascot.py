"""Seven, agent7's tiny pixel-art coding companion."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

# One character is one pixel. Keep this as the source of truth for the terminal,
# SVG, and PNG versions so Seven looks the same everywhere.
PIXEL_ROWS = (
    "....................",
    ".........OAAO.......",
    "..........OAO.......",
    "..........OA........",
    "...OOOOOOOOOOOOOO...",
    "..OLLLLLLLLLLLLLO...",
    "..OLLOOOOOOOOOOPPO..",
    "OSOLLODCDDDDDCDPOSOO",
    "OSOLLODWCDDDDWCDPOSO",
    "OSOLLODCCCDDCCCDPOSO",
    ".OOLLODDDDDDDDDDPOO.",
    "..OLLOOOOOOOOOOOPPO.",
    "..OPPPPPPPPPPPPPPPO.",
    "....OOOOOOOOOOOO....",
    ".....OOOOOOOOOO.....",
    "...OOOPDAAAAOSOOO...",
    ".OOOOPDDDDAOSOOOOO..",
    ".OO..OPDDAADOSO..LO.",
    ".....OOOOOOOOOO.....",
    ".....OPPO..OPPO.....",
)

COLORS = {
    "O": "#211c36",  # outline
    "P": "#a99be5",  # lavender shell
    "L": "#d4caff",  # highlights
    "S": "#796bb5",  # shadows
    "D": "#203b45",  # face and chest screens
    "C": "#8cf5c5",  # cursor-green eyes
    "W": "#f5f7ff",  # eye glints
    "A": "#ffb454",  # antenna and lucky number seven
}


def mascot_text(*, color: bool = True) -> Text:
    """Pack two vertical pixels into each terminal cell with half blocks."""
    art = Text(no_wrap=True)
    for row, (top, bottom) in enumerate(zip(PIXEL_ROWS[::2], PIXEL_ROWS[1::2])):
        if row:
            art.append("\n")
        art.append("  ")
        for upper, lower in zip(top, bottom):
            if upper == "." and lower == ".":
                art.append(" ")
            elif upper == ".":
                art.append("▄", style=COLORS[lower] if color else None)
            elif lower == "." or upper == lower:
                art.append("▀", style=COLORS[upper] if color else None)
            else:
                style = f"{COLORS[upper]} on {COLORS[lower]}" if color else None
                art.append("▀", style=style)
    return art


def print_mascot(console: Console) -> None:
    """Print the REPL welcome; one-shot and redirected output never call this."""
    console.print()
    console.print(mascot_text(color=console.color_system is not None))
    console.print("  [bold #c5b8f2]agent7[/bold #c5b8f2]  [dim]four tools, one loop[/dim]")
    console.print("  [dim]Seven · your tiny coding companion[/dim]")
    console.print()
