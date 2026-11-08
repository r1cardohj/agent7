"""`python -m agent7` — same entry point as the installed `agent7` script."""

from __future__ import annotations

from .cli import PROG, main

if __name__ == "__main__":
    main(prog_name=PROG)
