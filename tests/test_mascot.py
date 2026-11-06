import io
import unittest

from rich.console import Console

from agent7.mascot import COLORS, PIXEL_ROWS, mascot_text, print_mascot


class MascotTests(unittest.TestCase):
    def test_pixel_art_is_rectangular_and_uses_the_palette(self) -> None:
        self.assertEqual(len(PIXEL_ROWS), 20)
        self.assertTrue(all(len(row) == 20 for row in PIXEL_ROWS))
        self.assertTrue(set("".join(PIXEL_ROWS)) <= set(COLORS) | {"."})

    def test_terminal_art_packs_two_vertical_pixels_per_cell(self) -> None:
        art = mascot_text()
        lines = art.plain.splitlines()
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(len(line) == 22 for line in lines))
        self.assertIn("▀", art.plain)
        self.assertIn("▄", art.plain)
        self.assertTrue(art.spans)

    def test_uncolored_terminal_keeps_the_pixel_shape(self) -> None:
        output = io.StringIO()
        print_mascot(Console(file=output, force_terminal=True, color_system=None))
        self.assertIn("▀", output.getvalue())
        self.assertNotIn("\x1b[", output.getvalue())

    def test_welcome_includes_name_and_tagline(self) -> None:
        output = io.StringIO()
        print_mascot(Console(file=output, force_terminal=True, color_system="truecolor"))
        self.assertIn("agent7", output.getvalue())
        self.assertIn("Seven", output.getvalue())
        self.assertIn("four tools, one loop", output.getvalue())


if __name__ == "__main__":
    unittest.main()
