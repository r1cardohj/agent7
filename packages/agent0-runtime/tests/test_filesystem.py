"""Focused tests for filesystem diff preview statistics."""

import unittest
from pathlib import Path

from agent0_runtime.filesystem import _write_stats


class DiffRenderingTests(unittest.TestCase):
    def test_stats_only_no_hunk_body(self) -> None:
        before = "1\n2\n3\n4\n5\n"
        after = before.replace("3", "three")
        stats = _write_stats(Path("f.txt"), before, after)
        self.assertEqual(stats, "write  f.txt  +1 -1")
        self.assertNotIn("@@", stats)
        self.assertNotIn("+three", stats)

    def test_new_file_counts_adds(self) -> None:
        stats = _write_stats(Path("new.txt"), "", "a\nb\nc\n")
        self.assertEqual(stats, "write  new.txt  +3 -0")

    def test_edit_kind_in_stats(self) -> None:
        stats = _write_stats(Path("f.txt"), "a\n", "b\n", kind="edit")
        self.assertEqual(stats, "edit  f.txt  +1 -1")


if __name__ == "__main__":
    unittest.main()
