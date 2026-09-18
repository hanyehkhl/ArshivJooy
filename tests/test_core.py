"""تست‌های واحدِ بخش‌هایی که به مدل یا دیتابیس نیاز ندارند.

اجرا::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SUPPORTED_EXTENSIONS  # noqa: E402
from src.extractors import chunk_text, iter_supported_files, normalize_text  # noqa: E402
from src.search import SOURCE_CLIP_IMAGE, SOURCE_TEXT, SearchEngine  # noqa: E402


class TestNormalizeText(unittest.TestCase):
    def test_arabic_characters_are_normalized(self) -> None:
        self.assertEqual(normalize_text("كتاب عربي"), "کتاب عربی")

    def test_digits_are_normalized(self) -> None:
        self.assertEqual(normalize_text("سال ۱۴۰۲"), "سال 1402")

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(normalize_text("  a   b \n\n\n c "), "a b\n\nc")


class TestChunkText(unittest.TestCase):
    def test_short_text_is_single_chunk(self) -> None:
        self.assertEqual(chunk_text("سلام دنیا", 100, 20, 10), ["سلام دنیا"])

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_text("   ", 100, 20, 10), [])

    def test_long_text_is_split_and_covers_content(self) -> None:
        text = " ".join(f"کلمه{i}" for i in range(500))
        chunks = chunk_text(text, 200, 40, 100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        self.assertIn("کلمه0", chunks[0])
        self.assertIn("کلمه499", chunks[-1])

    def test_max_chunks_is_respected(self) -> None:
        text = "x " * 10_000
        self.assertEqual(len(chunk_text(text, 50, 10, 7)), 7)

    def test_overlap_larger_than_chunk_still_terminates(self) -> None:
        text = "y " * 2_000
        chunks = chunk_text(text, 60, 500, 50)
        self.assertGreater(len(chunks), 1)


class TestIterSupportedFiles(unittest.TestCase):
    def test_only_supported_extensions_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            wanted = [root / "a.txt", root / "sub" / "b.PDF", root / "sub" / "c.png"]
            unwanted = [root / "d.mp3", root / "sub" / "e.zip"]
            for path in wanted + unwanted:
                path.write_bytes(b"x")

            found = iter_supported_files(root, SUPPORTED_EXTENSIONS)
            self.assertEqual({p.name for p in found}, {"a.txt", "b.PDF", "c.png"})

    def test_missing_folder_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            iter_supported_files(Path("/definitely/not/here"), SUPPORTED_EXTENSIONS)


def _hit(path: str, distance: float, item_type: str = "document") -> dict:
    return {
        "id": path,
        "document": f"متن {path}",
        "metadata": {"path": path, "filename": Path(path).name, "type": item_type},
        "distance": distance,
    }


class TestFusion(unittest.TestCase):
    def test_item_matched_by_two_sources_ranks_first(self) -> None:
        text_hits = [_hit("/a.txt", 0.10), _hit("/b.txt", 0.15)]
        image_hits = [_hit("/b.txt", 0.20, "image")]

        results = SearchEngine._fuse(
            [(SOURCE_TEXT, 1.0, text_hits), (SOURCE_CLIP_IMAGE, 1.0, image_hits)],
            n_results=5,
        )
        self.assertEqual(results[0].path, "/b.txt")
        self.assertEqual(sorted(results[0].matched_by), [SOURCE_CLIP_IMAGE, SOURCE_TEXT])
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_duplicate_chunks_of_same_file_collapse(self) -> None:
        hits = [_hit("/a.txt", 0.1), _hit("/a.txt", 0.2), _hit("/b.txt", 0.3)]
        results = SearchEngine._fuse([(SOURCE_TEXT, 1.0, hits)], n_results=5)
        self.assertEqual([r.path for r in results], ["/a.txt", "/b.txt"])

    def test_empty_input(self) -> None:
        self.assertEqual(SearchEngine._fuse([], n_results=5), [])


if __name__ == "__main__":
    unittest.main()
