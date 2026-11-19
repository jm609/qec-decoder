import tempfile
from pathlib import Path
import unittest
import zipfile

from tools.prepare_overleaf_package import D7_FIGURE_DIR, D7_FIGURES, build_package


class CurrentManuscriptPackageTest(unittest.TestCase):
    def test_current_manuscripts_package_their_exact_sources_and_five_figures(self):
        for name in ("main_d7_en.tex", "main_d7_ko.tex"):
            with self.subTest(manuscript=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "package.zip"
                result = build_package(root / "package", archive, Path(name))
                self.assertTrue(result["pass"])
                self.assertEqual(result["missing_required"], [])
                self.assertEqual(result["figure_priority"], ["pdf"])
                with zipfile.ZipFile(archive) as package:
                    self.assertIsNone(package.testzip())
                    self.assertEqual(package.read(name), Path(name).read_bytes())
                    self.assertNotIn("main.bib", package.namelist())
                    self.assertEqual(len([p for p in package.namelist() if p.endswith(".pdf")]), 5)
                    for figure in D7_FIGURES:
                        path = D7_FIGURE_DIR / figure
                        self.assertEqual(package.read(path.as_posix()), path.read_bytes())

    def test_current_package_rejects_existing_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "package"
            archive = root / "package.zip"
            build_package(output, archive, Path("main_d7_en.tex"))
            original = archive.read_bytes()
            with self.assertRaises(FileExistsError):
                build_package(output, archive, Path("main_d7_en.tex"))
            self.assertEqual(archive.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
