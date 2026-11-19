import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from qec_detector_rtl.scripts.prepare_server_bundle import prepare_bundle
from qec_detector_rtl.scripts.run_sim import digest, run_sim, validate_vectors


VECTORS = Path(__file__).resolve().parents[1] / "artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors"


@unittest.skipUnless(VECTORS.is_dir(), "Frozen P1 vectors are not present")
class DetectorRTLTest(unittest.TestCase):
    def test_reject_corrupted_input_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "vectors"
            shutil.copytree(VECTORS, copied)
            (copied / "single_flip.measurements.01").write_text("0\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_vectors(copied)

    def test_server_bundle_is_portable_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "server_bundle"
            archive = prepare_bundle(VECTORS, root)
            validate_vectors(root / "vectors")
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            for name, checksum in manifest["files_sha256"].items():
                self.assertEqual(digest(root / name), checksum)
            with zipfile.ZipFile(archive) as zipped:
                self.assertTrue(all(name.startswith("server_bundle/") and "\\" not in name
                                    for name in zipped.namelist()))
                self.assertEqual(zipped.read("server_bundle/vectors/manifest.json"),
                                 (VECTORS / "manifest.json").read_bytes())
                self.assertIn("server_bundle/scripts/run_vcs.sh", zipped.namelist())
                self.assertNotIn(b"\r", zipped.read("server_bundle/scripts/run_vcs.sh"))
            with self.assertRaises(FileExistsError):
                prepare_bundle(VECTORS, root)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "Icarus is not present")
    def test_rtl_matches_frozen_oracle_and_rejects_wrong_expected_bit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            report = run_sim(VECTORS, root, "iverilog")
            self.assertEqual(report["status"], "pass")
            self.assertEqual(sum(case["shots"] for case in report["cases"]), 5410)
            self.assertEqual(sum(case["detector_bits"] for case in report["cases"]), 129840)
            self.assertNotEqual(report["negative_check_returncode"], 0)
            before = (root / "verification.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_sim(VECTORS, root, "iverilog")
            self.assertEqual((root / "verification.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
