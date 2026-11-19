import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import stim

from circuits import build_memory_circuit
from config import CircuitConfig
from qec_detector_rtl.model.generate_vectors import (
    convert_measurements, digest, extract_maps, generate_vectors, read_records,
)


class MeasurementToDetectionTest(unittest.TestCase):
    def test_boundaries_repeat_coordinates_and_single_bit_connections(self):
        circuit = build_memory_circuit(CircuitConfig())
        measurements, detectors = extract_maps(circuit)
        self.assertEqual(len(measurements), 33)
        self.assertEqual(len(detectors), 24)
        self.assertEqual([row["measurement_group"] for row in measurements],
                         [0] * 8 + [1] * 8 + [2] * 8 + [3] * 9)
        self.assertEqual(detectors[0].measurement_indices, (4,))
        self.assertEqual(detectors[4].measurement_indices, (0, 8))
        self.assertEqual(detectors[20].measurement_indices, (20, 27, 30))
        self.assertEqual(detectors[22].measurement_indices, (22, 28, 29, 31, 32))
        self.assertEqual(detectors[23].required_count, 30)
        self.assertEqual([d.coordinates[-1] for d in detectors],
                         [0] * 4 + [1] * 8 + [2] * 8 + [3] * 4)
        records = np.vstack((circuit.reference_sample(), np.eye(33, dtype=np.bool_)))
        expected = circuit.compile_m2d_converter().convert(measurements=records, append_observables=False)
        np.testing.assert_array_equal(convert_measurements(records, detectors, 33), expected)

    def test_nonzero_reference_and_duplicate_cancellation(self):
        circuit = stim.Circuit("X 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-1]")
        _, detectors = extract_maps(circuit)
        self.assertEqual(detectors[0].reference_parity, 1)
        self.assertEqual(detectors[1].measurement_indices, ())
        records = np.array([[0], [1]], dtype=np.bool_)
        np.testing.assert_array_equal(convert_measurements(records, detectors, 1),
                                      [[1, 0], [0, 0]])

    def test_reject_unsupported_or_nondeterministic_measurements(self):
        for text in ["R 0\nH 0\nM 0\nDETECTOR rec[-1]", "R 0\nMPP Z0\nDETECTOR rec[-1]"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_maps(stim.Circuit(text))

    def test_reject_invalid_records(self):
        _, detectors = extract_maps(stim.Circuit("M 0\nDETECTOR rec[-1]"))
        for records in [np.array([0]), np.array([[0, 1]]), np.array([[2]]), np.array([[0.5]])]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                convert_measurements(records, detectors, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.01"
            for text in ["00\n", "x\n", "\n", " 0\n"]:
                path.write_text(text, encoding="ascii")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_records(path, 1)

    def test_frozen_vectors_hashes_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            path = generate_vectors(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([case["shots"] for case in manifest["cases"]], [256, 34, 1024, 4096])
            self.assertTrue(all(case["python_stim_mismatches"] == 0 for case in manifest["cases"]))
            for name, checksum in manifest["files_sha256"].items():
                self.assertEqual(digest(path.parent / name), checksum)
            for case in manifest["cases"]:
                self.assertEqual(read_records(path.parent / case["measurements"], 33).shape,
                                 (case["shots"], 33))
                self.assertEqual(read_records(path.parent / case["expected_detectors"], 24).shape,
                                 (case["shots"], 24))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                generate_vectors(root)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
