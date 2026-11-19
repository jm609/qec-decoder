import math
import json
from pathlib import Path
import tempfile
import unittest
from itertools import zip_longest

import numpy as np
import stim

from config import make_stage_c_config
from logical_frame import infer_rotated_data_qubits
from noise_willowcore import bias_willowcore_data_paulis, build_willowcore_memory_circuit
from sample_dataset import save_family_dataset


def operations(circuit):
    for op in circuit:
        if op.name in {"DEPOLARIZE1", "PAULI_CHANNEL_1"}:
            for target in op.targets_copy():
                yield op.name, [target], op.gate_args_copy()
        else:
            yield op.name, op.targets_copy(), op.gate_args_copy()


class WillowCoreDataBiasTest(unittest.TestCase):
    def test_dataset_default_replay_and_bias_metadata(self):
        settings = dict(family="stage_c_corr", distance=3, rounds=3, basis="z",
                        variant="stim_rotated", target_mode="bell_pair_z_readout",
                        physical_error_rate=0.0015, shots=32, seed=1885000,
                        overwrite=False, local_seed=788500)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = save_family_dataset(out_root=root / "original", **settings)
            control = save_family_dataset(out_root=root / "control", data_pauli_z_bias=0.5, **settings)
            biased = save_family_dataset(out_root=root / "biased", data_pauli_z_bias=20.0, **settings)
            for name in ("circuit.stim", "detector_error_model.dem"):
                self.assertEqual((original / name).read_bytes(), (control / name).read_bytes())
            with np.load(original / "samples.npz") as a, np.load(control / "samples.npz") as b:
                self.assertEqual(a.files, b.files)
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])
            control_circuit = stim.Circuit.from_file(str(control / "circuit.stim"))
            biased_circuit = stim.Circuit.from_file(str(biased / "circuit.stim"))
            data = set(infer_rotated_data_qubits(control_circuit))
            original_rates = [args[0] for name, targets, args in operations(control_circuit)
                              if name == "DEPOLARIZE1" and targets[0].value in data]
            biased_rates = [math.fsum(args) for name, _, args in operations(biased_circuit)
                            if name == "PAULI_CHANNEL_1"]
            # Stim text serialization retains six significant digits.
            np.testing.assert_allclose(biased_rates, original_rates, rtol=1e-5, atol=0)
            for directory in (control, biased):
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["data_pauli_z_bias"], config["data_pauli_z_bias"])
                self.assertEqual(metadata["data_pauli_z_bias"]["eta"], 20.0 if directory == biased else 0.5)
                self.assertTrue(metadata["dem"]["kwargs"]["approximate_disjoint_errors"])
                self.assertEqual(metadata["circuit"]["num_observables"], 2)

    def test_bias_redistributes_only_data_qubit_depolarizing_faults(self):
        cfg = make_stage_c_config(distance=7, rounds=7, basis="z", local_seed=788500)
        control = build_willowcore_memory_circuit(cfg)
        data_qubits = set(infer_rotated_data_qubits(control))
        self.assertEqual(len(data_qubits), 49)
        self.assertEqual(bias_willowcore_data_paulis(control, distance=7, eta=0.5), control)

        for eta in (5.0, 20.0):
            with self.subTest(eta=eta):
                biased = bias_willowcore_data_paulis(control, distance=7, eta=eta)
                self.assertEqual(biased.without_noise(), control.without_noise())
                modified_sites = 0
                for original, modified in zip_longest(operations(control), operations(biased)):
                    if original[0] != "DEPOLARIZE1" or original[1][0].value not in data_qubits:
                        self.assertEqual(original, modified)
                        continue
                    self.assertEqual(modified[:2], ("PAULI_CHANNEL_1", original[1]))
                    px, py, pz = modified[2]
                    self.assertEqual(px, py)
                    self.assertTrue(math.isclose(math.fsum((px, py, pz)), original[2][0], rel_tol=1e-12))
                    self.assertAlmostEqual(pz / (px + py), eta)
                    modified_sites += 1
                self.assertGreater(modified_sites, 0)

        for eta in (float("nan"), float("inf"), 0.1):
            with self.subTest(invalid_eta=eta), self.assertRaises(ValueError):
                bias_willowcore_data_paulis(control, distance=7, eta=eta)


if __name__ == "__main__":
    unittest.main()
