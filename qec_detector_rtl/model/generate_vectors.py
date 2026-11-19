from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import stim

from circuits import build_memory_circuit
from config import CircuitConfig, SI1000Config
from noise_si1000 import apply_si1000_noise


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Detector:
    index: int
    measurement_indices: tuple[int, ...]
    coordinates: tuple[float, ...]
    reference_parity: int

    @property
    def required_count(self) -> int:
        return max(self.measurement_indices, default=-1) + 1


def extract_maps(circuit: stim.Circuit) -> tuple[list[dict], tuple[Detector, ...]]:
    """Extract single-qubit measurement records and deterministic rec-only detectors."""
    circuit.detector_error_model(allow_gauge_detectors=False)
    reference = circuit.reference_sample()
    measurements = []
    detectors = []
    group = 0
    for op in circuit.flattened():
        if op.num_measurements:
            if op.name not in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
                raise ValueError(f"Unsupported measurement instruction: {op.name}")
            for target in op.targets_copy():
                if not target.is_qubit_target:
                    raise ValueError(f"Unsupported measurement target: {target}")
                measurements.append({
                    "index": len(measurements), "qubit": target.value,
                    "gate": op.name, "measurement_group": group,
                })
            group += 1
        elif op.name == "DETECTOR":
            indices = []
            for target in op.targets_copy():
                if not target.is_measurement_record_target:
                    raise ValueError(f"Unsupported detector target: {target}")
                index = len(measurements) + target.value
                if not 0 <= index < len(measurements):
                    raise ValueError(f"Invalid measurement reference: {target}")
                indices.append(index)
            support = tuple(sorted(i for i, count in Counter(indices).items() if count % 2))
            parity = sum(int(reference[i]) for i in support) % 2
            detectors.append(Detector(
                len(detectors), support, tuple(op.gate_args_copy()), parity,
            ))
    if len(measurements) != circuit.num_measurements or len(detectors) != circuit.num_detectors:
        raise ValueError("Extracted circuit dimensions do not match Stim")
    return measurements, tuple(detectors)


def convert_measurements(
    measurements: np.ndarray, detectors: tuple[Detector, ...], measurement_count: int,
) -> np.ndarray:
    values = np.asarray(measurements)
    if values.ndim != 2 or values.shape[1] != measurement_count:
        raise ValueError(f"Expected a (shots, {measurement_count}) measurement array")
    if values.dtype.kind not in "biu" or not np.all((values == 0) | (values == 1)):
        raise ValueError("Measurement values must be integer or boolean 0/1")
    result = np.zeros((len(values), len(detectors)), dtype=np.bool_)
    for detector in detectors:
        result[:, detector.index] = np.logical_xor.reduce(
            values[:, detector.measurement_indices], axis=1, initial=False,
        ) ^ bool(detector.reference_parity)
    return result


def write_records(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as output:
        for row in values:
            output.write("".join("1" if bit else "0" for bit in row) + "\n")


def read_records(path: Path, width: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    if any(len(line) != width or set(line) - {"0", "1"} for line in lines):
        raise ValueError(f"Invalid {width}-bit records in {path}")
    return np.array([[char == "1" for char in line] for line in lines], dtype=np.bool_).reshape(-1, width)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_vectors(out_root: Path) -> Path:
    cfg = CircuitConfig(distance=3, rounds=3, basis="z", variant="stim_rotated")
    noise = SI1000Config(physical_error_rate=0.001)
    ideal = build_memory_circuit(cfg)
    noisy = apply_si1000_noise(ideal, noise)
    measurement_map, detectors = extract_maps(ideal)
    if (ideal.num_measurements, ideal.num_detectors, ideal.num_observables) != (33, 24, 1):
        raise ValueError("The fixed d3/r3/z circuit dimensions changed")
    if extract_maps(noisy) != (measurement_map, detectors):
        raise ValueError("Noise injection changed measurement or detector semantics")
    if noisy.without_noise().flattened() != ideal.flattened():
        raise ValueError("Noise injection changed the ideal circuit")
    reference = ideal.reference_sample()
    single_flip = np.tile(reference, (34, 1))
    single_flip[1:] ^= np.eye(33, dtype=np.bool_)
    cases = [
        ("noiseless", ideal, 1101801, ideal.compile_sampler(seed=1101801).sample(shots=256)),
        ("single_flip", ideal, None, single_flip),
        ("random_bits", ideal, 1101802,
         np.random.default_rng(1101802).integers(0, 2, (1024, 33), dtype=np.uint8).astype(np.bool_)),
        ("si1000", noisy, 1101803, noisy.compile_sampler(seed=1101803).sample(shots=4096)),
    ]
    out_root.mkdir(parents=True, exist_ok=False)
    vectors = out_root / "vectors"
    vectors.mkdir()
    (vectors / "ideal.stim").write_text(str(ideal) + "\n", encoding="ascii")
    (vectors / "si1000.stim").write_text(str(noisy) + "\n", encoding="ascii")
    write_records(vectors / "reference.01", reference[None, :])
    write_json(vectors / "measurement_map.json", measurement_map)
    write_json(vectors / "detector_map.json", [
        {**asdict(detector), "required_count": detector.required_count} for detector in detectors
    ])
    summaries = []
    for name, circuit, seed, measurements in cases:
        expected = circuit.compile_m2d_converter().convert(
            measurements=measurements, append_observables=False,
        )
        direct = convert_measurements(measurements, detectors, 33)
        mismatches = int(np.count_nonzero(direct != expected))
        if mismatches or (name == "noiseless" and expected.any()):
            raise ValueError(f"{name}: {mismatches} mismatches or nonzero noiseless detectors")
        measurement_file = f"{name}.measurements.01"
        detector_file = f"{name}.expected_detectors.01"
        write_records(vectors / measurement_file, measurements)
        write_records(vectors / detector_file, expected)
        if not np.array_equal(read_records(vectors / measurement_file, 33), measurements):
            raise ValueError(f"{name}: measurement file round trip failed")
        if not np.array_equal(read_records(vectors / detector_file, 24), expected):
            raise ValueError(f"{name}: detector file round trip failed")
        summaries.append({
            "name": name, "shots": len(measurements), "seed": seed,
            "measurements": measurement_file, "expected_detectors": detector_file,
            "python_stim_mismatches": mismatches,
            "detection_events": int(expected.sum()),
        })
    sources = [
        "config.py", "circuits.py", "logical_frame.py", "logical_targets.py",
        "noise_si1000.py", "requirements.txt", "qec_detector_rtl/model/generate_vectors.py",
    ]
    manifest = {
        "schema_version": 1, "status": "complete", "run_id": out_root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "circuit_config": asdict(cfg),
        "noise_config": asdict(noise), "noise_resolved_rates": noise.resolved_rates(),
        "python": sys.version, "numpy": np.__version__, "stim": stim.__version__,
        "measurement_count": 33, "detector_count": 24, "observable_count": 1,
        "bit_order": "leftmost character is index 0; one shot per line",
        "append_observables": False, "protocol_seed": 1101804, "cases": summaries,
        "source_sha256": {name: digest(ROOT / name) for name in sources},
        "files_sha256": {p.name: digest(p) for p in sorted(vectors.iterdir())},
    }
    path = vectors / "manifest.json"
    write_json(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze d3/r3/z measurement-to-detector vectors")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    print(generate_vectors(args.out_root))


if __name__ == "__main__":
    main()
