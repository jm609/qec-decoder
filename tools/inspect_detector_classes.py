from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the project root that contains the core source files.
    """
    required = {
        "config.py",
        "circuits.py",
        "sample_dataset.py",
        "noise_si1000.py",
        "noise_willowcore.py",
    }

    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if all((candidate / name).exists() for name in required):
            return candidate

    raise RuntimeError(
        "Could not locate repo root from this script location. "
        "Expected a parent directory containing config.py, circuits.py, "
        "sample_dataset.py, noise_si1000.py, and noise_willowcore.py."
    )


REPO_ROOT = _find_repo_root(Path(__file__).parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CircuitConfig
from circuits import (
    build_memory_circuit,
    export_detector_semantics,
    get_detector_coordinate_array,
)


def inspect_case(distance: int, rounds: int, basis: str, variant: str) -> None:
    cfg = CircuitConfig(
        distance=distance,
        rounds=rounds,
        basis=basis,
        variant=variant,
    )

    circuit = build_memory_circuit(cfg)
    coords = get_detector_coordinate_array(circuit, coord_dim=3)
    sem = export_detector_semantics(cfg, circuit)

    cls = sem["detector_checkerboard_class"]
    t = sem["detector_time_index"]
    b = sem["detector_boundary_flag"]
    f = sem["detector_final_round_flag"]
    dtype = sem["detector_type"]

    print("=" * 100)
    print(
        f"distance={distance} rounds={rounds} basis={basis} variant={variant} "
        f"num_detectors={len(coords)}"
    )
    print("repo_root =", REPO_ROOT)
    print("class_counts =", dict(Counter(int(v) for v in cls.tolist())))
    print("time_counts =", dict(Counter(int(v) for v in t.tolist())))
    print("boundary_counts =", dict(Counter(int(v) for v in b.tolist())))
    print("final_round_counts =", dict(Counter(int(v) for v in f.tolist())))
    print("detector_type_counts =", dict(Counter(int(v) for v in dtype.tolist())))
    print()

    for i in range(len(coords)):
        print(
            i,
            coords[i].tolist(),
            "class=", int(cls[i]),
            "time=", int(t[i]),
            "boundary=", int(b[i]),
            "final=", int(f[i]),
            "type=", int(dtype[i]),
        )


def compare_basis(distance: int, rounds: int, variant: str) -> None:
    cfg_z = CircuitConfig(
        distance=distance,
        rounds=rounds,
        basis="z",
        variant=variant,
    )
    cfg_x = CircuitConfig(
        distance=distance,
        rounds=rounds,
        basis="x",
        variant=variant,
    )

    circ_z = build_memory_circuit(cfg_z)
    circ_x = build_memory_circuit(cfg_x)

    coords_z = get_detector_coordinate_array(circ_z, coord_dim=3)
    coords_x = get_detector_coordinate_array(circ_x, coord_dim=3)

    sem_z = export_detector_semantics(cfg_z, circ_z)
    sem_x = export_detector_semantics(cfg_x, circ_x)

    same_shape = coords_z.shape == coords_x.shape
    same_coords = np.array_equal(coords_z, coords_x)
    same_checkerboard = np.array_equal(
        sem_z["detector_checkerboard_class"],
        sem_x["detector_checkerboard_class"],
    )
    same_time = np.array_equal(
        sem_z["detector_time_index"],
        sem_x["detector_time_index"],
    )
    same_boundary = np.array_equal(
        sem_z["detector_boundary_flag"],
        sem_x["detector_boundary_flag"],
    )
    same_final = np.array_equal(
        sem_z["detector_final_round_flag"],
        sem_x["detector_final_round_flag"],
    )

    print("=" * 100)
    print(
        f"basis comparison for distance={distance} rounds={rounds} variant={variant}"
    )
    print("same_shape =", same_shape)
    print("same_coords =", same_coords)
    print("same_checkerboard_class =", same_checkerboard)
    print("same_time_index =", same_time)
    print("same_boundary_flag =", same_boundary)
    print("same_final_round_flag =", same_final)
    print()


def main() -> None:
    distance = 3
    rounds = 3
    variant = "stim_rotated"

    inspect_case(distance=distance, rounds=rounds, basis="z", variant=variant)
    print()
    inspect_case(distance=distance, rounds=rounds, basis="x", variant=variant)
    print()
    compare_basis(distance=distance, rounds=rounds, variant=variant)


if __name__ == "__main__":
    main()