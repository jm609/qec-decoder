from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CircuitConfig
from circuits import build_memory_circuit
from logical_frame import audit_ideal_logical_frame_support


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the current single-basis memory scaffold supports "
            "a same-shot logical X/Z frame."
        )
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=["x", "z", "both"], default="both")
    parser.add_argument("--variant", choices=["stim_rotated", "xzzx"], default="stim_rotated")
    parser.add_argument("--sample-shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def _audit_one_basis(
    *,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
    sample_shots: int,
    seed: int,
) -> dict[str, Any]:
    cfg = CircuitConfig(
        distance=distance,
        rounds=rounds,
        basis=basis,
        variant=variant,
    )
    circuit = build_memory_circuit(cfg)
    report = audit_ideal_logical_frame_support(
        circuit=circuit,
        basis=basis,
        variant=variant,
        sample_shots=sample_shots,
        seed=seed,
    )
    return report.to_dict()


def main() -> None:
    args = _parse_args()
    bases = ["x", "z"] if args.basis == "both" else [args.basis]

    reports = {
        basis: _audit_one_basis(
            distance=args.distance,
            rounds=args.rounds,
            basis=basis,
            variant=args.variant,
            sample_shots=args.sample_shots,
            seed=args.seed,
        )
        for basis in bases
    }

    payload: dict[str, Any] = {
        "schema_version": "logical_frame_support_bundle.v1",
        "distance": args.distance,
        "rounds": args.rounds,
        "variant": args.variant,
        "sample_shots": args.sample_shots,
        "reports": reports,
    }

    if args.out_json is not None:
        _write_json(Path(args.out_json), payload)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
