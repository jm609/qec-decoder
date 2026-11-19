from __future__ import annotations

"""Run the three controlled MWPM conditions used by this review project."""

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circuits import build_memory_circuit  # noqa: E402
from config import make_stage_b_config, make_stage_c_config  # noqa: E402
from decoders.baseline_pymatching import (  # noqa: E402
    _decode_batch,
    _estimate_ler_per_cycle,
    _require_pymatching,
)
from noise_willowcore import apply_willowcore_noise  # noqa: E402


CSV_FIELDS = (
    "basis",
    "distance",
    "rounds",
    "condition",
    "noise_model",
    "decoder_prior",
    "p",
    "p_corr",
    "zz_factor",
    "shots",
    "failures",
    "p_L",
    "p_L_ci95_low",
    "p_L_ci95_high",
    "epsilon",
    "epsilon_ci95_low",
    "epsilon_ci95_high",
    "p_det",
    "mismatched_only_failures",
    "matched_only_failures",
    "decoder_disagreements",
    "num_detectors",
    "seed",
    "stop_reason",
)


def wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    if shots < 1 or not 0 <= failures <= shots:
        raise ValueError("require 0 <= failures <= shots and shots >= 1")
    z = 1.959963984540054
    rate = failures / shots
    denominator = 1.0 + z * z / shots
    center = (rate + z * z / (2.0 * shots)) / denominator
    spread = z * math.sqrt(rate * (1.0 - rate) / shots + z * z / (4.0 * shots * shots))
    return center - spread / denominator, center + spread / denominator


def _basis_sample_seed(seed: int, basis: str) -> int:
    return seed + (10_000 if basis == "z" else 0)


def _epsilon(rate: float, rounds: int) -> float:
    value = _estimate_ler_per_cycle(rate, rounds)
    return 0.5 if rate == 0.5 else (float("nan") if value is None else value)


def _sample_until(
    circuit: Any,
    matchings: list[Any],
    *,
    rounds: int,
    batch_shots: int,
    max_shots: int,
    target_failures: int,
    seed: int,
) -> list[dict[str, Any]]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    failures = [0] * len(matchings)
    discordant = [0, 0]
    shots = detector_ones = detector_bits = 0

    while shots < max_shots and min(failures) < target_failures:
        batch = min(batch_shots, max_shots - shots)
        detections, observables = sampler.sample(batch, separate_observables=True)
        observables = np.asarray(observables, dtype=np.uint8)
        if observables.ndim == 1:
            observables = observables.reshape(-1, 1)
        batch_failures = []
        for index, matching in enumerate(matchings):
            predictions = _decode_batch(matching, detections)
            failed = np.any(predictions != observables, axis=1)
            batch_failures.append(failed)
            failures[index] += int(failed.sum())
        if len(batch_failures) == 2:
            discordant[0] += int(np.count_nonzero(batch_failures[0] & ~batch_failures[1]))
            discordant[1] += int(np.count_nonzero(batch_failures[1] & ~batch_failures[0]))
        shots += batch
        detector_ones += int(np.count_nonzero(detections))
        detector_bits += int(detections.size)

    results = []
    for count in failures:
        rate = count / shots
        low, high = wilson_interval(count, shots)
        results.append(
            {
                "shots": shots,
                "failures": count,
                "p_L": rate,
                "p_L_ci95_low": low,
                "p_L_ci95_high": high,
                "epsilon": _epsilon(rate, rounds),
                "epsilon_ci95_low": _epsilon(low, rounds),
                "epsilon_ci95_high": _epsilon(high, rounds),
                "p_det": detector_ones / detector_bits,
                "mismatched_only_failures": discordant[0] if len(matchings) == 2 else None,
                "matched_only_failures": discordant[1] if len(matchings) == 2 else None,
                "decoder_disagreements": sum(discordant) if len(matchings) == 2 else None,
                "stop_reason": "target_failures" if min(failures) >= target_failures else "max_shots",
            }
        )
    return results


def run_basis(
    *,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    independent_p: float | None,
    zz_factor: float,
    batch_shots: int,
    max_shots: int,
    target_failures: int,
    config_seed: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    common = dict(
        distance=distance,
        rounds=rounds,
        basis=basis,
        shots=max_shots,
        local_seed=config_seed,
    )
    independent_p = p if independent_p is None else independent_p
    independent_cfg = make_stage_b_config(**common, physical_error_rate=independent_p)
    correlated_cfg = make_stage_c_config(
        **common, physical_error_rate=p, corr_seed=config_seed + 1
    )
    correlated_cfg = replace(
        correlated_cfg,
        willow=replace(
            correlated_cfg.willow,
            corr=replace(
                correlated_cfg.willow.corr,
                zz_factor=zz_factor,
                swap_factor=0.0,
                enable_swap_component=False,
            ),
        ),
    )
    correlated_cfg.validate()

    independent = apply_willowcore_noise(build_memory_circuit(independent_cfg), independent_cfg)
    correlated = apply_willowcore_noise(build_memory_circuit(correlated_cfg), correlated_cfg)
    independent_circuit = independent.noisy_circuit
    correlated_circuit = correlated.noisy_circuit

    topology_a = (independent_circuit.num_detectors, independent_circuit.num_observables)
    topology_c = (correlated_circuit.num_detectors, correlated_circuit.num_observables)
    if topology_a != topology_c:
        raise RuntimeError(f"decoder topology mismatch: independent={topology_a}, correlated={topology_c}")

    pymatching = _require_pymatching()
    independent_matching = pymatching.Matching.from_detector_error_model(
        independent_circuit.detector_error_model(decompose_errors=True)
    )
    correlated_matching = pymatching.Matching.from_detector_error_model(
        correlated_circuit.detector_error_model(decompose_errors=True)
    )

    independent_result = _sample_until(
        independent_circuit,
        [independent_matching],
        rounds=rounds,
        batch_shots=batch_shots,
        max_shots=max_shots,
        target_failures=target_failures,
        seed=sample_seed,
    )[0]
    correlated_results = _sample_until(
        correlated_circuit,
        [independent_matching, correlated_matching],
        rounds=rounds,
        batch_shots=batch_shots,
        max_shots=max_shots,
        target_failures=target_failures,
        seed=sample_seed + 1,
    )

    corr_values = list(correlated.corr_maps.p_corr_zz.values()) if correlated.corr_maps else []
    p_corr = sum(corr_values) / len(corr_values) if corr_values else 0.0
    specifications = (
        (
            "independent_matched",
            "stage_b_independent",
            "independent",
            independent_p,
            0.0,
            independent_result,
        ),
        ("correlated_mismatched", "stage_c_zz", "independent", p, p_corr, correlated_results[0]),
        ("correlated_matched", "stage_c_zz", "correlated_zz", p, p_corr, correlated_results[1]),
    )
    return [
        {
            "basis": basis,
            "distance": distance,
            "rounds": rounds,
            "condition": condition,
            "noise_model": noise_model,
            "decoder_prior": decoder_prior,
            "p": condition_p,
            "p_corr": condition_p_corr,
            "zz_factor": zz_factor,
            "num_detectors": int(independent_circuit.num_detectors),
            "seed": sample_seed if condition == "independent_matched" else sample_seed + 1,
            **result,
        }
        for condition, noise_model, decoder_prior, condition_p, condition_p_corr, result in specifications
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _self_test() -> None:
    low, high = wilson_interval(0, 100)
    assert abs(low) < 1e-15 and 0.0 < high < 0.05
    rows = run_basis(
        basis="z",
        distance=3,
        rounds=3,
        p=0.003,
        independent_p=None,
        zz_factor=0.05,
        batch_shots=32,
        max_shots=32,
        target_failures=32,
        config_seed=7,
        sample_seed=11,
    )
    assert len(rows) == 3
    assert rows[1]["shots"] == rows[2]["shots"]
    assert rows[1]["p_det"] == rows[2]["p_det"]
    assert rows[1]["decoder_disagreements"] == (
        rows[1]["mismatched_only_failures"] + rows[1]["matched_only_failures"]
    )
    assert rows[1]["p_corr"] > 0.0
    assert _basis_sample_seed(11, "z") == 10_011
    print("self-test passed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--basis", choices=("x", "z", "both"), default="both")
    parser.add_argument("--p", type=float, default=0.003)
    parser.add_argument(
        "--independent-p",
        type=float,
        help="Stage B p used for detection-probability matching; defaults to --p",
    )
    parser.add_argument("--zz-factor", type=float, default=0.05)
    parser.add_argument("--batch-shots", type=int, default=2_000)
    parser.add_argument("--max-shots", type=int, default=20_000)
    parser.add_argument("--target-failures", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    if args.out_csv is None:
        raise SystemExit("--out-csv is required unless --self-test is used")
    rounds = args.distance if args.rounds is None else args.rounds
    if min(args.distance, rounds, args.batch_shots, args.max_shots, args.target_failures) < 1:
        raise SystemExit("distance, rounds, shot limits, and target failures must be positive")
    if args.batch_shots > args.max_shots:
        raise SystemExit("--batch-shots must not exceed --max-shots")
    if args.zz_factor < 0.0:
        raise SystemExit("--zz-factor must be nonnegative")

    bases = ("x", "z") if args.basis == "both" else (args.basis,)
    rows = []
    for basis in bases:
        rows.extend(
            run_basis(
                basis=basis,
                distance=args.distance,
                rounds=rounds,
                p=args.p,
                independent_p=args.independent_p,
                zz_factor=args.zz_factor,
                batch_shots=args.batch_shots,
                max_shots=args.max_shots,
                target_failures=args.target_failures,
                config_seed=args.seed,
                sample_seed=_basis_sample_seed(args.seed, basis),
            )
        )
    write_csv(args.out_csv, rows, overwrite=args.overwrite)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
