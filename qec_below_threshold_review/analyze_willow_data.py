from __future__ import annotations

"""Evaluate observable predictions from the public Willow memory dataset."""

import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np

from run_experiment import wilson_interval


def _single_observable_from_bytes(data: bytes, shots: int, source: str) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    if packed.size != shots:
        raise ValueError(f"{source}: expected {shots} bytes, got {packed.size}")
    if np.any(packed & 0xFE):
        raise ValueError(f"{source}: non-zero padding bits")
    return packed & 1


def _load_single_observable_b8(path: Path, shots: int) -> np.ndarray:
    return _single_observable_from_bytes(path.read_bytes(), shots, str(path))


def _logical_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {predicted.shape}")
    shots = int(actual.size)
    failures = int(np.count_nonzero(actual ^ predicted))
    low, high = wilson_interval(failures, shots)
    return {
        "shots": shots,
        "failures": failures,
        "logical_failure_rate": failures / shots,
        "ci95_low": low,
        "ci95_high": high,
    }


def _parse_prediction(value: str) -> tuple[str, str]:
    label, separator, relative = value.partition("=")
    if not separator or not label or not relative:
        raise argparse.ArgumentTypeError("prediction must be LABEL=PATH")
    return label, relative


def _fit_epsilon(rounds: np.ndarray, rates: np.ndarray) -> tuple[float, float, float]:
    if rounds.size < 2 or rounds.shape != rates.shape:
        raise ValueError("at least two equally shaped round/rate values are required")
    if np.any((rates < 0) | (rates >= 0.5)):
        raise ValueError("logical failure rates must satisfy 0 <= p_L < 0.5")
    coefficients, covariance = np.polyfit(
        rounds, np.log1p(-2.0 * rates), 1, cov=True
    )
    slope, intercept = coefficients
    epsilon = -float(np.expm1(slope)) / 2.0
    epsilon_std = 0.5 * float(np.exp(slope)) * float(np.sqrt(covariance[0, 0]))
    return epsilon, float(intercept), epsilon_std


def _decode_detection_b8(
    data: bytes,
    *,
    shots: int,
    num_detectors: int,
    matching: object,
    batch_shots: int | None = None,
) -> np.ndarray:
    bytes_per_shot = (num_detectors + 7) // 8
    packed = np.frombuffer(data, dtype=np.uint8)
    if packed.size != shots * bytes_per_shot:
        raise ValueError("detection_events.b8 size does not match metadata and DEM")
    packed = packed.reshape(shots, bytes_per_shot)
    if batch_shots is None:
        batch_shots = max(1_000, min(shots, 60_000_000 // max(1, num_detectors)))
    predictions = np.empty(shots, dtype=np.uint8)
    for start in range(0, shots, batch_shots):
        stop = min(shots, start + batch_shots)
        unpacked = np.unpackbits(packed[start:stop], axis=1, bitorder="little")
        if np.any(unpacked[:, num_detectors:]):
            raise ValueError("detection_events.b8 has non-zero padding bits")
        decoded = np.asarray(matching.decode_batch(unpacked[:, :num_detectors]))
        if decoded.ndim == 2:
            if decoded.shape[1] != 1:
                raise ValueError(f"expected one observable, got {decoded.shape}")
            decoded = decoded[:, 0]
        predictions[start:stop] = decoded.astype(np.uint8, copy=False)
    return predictions


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analyze_archive(
    archive: Path,
    pathway: str,
    out_dir: Path,
    min_rounds: int,
    excluded_rounds: set[int],
    direct_si1000: bool,
) -> None:
    result_label = "direct_pymatching_si1000" if direct_si1000 else pathway
    raw_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        metadata_names = sorted(name for name in names if name.endswith("/metadata.json"))
        for metadata_name in metadata_names:
            base = metadata_name.rsplit("/", 1)[0]
            metadata = json.loads(zipped.read(metadata_name))
            rounds = int(metadata["rounds"])
            if rounds < min_rounds or rounds in excluded_rounds:
                continue
            shots = int(metadata["shots"])
            actual_name = f"{base}/obs_flips_actual.b8"
            actual = _single_observable_from_bytes(zipped.read(actual_name), shots, actual_name)
            if direct_si1000:
                import pymatching
                import stim

                noisy_name = f"{base}/circuit_noisy_si1000.stim"
                circuit = stim.Circuit(zipped.read(noisy_name).decode("utf-8"))
                dem = circuit.detector_error_model()
                matching = pymatching.Matching.from_detector_error_model(dem)
                detection_name = f"{base}/detection_events.b8"
                predicted = _decode_detection_b8(
                    zipped.read(detection_name),
                    shots=shots,
                    num_detectors=dem.num_detectors,
                    matching=matching,
                )
            else:
                prediction_name = f"{base}/decoding_results/{pathway}/obs_flips_predicted.b8"
                if prediction_name not in names:
                    continue
                predicted = _single_observable_from_bytes(
                    zipped.read(prediction_name), shots, prediction_name
                )
            summary = _logical_summary(actual, predicted)
            parts = base.split("/")
            raw_rows.append(
                {
                    "patch": parts[-3],
                    "basis": str(metadata["basis"]),
                    "distance": int(metadata["distance"]),
                    "rounds": rounds,
                    "pathway": result_label,
                    **summary,
                }
            )
            if direct_si1000 and len(raw_rows) % 25 == 0:
                print(f"decoded {len(raw_rows)} configurations", flush=True)

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (str(row["patch"]), str(row["basis"]), int(row["distance"]))
        grouped.setdefault(key, []).append(row)

    fit_rows: list[dict[str, object]] = []
    for (patch, basis, distance), rows in sorted(grouped.items()):
        total_points = len(rows)
        rows = [row for row in rows if float(row["logical_failure_rate"]) < 0.5]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda row: int(row["rounds"]))
        rounds = np.array([int(row["rounds"]) for row in rows], dtype=float)
        rates = np.array([float(row["logical_failure_rate"]) for row in rows])
        epsilon, intercept, epsilon_std = _fit_epsilon(rounds, rates)
        fit_rows.append(
            {
                "patch": patch,
                "basis": basis,
                "distance": distance,
                "pathway": result_label,
                "min_rounds": min_rounds,
                "excluded_rounds": ",".join(str(value) for value in sorted(excluded_rounds)),
                "points_total": total_points,
                "points_used": len(rows),
                "points_at_or_above_half_excluded": total_points - len(rows),
                "epsilon": epsilon,
                "epsilon_std": epsilon_std,
                "log_fidelity_intercept": intercept,
            }
        )

    distance_rows: list[dict[str, object]] = []
    for distance in sorted({int(row["distance"]) for row in fit_rows}):
        selected = [row for row in fit_rows if row["distance"] == distance]
        values = [float(row["epsilon"]) for row in selected]
        standard_deviations = [float(row["epsilon_std"]) for row in selected]
        distance_rows.append(
            {
                "distance": distance,
                "pathway": result_label,
                "code_basis_fits": len(values),
                "mean_epsilon": float(np.mean(values)),
                "mean_epsilon_std": float(np.sqrt(np.sum(np.square(standard_deviations))))
                / len(values),
            }
        )

    if len(distance_rows) < 2:
        raise ValueError("not enough distance summaries to fit Lambda")
    distances = np.array([int(row["distance"]) for row in distance_rows], dtype=float)
    mean_epsilon = np.array([float(row["mean_epsilon"]) for row in distance_rows])
    slope, intercept = np.polyfit(distances, np.log(mean_epsilon), 1)
    mean_std = np.array([float(row["mean_epsilon_std"]) for row in distance_rows])
    rng = np.random.default_rng(20260824)
    sampled_epsilon = rng.normal(mean_epsilon, mean_std, size=(20_000, len(distances)))
    sampled_epsilon = sampled_epsilon[np.all(sampled_epsilon > 0, axis=1)]
    centered_distance = distances - distances.mean()
    sampled_slopes = (
        np.log(sampled_epsilon) @ centered_distance
        / float(np.sum(np.square(centered_distance)))
    )
    sampled_lambda = np.exp(-2.0 * sampled_slopes)
    lambda_fit = float(np.exp(-2.0 * slope))
    summary = {
        "archive": str(archive),
        "pathway": result_label,
        "min_rounds": min_rounds,
        "excluded_rounds": sorted(excluded_rounds),
        "configurations": len(raw_rows),
        "lambda_fit": lambda_fit,
        "lambda_std": float(np.std(sampled_lambda, ddof=1)),
        "lambda_ci95_low": float(np.quantile(sampled_lambda, 0.025)),
        "lambda_ci95_high": float(np.quantile(sampled_lambda, 0.975)),
        "log_epsilon_intercept": float(intercept),
        "distances": distance_rows,
    }
    _write_csv(out_dir / "logical_error_by_configuration.csv", raw_rows)
    _write_csv(out_dir / "epsilon_by_patch_basis.csv", fit_rows)
    _write_csv(out_dir / "epsilon_by_distance.csv", distance_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _self_test() -> None:
    actual = np.array([0, 1, 1, 0], dtype=np.uint8)
    predicted = np.array([0, 0, 1, 1], dtype=np.uint8)
    summary = _logical_summary(actual, predicted)
    assert summary["failures"] == 2
    assert summary["logical_failure_rate"] == 0.5
    epsilon, intercept, epsilon_std = _fit_epsilon(
        np.array([10.0, 20.0, 30.0]),
        0.5 * (1.0 - np.power(1.0 - 2.0 * 0.01, [10.0, 20.0, 30.0])),
    )
    assert abs(epsilon - 0.01) < 1e-12
    assert abs(intercept) < 1e-12
    assert epsilon_std < 1e-12
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--prediction", action="append", type=_parse_prediction, default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--pathway")
    parser.add_argument("--direct-si1000", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--min-rounds", type=int, default=10)
    parser.add_argument("--exclude-rounds", type=int, nargs="*", default=[13])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.archive is not None:
        if args.out_dir is None or (args.pathway is None) == (not args.direct_si1000):
            parser.error(
                "--archive requires --out-dir and exactly one of --pathway or --direct-si1000"
            )
        _analyze_archive(
            args.archive,
            args.pathway or "",
            args.out_dir,
            args.min_rounds,
            set(args.exclude_rounds),
            args.direct_si1000,
        )
        return
    if args.data_dir is None or args.out_json is None or not args.prediction:
        parser.error("--data-dir, --prediction, and --out-json are required")

    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    shots = int(metadata["shots"])
    actual = _load_single_observable_b8(args.data_dir / "obs_flips_actual.b8", shots)
    results = {
        label: _logical_summary(
            actual,
            _load_single_observable_b8(args.data_dir / relative, shots),
        )
        for label, relative in args.prediction
    }
    payload = {"metadata": metadata, "predictions": results}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
