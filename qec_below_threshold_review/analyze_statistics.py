from __future__ import annotations

"""Bootstrap fitted QEC metrics and correct paired decoder comparisons."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from plot_baseline import _crossing
from plot_correlated import _exact_pvalue, _lambda_fit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _holm_adjust(pvalues: list[float]) -> list[float]:
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _bootstrap_lambda(
    rows: list[dict[str, str]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row["distance"]))
    points = [(int(row["distance"]), float(row["epsilon"])) for row in rows]
    point_estimate = _lambda_fit(points)
    sampled_epsilon = []
    for row in rows:
        shots = int(row["shots"])
        rate = int(row["failures"]) / shots
        sampled_rate = rng.binomial(shots, rate, size=replicates) / shots
        sampled_rate = np.minimum(sampled_rate, 0.5)
        rounds = int(row["rounds"])
        epsilon = 0.5 * (1.0 - np.power(1.0 - 2.0 * sampled_rate, 1.0 / rounds))
        sampled_epsilon.append(np.maximum(epsilon, np.finfo(float).tiny))
    distances = np.asarray([point[0] for point in points], dtype=float)
    centered = distances - distances.mean()
    slopes = (centered[:, None] * np.log(np.asarray(sampled_epsilon))).sum(axis=0)
    slopes /= float(np.square(centered).sum())
    return point_estimate, np.exp(-2.0 * slopes)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def _baseline_bootstrap(
    rows: list[dict[str, str]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["basis"], float(row["p"]))].append(row)

    lambda_rows = []
    samples_by_basis: dict[str, dict[float, np.ndarray]] = defaultdict(dict)
    for (basis, p), group in sorted(groups.items()):
        estimate, samples = _bootstrap_lambda(group, replicates=replicates, rng=rng)
        low, high = _ci(samples)
        lambda_rows.append(
            {"basis": basis, "p": p, "lambda_fit": estimate, "ci95_low": low, "ci95_high": high}
        )
        samples_by_basis[basis][p] = samples

    threshold_rows = []
    for basis in ("x", "z"):
        ordered = sorted(samples_by_basis[basis].items())
        point_values = [
            (float(row["p"]), float(row["lambda_fit"]))
            for row in lambda_rows
            if row["basis"] == basis
        ]
        point_estimate = _crossing(point_values)
        threshold_samples = []
        for replicate in range(replicates):
            points = [(p, samples[replicate]) for p, samples in ordered]
            crossing = _crossing(points)
            if crossing is not None:
                threshold_samples.append(crossing)
        sample_array = np.asarray(threshold_samples, dtype=float)
        low, high = _ci(sample_array)
        threshold_rows.append(
            {
                "basis": basis,
                "threshold": point_estimate,
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_fraction": len(threshold_samples) / replicates,
            }
        )
    return lambda_rows, threshold_rows


def _correlated_bootstrap(
    rows: list[dict[str, str]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    paired_rows = []
    for row in rows:
        groups[(row["basis"], row["condition"])].append(row)
        if row["condition"] == "correlated_mismatched":
            first = int(row["mismatched_only_failures"])
            second = int(row["matched_only_failures"])
            paired_rows.append(
                {
                    "basis": row["basis"],
                    "distance": int(row["distance"]),
                    "mismatched_only_failures": first,
                    "matched_only_failures": second,
                    "net_gain": first - second,
                    "p_raw": _exact_pvalue(first, second),
                }
            )

    lambda_rows = []
    for (basis, condition), group in sorted(groups.items()):
        estimate, samples = _bootstrap_lambda(group, replicates=replicates, rng=rng)
        low, high = _ci(samples)
        lambda_rows.append(
            {
                "basis": basis,
                "condition": condition,
                "lambda_fit": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    pvalues = [float(row["p_raw"]) for row in paired_rows]
    holm = _holm_adjust(pvalues)
    for row, adjusted in zip(paired_rows, holm):
        row["p_holm_6"] = adjusted
        row["p_bonferroni_6"] = min(1.0, 6.0 * float(row["p_raw"]))
    return lambda_rows, paired_rows


def _sensitivity_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        groups[(row["basis"], float(row["zz_factor"]))][row["condition"]] = row

    output = []
    for (basis, zz_factor), conditions in sorted(groups.items()):
        mismatched = conditions["correlated_mismatched"]
        matched = conditions["correlated_matched"]
        first = int(mismatched["mismatched_only_failures"])
        second = int(mismatched["matched_only_failures"])
        mismatch_rate = float(mismatched["p_L"])
        matched_rate = float(matched["p_L"])
        output.append(
            {
                "basis": basis,
                "zz_factor": zz_factor,
                "p_corr": float(mismatched["p_corr"]),
                "shots": int(mismatched["shots"]),
                "p_L_mismatched": mismatch_rate,
                "p_L_matched": matched_rate,
                "relative_reduction": (mismatch_rate - matched_rate) / mismatch_rate,
                "mismatched_only_failures": first,
                "matched_only_failures": second,
                "net_gain": first - second,
                "p_raw": _exact_pvalue(first, second),
            }
        )
    holm = _holm_adjust([float(row["p_raw"]) for row in output])
    for row, adjusted in zip(output, holm):
        row["p_holm_8"] = adjusted
        row["p_bonferroni_8"] = min(1.0, 8.0 * float(row["p_raw"]))
    return output


def _self_test() -> None:
    assert _exact_pvalue(5, 0) == 0.0625
    adjusted = _holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--correlated-csv", type=Path)
    parser.add_argument("--sensitivity-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not all((args.baseline_csv, args.correlated_csv, args.sensitivity_csv, args.out_dir)):
        raise SystemExit("baseline, correlated, sensitivity CSVs and --out-dir are required")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")

    rng = np.random.default_rng(args.seed)
    baseline_lambda, thresholds = _baseline_bootstrap(
        _read_csv(args.baseline_csv), replicates=args.bootstrap_replicates, rng=rng
    )
    correlated_lambda, paired = _correlated_bootstrap(
        _read_csv(args.correlated_csv), replicates=args.bootstrap_replicates, rng=rng
    )
    sensitivity = _sensitivity_summary(_read_csv(args.sensitivity_csv))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "baseline_lambda_bootstrap.csv", baseline_lambda)
    _write_csv(args.out_dir / "threshold_bootstrap.csv", thresholds)
    _write_csv(args.out_dir / "correlated_lambda_bootstrap.csv", correlated_lambda)
    _write_csv(args.out_dir / "paired_adjusted.csv", paired)
    _write_csv(args.out_dir / "sensitivity_summary.csv", sensitivity)
    summary = {
        "method": "parametric binomial bootstrap of aggregated shot failures",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "thresholds": thresholds,
        "correlated_lambda": correlated_lambda,
        "paired_adjusted": paired,
        "sensitivity": sensitivity,
    }
    (args.out_dir / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
