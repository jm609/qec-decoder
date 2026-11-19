from __future__ import annotations

"""Summarize and plot the p_det-matched correlated-noise experiment."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CONDITIONS = (
    "independent_matched",
    "correlated_mismatched",
    "correlated_matched",
)


def _exact_pvalue(first_only: int, second_only: int) -> float:
    total = first_only + second_only
    if total == 0:
        return 1.0
    tail = min(first_only, second_only)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total)


def _lambda_fit(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    distances = np.asarray([distance for distance, _ in points], dtype=float)
    epsilons = np.asarray([epsilon for _, epsilon in points], dtype=float)
    return math.exp(-2.0 * float(np.polyfit(distances, np.log(epsilons), 1)[0]))


def _read_groups(path: Path) -> dict[tuple[str, int], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        groups[(row["basis"], int(row["distance"]))][row["condition"]] = row
    for key, conditions in groups.items():
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"missing condition for {key}: {sorted(conditions)}")
    return groups


def summarize(
    groups: dict[tuple[str, int], dict[str, dict[str, str]]],
    *,
    pdet_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    lambdas: dict[str, dict[str, float]] = {}
    for basis in ("x", "z"):
        lambdas[basis] = {
            condition: _lambda_fit(
                [
                    (distance, float(groups[(basis, distance)][condition]["epsilon"]))
                    for distance in (3, 5, 7)
                ]
            )
            for condition in CONDITIONS
        }

    output = []
    for (basis, distance), conditions in sorted(groups.items()):
        independent = conditions["independent_matched"]
        mismatched = conditions["correlated_mismatched"]
        matched = conditions["correlated_matched"]
        pdet_delta = float(mismatched["p_det"]) - float(independent["p_det"])
        if abs(pdet_delta) > pdet_tolerance:
            raise ValueError(
                f"p_det mismatch exceeds tolerance for basis={basis}, d={distance}: {pdet_delta}"
            )
        mismatch_only = int(mismatched["mismatched_only_failures"])
        matched_only = int(mismatched["matched_only_failures"])
        output.append(
            {
                "basis": basis,
                "distance": distance,
                "independent_p": float(independent["p"]),
                "correlated_p": float(mismatched["p"]),
                "p_corr": float(mismatched["p_corr"]),
                "zz_factor": float(mismatched["zz_factor"]),
                "shots": int(mismatched["shots"]),
                "pdet_independent": float(independent["p_det"]),
                "pdet_correlated": float(mismatched["p_det"]),
                "pdet_delta": pdet_delta,
                "epsilon_independent": float(independent["epsilon"]),
                "epsilon_mismatched": float(mismatched["epsilon"]),
                "epsilon_matched": float(matched["epsilon"]),
                "mismatched_only_failures": mismatch_only,
                "matched_only_failures": matched_only,
                "paired_net_gain": mismatch_only - matched_only,
                "paired_exact_pvalue": _exact_pvalue(mismatch_only, matched_only),
                "lambda_independent": lambdas[basis]["independent_matched"],
                "lambda_mismatched": lambdas[basis]["correlated_mismatched"],
                "lambda_matched": lambdas[basis]["correlated_matched"],
            }
        )
    return output, lambdas


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(
    groups: dict[tuple[str, int], dict[str, dict[str, str]]],
    lambdas: dict[str, dict[str, float]],
    output_png: Path,
) -> None:
    styles = {
        "independent_matched": ("Independent", "#4477AA", "o"),
        "correlated_mismatched": ("ZZ + mismatched prior", "#EE6677", "s"),
        "correlated_matched": ("ZZ + matched prior", "#228833", "^"),
    }
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, constrained_layout=True)
    for axis, basis in zip(axes, ("x", "z")):
        for condition in CONDITIONS:
            label, color, marker = styles[condition]
            rows = [groups[(basis, distance)][condition] for distance in (3, 5, 7)]
            epsilon = np.asarray([float(row["epsilon"]) for row in rows])
            low = np.asarray([float(row["epsilon_ci95_low"]) for row in rows])
            high = np.asarray([float(row["epsilon_ci95_high"]) for row in rows])
            axis.errorbar(
                (3, 5, 7),
                epsilon,
                yerr=np.vstack((epsilon - low, high - epsilon)),
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2.5,
                capsize=4,
                label=f"{label} (Λ={lambdas[basis][condition]:.2f})",
            )
        axis.set(
            title=f"{basis.upper()} memory",
            xlabel="Code distance d",
            xticks=(3, 5, 7),
            yscale="log",
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Logical error per cycle ε")
    fig.suptitle("Correlated-ZZ stress test at matched detection probability", fontsize=18)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)


def _self_test() -> None:
    assert _exact_pvalue(5, 0) == 0.0625
    assert _exact_pvalue(3, 3) == 1.0
    assert abs(_lambda_fit([(3, 0.04), (5, 0.02), (7, 0.01)]) - 2.0) < 1e-12
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--pdet-tolerance", type=float, default=0.0005)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not all((args.input_csv, args.output_png, args.summary_csv)):
        raise SystemExit("--input-csv, --output-png, and --summary-csv are required")
    groups = _read_groups(args.input_csv)
    rows, lambdas = summarize(groups, pdet_tolerance=args.pdet_tolerance)
    _write_csv(args.summary_csv, rows)
    plot(groups, lambdas, args.output_png)
    print(json.dumps(lambdas, indent=2))


if __name__ == "__main__":
    main()
