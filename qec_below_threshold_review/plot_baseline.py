from __future__ import annotations

"""Plot baseline logical error scaling and fitted suppression factors."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_rows(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return [
        {
            "basis": row["basis"],
            "distance": int(row["distance"]),
            "p": float(row["p"]),
            "epsilon": float(row["epsilon"]),
            "epsilon_low": float(row["epsilon_ci95_low"]),
            "epsilon_high": float(row["epsilon_ci95_high"]),
        }
        for row in rows
    ]


def _lambda_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, float], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["basis"]), float(row["p"]))].append(
            (int(row["distance"]), float(row["epsilon"]))
        )

    output = []
    for (basis, p), points in sorted(grouped.items()):
        points.sort()
        if [distance for distance, _ in points] != [3, 5, 7]:
            raise ValueError(f"expected distances 3,5,7 for basis={basis}, p={p}")
        distances = np.asarray([distance for distance, _ in points], dtype=float)
        epsilons = np.asarray([epsilon for _, epsilon in points], dtype=float)
        slope = float(np.polyfit(distances, np.log(epsilons), 1)[0])
        output.append(
            {
                "basis": basis,
                "p": p,
                "lambda_fit": math.exp(-2.0 * slope),
                "lambda_3_to_5": epsilons[0] / epsilons[1],
                "lambda_5_to_7": epsilons[1] / epsilons[2],
            }
        )
    return output


def _crossing(points: list[tuple[float, float]]) -> float | None:
    for (p0, value0), (p1, value1) in zip(points, points[1:]):
        if (value0 - 1.0) * (value1 - 1.0) <= 0.0 and value0 != value1:
            return p0 + (1.0 - value0) * (p1 - p0) / (value1 - value0)
    return None


def _write_lambda_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(input_csv: Path, output_png: Path, lambda_csv: Path) -> dict[str, float | None]:
    rows = _read_rows(input_csv)
    lambdas = _lambda_rows(rows)
    _write_lambda_csv(lambda_csv, lambdas)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    fig = plt.figure(figsize=(11.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
    colors = {3: "#4477AA", 5: "#EE6677", 7: "#228833"}
    markers = {3: "o", 5: "s", 7: "^"}
    for column, basis in enumerate(("x", "z")):
        axis = fig.add_subplot(grid[0, column])
        for distance in (3, 5, 7):
            selected = sorted(
                (row for row in rows if row["basis"] == basis and row["distance"] == distance),
                key=lambda row: float(row["p"]),
            )
            x = np.asarray([row["p"] for row in selected], dtype=float)
            y = np.asarray([row["epsilon"] for row in selected], dtype=float)
            low = np.asarray([row["epsilon_low"] for row in selected], dtype=float)
            high = np.asarray([row["epsilon_high"] for row in selected], dtype=float)
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                marker=markers[distance],
                markersize=6,
                linewidth=2.2,
                capsize=3,
                color=colors[distance],
                label=f"d={distance}",
            )
        axis.set(xscale="log", yscale="log", title=f"{basis.upper()} memory", xlabel="Noise parameter p")
        axis.grid(True, which="both", alpha=0.25)
        if column == 0:
            axis.set_ylabel("Logical error per cycle ε")
            axis.legend(frameon=False)

    lambda_axis = fig.add_subplot(grid[1, :])
    crossings: dict[str, float | None] = {}
    for basis, marker, color in (("x", "o", "#4477AA"), ("z", "s", "#CC6677")):
        selected = [row for row in lambdas if row["basis"] == basis]
        points = [(float(row["p"]), float(row["lambda_fit"])) for row in selected]
        lambda_axis.plot(
            *zip(*points),
            marker=marker,
            markersize=7,
            linewidth=2.5,
            color=color,
            label=f"{basis.upper()} fit",
        )
        crossings[basis] = _crossing(points)
        if crossings[basis] is not None:
            lambda_axis.axvline(crossings[basis], color=color, linestyle=":", alpha=0.7)
    lambda_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    lambda_axis.set(xscale="log", xlabel="Noise parameter p", ylabel="Suppression factor Λ")
    lambda_axis.grid(True, which="both", alpha=0.25)
    lambda_axis.legend(frameon=False)
    fig.suptitle("Rotated surface-code baseline (20,000 shots per point)", fontsize=18)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    return {f"{basis}_crossing": crossing for basis, crossing in crossings.items()}


def _self_test() -> None:
    rows = [
        {"basis": "x", "p": 0.001, "distance": distance, "epsilon": epsilon}
        for distance, epsilon in ((3, 0.04), (5, 0.02), (7, 0.01))
    ]
    result = _lambda_rows(rows)
    assert len(result) == 1 and abs(float(result[0]["lambda_fit"]) - 2.0) < 1e-12
    assert _crossing([(0.008, 1.1), (0.009, 0.9)]) == 0.0085
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--lambda-csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not all((args.input_csv, args.output_png, args.lambda_csv)):
        raise SystemExit("--input-csv, --output-png, and --lambda-csv are required")
    print(json.dumps(plot(args.input_csv, args.output_png, args.lambda_csv), indent=2))


if __name__ == "__main__":
    main()
