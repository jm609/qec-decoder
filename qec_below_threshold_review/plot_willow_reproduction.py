from __future__ import annotations

"""Plot the public Willow-data reproduction against the paper references."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PAPER_ROWS = (
    {
        "decoder": "Paper neural decoder",
        "lambda": 2.14,
        "epsilon_d3": 0.00650,
        "epsilon_d5": 0.00303,
        "epsilon_d7": 0.00143,
        "source": "Nature Table S1",
    },
    {
        "decoder": "Paper Libra (101 ensemble)",
        "lambda": 2.04,
        "epsilon_d3": 0.00712,
        "epsilon_d5": 0.00349,
        "epsilon_d7": 0.00171,
        "source": "Nature Table S1",
    },
)


def _summary_row(path: Path, decoder: str, source: str) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    epsilon = {
        int(row["distance"]): float(row["mean_epsilon"])
        for row in summary["distances"]
    }
    if set(epsilon) != {3, 5, 7}:
        raise ValueError(f"expected distances 3, 5, 7 in {path}")
    return {
        "decoder": decoder,
        "lambda": float(summary["lambda_fit"]),
        "epsilon_d3": epsilon[3],
        "epsilon_d5": epsilon[5],
        "epsilon_d7": epsilon[7],
        "source": source,
    }


def plot(input_root: Path, output_png: Path, output_csv: Path) -> None:
    rows = [
        *PAPER_ROWS,
        _summary_row(
            input_root / "libra_rl" / "summary.json",
            "Public Libra (51 ensemble)",
            "Zenodo predictions, recomputed",
        ),
        _summary_row(
            input_root / "direct_pymatching_si1000" / "summary.json",
            "Direct PyMatching + SI1000",
            "Zenodo syndromes, direct decode",
        ),
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

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
    fig, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    styles = (("o", "-"), ("s", "--"), ("D", "-"), ("^", ":"))
    for row, (marker, line_style) in zip(rows, styles):
        axis.plot(
            (3, 5, 7),
            (row["epsilon_d3"], row["epsilon_d5"], row["epsilon_d7"]),
            marker=marker,
            linestyle=line_style,
            linewidth=2.8,
            markersize=8,
            label=f'{row["decoder"]}  (Λ={float(row["lambda"]):.3f})',
        )
    axis.set(
        xticks=(3, 5, 7),
        yscale="log",
        xlabel="Code distance d",
        ylabel="Logical error per cycle ε",
        title="Willow 105Q public-data decoder comparison",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)


def _self_test() -> None:
    assert {row["decoder"] for row in PAPER_ROWS} == {
        "Paper neural decoder",
        "Paper Libra (101 ensemble)",
    }
    assert all(float(row["epsilon_d7"]) < float(row["epsilon_d3"]) for row in PAPER_ROWS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not all((args.input_root, args.output_png, args.output_csv)):
        parser.error("--input-root, --output-png, and --output-csv are required")
    plot(args.input_root, args.output_png, args.output_csv)


if __name__ == "__main__":
    main()
