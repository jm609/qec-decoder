"""Render crossed d7 diagnostics from saved JSON without loading PyTorch."""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(path: Path, cells: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    pairs = ('g0_d0', 'g0_d1', 'g1_d0', 'g1_d1')
    labels = ('G0 / D0', 'G0 / D1', 'G1 / D0', 'G1 / D1')
    x = np.arange(len(labels))
    branches = (('top1', 'Top 1'), ('branch', 'After margin'), ('final', 'Final policy'))
    for axs, method in zip(axes, ('mwpm', 'correlated')):
        rows = [cells[f'{key}_{method}'] for key in pairs]
        for i, (branch, title) in enumerate(branches):
            position = x + (i - 1) * 0.24
            axs[0].bar(position, [row[f'{branch}_gain_pp'] for row in rows], width=0.23, label=title)
            harm = [row[branch]['harm_pct_of_edits'] for row in rows]
            eligible = [k for k, value in enumerate(harm) if value is not None]
            axs[2].bar(position[eligible], [harm[k] for k in eligible], width=0.23, label=title)
        axs[1].bar(x, [row['oracle_gain_pp'] for row in rows], width=0.55, color='#317759')
        axs[0].axhline(0, color='#555555', linewidth=0.6)
        axs[0].legend(fontsize=8)
        axs[2].text(0.01, 0.99, 'Omitted when no edits were applied', fontsize=7, va='top', transform=axs[2].transAxes)
        for ax, title, ylabel in zip(axs, ('Net gain', 'Candidate oracle', 'Harm per applied edit'), ('pp', 'pp', '%')):
            ax.set_xticks(x, labels)
            ax.set_title(f'{method.upper()} | {title}')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    render(args.summary.with_name('crossed_diagnostics.png'), json.loads(args.summary.read_text(encoding='utf-8'))['cells'])
