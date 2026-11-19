"""Attribute stored d7 selector errors to candidate coverage, ranking, or no-edit."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

LABELS = ('I', 'X', 'Z', 'Y')
CELLS = ('d0_mwpm', 'd1_mwpm', 'd0_correlated', 'd1_correlated')
BRANCHES = ('top1', 'branch', 'final')


def class4(bits: np.ndarray) -> np.ndarray:
    # Saved observable order is [Z flip, X flip].
    if bits.ndim != 2 or bits.shape[1] != 2 or np.any((bits != 0) & (bits != 1)):
        raise ValueError('Expected two binary observables per prediction')
    bits = bits.astype(np.uint8, copy=False)
    return 2 * bits[:, 0] + bits[:, 1]


def branch_counts(
    raw: np.ndarray, truth: np.ndarray, predicted: np.ndarray, edited: np.ndarray,
    reachable: np.ndarray, reachable_by_edit: np.ndarray,
) -> dict[str, int]:
    raw_correct = raw == truth
    selected_correct = predicted == truth
    harmed = raw_correct & ~selected_correct
    missed = ~raw_correct & ~selected_correct & reachable
    no_candidate = ~reachable
    assert not np.any(harmed & ~edited)
    assert not np.any(no_candidate & (selected_correct | raw_correct))
    fixed_oracle = np.where(edited, reachable_by_edit, raw_correct)
    original_wrong = ~raw_correct
    fixed_n = int(fixed_oracle.sum())
    counts = {
        'shots': len(truth),
        'raw_correct': int(raw_correct.sum()),
        'selected_correct': int(selected_correct.sum()),
        'raw_wrong': int(original_wrong.sum()),
        'edited': int(edited.sum()),
        'fixed_edit_oracle_correct': fixed_n,
        'oracle_correct': int(reachable.sum()),
        'both_correct': int(np.sum(raw_correct & selected_correct)),
        'improved': int(np.sum(original_wrong & selected_correct)),
        'harm_alternative_edit': int(np.sum(harmed & reachable_by_edit)),
        'harm_requires_no_edit': int(np.sum(harmed & ~reachable_by_edit)),
        'missed_recoverable_edited': int(np.sum(missed & edited)),
        'missed_recoverable_no_edit': int(np.sum(missed & ~edited)),
        'no_correct_candidate': int(no_candidate.sum()),
    }
    outcomes = ('both_correct', 'improved', 'harm_alternative_edit', 'harm_requires_no_edit',
                'missed_recoverable_edited', 'missed_recoverable_no_edit', 'no_correct_candidate')
    assert len(truth) == sum(counts[key] for key in outcomes)
    fixed_edit_gain = counts['harm_alternative_edit'] + counts['missed_recoverable_edited']
    edit_decision_gain = counts['harm_requires_no_edit'] + counts['missed_recoverable_no_edit']
    assert fixed_n - counts['selected_correct'] == fixed_edit_gain
    assert counts['oracle_correct'] - fixed_n == edit_decision_gain
    return counts


def analyze_cell(arrays, cell: str) -> dict:
    raw, truth, candidate = (class4(arrays[key]) for key in (f'{cell}_raw', 'truth', f'{cell}_predicted'))
    shot, weights, utility = (arrays[key] for key in ('shot_index', 'weights', 'utility'))
    identity = np.flatnonzero(weights == 0)
    np.testing.assert_array_equal(shot[identity], np.arange(len(truth)))
    np.testing.assert_array_equal(candidate[identity], raw)
    if arrays[f'{cell}_failure'].any():
        raise ValueError('This diagnostic expects the verified run with zero matching failures')
    class_scores = np.full((len(truth), len(LABELS)), -np.inf, dtype=utility.dtype)
    np.maximum.at(class_scores, (shot, candidate), utility)
    rows = np.arange(len(truth))
    correct = truth[shot] == candidate
    reachable = np.isfinite(class_scores[rows, truth])
    reachable_by_edit = np.bincount(shot[correct & (weights != 0)], minlength=len(truth)) != 0
    output = {}
    for route in BRANCHES:
        chosen = arrays[f'{route}_rows']
        np.testing.assert_array_equal(shot[chosen], rows)
        output[route] = branch_counts(raw, truth, candidate[chosen], weights[chosen] != 0, reachable, reachable_by_edit)

    top = arrays['top1_rows']
    selected = candidate[top]
    max_utility = class_scores.max(axis=1)
    np.testing.assert_array_equal(class_scores[rows, selected], max_utility)
    missed = (raw != truth) & (selected != truth) & reachable
    masks = {
        'harm': (raw == truth) & (selected != truth),
        'missed_recoverable': missed,
        'no_correct_candidate': ~reachable,
        'improved': (raw != truth) & (selected == truth),
    }
    triple = 16 * raw + 4 * selected + truth
    output['class_triples'] = {name: np.bincount(triple[mask], minlength=64).tolist() for name, mask in masks.items()}
    top_classes = (class_scores == max_utility[:, None]).sum(axis=1)
    correct_utility = class_scores[rows, truth]
    output['class_scores'] = {
        'top_utility_ties_across_classes': int(np.sum(top_classes > 1)),
        'available_class_histogram': np.bincount(np.isfinite(class_scores).sum(axis=1), minlength=5).tolist(),
        'missed_correct_class_utility_gap_sum': float(np.sum(max_utility[missed] - correct_utility[missed], dtype=np.float64)),
    }
    return output


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    # This run stores one compressed prediction payload per model/map/strength.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records: list[dict]) -> dict:
    totals = {}
    for route in BRANCHES:
        counts = Counter()
        for row in records:
            counts.update(row[route])
        n = counts['shots']
        totals[route] = {**counts,
            'gain_pp': 100*(counts['selected_correct']-counts['raw_correct'])/n,
            'oracle_gain_pp': 100*(counts['oracle_correct']-counts['raw_correct'])/n,
            'fixed_edit_oracle_gain_pp': 100*(counts['fixed_edit_oracle_correct']-counts['raw_correct'])/n}
    histograms = {}
    for outcome in records[0]['class_triples']:
        vector = np.array([row['class_triples'][outcome] for row in records]).sum(axis=0)
        histograms[outcome] = [{'raw': LABELS[i//16], 'selected': LABELS[(i//4)%4], 'truth': LABELS[i%4], 'count': int(vector[i])}
                               for i in np.argsort(-vector, kind='stable') if vector[i]]
    totals['class_triples'] = histograms
    scores = [row['class_scores'] for row in records]
    missed = totals['top1']['missed_recoverable_edited'] + totals['top1']['missed_recoverable_no_edit']
    totals['class_scores'] = {
        'top_utility_ties_across_classes': sum(row['top_utility_ties_across_classes'] for row in scores),
        'available_class_histogram': np.array([row['available_class_histogram'] for row in scores]).sum(axis=0).tolist(),
        'missed_correct_class_utility_gap_mean': sum(row['missed_correct_class_utility_gap_sum'] for row in scores) / missed if missed else None,
    }
    return totals


def run(source: Path, out: Path) -> None:
    config = load(source / 'frozen_evaluation.json')
    lock = load(source / 'verification.json')
    assert lock['summary_sha256'] == digest(source / 'summary.json')
    summary = load(source / 'summary.json')
    assert summary['frozen_evaluation_sha256'] == digest(source / 'frozen_evaluation.json')
    all_cells = {f'{g}_{cell}': [] for g in ('g0', 'g1') for cell in CELLS}
    panel = {}
    for seed in config['evaluation_local_seeds']:
        base = source / 'evaluation' / str(seed)
        result = load(base / 'result.json')
        assert result['local_seed'] == seed
        for strength, models in result['data'].items():
            for model, payload in models.items():
                path = base / strength / payload['file']
                assert digest(path) == payload['sha256']
                with np.load(path) as arrays:
                    assert len(arrays['truth']) == config['shots_per_strength']
                    for cell in CELLS:
                        item = analyze_cell(arrays, cell)
                        expected = payload['cells'][cell]
                        for route in BRANCHES:
                            counts = item[route]
                            assert counts['raw_correct'] == expected['raw_correct']
                            assert counts['oracle_correct'] == expected['oracle_correct']
                            assert counts['selected_correct'] == expected[route]['selected_correct']
                            assert counts['edited'] == expected[route]['applied_edits']
                            assert counts['improved'] == expected[route]['improved']
                            assert counts['harm_alternative_edit'] + counts['harm_requires_no_edit'] == expected[route]['harmed']
                        key = f'{strength}_{cell}'
                        all_cells[key].append(item)
                        panel.setdefault(model, {}).setdefault(key, []).append(item)
        print(f'Map {seed} replayed', flush=True)
    totals = {key: summarize(items) for key, items in all_cells.items()}
    for key, totals_by_route in totals.items():
        for route in BRANCHES:
            np.testing.assert_allclose(totals_by_route[route]['gain_pp'], summary['cells'][key][f'{route}_gain_pp'], atol=1e-12)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump({'source_summary_sha256': lock['summary_sha256'], 'conda_prefix': os.environ['CONDA_PREFIX'],
                   'map_ids': config['evaluation_local_seeds'], 'shots_per_map_strength': config['shots_per_strength'],
                   'cells': totals, 'models': {model: {k: summarize(v) for k, v in cells.items()} for model, cells in panel.items()}},
                  stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    run(args.source, args.out)
