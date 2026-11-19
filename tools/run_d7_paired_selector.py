"""Fit a d7 selector head for Z versus raw-preserving edits on development maps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.rotated_rect import build_rectangular_syndrome_layout
from tools import probe_d7_z_transition as probe
from tools.analyze_d7_logical_transitions import class4, digest, load


@dataclass(frozen=True)
class Cases:
    raw: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    weight: np.ndarray
    top: np.ndarray
    margin: np.ndarray
    no_edit: np.ndarray
    pair_shots: np.ndarray
    preserving_rows: np.ndarray
    control: np.ndarray
    patch_events: np.ndarray

    def outcomes(self) -> np.ndarray:
        return correctness_delta(self.raw[self.pair_shots], self.prediction[self.top[self.pair_shots]],
                                 self.truth[self.pair_shots])


def correctness_delta(raw: np.ndarray, selected: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (selected == truth).astype(np.int8) - (raw == truth)


def preserving_rows(raw: np.ndarray, prediction: np.ndarray, shot: np.ndarray, weights: np.ndarray,
                    utility: np.ndarray) -> np.ndarray:
    candidates = np.flatnonzero((weights != 0) & (prediction == raw[shot]))
    maxima = np.full(len(raw), -np.inf)
    np.maximum.at(maxima, shot[candidates], utility[candidates])
    winners = candidates[utility[candidates] == maxima[shot[candidates]]]
    rows = np.full(len(raw), len(shot), dtype=np.intp)
    np.minimum.at(rows, shot[winners], winners)
    return rows


def make_cases(arrays, events: np.ndarray, lookup: np.ndarray, order: np.ndarray, model_index: int) -> Cases:
    shot, utility, weight, top, margin = (arrays[key] for key in (
        'shot_index', 'utility', 'weights', 'top1_rows', 'branch_rows'))
    raw, prediction = (class4(arrays[key]) for key in ('d0_mwpm_raw', 'd0_mwpm_predicted'))
    if arrays['d0_mwpm_failure'].any():
        raise ValueError('The development payloads must have zero matching failures')
    no_edit = np.flatnonzero(weight == 0)
    for rows in (no_edit, top, margin):
        np.testing.assert_array_equal(shot[rows], np.arange(len(raw)))
    np.testing.assert_array_equal(prediction[no_edit], raw)
    keep = preserving_rows(raw, prediction, shot, weight, utility)
    paired = np.flatnonzero((weight[top] != 0) & ((prediction[top] ^ raw) == 2) & (keep < len(shot)))
    z, keep = top[paired], keep[paired]
    np.testing.assert_array_equal(shot[keep], paired)
    max_detectors = events.shape[1]
    masks_z = np.unpackbits(arrays['packed_masks'][z], axis=1, count=max_detectors)
    masks_keep = np.unpackbits(arrays['packed_masks'][keep], axis=1, count=max_detectors)
    for masks, rows in ((masks_z, z), (masks_keep, keep)):
        np.testing.assert_array_equal(masks.sum(axis=1), weight[rows])
    geometry_z, patch_z = probe.patch_features(events[paired], masks_z, lookup, order)
    geometry_keep, patch_keep = probe.patch_features(events[paired], masks_keep, lookup, order)
    model_bits = np.broadcast_to(np.eye(len(probe.CHECKPOINTS))[model_index], (len(paired), len(probe.CHECKPOINTS)))
    control = np.column_stack((utility[z] - utility[keep], utility[z] - utility[no_edit[paired]], weight[z],
                               weight[keep], np.eye(4)[raw[paired]], model_bits, geometry_z, geometry_keep))
    return Cases(raw, class4(arrays['truth']), prediction, weight, top, margin, no_edit, paired, keep,
                 control, np.column_stack((patch_z, patch_keep)))


def load_map(source: Path, seed: int, input_hashes: dict) -> list[Cases]:
    base = source / 'development' / str(seed)
    case_root = base / 'g1'
    record = load(case_root / 'case.json')
    assert record['settings']['local_seed'] == seed and record['settings']['zz_factor'] == 0.25
    samples, = (name for name in record['sha256'] if name.endswith('/samples.npz'))
    sample = case_root / samples
    assert digest(sample) == input_hashes[sample.relative_to(source).as_posix()] == record['sha256'][samples]
    with np.load(sample) as arrays:
        lookup, order = probe.patch_lookup(build_rectangular_syndrome_layout(arrays))
        events, truth = arrays['detector_events'], arrays['observable_flips']
    models = load(base / 'result.json')['data']['g1']
    cases = []
    for index, model in enumerate(probe.CHECKPOINTS):
        path = case_root / models[model]['file']
        assert digest(path) == input_hashes[path.relative_to(source).as_posix()] == models[model]['sha256']
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays['truth'], truth)
            cases.append(make_cases(arrays, events, lookup, order, index))
    return cases


def fit_head(features: np.ndarray, outcome: np.ndarray) -> dict:
    pivotal = outcome != 0
    train = features[pivotal]
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1.0)
    x = np.column_stack((np.ones(len(train)), (train - center) / scale))
    y = outcome[pivotal] == 1

    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ beta
        penalty = np.r_[0.0, beta[1:]]
        value = np.mean(np.logaddexp(0.0, logits) - y * logits) + probe.L2 / 2 * (penalty @ penalty)
        gradient = x.T @ (expit(logits) - y) / len(y) + probe.L2 * penalty
        return float(value), gradient

    fitted = minimize(loss, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f'Selector head did not converge: {fitted.message}')
    return {'center': center.tolist(), 'scale': scale.tolist(), 'intercept': float(fitted.x[0]),
            'coefficients': fitted.x[1:].tolist()}


def head_score(features: np.ndarray, state: dict) -> np.ndarray:
    scaled = (features - np.asarray(state['center'])) / np.asarray(state['scale'])
    return scaled @ np.asarray(state['coefficients']) + state['intercept']


def select_rows(case: Cases, accept_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paired = case.pair_shots
    rejected = paired[~accept_z]
    fixed = case.top.copy()
    fixed[rejected] = case.preserving_rows[~accept_z]
    free = case.margin.copy()
    free[paired] = np.where(accept_z, case.top[paired], case.no_edit[paired])
    np.testing.assert_array_equal(case.weight[fixed] != 0, case.weight[case.top] != 0)
    return fixed, free


def count_routes(cases: list[Cases], rows: list[np.ndarray]) -> dict:
    totals = dict(shots=0, edits=0, corrections=0, harms=0, changed_rows_vs_top1=0)
    for case, selected in zip(cases, rows, strict=True):
        delta = correctness_delta(case.raw, case.prediction[selected], case.truth)
        totals['shots'] += len(delta)
        totals['edits'] += int(np.count_nonzero(case.weight[selected]))
        totals['corrections'] += int(np.sum(delta == 1))
        totals['harms'] += int(np.sum(delta == -1))
        totals['changed_rows_vs_top1'] += int(np.sum(selected != case.top))
    totals['gain_pp'] = 100 * (totals['corrections'] - totals['harms']) / totals['shots']
    return totals


def features(case: Cases, local_events: bool) -> np.ndarray:
    return np.column_stack((case.control, case.patch_events)) if local_events else case.control


def compare_fold(train: list[Cases], test: list[Cases]) -> dict:
    outcome = np.concatenate([case.outcomes() for case in train])
    margin_accepts = np.concatenate([case.margin[case.pair_shots] == case.top[case.pair_shots] for case in train])
    train_recovery_target = int(np.sum((outcome == 1) & margin_accepts))
    result = {
        'paired_shots': sum(len(case.pair_shots) for case in test),
        'training_recovery_target': train_recovery_target,
        'top1': count_routes(test, [case.top for case in test]),
        'margin': count_routes(test, [case.margin for case in test]),
        'heads': {},
    }
    for with_events in (False, True):
        x = np.concatenate([features(case, with_events) for case in train])
        head = fit_head(x, outcome)
        train_scores = head_score(x, head)
        test_scores = [head_score(features(case, with_events), head) for case in test]
        cutoffs = {'zero_log_odds': 0.0,
                   'training_margin_matched': probe.recovery_cutoff(train_scores, outcome, train_recovery_target)}
        thresholds = {}
        for name, cutoff in cutoffs.items():
            selected = [select_rows(case, score >= cutoff) for case, score in zip(test, test_scores, strict=True)]
            fixed, free = zip(*selected)
            thresholds[name] = {'cutoff': cutoff, 'fixed': count_routes(test, fixed), 'free': count_routes(test, free)}
        result['heads']['detector_events' if with_events else 'control'] = {'state': head, 'thresholds': thresholds}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--probe-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    screening = load(args.probe_summary)
    assert screening['development_maps'] == list(probe.MAPS)
    cases = {seed: load_map(args.source, seed, screening['input_sha256']) for seed in probe.MAPS}
    folds = {str(test): {'fit_map': train, **compare_fold(cases[train], cases[test])}
             for train, test in (probe.MAPS, probe.MAPS[::-1])}
    summary = {
        'primary': 'zero log-odds, same originally edited shots, paired selector versus original top1',
        'secondary': 'allow no edit; also report both paths with a cutoff matched to training-map margin recoveries',
        'runner_sha256': digest(Path(__file__)), 'probe_summary_sha256': digest(args.probe_summary),
        'probe_source_sha256': digest(Path(probe.__file__)), 'conda_prefix': sys.prefix,
        'input_sha256': screening['input_sha256'], 'l2': probe.L2, 'folds': folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    for seed, fold in folds.items():
        print(seed, 'top1', fold['top1'], 'margin', fold['margin'])
        for head, result in fold['heads'].items():
            print(head, result['thresholds'])


if __name__ == '__main__':
    main()
