"""Test local detector-event features on two held-out-by-map development folds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.rotated_rect import build_rectangular_syndrome_layout
from tools.analyze_d7_logical_transitions import class4, digest, load

MAPS = (783000, 783001)
CHECKPOINTS = ('2', '11', '54', '57')
RADIUS = 1
L2 = 0.01


@dataclass(frozen=True)
class Cohort:
    control: np.ndarray
    patch_events: np.ndarray
    outcome: np.ndarray  # +1 repairs a raw error, -1 harms a raw correct prediction.
    margin_accepts: np.ndarray


def patch_lookup(layout) -> tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack((layout.detector_time_index, layout.row_index_by_detector,
                              layout.col_index_by_detector)).astype(np.intp)
    offsets = np.array(list(product(range(-RADIUS, RADIUS + 1), repeat=3)))
    padded = np.pad(layout.detector_index_volume, RADIUS, constant_values=-1)
    indices = coords[:, None, :] + offsets[None, :, :] + RADIUS
    lookup = padded[tuple(indices.transpose(2, 0, 1))]
    order = np.lexsort(tuple(coords.T[::-1]))
    return lookup, order


def patch_features(events: np.ndarray, masks: np.ndarray, lookup: np.ndarray,
                   detector_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert masks.any(axis=1).all()
    anchors = detector_order[np.argmax(masks[:, detector_order], axis=1)]
    indices = lookup[anchors]
    valid = indices >= 0
    row = np.arange(len(events))[:, None]
    geometry = np.column_stack((valid, masks[row, indices] & valid))
    observations = np.column_stack((events[row, indices] & valid, (masks & events).sum(axis=1)))
    return geometry, observations


def load_cohort(source: Path, seed: int, hashes: dict) -> Cohort:
    base = source / 'development' / str(seed)
    result = load(base / 'result.json')
    case_root = base / 'g1'
    case = load(case_root / 'case.json')
    assert case['settings']['local_seed'] == result['local_seed'] == seed
    assert case['settings']['zz_factor'] == 0.25
    samples, = (name for name in case['sha256'] if name.endswith('/samples.npz'))
    sample_path = case_root / samples
    sample_hash = digest(sample_path)
    assert sample_hash == case['sha256'][samples]
    hashes[sample_path.relative_to(source).as_posix()] = sample_hash
    with np.load(sample_path) as arrays:
        lookup, detector_order = patch_lookup(build_rectangular_syndrome_layout(arrays))
        events, truth_bits = arrays['detector_events'], arrays['observable_flips']
    checkpoints = result['data']['g1']
    assert set(checkpoints) == set(CHECKPOINTS)
    chunks = []
    for index, model in enumerate(CHECKPOINTS):
        payload = checkpoints[model]
        path = case_root / payload['file']
        payload_hash = digest(path)
        assert payload_hash == payload['sha256']
        hashes[path.relative_to(source).as_posix()] = payload_hash
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays['truth'], truth_bits)
            raw, truth = class4(arrays['d0_mwpm_raw']), class4(truth_bits)
            top, branch, utility, weight, shot_index = (arrays[key] for key in (
                'top1_rows', 'branch_rows', 'utility', 'weights', 'shot_index'))
            predicted = class4(arrays['d0_mwpm_predicted'])
            identity = np.flatnonzero(weight == 0)
            np.testing.assert_array_equal(shot_index[identity], np.arange(len(events)))
            np.testing.assert_array_equal(shot_index[top], np.arange(len(events)))
            np.testing.assert_array_equal(shot_index[branch], np.arange(len(events)))
            np.testing.assert_array_equal(predicted[identity], raw)
            cohort = np.flatnonzero((weight[top] > 0) & ((raw ^ predicted[top]) == 2))
            chosen = top[cohort]
            margin_accepts = branch[cohort] == chosen
            assert np.all(margin_accepts | (branch[cohort] == identity[cohort]))
            assert not arrays['d0_mwpm_failure'][chosen].any()
            masks = np.unpackbits(arrays['packed_masks'][chosen], axis=1, count=events.shape[1])
            np.testing.assert_array_equal(masks.sum(axis=1), weight[chosen])

        geometry, patch = patch_features(events[cohort], masks, lookup, detector_order)
        gap = utility[chosen] - utility[identity[cohort]]
        checkpoint_bits = np.broadcast_to(np.eye(len(CHECKPOINTS))[index], (len(cohort), len(CHECKPOINTS)))
        controls = np.column_stack((gap, weight[chosen], np.eye(4)[raw[cohort]], checkpoint_bits, geometry))
        outcome = (predicted[chosen] == truth[cohort]).astype(np.int8) - (raw[cohort] == truth[cohort])
        chunks.append(Cohort(controls, patch, outcome, margin_accepts))
    return Cohort(
        np.concatenate([chunk.control for chunk in chunks]),
        np.concatenate([chunk.patch_events for chunk in chunks]),
        np.concatenate([chunk.outcome for chunk in chunks]),
        np.concatenate([chunk.margin_accepts for chunk in chunks]),
    )


def fit_and_score(train: np.ndarray, outcomes: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pivotal = outcomes != 0
    center = train[pivotal].mean(axis=0)
    scale = np.maximum(train[pivotal].std(axis=0), 1.0)
    x = np.column_stack((np.ones(len(train)), (train - center) / scale))
    fit_x, y = x[pivotal], outcomes[pivotal] == 1

    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = fit_x @ beta
        penalty = np.r_[0.0, beta[1:]]
        cost = np.mean(np.logaddexp(0.0, logits) - y * logits) + L2 / 2 * (penalty @ penalty)
        grad = fit_x.T @ (expit(logits) - y) / len(y) + L2 * penalty
        return float(cost), grad

    result = minimize(loss, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True)
    if not result.success:
        raise RuntimeError(f'Probe did not converge: {result.message}')
    test_x = np.column_stack((np.ones(len(test)), (test - center) / scale))
    return x @ result.x, test_x @ result.x


def recovery_cutoff(scores: np.ndarray, outcome: np.ndarray, target: int) -> float:
    # A threshold admits all ties, including ties between checkpoints on the same shot.
    repairs = np.sort(scores[outcome == 1])[::-1]
    if not 1 <= target <= len(repairs):
        raise ValueError('Recovery target must be attainable and positive')
    return float(repairs[target - 1])


def counts(outcomes: np.ndarray, accepted: np.ndarray) -> dict:
    return {'repairs': int(np.sum(accepted & (outcomes == 1))),
            'harms': int(np.sum(accepted & (outcomes == -1))),
            'both_wrong': int(np.sum(accepted & (outcomes == 0)))}


def compare_fold(train: Cohort, test: Cohort) -> dict:
    margin = counts(test.outcome, test.margin_accepts)
    train_target = counts(train.outcome, train.margin_accepts)['repairs']
    result = {'all_z_edits': counts(test.outcome, np.ones(len(test.outcome), dtype=bool)),
              'stored_margin': margin, 'training_recovery_target': train_target, 'probes': {}}
    for name, train_x, test_x in (
        ('control', train.control, test.control),
        ('local_events', np.column_stack((train.control, train.patch_events)),
         np.column_stack((test.control, test.patch_events))),
    ):
        train_scores, test_scores = fit_and_score(train_x, train.outcome, test_x)
        frozen = recovery_cutoff(train_scores, train.outcome, train_target)
        # The matched test-fold cutoff describes the ranking curve only.
        matched = recovery_cutoff(test_scores, test.outcome, margin['repairs'])
        pivotal = test.outcome != 0
        positive = test.outcome[pivotal] == 1
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        ranks = rankdata(test_scores[pivotal])
        auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        result['probes'][name] = {
            'auc_repair_vs_harm': float(auc),
            'matched_repair_counts': counts(test.outcome, test_scores >= matched),
            'train_locked_cutoff': frozen,
            'train_locked_counts': counts(test.outcome, test_scores >= frozen),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    hashes = {}
    cohorts = [load_cohort(args.source, seed, hashes) for seed in MAPS]
    summary = {
        'development_maps': MAPS, 'model_seeds': CHECKPOINTS, 'patch_radius': RADIUS, 'l2': L2,
        'runner_sha256': digest(Path(__file__)), 'input_sha256': hashes, 'conda_prefix': sys.prefix,
        'folds': {str(seed): {'fit_map': MAPS[1 - i], **compare_fold(cohorts[1 - i], cohorts[i])}
                  for i, seed in enumerate(MAPS)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(json.dumps(summary['folds'], indent=2))


if __name__ == '__main__':
    main()
