"""Diagnose Z-transition ranking from cross-validated and frozen d7 scores."""

import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_adoption as adoption

rv = adoption.rv
Z = 2
TOP_FRACTIONS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.10)
OUT = ROOT / 'artifacts/eval/d7_adoption_z_scores/20261117'


def score_rows(raw, truth, support, probabilities, selected):
    assert probabilities.shape == support.shape
    assert probabilities.shape[1] == len(raw) and truth.shape == raw.shape
    return dict(raw=raw, truth=truth, available=support[..., Z],
                score=probabilities[..., Z] - probabilities[..., 0], selected=selected)


def z_outcomes(raw, truth):
    return ((raw ^ Z) == truth).astype(np.int8) - (raw == truth)


def counts(outcome, choose):
    selected = outcome[choose]
    improved = int(np.count_nonzero(selected == 1))
    harmed = int(np.count_nonzero(selected == -1))
    net = improved - harmed
    values = (outcome * choose).reshape(outcome.shape[0], -1).mean(axis=1) * 100
    center = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(changed=len(selected), improved=improved, harmed=harmed,
                neutral=len(selected) - improved - harmed, net=net,
                gain_pp=center, interval_95_pp=[center-half, center+half],
                map_signs=[int(np.count_nonzero(values > 0)), int(np.count_nonzero(values == 0)),
                           int(np.count_nonzero(values < 0))])


def diagnose(rows, thresholds):
    available = np.stack([row['available'] for row in rows])
    score = np.stack([row['score'] for row in rows])
    outcome = np.broadcast_to(np.stack([z_outcomes(row['raw'], row['truth']) for row in rows])[:, None, :],
                              available.shape)
    actual = np.stack([(row['selected'] ^ row['raw']) == Z for row in rows])
    assert not np.any(actual & ~available)
    pivotal = available & (outcome != 0)
    positive = outcome[pivotal] > 0
    rank = rankdata(score[pivotal], method='average')
    pos, neg = int(positive.sum()), int((~positive).sum())
    auc = (rank[positive].sum() - pos*(pos+1)/2) / (pos * neg)
    thresholds_result = {str(value): counts(outcome, available & (score > value)) for value in thresholds}
    quantiles = {}
    descending = np.sort(score[available])[::-1]
    for fraction in TOP_FRACTIONS:
        count = int(np.ceil(fraction * available.size))
        cutoff = float(descending[count-1])
        quantiles[str(fraction)] = dict(score_boundary=cutoff,
                                       **counts(outcome, available & (score >= cutoff)))
    return dict(pairs=available.size, z_available=int(available.sum()), possible_repairs=pos,
                harmful_if_flipped=neg, pivotal_auc=float(auc), frozen_or_oof_policy=counts(outcome, actual),
                by_threshold=thresholds_result, by_top_fraction_of_all_pairs=quantiles)


def load_independent(report, protocol, input_hashes):
    rows = {}
    for result in report['maps']:
        assert result['protocol_sha256'] == input_hashes['protocol']
        seed = str(result['local_seed'])
        path = adoption.OUT/'evaluation'/seed/'predictions.npz'
        assert rv.digest(path) == result['sha256']
        input_hashes[path.relative_to(ROOT).as_posix()] = result['sha256']
        with np.load(path) as data:
            support = data['class_rows'] >= 0
            probability = data['joint_proba']
            selected = data['raw'] ^ adoption.decide(probability, support, protocol['threshold']['joint'])
            np.testing.assert_array_equal(selected, data['joint'])
            assert adoption.counts(data['raw'], data['truth'], selected) == result['methods']['joint']
            rows[seed] = score_rows(data['raw'], data['truth'], support, probability, selected)
    assert len(rows) == 24
    return rows


def load_development(development, protocol):
    maps = {}
    for path, expected in development['inputs_sha256'].items():
        file = ROOT/path
        assert rv.digest(file) == expected
        with np.load(file) as replay:
            maps[file.parent.name] = adoption.feature_rows(file.parent, replay=replay)
    assert maps.keys() == development['folds'].keys()
    rows = {}
    for seed, held in maps.items():
        heads = adoption.fit([m for name, m in maps.items() if name != seed])
        prob, support = adoption.predict(held, heads)['joint']
        raw, truth = held['raw'], held['truth']
        for index, threshold in enumerate(development['threshold_grid']):
            selected = raw ^ adoption.decide(prob, support, threshold)
            expected = development['folds'][seed]['joint'][index]
            assert adoption.counts(raw, truth, selected) == expected, (seed, threshold)
        selected = raw ^ adoption.decide(prob, support, protocol['threshold']['joint'])
        rows[seed] = score_rows(raw, truth, support, prob, selected)
        print(f'Development fold {seed}: original threshold-grid counts reproduced', flush=True)
    return rows


def main():
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        raise RuntimeError('Activate Anaconda fp first')
    adoption.sedp.torch.set_num_threads(adoption.gen.THREADS)
    paths = {name: adoption.OUT/f'{name}.json' for name in ('protocol','summary','development')}
    hashes = {name: rv.digest(path) for name, path in paths.items()}
    report, protocol, development = [rv.load(paths[key]) for key in ('summary','protocol','development')]
    assert report['protocol_sha256'] == hashes['protocol']
    assert protocol['sources'][paths['development'].relative_to(ROOT).as_posix()] == hashes['development']
    assert protocol['threshold'] == development['threshold']
    for source, digest in protocol['sources'].items():
        assert rv.digest(ROOT/source) == digest, source
    data = {'independent': load_independent(report, protocol, hashes),
            'development': load_development(development, protocol)}
    analysis = {name: diagnose(list(rows.values()), development['threshold_grid'])
                for name, rows in data.items()}
    OUT.mkdir(parents=True, exist_ok=False)
    for name, rows in data.items():
        with (OUT/f'{name}.npz').open('xb') as stream:
            np.savez_compressed(stream, **{f'{seed}_{field}': array for seed, row in rows.items()
                                          for field, array in row.items()})
    rv.save(OUT/'summary.json', dict(input_sha256=hashes, code_sha256=rv.digest(Path(__file__)),
            scores_sha256={name: rv.digest(OUT/f'{name}.npz') for name in data}, method=dict(
                top_fraction_denominator='all model-shot pairs, ties included at rank boundary',
                fixed_threshold='strictly greater than threshold; Z candidate required',
                intervals='descriptive 95% map-level t intervals; no threshold selected'),
            **analysis))
    for name, result in analysis.items():
        print(name, 'AUC', result['pivotal_auc'], 'policy', result['frozen_or_oof_policy'], flush=True)


if __name__ == '__main__':
    main()
