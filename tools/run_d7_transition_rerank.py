"""Calibrate logical-transition scores and test on fresh d7 strong-ZZ maps."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_d7_logical_transitions import class4
from tools.run_d7_matching_rerank import MAPS as MATCHING_MAPS, PREVIOUS, rerank_rows
from tools import run_predecoder_noise_shift as shift

rv, sedp = shift.rv, shift.sedp
MAPS = tuple(range(785100, 785108))
SHOTS = 1024
L2 = 0.01
INTERCEPT_CLASSES = 3
BOOTSTRAP_SEED = 1787399
BOOTSTRAP_REPS = 20000
ROUTES = ('top1', 'fixed', 'free')


def candidate_classes(arrays) -> tuple[np.ndarray, np.ndarray]:
    raw = class4(arrays['d0_mwpm_raw'])
    predicted = class4(arrays['d0_mwpm_predicted'])
    return raw, predicted ^ raw[arrays['shot_index']]


def development_rows(arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, transition = candidate_classes(arrays)
    shot, utility = arrays['shot_index'], arrays['utility'].astype(np.float64)
    identity = np.flatnonzero(arrays['weights'] == 0)
    np.testing.assert_array_equal(shot[identity], np.arange(len(raw)))
    np.testing.assert_array_equal(transition[identity], 0)
    scores = np.full((len(raw), 4), -np.inf)
    np.maximum.at(scores, (shot, transition), utility - utility[identity][shot])
    return raw, scores, raw ^ class4(arrays['truth'])


def fit_scores(raw: np.ndarray, scores: np.ndarray, target: np.ndarray) -> dict:
    reachable = np.isfinite(scores[np.arange(len(raw)), target])
    raw, scores, target = raw[reachable], scores[reachable], target[reachable]
    row = np.arange(len(target))
    initial = np.zeros(1 + 4 * INTERCEPT_CLASSES)
    initial[0] = 1.0

    def loss(params: np.ndarray) -> float:
        bias = np.pad(params[1:].reshape(4, INTERCEPT_CLASSES), ((0, 0), (1, 0)))
        adjusted = params[0] * scores + bias[raw]
        cross_entropy = (logsumexp(adjusted, axis=1) - adjusted[row, target]).mean()
        return float(cross_entropy + L2 / 2 * np.sum((params - initial) ** 2))

    result = minimize(loss, initial, method='L-BFGS-B', bounds=[(1e-6, None)] + [(None, None)] * (len(initial) - 1))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Transition calibration did not converge: {result.message}')
    return {
        'utility_scale': float(result.x[0]),
        'transition_bias_by_raw_class': np.pad(result.x[1:].reshape(4, INTERCEPT_CLASSES), ((0, 0), (1, 0))).tolist(),
        'training_pairs': len(target),
        'unreachable_pairs': int((~reachable).sum()),
        'initial_objective': loss(initial),
        'fitted_objective': float(result.fun),
    }


def select_rows(arrays, calibration: dict) -> dict[str, np.ndarray]:
    raw, transition = candidate_classes(arrays)
    shot, weight, top1 = (arrays[key] for key in ('shot_index', 'weights', 'top1_rows'))
    bias = np.asarray(calibration['transition_bias_by_raw_class'])[raw[shot], transition]
    utility = calibration['utility_scale'] * arrays['utility'].astype(np.float64)
    return {'top1': top1, **{
        key: rerank_rows(utility, bias, shot, weight, top1, 1.0, fixed_edit_rate=key == 'fixed')
        for key in ('fixed', 'free')
    }}


def compare(arrays, calibration: dict) -> dict:
    rows = select_rows(arrays, calibration)
    raw, candidates, truth = (arrays[key] for key in ('d0_mwpm_raw', 'd0_mwpm_predicted', 'truth'))
    raw_correct = np.all(raw == truth, axis=1)
    original_correct = np.all(candidates[rows['top1']] == truth, axis=1)
    metrics = {}
    for key, selected in rows.items():
        predictions = candidates[selected]
        correct = np.all(predictions == truth, axis=1)
        counts = rv.paired_counts(raw, predictions, truth)
        counts.update(
            retained_corrections=int((~raw_correct & original_correct & correct).sum()),
            lost_corrections=int((~raw_correct & original_correct & ~correct).sum()),
            new_corrections=int((~raw_correct & ~original_correct & correct).sum()),
            avoided_harms=int((raw_correct & ~original_correct & correct).sum()),
            new_harms=int((raw_correct & original_correct & ~correct).sum()),
            edited_shots=int((arrays['weights'][selected] != 0).sum()),
            changed_rows=int((selected != rows['top1']).sum()),
        )
        metrics[key] = counts
    return metrics


def calibrate(out: Path, protocol: dict) -> dict:
    batches, hashes = [], {}
    for seed in protocol['development_maps']:
        base = PREVIOUS / 'development' / str(seed)
        record = rv.load(base / 'result.json')['data']['g1']
        for model in shift.PANEL:
            payload = record[str(model)]
            path = base / 'g1' / payload['file']
            digest = rv.digest(path)
            assert digest == payload['sha256']
            hashes[path.relative_to(ROOT).as_posix()] = digest
            with np.load(path) as arrays:
                batches.append(development_rows(arrays))
    frozen = fit_scores(*(np.concatenate(values) for values in zip(*batches)))
    frozen.update(protocol_sha256=rv.digest(out / 'protocol.json'), development_predictions_sha256=hashes)
    rv.save(out / 'calibration.json', frozen)
    return frozen


def evaluate_map(out: Path, seed: int, protocol: dict, calibration: dict) -> dict:
    sedp.torch.set_num_threads(2)
    path = out / 'evaluation' / str(seed)
    targets, raw, validation = shift.make_pair(path, seed, protocol['shots_per_map'])
    assert validation['sampler_seeds'] == protocol['sampler_seeds'][str(seed)]
    marker = path / 'result.json'
    if marker.exists():
        record = rv.load(marker)
        assert record['local_seed'] == seed
        assert record['calibration_sha256'] == rv.digest(out / 'calibration.json')
        for model in record['models'].values():
            assert rv.digest(path / 'g1' / model['file']) == model['sha256']
        return record

    decoder = shift.open_decoders(raw[:1])[shift.D0_MWPM]
    entry = sedp._prepare_edit_family(targets[1], fill_value=-0.5, max_shots=None)
    entry, context = shift.with_context(entry, rv.load(raw[0] / 'metadata.json'), protocol['context_event_fraction'])
    models = {}
    for seed_model, checkpoint in rv.checkpoints(shift.REVALIDATION, 7, shift.PANEL).items():
        file = path / 'g1' / f'predictions_seed{seed_model}.npz'
        payload = shift.write_case_predictions(file, entry, {shift.D0_MWPM: decoder}, rv.load_selector(checkpoint), adopt=False)
        with np.load(file) as arrays:
            routes = compare(arrays, calibration)
        original = payload['cells'][shift.D0_MWPM]['top1']
        for key in ('selected_correct', 'improved', 'harmed'):
            assert routes['top1'][key] == original[key]
        models[str(seed_model)] = {**payload, 'routes': routes}
    record = {'local_seed': seed, 'validation': validation, 'context': context, 'models': models,
              'calibration_sha256': rv.digest(out / 'calibration.json')}
    rv.save(marker, record)
    return record


def summarize(results: list[dict], calibration_sha256: str) -> dict:
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(results), (BOOTSTRAP_REPS, len(results)))
    maps = {str(item['local_seed']): list(item['models'].values()) for item in results}
    models = [model for group in maps.values() for model in group]
    summary = {'calibration_sha256': calibration_sha256, 'candidate_rows': sum(
        m['cells'][shift.D0_MWPM]['candidate_rows'] for m in models), 'candidate_decode_fallbacks': sum(
        m['cells'][shift.D0_MWPM]['candidate_fallback_rows'] for m in models), 'routes': {}}
    for route in ROUTES:
        keys = models[0]['routes'][route].keys() - {'gain_pp'}
        total = {key: sum(model['routes'][route][key] for model in models) for key in sorted(keys)}
        gain = np.array([np.mean([m['routes'][route]['gain_pp'] for m in group]) for group in maps.values()])
        base = np.array([np.mean([m['routes']['top1']['gain_pp'] for m in group]) for group in maps.values()])
        delta = gain - base
        total.update(gain_vs_raw_pp=float(gain.mean()), gain_vs_top1_pp=float(delta.mean()),
                     interval_vs_raw_95_pp=np.quantile(gain[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                     interval_vs_top1_95_pp=np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                     map_effects_vs_top1_pp=dict(zip(maps, map(float, delta))))
        summary['routes'][route] = total
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    out = args.out.resolve()
    dev = rv.load(PREVIOUS / 'development.json')
    assert dev['source_sha256'] == shift.source_hashes()
    assert dev['checkpoint_sha256'] == shift.panel_hashes()
    plan = shift.SEED_PLAN
    assert not set(MAPS).intersection((*MATCHING_MAPS, *plan['development_local_seeds'], *plan['evaluation_local_seeds']))
    protocol = {
        'primary': 'G1 D0 MWPM, edit-shot-fixed rerank minus original top1, paired by map',
        'secondary': 'Allow no-edit; compare to top1 and raw PyMatching',
        'readout': 'Net gain, harms and retained/lost/new corrections; report raw comparison separately from ranking contrast',
        'scoring': 'Positive shared utility scale plus bias by raw class and XOR transition; no-change bias fixed at zero',
        'fit': 'L2-regularized conditional cross entropy over available logical classes, dev G1 only',
        'class_order': ['I', 'X', 'Z', 'Y'], 'l2': L2, 'development_maps': plan['development_local_seeds'],
        'evaluation_maps': list(MAPS), 'shots_per_map': SHOTS,
        'sampler_seeds': {str(m): [plan['sampler_base'] + 2 * (m - plan['development_local_seeds'][0]) + g
                                  for g in (0, 1)] for m in MAPS},
        'context_event_fraction': dev['context_event_fraction'], 'checkpoint_sha256': dev['checkpoint_sha256'],
        'development_sha256': rv.digest(PREVIOUS / 'development.json'),
        'runner_sha256': rv.digest(Path(__file__)), 'conda_prefix': os.environ['CONDA_PREFIX'],
        'helper_sha256': {name: rv.digest(ROOT / 'tools' / name) for name in
                          ('analyze_d7_logical_transitions.py', 'run_d7_matching_rerank.py')},
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_repetitions': BOOTSTRAP_REPS,
        'versions': {name: version(name) for name in ('torch', 'stim', 'pymatching', 'numpy', 'scipy')},
    }
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'protocol.json'
    if marker.exists():
        assert rv.load(marker) == protocol
    else:
        rv.save(marker, protocol)
    lock = out / 'calibration.json'
    frozen = rv.load(lock) if lock.exists() else calibrate(out, protocol)
    assert frozen['protocol_sha256'] == rv.digest(marker)
    calibration_hash = rv.digest(lock)
    print(f'Calibration frozen: {calibration_hash}', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(evaluate_map, out, seed, protocol, frozen) for seed in MAPS]
        for task in as_completed(tasks):
            record = task.result()
            results.append(record)
            print(f'Map {record["local_seed"]} verified ({len(results)}/{len(MAPS)})', flush=True)
    summary = summarize(sorted(results, key=lambda item: item['local_seed']), calibration_hash)
    report = out / 'summary.json'
    if report.exists():
        assert rv.load(report) == summary
    else:
        rv.save(report, summary)
    print(f'Done: {report}', flush=True)


if __name__ == '__main__':
    main()
