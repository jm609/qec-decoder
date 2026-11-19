"""Evaluate the frozen paired d7 selector on new strong-ZZ maps."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_d7_paired_selector as paired
from tools import run_predecoder_noise_shift as shift

rv, sedp = shift.rv, shift.sedp
SOURCE = ROOT / 'artifacts/eval/noise_shift_d7/20261115_retrained'
DEVELOPMENT = ROOT / 'artifacts/eval/d7_paired_selector/20261116/summary.json'
MAPS = tuple(range(786100, 786108))
SHOTS = 2048
BOOTSTRAP_SEED = 1789399
BOOTSTRAP_REPS = 20000
ROUTES = ('top1', 'margin', 'fixed', 'free')


def freeze_head(out: Path, development: dict) -> dict:
    lock = out / 'calibration.json'
    protocol_hash = rv.digest(out / 'protocol.json')
    if lock.exists():
        frozen = rv.load(lock)
        assert frozen['protocol_sha256'] == protocol_hash
        return frozen

    assert not (out / 'evaluation').exists()
    cases = [case for seed in paired.probe.MAPS for case in paired.load_map(SOURCE, seed, development['input_sha256'])]
    labels = np.concatenate([case.outcomes() for case in cases])
    features = np.concatenate([paired.features(case, local_events=True) for case in cases])
    frozen = {
        'protocol_sha256': protocol_hash, 'development_input_sha256': development['input_sha256'],
        'training_pairs': len(labels), 'pivotal_pairs': int(np.count_nonzero(labels)),
        'cutoff': 0.0, 'head': paired.fit_head(features, labels),
    }
    rv.save(lock, frozen)
    return frozen


def score_payload(path: Path, raw_data: dict, lookup: np.ndarray, order: np.ndarray,
                  model_index: int, head: dict, cutoff: float) -> dict:
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['truth'], raw_data['observable_flips'])
        labels = saved['truth']
        arrays = {name: saved[name] for name in saved.files if name != 'truth'}
    # Construct the choices with dummy labels. Restore truth only for scoring them.
    arrays['truth'] = np.zeros_like(labels)
    case = paired.make_cases(arrays, raw_data['detector_events'], lookup, order, model_index)
    logits = paired.head_score(paired.features(case, local_events=True), head)
    fixed, free = paired.select_rows(case, logits >= cutoff)
    case = replace(case, truth=paired.class4(labels))
    routes = dict(zip(ROUTES, (case.top, case.margin, fixed, free), strict=True))
    return {
        'paired_shots': len(case.pair_shots),
        'top1_z_shots': int(np.count_nonzero((case.weight[case.top] != 0) & ((case.raw ^ case.prediction[case.top]) == 2))),
        'accepted_z': int(np.count_nonzero(logits >= cutoff)),
        'fixed_free_prediction_differences': int(np.count_nonzero(case.prediction[fixed] != case.prediction[free])),
        'routes': {route: paired.count_routes([case], [rows]) for route, rows in routes.items()},
    }


def evaluate_map(out: Path, seed: int, protocol: dict, frozen: dict, replay: bool) -> dict:
    sedp.torch.set_num_threads(2)
    path = out / 'evaluation' / str(seed)
    strength = path / 'g1'
    if replay:
        for g in ('g0', 'g1'):
            assert (path / g / 'case.json').is_file()
        assert (path / 'result.json').is_file()
    targets, raw, validation = shift.make_pair(path, seed, protocol['shots_per_map'])
    assert validation['sampler_seeds'] == protocol['sampler_seeds'][str(seed)]
    with np.load(raw[1] / 'samples.npz') as samples:
        lookup, order = paired.probe.patch_lookup(paired.build_rectangular_syndrome_layout(samples))
        raw_data = {key: samples[key] for key in ('detector_events', 'observable_flips')}

    files = {model: strength / f'predictions_seed{model}.json' for model in paired.probe.CHECKPOINTS}
    missing = [int(model) for model, file in files.items() if not file.exists()]
    if missing:
        assert not replay and not (path / 'result.json').exists()
        decoder = shift.open_decoders(raw[:1])[shift.D0_MWPM]
        entry = sedp._prepare_edit_family(targets[1], fill_value=-0.5, max_shots=None)
        entry, context = shift.with_context(entry, rv.load(raw[0] / 'metadata.json'), protocol['context_event_fraction'])
        predictors = rv.checkpoints(shift.REVALIDATION, 7, missing)
        for model, checkpoint in predictors.items():
            file = strength / f'predictions_seed{model}.npz'
            payload = shift.write_case_predictions(file, entry, {shift.D0_MWPM: decoder}, rv.load_selector(checkpoint), adopt=False)
            rv.save(files[str(model)], {**payload, 'context': context})

    records = {}
    for index, (model, file) in enumerate(files.items()):
        payload = rv.load(file)
        prediction = strength / payload['file']
        assert rv.digest(prediction) == payload['sha256']
        metrics = score_payload(prediction, raw_data, lookup, order, index, frozen['head'], frozen['cutoff'])
        for route, previous in (('top1', 'top1'), ('margin', 'branch')):
            source = payload['cells'][shift.D0_MWPM][previous]
            result = metrics['routes'][route]
            assert (result['corrections'], result['harms'], result['edits']) == (
                source['improved'], source['harmed'], source['applied_edits'])
        records[model] = {**metrics, 'sha256': payload['sha256'], 'context': payload['context'],
                          'candidate_rows': payload['cells'][shift.D0_MWPM]['candidate_rows']}

    record = {'local_seed': seed, 'models': records, 'calibration_sha256': rv.digest(out / 'calibration.json')}
    marker = path / 'result.json'
    if marker.exists():
        assert rv.load(marker) == record
    else:
        rv.save(marker, record)
    return record


def summarize(results: list[dict], calibration_hash: str) -> dict:
    maps = {str(row['local_seed']): list(row['models'].values()) for row in results}
    flat = [model for models in maps.values() for model in models]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(maps), (BOOTSTRAP_REPS, len(maps)))
    effects = {
        route: np.array([np.mean([model['routes'][route]['gain_pp'] for model in models]) for models in maps.values()])
        for route in ROUTES
    }
    report = {key: sum(row[key] for row in flat) for key in (
        'candidate_rows', 'top1_z_shots', 'paired_shots', 'accepted_z', 'fixed_free_prediction_differences')}
    report.update(calibration_sha256=calibration_hash, routes={})
    for route, values in effects.items():
        totals = {
            key: sum(row['routes'][route][key] for row in flat)
            for key in ('shots', 'edits', 'corrections', 'harms', 'changed_rows_vs_top1')
        }
        totals['map_gains_pp'] = dict(zip(maps, map(float, values), strict=True))
        for comparator, base in (('raw', 0.0), ('top1', effects['top1']), ('margin', effects['margin'])):
            delta = values - base
            totals[f'gain_vs_{comparator}_pp'] = float(delta.mean())
            totals[f'interval_vs_{comparator}_95_pp'] = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
        report['routes'][route] = totals
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--replay', action='store_true', help='Recompute the results from saved candidate predictions')
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    out = args.out.resolve()
    development = rv.load(DEVELOPMENT)
    dev = rv.load(SOURCE / 'development.json')
    assert development['runner_sha256'] == rv.digest(Path(paired.__file__))
    assert development['probe_source_sha256'] == rv.digest(Path(paired.probe.__file__))
    assert dev['source_sha256'] == shift.source_hashes()
    assert dev['checkpoint_sha256'] == shift.panel_hashes()
    plan = shift.SEED_PLAN
    used_maps = set(plan['development_local_seeds'] + plan['evaluation_local_seeds'])
    for run in ('d7_matching_rerank', 'd7_transition_rerank'):
        used_maps.update(rv.load(ROOT / 'artifacts/eval' / run / '20261116/protocol.json')['evaluation_maps'])
    assert not set(MAPS).intersection(used_maps)
    protocol = {
        'primary': 'G1 D0 MWPM free paired selector minus raw PyMatching; map-level paired bootstrap',
        'secondary': 'Edit-shot-fixed rerank and comparisons with original top1 and margin',
        'rule': 'Detector-event head, zero log-odds threshold, trained on both frozen development maps',
        'positive_evidence': 'Primary 95% bootstrap interval lies entirely above zero',
        'development_maps': list(paired.probe.MAPS), 'evaluation_maps': list(MAPS), 'shots_per_map': SHOTS,
        'sampler_seeds': {str(m): [plan['sampler_base'] + 2 * (m - plan['development_local_seeds'][0]) + g
                                  for g in (0, 1)] for m in MAPS},
        'development_summary_sha256': rv.digest(DEVELOPMENT), 'checkpoint_sha256': dev['checkpoint_sha256'],
        'context_event_fraction': dev['context_event_fraction'], 'runner_sha256': rv.digest(Path(__file__)),
        'conda_prefix': sys.prefix, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_repetitions': BOOTSTRAP_REPS,
        'versions': {name: version(name) for name in ('stim', 'pymatching', 'torch', 'numpy', 'scipy')},
    }
    if args.replay:
        assert (out / 'summary.json').is_file() and (out / 'calibration.json').is_file()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'protocol.json'
    if marker.exists():
        assert rv.load(marker) == protocol
    else:
        rv.save(marker, protocol)
    frozen = freeze_head(out, development)
    calibration_hash = rv.digest(out / 'calibration.json')
    print(f'Frozen head {calibration_hash}; {len(MAPS)} maps x {SHOTS} shots', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(evaluate_map, out, seed, protocol, frozen, args.replay) for seed in MAPS]
        for task in as_completed(tasks):
            record = task.result()
            results.append(record)
            print(f'Map {record["local_seed"]} verified ({len(results)}/{len(MAPS)})', flush=True)
    summary = summarize(sorted(results, key=lambda x: x['local_seed']), calibration_hash)
    result_file = out / 'summary.json'
    if result_file.exists():
        assert rv.load(result_file) == summary
    else:
        rv.save(result_file, summary)
    print(f'Done: {result_file}', flush=True)


if __name__ == '__main__':
    main()
