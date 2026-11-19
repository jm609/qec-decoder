"""Evaluate the frozen paired selector at G0 and measure its CPU inference cost."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_paired_selector as strong

paired, shift, rv, sedp = strong.paired, strong.shift, strong.rv, strong.sedp
STRONG = ROOT / 'artifacts/eval/d7_paired_selector_independent/20261116'
BENCH_SHOTS = shift.BATCH
BENCH_REPS = 3
BENCH_MODEL = 2


class TimedDecoder(shift.Decoder):
    matching_seconds = 0.0

    def decode_batch(self, events: np.ndarray) -> np.ndarray:
        started = perf_counter()
        try:
            return super().decode_batch(events)
        finally:
            self.matching_seconds += perf_counter() - started

    def decode(self, event: np.ndarray) -> np.ndarray:
        started = perf_counter()
        try:
            return super().decode(event)
        finally:
            self.matching_seconds += perf_counter() - started


def decoded_from_transition(bundle, raw: np.ndarray) -> np.ndarray:
    offsets = sedp._transition_feature_offsets(candidate_feature_dim=bundle.candidate_features.shape[1])
    delta = bundle.candidate_features[:, offsets['obs_delta']]
    if np.any((delta != 0) & (delta != 1)) or bundle.decode_fallback_count:
        raise ValueError('Expected binary transition features from successful D0 candidate matching')
    return raw[bundle.shot_indices] ^ delta.astype(np.uint8)


def infer(entry, matching, predictor: tuple, model_index: int, lookup: np.ndarray,
          order: np.ndarray, frozen: dict) -> tuple[dict, dict, dict]:
    if predictor[2]['selector_target_mode'] != sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM:
        raise ValueError('The frozen selector must encode its candidate matching predictions')
    decoder = TimedDecoder(matching, False)
    events = entry.detector_events
    started = perf_counter()
    raw = decoder.decode_batch(events)
    raw_seconds = perf_counter() - started
    raw_counts = decoder.counts()
    entry = replace(entry, matching=decoder, baseline_predicted_observables=raw)

    chunks = {}
    offset = 0
    for lo in range(0, len(raw), shift.BATCH):
        hi = min(lo + shift.BATCH, len(raw))
        bundle, utility, margin = rv.score_bundle(entry, lo, hi, predictor)
        _, top = shift.top_rows(bundle, utility)
        branch = shift.selected_rows(bundle, utility, margin)
        preds = decoded_from_transition(bundle, raw[lo:hi])
        payload = {'shot_index': lo + bundle.shot_indices, 'weights': bundle.candidate_edit_weight,
                   'packed_masks': np.packbits(bundle.candidate_edit_mask, axis=1), 'utility': utility,
                   'top1_rows': top + offset, 'branch_rows': branch + offset, 'd0_mwpm_predicted': preds}
        for key, value in payload.items():
            chunks.setdefault(key, []).append(value)
        offset += len(utility)

    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    arrays.update(d0_mwpm_raw=raw, truth=np.zeros_like(raw), d0_mwpm_failure=np.zeros(offset, dtype=bool))
    case = paired.make_cases(arrays, events, lookup, order, model_index)
    log_odds = paired.head_score(paired.features(case, local_events=True), frozen['head'])
    fixed, free = paired.select_rows(case, log_odds >= frozen['cutoff'])
    total_seconds = perf_counter() - started
    arrays.update(truth=entry.observable_flips, fixed_rows=fixed, free_rows=free)
    timing = {'raw_seconds': raw_seconds, 'selector_seconds': total_seconds, 'matching_seconds': decoder.matching_seconds,
              'raw_counts': raw_counts, 'selector_counts': decoder.counts()}
    case = replace(case, truth=paired.class4(entry.observable_flips))
    return arrays, score(case, arrays), timing


def score(case, arrays: dict) -> dict:
    fixed, free = arrays['fixed_rows'], arrays['free_rows']
    routes = (case.top, case.margin, fixed, free)
    for rows in routes:
        np.testing.assert_array_equal(arrays['shot_index'][rows], np.arange(len(case.raw)))
    np.testing.assert_array_equal(case.weight[fixed] != 0, case.weight[case.top] != 0)
    return {
        'candidate_rows': len(case.prediction),
        'paired_shots': len(case.pair_shots),
        'top1_z_shots': int(np.count_nonzero((case.weight[case.top] != 0) & ((case.raw ^ case.prediction[case.top]) == 2))),
        'accepted_z': int(np.sum(free[case.pair_shots] == case.top[case.pair_shots])),
        'fixed_free_prediction_differences': int(np.count_nonzero(case.prediction[fixed] != case.prediction[free])),
        'routes': {name: paired.count_routes([case], [rows]) for name, rows in zip(strong.ROUTES, routes, strict=True)},
    }


def load_pair(seed: int, protocol: dict):
    path = STRONG / 'evaluation' / str(seed)
    assert all((path / g / 'case.json').is_file() for g in ('g0', 'g1'))
    targets, raw, validation = shift.make_pair(path, seed, protocol['shots_per_map'])
    assert validation['sampler_seeds'] == protocol['sampler_seeds'][str(seed)]
    return targets, raw, shift.open_decoders(raw[:1])[shift.D0_MWPM].matching


def load_layout(raw: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(raw / 'samples.npz') as samples:
        return paired.probe.patch_lookup(paired.build_rectangular_syndrome_layout(samples))


def evaluate_map(out: Path, seed: int, protocol: dict, frozen: dict, replay: bool) -> dict:
    sedp.torch.set_num_threads(2)
    targets, raw, matching = load_pair(seed, protocol)
    lookup, order = load_layout(raw[0])
    entry = sedp._prepare_edit_family(targets[0], fill_value=-0.5, max_shots=None)
    entry, _ = shift.with_context(entry, rv.load(raw[0] / 'metadata.json'), protocol['context_event_fraction'])
    path = out / 'evaluation' / str(seed)
    path.mkdir(parents=True, exist_ok=True)
    models = {}
    for index, model in enumerate(paired.probe.CHECKPOINTS):
        file = path / f'predictions_seed{model}.npz'
        marker = file.with_suffix('.json')
        if not marker.exists():
            assert not replay
            predictor = rv.load_selector(rv.checkpoints(shift.REVALIDATION, 7, [int(model)])[int(model)])
            arrays, result, timing = infer(entry, matching, predictor, index, lookup, order, frozen)
            np.savez_compressed(file, **arrays)
            rv.save(marker, {**result, 'sha256': rv.digest(file), 'timing': timing})
        record = rv.load(marker)
        assert record['sha256'] == rv.digest(file)
        if replay:
            with np.load(file) as saved:
                arrays = dict(saved)
            case = paired.make_cases(arrays, entry.detector_events, lookup, order, index)
            log_odds = paired.head_score(paired.features(case, local_events=True), frozen['head'])
            fixed, free = paired.select_rows(case, log_odds >= frozen['cutoff'])
            np.testing.assert_array_equal(fixed, arrays['fixed_rows'])
            np.testing.assert_array_equal(free, arrays['free_rows'])
            assert all(record[key] == value for key, value in score(case, arrays).items())
        models[model] = record
    return {'local_seed': seed, 'models': models}


def benchmark(out: Path, protocol: dict, frozen: dict) -> None:
    sedp.torch.set_num_threads(2)
    seed = protocol['evaluation_maps'][0]
    targets, raw, matching = load_pair(seed, protocol)
    predictor = rv.load_selector(rv.checkpoints(shift.REVALIDATION, 7, [BENCH_MODEL])[BENCH_MODEL])
    baseline_context = rv.load(raw[0] / 'metadata.json')
    results = {}
    for index, strength in enumerate(('g0', 'g1')):
        entry = sedp._prepare_edit_family(targets[index], fill_value=-0.5, max_shots=BENCH_SHOTS)
        entry, _ = shift.with_context(entry, baseline_context, protocol['context_event_fraction'])
        lookup, order = load_layout(raw[index])
        observations = []
        for rep in range(BENCH_REPS + 1):
            arr, _, timing = infer(entry, matching, predictor, 0, lookup, order, frozen)
            if rep:
                observations.append(timing)
        base = STRONG / 'evaluation' / str(seed) / 'g1' if strength == 'g1' else out / 'evaluation' / str(seed)
        prediction = base / f'predictions_seed{BENCH_MODEL}.npz'
        assert rv.digest(prediction) == rv.load(prediction.with_suffix('.json'))['sha256']
        with np.load(prediction) as saved:
            n = len(arr['utility'])
            for key in ('utility', 'weights', 'shot_index', 'packed_masks', 'd0_mwpm_predicted'):
                np.testing.assert_array_equal(arr[key], saved[key][:n])
            for key in ('top1_rows', 'branch_rows', 'd0_mwpm_raw'):
                np.testing.assert_array_equal(arr[key], saved[key][:BENCH_SHOTS])
        raw_median = float(np.median([row['raw_seconds'] for row in observations]))
        full_median = float(np.median([row['selector_seconds'] for row in observations]))
        results[strength] = {'raw_median_seconds': raw_median, 'selector_median_seconds': full_median,
                             'selector_over_raw': full_median / raw_median, 'repetitions': observations}
        print(f'{strength} timing complete', flush=True)
    rv.save(out / 'benchmark.json', {'map': seed, 'checkpoint': BENCH_MODEL, 'shots': BENCH_SHOTS, 'cpu_threads': 2,
                                     'cpu': os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                                     'reference_predictions_equal': True, 'strengths': results})


def summarize(results: list[dict], protocol: dict, frozen_hash: str) -> dict:
    summary = strong.summarize(results, frozen_hash)
    g1 = rv.load(STRONG / 'summary.json')['routes']['free']['map_gains_pp']
    g0 = summary['routes']['free']['map_gains_pp']
    delta = np.array([g1[str(seed)] - g0[str(seed)] for seed in protocol['evaluation_maps']])
    draws = np.random.default_rng(strong.BOOTSTRAP_SEED).integers(0, len(delta), (strong.BOOTSTRAP_REPS, len(delta)))
    summary['g1_minus_g0_pp'] = float(delta.mean())
    summary['g1_minus_g0_interval_95_pp'] = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
    timings = [model['timing'] for row in results for model in row['models'].values()]
    summary['matching_counts'] = {name: {key: sum(timing[name][key] for timing in timings)
                                         for key in ('batch_calls', 'single_calls', 'rows')}
                                  for name in ('raw_counts', 'selector_counts')}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'replay', 'benchmark'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    protocol = rv.load(STRONG / 'protocol.json')
    frozen = rv.load(STRONG / 'calibration.json')
    frozen_hash = rv.digest(STRONG / 'calibration.json')
    assert frozen['protocol_sha256'] == rv.digest(STRONG / 'protocol.json')
    assert frozen_hash == rv.load(STRONG / 'summary.json')['calibration_sha256'] and frozen['cutoff'] == 0.0
    assert protocol['runner_sha256'] == rv.digest(Path(strong.__file__))
    assert protocol['checkpoint_sha256'] == shift.panel_hashes()
    assert rv.load(strong.SOURCE / 'development.json')['source_sha256'] == shift.source_hashes()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'primary': 'G0 D0 MWPM free paired selector minus raw PyMatching on the matching G1 map panel',
                'contrast': 'G1 minus G0 of the gain over the same-strength raw decoder; paired by local map',
                'inference': 'Reuse candidate observables encoded by the original selector in its transition features',
                'runtime': 'Single process, warm D0 and checkpoint, 512 cached-input shots, median of three trials per strength',
                'strong_protocol_sha256': frozen['protocol_sha256'], 'calibration_sha256': frozen_hash,
                'strong_summary_sha256': rv.digest(STRONG / 'summary.json'), 'runner_sha256': rv.digest(Path(__file__)),
                'mkl_threading_layer': os.environ.get('MKL_THREADING_LAYER')}
    marker = out / 'protocol.json'
    if marker.exists():
        assert rv.load(marker) == manifest
    else:
        assert args.phase == 'evaluate'
        rv.save(marker, manifest)
    if args.phase == 'benchmark':
        benchmark(out, protocol, frozen)
        return

    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(evaluate_map, out, seed, protocol, frozen, args.phase == 'replay')
                 for seed in protocol['evaluation_maps']]
        for task in as_completed(tasks):
            result = task.result()
            records.append(result)
            print(f'Map {result["local_seed"]} verified ({len(records)}/{len(tasks)})', flush=True)
    summary = summarize(sorted(records, key=lambda row: row['local_seed']), protocol, frozen_hash)
    result_file = out / 'summary.json'
    if result_file.exists():
        assert rv.load(result_file) == summary
    else:
        rv.save(result_file, summary)
    print(f'Done: {result_file}', flush=True)


if __name__ == '__main__':
    main()
