"""Test whether a matching-cost feature improves the fixed d7 candidate ordering."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_predecoder_noise_shift as shift

rv, sedp = shift.rv, shift.sedp
PREVIOUS = ROOT / 'artifacts/eval/noise_shift_d7/20261115_retrained'
COEFFICIENTS = (0.25, 0.5, 1.0, 2.0, 4.0)
MAPS = tuple(range(784100, 784108))
SHOTS = 1024
BOOTSTRAP_SEED = 1784199
BOOTSTRAP_REPS = 20000


def rerank_rows(logits: np.ndarray, feature: np.ndarray, shot: np.ndarray, weight: np.ndarray,
                baseline: np.ndarray, coefficient: float, *, fixed_edit_rate: bool) -> np.ndarray:
    adjusted = logits.astype(np.float64) + coefficient * feature
    if fixed_edit_rate:
        eligible = (weight != 0) == (weight[baseline][shot] != 0)
        adjusted[~eligible] = -np.inf
    maximum = np.full(len(baseline), -np.inf)
    np.maximum.at(maximum, shot, adjusted)
    rows = np.flatnonzero(adjusted == maximum[shot])
    selected = np.full(len(baseline), len(shot), dtype=np.int64)
    np.minimum.at(selected, shot[rows], rows)
    np.testing.assert_array_equal(shot[selected], np.arange(len(baseline)))
    if fixed_edit_rate:
        np.testing.assert_array_equal(weight[selected] != 0, weight[baseline] != 0)
    return selected


def matching_feature(matching, events: np.ndarray, masks: np.ndarray, shot: np.ndarray) -> tuple:
    raw, raw_cost = matching.decode_batch(events, return_weights=True)
    predicted, cost = raw[shot].copy(), raw_cost[shot].copy()
    failure = np.zeros(len(shot), dtype=bool)
    edited = np.flatnonzero(masks.any(axis=1))
    syndrome = events[shot[edited]] ^ masks[edited]
    if edited.size:
        try:
            predicted[edited], cost[edited] = matching.decode_batch(syndrome, return_weights=True)
        except ValueError:
            for row, detector_events in zip(edited, syndrome):
                try:
                    predicted[row], cost[row] = matching.decode(detector_events, return_weight=True)
                except ValueError as error:
                    if 'perfect matching' not in str(error).lower():
                        raise
                    failure[row] = True
    feature = np.clip((raw_cost[shot] - cost) / np.maximum(np.abs(raw_cost[shot]), 1.0), -1, 1)
    assert np.isfinite(feature).all()
    return raw, predicted, feature, failure


def calibrate(out: Path, protocol: dict) -> dict:
    gains = {coefficient: [] for coefficient in COEFFICIENTS}
    by_map = {}
    for seed in protocol['development_maps']:
        path = PREVIOUS / 'development' / str(seed)
        raw_dirs = [Path(rv.load(path / f'g{g}' / rv.load(path / f'g{g}/case.json')['target'] / 'metadata.json')['source_family_dir'])
                    for g in (0, 1)]
        matching = shift.open_decoders(raw_dirs[:1])[shift.D0_MWPM].matching
        record = rv.load(path / 'result.json')
        by_map[str(seed)] = {}
        with np.load(raw_dirs[1] / 'samples.npz') as samples:
            events = samples['detector_events']
            truth = samples['observable_flips']
        for model in shift.PANEL:
            model_record = record['data']['g1'][str(model)]
            file = path / 'g1' / model_record['file']
            assert rv.digest(file) == model_record['sha256']
            with np.load(file) as saved:
                logits, shot, weight, top1 = (saved[k] for k in ('utility', 'shot_index', 'weights', 'top1_rows'))
                masks = np.unpackbits(saved['packed_masks'], axis=1, count=events.shape[1])
                raw, decoded, feature, _ = matching_feature(matching, events, masks, shot)
                np.testing.assert_array_equal(raw, saved['d0_mwpm_raw'])
                np.testing.assert_array_equal(decoded, saved['d0_mwpm_predicted'])
                np.testing.assert_array_equal(truth, saved['truth'])
            same = rerank_rows(logits, feature, shot, weight, top1, 0.0, fixed_edit_rate=True)
            np.testing.assert_array_equal(same, top1)
            scores = {}
            for coefficient in COEFFICIENTS:
                rows = rerank_rows(logits, feature, shot, weight, top1, coefficient, fixed_edit_rate=True)
                gain = rv.paired_counts(decoded[top1], decoded[rows], truth)['gain_pp']
                gains[coefficient].append(gain)
                scores[str(coefficient)] = gain
            by_map[str(seed)][str(model)] = scores
    averages = {coefficient: float(np.mean(values)) for coefficient, values in gains.items()}
    chosen = max(COEFFICIENTS, key=lambda coefficient: (averages[coefficient], -coefficient))
    result = {'coefficient': chosen, 'development_mean_gain_pp': averages, 'by_map_and_seed': by_map,
              'protocol_sha256': rv.digest(out / 'protocol.json')}
    rv.save(out / 'calibration.json', result)
    print(f'Coefficient frozen at {chosen:g}; development ranking contrast {averages[chosen]:+.3f} pp', flush=True)
    return result


def evaluate_strength(path: Path, entry, matching, predictors: dict, coefficient: float) -> dict:
    raw = matching.decode_batch(entry.detector_events)
    entry = replace(entry, matching=matching, baseline_predicted_observables=raw)
    truth = entry.observable_flips
    arrays = {'raw': raw, 'truth': truth}
    output = {}
    for seed, predictor in predictors.items():
        chunks = {key: [] for key in ('top1', 'fixed', 'free', 'eligible', 'free_eligible', 'oracle')}
        changed = {'fixed_other_edit': 0, 'free_edit_to_raw': 0, 'free_raw_to_edit': 0, 'free_other_edit': 0}
        candidates = fallbacks = 0
        for lo in range(0, len(truth), shift.BATCH):
            hi = min(lo + shift.BATCH, len(truth))
            bundle, logits, _ = rv.score_bundle(entry, lo, hi, predictor)
            _, top = shift.top_rows(bundle, logits)
            mask, shot, weight = bundle.candidate_edit_mask, bundle.shot_indices, bundle.candidate_edit_weight
            raw_chunk, decoded, feature, failure = matching_feature(matching, entry.detector_events[lo:hi], mask, shot)
            np.testing.assert_array_equal(raw_chunk, raw[lo:hi])
            fixed = rerank_rows(logits, feature, shot, weight, top, coefficient, fixed_edit_rate=True)
            free = rerank_rows(logits, feature, shot, weight, top, coefficient, fixed_edit_rate=False)
            old_edits, free_edits = weight[top] != 0, weight[free] != 0
            changed['fixed_other_edit'] += int(np.count_nonzero(fixed != top))
            changed['free_edit_to_raw'] += int(np.count_nonzero(old_edits & ~free_edits))
            changed['free_raw_to_edit'] += int(np.count_nonzero(~old_edits & free_edits))
            changed['free_other_edit'] += int(np.count_nonzero(old_edits & free_edits & (free != top)))
            oracle = np.zeros(hi - lo, dtype=bool)
            np.logical_or.at(oracle, shot, np.all(decoded == truth[lo:hi][shot], axis=1))
            for name, values in [('top1', decoded[top]), ('fixed', decoded[fixed]), ('free', decoded[free]),
                                 ('eligible', old_edits), ('free_eligible', free_edits), ('oracle', oracle)]:
                chunks[name].append(values)
            candidates += len(shot)
            fallbacks += int(failure.sum())
        selected = {key: np.concatenate(value) for key, value in chunks.items()}
        arrays.update({f'seed{seed}_{key}': value for key, value in selected.items()})
        metrics = {key: rv.paired_counts(raw, selected[key], truth) for key in ('top1', 'fixed', 'free')}
        fixed_delta = rv.paired_counts(selected['top1'], selected['fixed'], truth)
        oracle_correct = int(selected['oracle'].sum())
        assert max(row['selected_correct'] for row in metrics.values()) <= oracle_correct
        output[str(seed)] = {**metrics, 'fixed_vs_top1': fixed_delta, 'changed_actions': changed,
                             'top1_edit_count': int(selected['eligible'].sum()),
                             'free_edit_count': int(selected['free_eligible'].sum()),
                             'oracle_gain_pp': 100 * (oracle_correct - metrics['top1']['raw_correct']) / len(truth),
                             'candidate_rows': candidates, 'candidate_decode_fallbacks': fallbacks}
    np.savez_compressed(path, **arrays)
    return {'prediction_sha256': rv.digest(path), 'checkpoints': output}


def evaluate_map(out: Path, seed: int, protocol: dict, coefficient: float) -> dict:
    sedp.torch.set_num_threads(2)
    root = out / 'evaluation' / str(seed)
    marker = root / 'result.json'
    if marker.exists():
        record = rv.load(marker)
        for strength in ('g0', 'g1'):
            assert record[strength]['prediction_sha256'] == rv.digest(root / strength / 'predictions.npz')
        return record
    targets, raw, validation = shift.make_pair(root, seed, protocol['shots_per_strength'])
    assert validation['sampler_seeds'] == protocol['sampler_seeds'][str(seed)]
    matching = shift.open_decoders(raw[:1])[shift.D0_MWPM].matching
    predictors = {model: rv.load_selector(checkpoint)
                  for model, checkpoint in rv.checkpoints(shift.REVALIDATION, 7, shift.PANEL).items()}
    context = rv.load(raw[0] / 'metadata.json')
    result = {'local_seed': seed, 'validation': validation}
    for g, target in enumerate(targets):
        entry = sedp._prepare_edit_family(target, fill_value=-0.5, max_shots=None)
        entry, fixed_context = shift.with_context(entry, context, protocol['context_event_fraction'])
        result[f'g{g}'] = evaluate_strength(root / f'g{g}/predictions.npz', entry, matching, predictors, coefficient)
        result[f'g{g}']['context'] = fixed_context
    assert result['g0']['context'] == result['g1']['context']
    rv.save(marker, result)
    return result


def summarize(out: Path, results: list[dict], calibration: dict) -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(results), size=(BOOTSTRAP_REPS, len(results)))
    summary = {'coefficient': calibration['coefficient'], 'strengths': {}}
    for strength in ('g0', 'g1'):
        groups = [result[strength]['checkpoints'] for result in results]
        flattened = [row for models in groups for row in models.values()]
        differences = np.array([np.mean([row['fixed_vs_top1']['gain_pp'] for row in models.values()]) for models in groups])
        summary['strengths'][strength] = {
            'fixed_vs_top1_pp': float(differences.mean()),
            'interval_95_pp': np.quantile(differences[draw].mean(axis=1), [0.025, 0.975]).tolist(),
            'map_effects_pp': dict(zip((r['local_seed'] for r in results), map(float, differences))),
            'gain_vs_raw_pp': {branch: float(np.mean([row[branch]['gain_pp'] for row in flattened]))
                               for branch in ('top1', 'fixed', 'free')},
            'oracle_gain_pp': float(np.mean([row['oracle_gain_pp'] for row in flattened])),
            'edit_counts': {key: sum(row[key] for row in flattened) for key in ('top1_edit_count', 'free_edit_count')},
            'changed_actions': {key: sum(row['changed_actions'][key] for row in flattened)
                                for key in flattened[0]['changed_actions']},
            'candidate_decode_fallbacks': sum(row['candidate_decode_fallbacks'] for row in flattened),
        }
    rv.save(out / 'summary.json', summary)
    print(f'Done: {out / "summary.json"}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    sedp.torch.set_num_threads(2)
    out = args.out.resolve()
    dev = rv.load(PREVIOUS / 'development.json')
    assert dev['source_sha256'] == shift.source_hashes()
    assert dev['checkpoint_sha256'] == shift.panel_hashes()
    assert not set(MAPS).intersection(shift.SEED_PLAN['development_local_seeds'] + shift.SEED_PLAN['evaluation_local_seeds'])
    protocol = {'primary': 'G1, D0 MWPM, fixed-edit rerank minus original top1, paired by map',
                'coefficient_grid': list(COEFFICIENTS), 'development_maps': shift.SEED_PLAN['development_local_seeds'],
                'evaluation_maps': list(MAPS), 'shots_per_strength': SHOTS,
                'sampler_seeds': {str(m): [shift.SEED_PLAN['sampler_base'] + 2 * (m-shift.SEED_PLAN['development_local_seeds'][0]) + g
                                          for g in (0, 1)] for m in MAPS},
                'context_event_fraction': dev['context_event_fraction'], 'checkpoint_sha256': dev['checkpoint_sha256'],
                'development_sha256': rv.digest(PREVIOUS / 'development.json'),
                'runner_sha256': rv.digest(Path(__file__)), 'bootstrap_seed': BOOTSTRAP_SEED,
                'bootstrap_repetitions': BOOTSTRAP_REPS, 'conda_prefix': os.environ['CONDA_PREFIX'],
                'versions': {name: version(name) for name in ('torch', 'stim', 'pymatching', 'numpy')}}
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'protocol.json'
    if marker.exists():
        assert rv.load(marker) == protocol
    else:
        rv.save(marker, protocol)
    locked = out / 'calibration.json'
    calibration = rv.load(locked) if locked.exists() else calibrate(out, protocol)
    assert calibration['protocol_sha256'] == rv.digest(marker)
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(evaluate_map, out, seed, protocol, calibration['coefficient']) for seed in MAPS]
        results = []
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f'Evaluation map {result["local_seed"]} done ({len(results)}/{len(MAPS)})', flush=True)
    if not (out / 'summary.json').exists():
        summarize(out, sorted(results, key=lambda result: result['local_seed']), calibration)


if __name__ == '__main__':
    main()
