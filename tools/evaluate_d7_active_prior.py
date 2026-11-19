"""Cross DEM and decoding mode for the active d7 margin and paired policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_confidence_development as dev
from tools.collect_d7_confidence_candidates import CONTEXT
from tools.evaluate_d7_paired_selector_weak import decoded_from_transition

rv, shift, paired, sedp = dev.rv, dev.shift, dev.paired, dev.shift.sedp
OUT = ROOT / 'artifacts/eval/d7_active_prior/20261117'
CELLS = ('d0_mwpm', 'd1_mwpm', 'd0_correlated', 'd1_correlated')
POLICIES = ('margin', 'paired')


def rescore(bundle, events, raw, decoder, predictor):
    indices = bundle.shot_indices
    baseline = raw[indices]
    predicted, failed = shift.decode_edits(decoder, events[indices], bundle.candidate_edit_mask, baseline)
    if failed.any():
        raise RuntimeError(f'Cannot form paired cases with {int(failed.sum())} failed candidate matches')
    start = sedp._transition_feature_offsets(candidate_feature_dim=bundle.candidate_features.shape[1])['obs_delta'].start
    features = bundle.candidate_features.copy()
    features[:, start:] = sedp._candidate_transition_feature_matrix(
        baseline_predicted_observables=baseline, edited_predicted_observables=predicted,
    )
    rescored = replace(bundle, candidate_features=features, decode_fallback_count=0)
    utility = sedp._selector_logits_for_bundle(
        selector=predictor[1], bundle=rescored, batch_size=64, device=sedp.torch.device('cpu'),
    )
    return rescored, utility, predicted


def choose(bundle, utility, prediction, raw, events, layout, lookup, order, seed, head, margin):
    _, top = shift.top_rows(bundle, utility)
    branch = shift.selected_rows(bundle, utility, margin)
    masks = np.packbits(bundle.candidate_edit_mask, axis=1)
    arrays = {
        'shot_index': bundle.shot_indices, 'weights': bundle.candidate_edit_weight, 'packed_masks': masks,
        'utility': utility, 'top1_rows': top, 'branch_rows': branch, 'truth': np.zeros_like(raw),
        'd0_mwpm_raw': raw, 'd0_mwpm_predicted': prediction,
        'd0_mwpm_failure': np.zeros(len(prediction), dtype=bool),
    }
    case, features, _ = dev.make_features(arrays, events, layout.detector_time_index, lookup, order,
                                          seed, None, None, None)
    accept = paired.head_score(features[dev.PAIR], head) >= 0
    _, free = paired.select_rows(case, accept)
    rows = {'margin': branch, 'paired': free}
    return ({name: bundle.candidate_edit_mask[indices] for name, indices in rows.items()},
            {name: prediction[indices] for name, indices in rows.items()},
            {'paired_eligible': len(accept), 'paired_accept_z': int(accept.sum())})


def inputs(root, strength, decoders):
    case = root / strength
    marker = rv.load(case / 'case.json')
    for name, expected in marker['sha256'].items():
        assert rv.digest(case / name) == expected, name
    raw_dir = dev.data_root(case)
    with np.load(raw_dir / 'samples.npz') as samples:
        layout = paired.build_rectangular_syndrome_layout(samples)
        np.testing.assert_array_equal(samples['observable_flips'],
                                      np.column_stack((samples['logical_z_flip'], samples['logical_x_flip'])))
    lookup, order = paired.probe.patch_lookup(layout)
    entry = sedp._prepare_edit_family(case / marker['target'], fill_value=-0.5, max_shots=None)
    entry, context = shift.with_context(entry, rv.load(dev.data_root(root / 'g0') / 'metadata.json'), CONTEXT)
    raw = {name: decoder.decode_batch(entry.detector_events) for name, decoder in decoders.items()}
    return entry, layout, lookup, order, context, raw


def score_map(out, role, spec, dev_summary, registry):
    sedp.torch.set_num_threads(2)
    seed = str(spec['local_seed'])
    root = dev.OUT / role / seed
    prior = rv.load(root / 'result.json')
    assert rv.digest(root / 'result.json') == dev_summary['candidate_results_sha256'][seed]
    assert prior['spec'] == spec
    for index, strength in enumerate(('g0', 'g1')):
        settings = rv.load(root / strength / 'case.json')['settings']
        assert settings == dict(distance=7, local_seed=spec['local_seed'], sampler_seed=spec[f'{strength}_sampler_seed'],
                                shots=spec['shots_per_strength'], zz_factor=(shift.G0, shift.G1)[index])
    head = (dev_summary['fold_heads'][seed] if role == 'development' else dev_summary['refit_heads'])[dev.PAIR]
    checkpoints = rv.checkpoints(shift.REVALIDATION, 7, list(range(8)))
    for model, record in registry['models'].items():
        assert rv.digest(ROOT / record['checkpoint']) == record['sha256'], model
    predictors = {model: rv.load_selector(checkpoint) for model, checkpoint in checkpoints.items()}
    decoders = shift.open_decoders([dev.data_root(root / strength) for strength in ('g0', 'g1')])
    started = perf_counter()
    for strength in ('g0', 'g1'):
        entry, layout, lookup, order, context, raw = inputs(root, strength, decoders)
        assert context == prior['context']
        events, truth = entry.detector_events, entry.observable_flips
        directory = out / role / seed / strength
        directory.mkdir(parents=True, exist_ok=True)
        for model, predictor in predictors.items():
            path = directory / f'seed{model}.npz'
            if path.is_file() and path.with_suffix('.json').is_file():
                assert rv.load(path.with_suffix('.json'))['sha256'] == rv.digest(path)
                continue
            if path.exists():
                raise FileExistsError(path)
            assert predictor[2]['selector_target_mode'] == sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM
            score_start = perf_counter()
            cached = prior['data'][strength][str(model)]
            old_path = root / strength / cached['file']
            assert rv.digest(old_path) == cached['sha256']
            with np.load(old_path) as old:
                saved = dict(old)
            np.testing.assert_array_equal(truth, saved['truth'])
            np.testing.assert_array_equal(raw[CELLS[0]], saved['d0_mwpm_raw'])
            data = {f'{cell}_raw': pred for cell, pred in raw.items()}
            data['truth'] = truth
            parts = {cell: {'masks': {p: [] for p in POLICIES}, 'pred': {p: [] for p in POLICIES},
                            'paired_eligible': 0, 'paired_accept_z': 0} for cell in CELLS}
            base_entry = replace(entry, matching=decoders[CELLS[0]], baseline_predicted_observables=raw[CELLS[0]])
            offset = 0
            for lo in range(0, len(events), shift.BATCH):
                hi = min(lo + shift.BATCH, len(events))
                batch = events[lo:hi]
                bundle, base_utility, margin = rv.score_bundle(base_entry, lo, hi, predictor)
                n = len(base_utility)
                base_predictions = decoded_from_transition(bundle, raw[CELLS[0]][lo:hi])
                for actual, expected in (
                    (base_utility, saved['utility'][offset:offset+n]),
                    (np.packbits(bundle.candidate_edit_mask, axis=1), saved['packed_masks'][offset:offset+n]),
                    (base_predictions, saved['d0_mwpm_predicted'][offset:offset+n]),
                ):
                    np.testing.assert_array_equal(actual, expected)
                for cell in CELLS:
                    baseline = raw[cell][lo:hi]
                    if cell == CELLS[0]:
                        full, utility, pred = bundle, base_utility, base_predictions
                    else:
                        full, utility, pred = rescore(bundle, batch, baseline, decoders[cell], predictor)
                    masks, predictions, counts = choose(full, utility, pred, baseline, batch, layout,
                                                        lookup, order, model, head, margin)
                    for policy in POLICIES:
                        parts[cell]['masks'][policy].append(masks[policy])
                        parts[cell]['pred'][policy].append(predictions[policy])
                    for name, count in counts.items():
                        parts[cell][name] += count
                offset += n
            assert offset == len(saved['utility'])
            anchor_rows = saved['branch_rows']
            anchor = {pol: np.concatenate(parts[CELLS[0]]['masks'][pol]) for pol in POLICIES}
            np.testing.assert_array_equal(np.packbits(anchor['margin'], axis=1), saved['packed_masks'][anchor_rows])
            counts = {cell: {key: value for key, value in parts[cell].items() if key.startswith('paired_')} for cell in CELLS}
            for cell in CELLS:
                for policy in POLICIES:
                    full_mask = np.concatenate(parts[cell]['masks'][policy])
                    full_pred = np.concatenate(parts[cell]['pred'][policy])
                    fixed_pred, failed = shift.decode_edits(decoders[cell], events, anchor[policy], raw[cell])
                    checked, full_failed = shift.decode_edits(decoders[cell], events, full_mask, raw[cell])
                    np.testing.assert_array_equal(full_pred, checked)
                    if failed.any() or full_failed.any():
                        raise RuntimeError(f'Selected mask matching failed in {cell}')
                    data[f'{cell}_{policy}_full_mask'] = np.packbits(full_mask, axis=1)
                    data[f'{cell}_{policy}_full'] = full_pred
                    data[f'{cell}_{policy}_fixed'] = fixed_pred
            with path.open('xb') as stream:
                np.savez_compressed(stream, **data)
            rv.save(path.with_suffix('.json'), {'sha256': rv.digest(path), 'cells': counts,
                                               'wall_seconds': perf_counter() - score_start})
            print(f'{role} {seed} {strength} seed{model}: {perf_counter()-score_start:.1f} s', flush=True)
    return {'seed': seed, 'seconds': perf_counter()-started}


def verify_reuse(state):
    """Compare transition-only rescoring with direct inference in all four cells."""
    sedp.torch.set_num_threads(2)
    root = dev.OUT / 'development/787000'
    decoders = shift.open_decoders([dev.data_root(root / g) for g in ('g0', 'g1')])
    head = state['fold_heads']['787000'][dev.PAIR]
    predictor = rv.load_selector(rv.checkpoints(shift.REVALIDATION, 7, [2])[2])
    measurements = {}
    for strength in ('g0', 'g1'):
        entry, layout, lookup, order, _, raw = inputs(root, strength, decoders)
        shots = 64
        events = entry.detector_events[:shots]
        base = replace(entry, matching=decoders[CELLS[0]], baseline_predicted_observables=raw[CELLS[0]])
        bundle, _, margin = rv.score_bundle(base, 0, shots, predictor)
        for cell in CELLS[1:]:
            baseline = raw[cell][:shots]
            direct = replace(entry, matching=decoders[cell], baseline_predicted_observables=raw[cell])
            expected, utility, _ = rv.score_bundle(direct, 0, shots, predictor)
            expected_pred = decoded_from_transition(expected, baseline)
            rescored, logits, pred = rescore(bundle, events, baseline, decoders[cell], predictor)
            for attr in ('shot_features', 'candidate_features', 'candidate_edit_mask', 'candidate_edit_weight',
                         'shot_indices'):
                np.testing.assert_array_equal(getattr(expected, attr), getattr(rescored, attr), err_msg=attr)
            np.testing.assert_array_equal(logits, utility)
            np.testing.assert_array_equal(pred, expected_pred)
            masks, predictions, counts = choose(rescored, logits, pred, baseline, events,
                                                layout, lookup, order, 2, head, margin)
            targets, outcomes, expected_counts = choose(expected, utility, expected_pred, baseline, events,
                                                         layout, lookup, order, 2, head, margin)
            assert counts == expected_counts
            for policy in POLICIES:
                np.testing.assert_array_equal(masks[policy], targets[policy])
                np.testing.assert_array_equal(predictions[policy], outcomes[policy])
            measurements[f'{strength}/{cell}'] = {'shots': shots, 'candidates': len(logits),
                                                   'paired_eligible': counts['paired_eligible']}
        print(f'Rescoring replay verified for {strength}', flush=True)
    return {'map': 787000, 'checkpoint': 2, 'replay_cells': measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run',))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dev.OUT / 'stage2_summary.json'
    registry = dev.OUT / 'data_registry.json'
    state, data = rv.load(summary), rv.load(registry)
    assert state['selection']['chosen'] == 'margin' and rv.digest(registry) == state['registry_sha256']
    protocol = {'stage2_summary_sha256': rv.digest(summary), 'registry_sha256': rv.digest(registry),
                'source_sha256': rv.digest(Path(__file__)), 'maps': data['reserved'], 'cells': list(CELLS),
                'policies': list(POLICIES), 'jobs': args.jobs, 'torch_threads': 2, 'paired_cutoff': 0,
                'head_on_development': 'stage2 held-out-map fold head', 'head_on_calibration': 'stage2 refit head',
                'versions': {name: version(name) for name in ('numpy', 'scipy', 'stim', 'pymatching', 'torch')}}
    marker = OUT / 'protocol.json'
    if not marker.exists():
        rv.save(marker, protocol)
    else:
        assert rv.load(marker) == protocol
    replay = OUT / 'rescore_replay.json'
    if not replay.exists():
        rv.save(replay, verify_reuse(state))
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(score_map, OUT, role, spec, state, data)
                   for role, maps in data['reserved'].items() for spec in maps]
        for future in as_completed(pending):
            result = future.result()
            print(f'Map {result["seed"]} complete ({perf_counter()-started:.1f} s elapsed)', flush=True)


if __name__ == '__main__':
    main()
