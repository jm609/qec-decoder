"""Compare frozen learned motifs with label-free local syndrome edits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.stats import t
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.rotated_rect import build_rectangular_syndrome_layout
from tools import benchmark_d7_final as bench

final, rv, sedp, shift = bench.final, bench.rv, bench.sedp, bench.shift
OUT = ROOT / 'artifacts/eval/d7_candidate_generators/20261117'
PLAN = ROOT / 'docs/thesis/PREDECODER_D7_CANDIDATE_GENERATOR_PROTOCOL.md'
DEVELOPMENT = tuple(range(787000, 787004))
BUDGETS = (4, 8, 16, 32)
RADII = (2, 4)
CAP = BUDGETS[-1]
THREADS = 2
SHOTS = 512
MODELS = tuple(range(8))
MAPS = tuple(dict(local_seed=788200 + i, g0_sampler=1794200 + 2*i, g1_sampler=1794201 + 2*i)
             for i in range(24))


def local_patterns(layout, radius):
    coords = np.column_stack((layout.detector_time_index, layout.row_index_by_detector,
                             layout.col_index_by_detector))
    first, second = np.triu_indices(len(coords), k=1)
    distance = np.abs(coords[first] - coords[second]).sum(axis=1)
    keep = np.flatnonzero(distance <= radius)
    keep = keep[np.argsort(distance[keep], kind='stable')]
    detectors = np.arange(len(coords))
    return np.r_[detectors, first[keep]], np.r_[detectors, second[keep]]


def local_masks(events, patterns):
    first, second = patterns
    # Stable sorting breaks ties by radius and detector indices in the pattern table.
    fired = events[:, first].astype(np.int8) + events[:, second]
    scores = np.where(first == second, np.where(fired == 2, 1, -1), fired - 1 + (fired == 2))
    rows = np.argsort(-scores, axis=1, kind='stable')[:, :CAP]
    active = np.take_along_axis(fired, rows, axis=1) > 0
    masks = np.zeros((len(events), CAP + 1, events.shape[1]), dtype=np.uint8)
    shot_idx, rank = np.nonzero(active)
    selected = rows[shot_idx, rank]
    masks[shot_idx, rank + 1, first[selected]] = 1
    masks[shot_idx, rank + 1, second[selected]] = 1
    return masks


def learned_patterns(layout, events, kwargs):
    assert kwargs['selector_policy_candidate_mode'] == sedp.SELECTOR_POLICY_CANDIDATE_MODE_NONE
    assert kwargs['local_motif_top_k'] == CAP
    entry = SimpleNamespace(detector_events=events, detector_time_index=layout.detector_time_index,
                            row_index_by_detector=layout.row_index_by_detector,
                            col_index_by_detector=layout.col_index_by_detector)
    placements = sedp._build_local_motif_placements(entry=entry, vocabulary=kwargs['local_motif_vocabulary'])
    assert placements.detector_index_lists[0] == ()
    ordered = sorted(range(1, len(placements.detector_index_lists)),
                     key=lambda index: len(placements.detector_index_lists[index]))
    keys = [placements.detector_index_lists[index] for index in ordered]
    assert len(keys) >= CAP and all(1 <= len(key) <= 2 for key in keys)
    return (placements.mask_table[ordered], np.array([key[0] for key in keys]),
            np.array([key[-1] for key in keys]), np.array([len(key) == 2 for key in keys]))


def learned_masks(events, layout, static, context, model, patterns):
    table, first, second, pair = patterns
    x = bench.tensor(events, layout, static, context)
    logits = sedp._collect_outputs(model=model, x=x, batch_size=64, device=sedp.torch.device('cpu'))
    probs = sedp.common._sigmoid_np(logits['edit_logits'])
    detector_probs = probs[:, layout.detector_time_index, layout.row_index_by_detector,
                           layout.col_index_by_detector]
    # Existing rank is sum of float32 probabilities, with weight and placement order as ties.
    scores = detector_probs[:, first] + detector_probs[:, second] * pair
    rows = np.argsort(-scores, axis=1, kind='stable')[:, :CAP]
    masks = np.zeros((len(events), CAP + 1, events.shape[1]), dtype=np.uint8)
    masks[:, 1:] = table[rows]
    return masks


def decode_masks(events, masks, raw, decoder):
    shots, ranks = np.nonzero(np.any(masks, axis=2))
    predictions = np.full(masks.shape[:2], 255, dtype=np.uint8)
    predictions[:, 0] = raw
    decoded, failures = shift.decode_edits(decoder, events[shots], masks[shots, ranks],
                                           np.zeros((len(shots), 2), dtype=np.uint8))
    predictions[shots[~failures], ranks[~failures]] = bench.prior.paired.class4(decoded[~failures])
    return predictions, int(failures.sum())


def measures(truth, raw, classes, masks, gen_seconds, setup_seconds, decode_seconds, failures):
    wrong = truth != raw
    assert np.any(wrong)
    weights = masks.sum(axis=2)
    budgets = {}
    for k in BUDGETS:
        labels = classes[:, :k + 1]
        recovered = np.any(labels == truth[:, None], axis=1)
        nclasses = sum(np.any(labels == label, axis=1) for label in range(4))
        budgets[str(k)] = dict(
            recovered_raw_errors=int(np.sum(recovered & wrong)),
            recovery_pp=100 * float(recovered[wrong].mean()),
            oracle_accuracy_pp=100 * float(recovered.mean()),
            mean_classes=float(nclasses.mean()),
            class_histogram=np.bincount(nclasses, minlength=5)[1:].tolist(),
            all_four_raw_errors=int(np.sum((nclasses == 4) & wrong)),
            mean_edits=float((weights[:, 1:k+1] != 0).sum(axis=1).mean()),
        )
    return dict(budgets=budgets, total_shots=len(truth), raw_errors=int(wrong.sum()),
                edit_weight_histogram=np.bincount(weights[:, 1:].ravel(), minlength=3).tolist(),
                setup_seconds=setup_seconds, gen_seconds=gen_seconds, decode_seconds=decode_seconds,
                infeasible_candidates=failures)


def run_map(base, radii, checkpoints, *, replay=False):
    g0, g1 = (bench.dev.data_root(base / strength) for strength in ('g0', 'g1'))
    for strength, directory in zip(('g0', 'g1'), (g0, g1)):
        case_root = base / strength
        hashes = rv.load(case_root/'case.json')['sha256']
        for name in ('samples.npz', 'metadata.json', 'circuit.stim', 'detector_error_model.dem'):
            path = directory / name
            assert rv.digest(path) == hashes[path.relative_to(case_root).as_posix()], path
    ideal = [stim.Circuit.from_file(str(directory/'circuit.stim')).without_noise() for directory in (g0, g1)]
    assert ideal[0] == ideal[1] and ideal[1].num_observables == 2
    decoder = shift.open_decoders([g0])[shift.D0_MWPM]
    with np.load(g1/'samples.npz') as samples:
        events = samples['detector_events']
        truth = samples['observable_flips']
        np.testing.assert_array_equal(truth, np.column_stack((samples['logical_z_flip'], samples['logical_x_flip'])))
        layout = build_rectangular_syndrome_layout(samples)
    raw = decoder.decode_batch(events)
    labels, raw_classes = map(bench.prior.paired.class4, (truth, raw))
    static, _ = sedp.volume_common._build_static_planes(layout)
    meta = rv.load(g0/'metadata.json')
    context, names = sedp.volume_common._build_context_planes(
        {**meta, 'qc_stats': {'detector_event_fraction': final.CONTEXT}}, layout,
    )
    assert np.all(context[names.index('reserved_label_context')] == 0)
    rows, arrays = {}, {'truth': labels, 'raw': raw_classes}

    def score(name, generate, setup_seconds):
        started = perf_counter()
        masks = generate()
        generated = perf_counter()
        classes, failures = decode_masks(events, masks, raw_classes, decoder)
        decoded = perf_counter()
        rows[name] = measures(labels, raw_classes, classes, masks, generated - started,
                              setup_seconds, decoded - generated, failures)
        arrays[name + '_classes'] = classes
        packed = np.packbits(masks, axis=2)
        arrays[name + '_masks'] = packed
        return packed, classes

    for radius in radii:
        started = perf_counter()
        patterns = local_patterns(layout, radius)
        packed, _ = score(f'local{radius}', lambda: local_masks(events, patterns), perf_counter() - started)
        if replay:
            edits = np.unpackbits(packed[:, 1:], axis=2, count=events.shape[1])
            has_edit = np.any(edits, axis=2)
            np.testing.assert_array_equal(has_edit, np.any(edits & events[:, None, :], axis=2))
            assert np.all(np.diff(has_edit.astype(np.int8), axis=1) <= 0)
            assert all(len(np.unique(shot[used], axis=0)) == used.sum() for shot, used in zip(packed[:, 1:], has_edit))
    for seed in MODELS:
        model, _, kwargs, _ = rv.load_selector(checkpoints[seed])
        started = perf_counter()
        patterns = learned_patterns(layout, events, kwargs)
        packed, classes = score(f'learned{seed}', lambda: np.concatenate([
            learned_masks(events[lo:lo + shift.BATCH], layout, static, context, model, patterns)
            for lo in range(0, len(events), shift.BATCH)
        ]), perf_counter() - started)
        if replay:
            path = base / 'g1' / f'predictions_seed{seed}.npz'
            assert rv.digest(path) == rv.load(path.with_suffix('.json'))['sha256']
            with np.load(path) as saved:
                np.testing.assert_array_equal(truth, saved['truth'])
                np.testing.assert_array_equal(raw, saved[shift.D0_MWPM + '_raw'])
                np.testing.assert_array_equal(saved['shot_index'], np.repeat(np.arange(len(events)), CAP + 1))
                np.testing.assert_array_equal(packed.reshape(-1, packed.shape[-1]), saved['packed_masks'])
                assert not np.any(saved[shift.D0_MWPM + '_failure'])
                expected = bench.prior.paired.class4(saved[shift.D0_MWPM + '_predicted'])
                np.testing.assert_array_equal(classes.ravel(), expected)
    return rows, arrays


def checkpoints():
    return rv.checkpoints(shift.REVALIDATION, 7, list(MODELS))


def development():
    source = final.dev.OUT/'development'
    panel = checkpoints()
    results = {}
    for seed in DEVELOPMENT:
        results[str(seed)], _ = run_map(source/str(seed), RADII, panel, replay=True)
        print(f'Development map {seed}: exact replay for all 8 learned generators', flush=True)
    means = {radius: float(np.mean([row[f'local{radius}']['budgets'][str(CAP)]['recovery_pp']
                                   for row in results.values()])) for radius in RADII}
    chosen = max(RADII, key=means.__getitem__)
    OUT.mkdir(parents=True, exist_ok=True)
    rv.save(OUT/'development.json', dict(results=results, local_means=means, radius=chosen, exact_replay=True))
    print(f'Development recovery at K={CAP}: {means}; selected radius {chosen}', flush=True)


def freeze():
    dev_path = OUT/'development.json'
    state = rv.load(dev_path)
    assert state['exact_replay'] and state['radius'] in RADII
    used_local, used_sampler = set(), set()
    for path in (ROOT/'artifacts/eval').rglob('case.json'):
        spec = rv.load(path).get('settings', {})
        used_local.add(spec.get('local_seed'))
        used_sampler.add(spec.get('sampler_seed'))
    registry = rv.load(final.dev.OUT/'data_registry.json')
    for spec in registry['training_sample_seeds']:
        used_local.add(spec['local_seed'])
        used_sampler.add(spec['seed'])
    assert not ({spec['local_seed'] for spec in MAPS} & used_local)
    assert not ({spec[key] for spec in MAPS for key in ('g0_sampler', 'g1_sampler')} & used_sampler)
    reference = rv.load(final.OUT/'protocol.json')
    assert reference['models'] == registry['models'] and reference['zz_factors'] == [shift.G0, shift.G1]
    sources = {path: rv.digest(ROOT/path) for path in reference['source_sha256']}
    for path in (Path(__file__), Path(bench.__file__), PLAN, dev_path, final.OUT/'protocol.json'):
        sources[path.relative_to(ROOT).as_posix()] = rv.digest(path)
    rv.save(OUT/'protocol.json', dict(
        sources=sources, maps=MAPS, models=reference['models'], radius=state['radius'],
        g1_shots=SHOTS, g0_shots=32, cell=shift.D0_MWPM, zz_factors=[shift.G0, shift.G1],
        context=final.CONTEXT, budgets=BUDGETS, primary_budget=CAP, level=0.95,
        versions={name: version(name) for name in final.VERSIONS}, threads=THREADS,
    ))
    print(f'Frozen {len(MAPS)} new maps with local radius {state["radius"]}', flush=True)


def evaluate_map(spec, protocol, protocol_hash):
    sedp.torch.set_num_threads(protocol['threads'])
    seed = spec['local_seed']
    base = OUT/'evaluation'/str(seed)
    marker = base/'result.json'
    predictions = base/'predictions.npz'
    if marker.exists():
        result = rv.load(marker)
        assert result['protocol_sha256'] == protocol_hash and result['sha256'] == rv.digest(predictions)
        return result
    for strength, factor in zip(('g0', 'g1'), protocol['zz_factors']):
        rv.generate_case(base/strength, distance=7, local_seed=seed,
                         sampler_seed=spec[strength + '_sampler'], shots=protocol[strength + '_shots'],
                         zz_factor=factor)
    assert not predictions.exists(), predictions
    rows, arrays = run_map(base, (protocol['radius'],), checkpoints())
    assert len(arrays['truth']) == protocol['g1_shots']
    with predictions.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(local_seed=seed, rows=rows, protocol_sha256=protocol_hash, sha256=rv.digest(predictions))
    rv.save(marker, result)
    print(f'Evaluation map {seed} complete', flush=True)
    return result


def summarize(results, protocol):
    summary = {}
    fields = ('recovery_pp', 'oracle_accuracy_pp', 'mean_classes', 'mean_edits')
    local_name = f'local{protocol["radius"]}'
    for k in protocol['budgets']:
        key = str(k)
        per_map = {'learned': [], 'local': []}
        for result in results:
            rows = result['rows']
            per_map['learned'].append([np.mean([rows[f'learned{i}']['budgets'][key][field]
                                              for i in MODELS]) for field in fields])
            per_map['local'].append([rows[local_name]['budgets'][key][field] for field in fields])
        learned, local = map(np.asarray, per_map.values())
        difference = learned[:, 0] - local[:, 0]
        half = t.ppf((1 + protocol['level']) / 2, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
        summary[key] = dict(
            learned=dict(zip(fields, learned.mean(axis=0).tolist())),
            local=dict(zip(fields, local.mean(axis=0).tolist())),
            recovery_difference_pp=float(difference.mean()),
            interval_pp=[float(difference.mean() - half), float(difference.mean() + half)],
            per_map_recovery_difference_pp=difference.tolist(),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('develop', 'freeze', 'run'))
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate Anaconda fp first')
    assert os.getenv('MKL_THREADING_LAYER') == 'SEQUENTIAL'
    sedp.torch.set_num_threads(THREADS)
    if args.mode == 'develop':
        development()
    elif args.mode == 'freeze':
        freeze()
    else:
        path = OUT/'protocol.json'
        protocol = rv.load(path)
        for source, digest in protocol['sources'].items():
            assert rv.digest(ROOT/source) == digest, source
        for model in protocol['models'].values():
            assert rv.digest(ROOT/model['checkpoint']) == model['sha256']
        assert protocol['versions'] == {name: version(name) for name in final.VERSIONS}
        digest = rv.digest(path)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate_map, spec, protocol, digest) for spec in protocol['maps']]
            results = [future.result() for future in futures]
        summary = summarize(results, protocol)
        rv.save(OUT/'summary.json', dict(protocol_sha256=digest, budgets=summary, maps=results))
        print(f'Primary result at K={CAP}: {summary[str(CAP)]}', flush=True)


if __name__ == '__main__':
    main()
