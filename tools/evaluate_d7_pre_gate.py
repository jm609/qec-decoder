"""Freeze and evaluate a cheap gate before d7 syndrome edit generation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import analyze_d7_adoption_z_scores as z_scores
from tools.d7_boundary_matching_costs import BoundaryCosts

adoption, rv = z_scores.adoption, z_scores.rv
gen, shift, sedp = adoption.gen, adoption.shift, adoption.sedp
OUT = ROOT / 'artifacts/eval/d7_pre_gate/20261117'
PLAN = ROOT / 'docs/thesis/PREDECODER_D7_PRE_GATE_PROTOCOL.md'
MAPS = tuple(dict(local_seed=788400 + i, g0_sampler=1794400 + 2*i, g1_sampler=1794401 + 2*i,
                  timing_checkpoint=i % len(gen.MODELS)) for i in range(24))
TARGET_RATES = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90)
L2 = 0.10
REPS = 3
MODES = ('raw', 'late', 'gated')


def round_counts(events, layout):
    return np.column_stack([events[:, layout.detector_time_index == step].sum(axis=1)
                            for step in range(layout.time_steps)])


def fit(features, outcome):
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    x = (features - center) / scale
    coeff = np.linalg.solve(x.T @ x / len(x) + L2*np.eye(x.shape[1]), x.T @ outcome / len(x))
    return dict(center=center.tolist(), scale=scale.tolist(), coeff=coeff.tolist(), intercept=float(outcome.mean()))


def score(features, gate):
    return (features - np.asarray(gate['center'])) / np.asarray(gate['scale']) @ gate['coeff'] + gate['intercept']


def cutoff(train_scores, rate):
    return float(np.quantile(train_scores, 1 - rate))


def outcome(raw, truth, late):
    return (late == truth).astype(np.int8) - (raw == truth)


def gate_effect(delta, admitted):
    blocked = delta[..., ~admitted]
    harms = int(np.count_nonzero(blocked == -1))
    missed = int(np.count_nonzero(blocked == 1))
    return dict(blocked_harms=harms, missed_repairs=missed, net=harms - missed,
                gain_pp=100*(harms - missed)/delta.size, admitted=int(admitted.sum()),
                physical_shots=len(admitted))


def open_case(base):
    locations = []
    for name in ('g0', 'g1'):
        case = base/name
        source = adoption.bench.dev.data_root(case)
        hashes = rv.load(case/'case.json')['sha256']
        for file in ('samples.npz', 'metadata.json', 'detector_error_model.dem', 'circuit.stim'):
            path = source/file
            assert rv.digest(path) == hashes[path.relative_to(case).as_posix()]
        locations.append(source)
    g0, g1 = locations
    matching = shift.open_decoders([g0])[shift.D0_MWPM].matching
    with np.load(g1/'samples.npz') as samples:
        events = samples['detector_events']
        layout = gen.build_rectangular_syndrome_layout(samples)
        truth = adoption.bench.prior.paired.class4(samples['observable_flips'])
        np.testing.assert_array_equal(truth, 2*samples['logical_z_flip'] + samples['logical_x_flip'])
    return events, truth, layout, matching, rv.load(g0/'metadata.json')


def develop():
    source = z_scores.OUT/'development.npz'
    diagnostic = rv.load(z_scores.OUT/'summary.json')
    assert rv.digest(source) == diagnostic['scores_sha256']['development']
    maps = {}
    with np.load(source) as rows:
        for spec in gen.MAPS:
            seed = str(spec['local_seed'])
            events, truth, layout, matching, _ = open_case(gen.OUT/'evaluation'/seed)
            raw, weight = matching.decode_batch(events, return_weights=True)
            raw = adoption.bench.prior.paired.class4(raw)
            np.testing.assert_array_equal(raw, rows[seed+'_raw'])
            np.testing.assert_array_equal(truth, rows[seed+'_truth'])
            late = rows[seed+'_selected']
            delta = outcome(raw, truth, late)
            maps[seed] = np.column_stack((weight, round_counts(events, layout))), delta
    cv = {}
    for hold, (features, delta) in maps.items():
        train = [value for key, value in maps.items() if key != hold]
        x = np.concatenate([f for f, _ in train])
        y = np.concatenate([d.mean(axis=0) for _, d in train])
        state = fit(x, y)
        train_scores, test_scores = score(x, state), score(features, state)
        cv[hold] = {str(rate): gate_effect(delta, test_scores > cutoff(train_scores, rate))
                    for rate in TARGET_RATES}
        cv[hold]['never'] = gate_effect(delta, np.zeros(len(features), dtype=bool))
        cv[hold]['always'] = gate_effect(delta, np.ones(len(features), dtype=bool))
    totals = {key: {field: sum(rows[key][field] for rows in cv.values())
                    for field in ('blocked_harms', 'missed_repairs', 'net', 'admitted', 'physical_shots')}
              for key in next(iter(cv.values()))}
    selected = max(TARGET_RATES, key=lambda rate: (totals[str(rate)]['net'], -rate))
    full_x = np.concatenate([features for features, _ in maps.values()])
    full_y = np.concatenate([delta.mean(axis=0) for _, delta in maps.values()])
    gate = fit(full_x, full_y)
    gate['cutoff'] = cutoff(score(full_x, gate), selected)
    OUT.mkdir(parents=True, exist_ok=True)
    rv.save(OUT/'development.json', dict(development_sha256=rv.digest(source), folds=cv, totals=totals,
            target_rates=TARGET_RATES, selected_rate=selected, gate=gate))
    print(f'Development gate {selected}: {totals[str(selected)]}, cutoff {gate["cutoff"]}', flush=True)


def freeze():
    dev = OUT/'development.json'
    state = rv.load(dev)
    prior_path = adoption.OUT/'protocol.json'
    prior = rv.load(prior_path)
    used_local, used_sampler = set(), set()
    for path in (ROOT/'artifacts/eval').rglob('case.json'):
        spec = rv.load(path).get('settings', {})
        used_local.add(spec.get('local_seed'))
        used_sampler.add(spec.get('sampler_seed'))
    for spec in rv.load(adoption.final.dev.OUT/'data_registry.json')['training_sample_seeds']:
        used_local.add(spec['local_seed'])
        used_sampler.add(spec['seed'])
    assert not ({m['local_seed'] for m in MAPS} & used_local)
    assert not ({m[key] for m in MAPS for key in ('g0_sampler','g1_sampler')} & used_sampler)
    for path, digest in prior['sources'].items():
        assert rv.digest(ROOT/path) == digest, path
    extra = [Path(__file__), PLAN, dev, prior_path, z_scores.OUT/'summary.json', Path(z_scores.__file__)]
    hashes = {path.relative_to(ROOT).as_posix(): rv.digest(path) for path in extra}
    protocol = dict(sources={**prior['sources'], **hashes}, maps=MAPS, g0_shots=32, g1_shots=gen.SHOTS,
                    zz_factors=prior['zz_factors'], models=prior['models'], head=prior['heads']['joint'],
                    late_cutoff=prior['threshold']['joint'], gate=state['gate'], level=0.95,
                    timing_repeats=REPS, threads=gen.THREADS, l2=L2, versions=prior['versions'])
    assert protocol['late_cutoff'] == 0.025
    rv.save(OUT/'protocol.json', protocol)
    print(f'Frozen {len(MAPS)} new maps, gate cutoff {state["gate"]["cutoff"]}', flush=True)


def preparation(events, layout, meta, panel):
    static, _ = sedp.volume_common._build_static_planes(layout)
    context, names = sedp.volume_common._build_context_planes(
        {**meta, 'qc_stats': {'detector_event_fraction': adoption.final.CONTEXT}}, layout,
    )
    assert np.all(context[names.index('reserved_label_context')] == 0)
    prepared = {}
    for seed, checkpoint in panel.items():
        model, _, kwargs, _ = rv.load_selector(checkpoint)
        prepared[seed] = model, gen.learned_patterns(layout, events, kwargs)
    return static, context, prepared


def active_inputs(events, raw, first_weight, per_round, admitted, solver):
    if not admitted.any():
        return None
    event_slice, raw_slice = events[admitted], raw[admitted]
    costs = solver.solve(event_slice)
    shot = np.arange(len(event_slice))
    best_weight = costs[shot, raw_slice]
    np.testing.assert_allclose(best_weight, first_weight[admitted], atol=3e-4, rtol=0)
    np.testing.assert_allclose(best_weight, costs.min(axis=1), atol=3e-4, rtol=0)
    relative = costs[shot[:, None], raw_slice[:, None] ^ adoption.CLASSES] - best_weight[:, None]
    reachable = np.isfinite(relative[:, 1:])
    raw_features = np.column_stack((best_weight, np.eye(4)[raw_slice][:,1:], per_round[admitted],
                                    np.where(reachable, relative[:, 1:], 0), reachable))
    return event_slice, raw_slice, best_weight, raw_features


def apply_late(raw, admitted, active, layout, matching, prepared, head, late_cutoff):
    output = raw.copy()
    if active is None:
        return output
    event_slice, raw_slice, best_weight, raw_features = active
    static, context, model, patterns = prepared
    masks = gen.learned_masks(event_slice, layout, static, context, model, patterns)
    features, rows, _ = adoption.candidate_features(event_slice, masks, matching, raw_slice, best_weight)
    probability = adoption.probabilities(np.column_stack((raw_features, features)), head)
    output[admitted] = raw_slice ^ adoption.decide(probability, rows >= 0, late_cutoff)
    return output


def infer(mode, events, layout, matching, solver, prepared, head, gate, late_cutoff):
    if mode == 'raw':
        raw = adoption.bench.prior.paired.class4(matching.decode_batch(events))
        return raw, raw, np.zeros(len(events), dtype=bool)
    decoded, first_weight = matching.decode_batch(events, return_weights=True)
    raw = adoption.bench.prior.paired.class4(decoded)
    per_round = round_counts(events, layout)
    admitted = (score(np.column_stack((first_weight, per_round)), gate) > gate['cutoff']
                if mode == 'gated' else np.ones(len(events), dtype=bool))
    active = active_inputs(events, raw, first_weight, per_round, admitted, solver)
    output = apply_late(raw, admitted, active, layout, matching, prepared, head, late_cutoff)
    return raw, output, admitted


def map_accuracy(spec, protocol, protocol_hash):
    sedp.torch.set_num_threads(protocol['threads'])
    seed = spec['local_seed']
    base = OUT/'evaluation'/str(seed)
    file, marker = base/'predictions.npz', base/'accuracy.json'
    if marker.exists():
        row = rv.load(marker)
        assert row['protocol_sha256'] == protocol_hash and row['sha256'] == rv.digest(file)
        return row
    for name, zz in zip(('g0','g1'), protocol['zz_factors']):
        rv.generate_case(base/name, distance=7, local_seed=seed, sampler_seed=spec[name+'_sampler'],
                         shots=protocol[name+'_shots'], zz_factor=zz)
    events, truth, layout, matching, meta = open_case(base)
    assert len(events) == protocol['g1_shots']
    static, context, models = preparation(events, layout, meta, gen.checkpoints())
    solver = BoundaryCosts(matching)
    raw = adoption.bench.prior.paired.class4(matching.decode_batch(events))
    weighted, first_weight = matching.decode_batch(events, return_weights=True)
    np.testing.assert_array_equal(raw, adoption.bench.prior.paired.class4(weighted))
    per_round = round_counts(events, layout)
    all_shots = np.ones(len(events), dtype=bool)
    gate = protocol['gate']
    mask = score(np.column_stack((first_weight, per_round)), gate) > gate['cutoff']
    # Accuracy shares model-independent inputs. Timing recomputes all work for each route.
    full = active_inputs(events, raw, first_weight, per_round, all_shots, solver)
    gated_inputs = active_inputs(events, raw, first_weight, per_round, mask, solver)
    late, gated = [], []
    for model, patterns in models.values():
        args = (layout, matching, (static, context, model, patterns), protocol['head'], protocol['late_cutoff'])
        p_late = apply_late(raw, all_shots, full, *args)
        p_gate = apply_late(raw, mask, gated_inputs, *args)
        np.testing.assert_array_equal(p_gate, np.where(mask, p_late, raw))
        late.append(p_late)
        gated.append(p_gate)
    late, gated = np.stack(late), np.stack(gated)
    effect = gate_effect(outcome(raw, truth, late), mask)
    late_counts, gated_counts = (adoption.counts(raw, truth, output) for output in (late, gated))
    assert (gated_counts['improved'] - gated_counts['harmed']) - (late_counts['improved'] - late_counts['harmed']) == effect['net']
    with file.open('xb') as stream:
        np.savez_compressed(stream, raw=raw, truth=truth, late=late, gated=gated, admitted=mask)
    row = dict(seed=seed, shots=len(raw), raw_correct=int((raw == truth).sum()), late=late_counts,
               gated=gated_counts, effect=effect, protocol_sha256=protocol_hash, sha256=rv.digest(file))
    rv.save(marker, row)
    print(f'Accuracy map {seed} complete', flush=True)
    return row


def map_timing(spec, protocol, protocol_hash, index):
    seed, checkpoint = spec['local_seed'], spec['timing_checkpoint']
    base = OUT/'evaluation'/str(seed)
    marker, file = base/'timing.json', base/'predictions.npz'
    accuracy = rv.load(base/'accuracy.json')
    assert accuracy['protocol_sha256'] == protocol_hash and accuracy['sha256'] == rv.digest(file)
    if marker.exists():
        row = rv.load(marker)
        assert row['protocol_sha256'] == protocol_hash and row['accuracy_sha256'] == accuracy['sha256']
        return row
    events, _, layout, matching, meta = open_case(base)
    hashes = rv.checkpoints(shift.REVALIDATION, 7, [checkpoint])
    static, context, models = preparation(events, layout, meta, hashes)
    model, patterns = models[checkpoint]
    shared = dict(events=events, layout=layout, matching=matching, solver=BoundaryCosts(matching),
                  prepared=(static, context, model, patterns), head=protocol['head'], gate=protocol['gate'],
                  late_cutoff=protocol['late_cutoff'])
    with np.load(file) as saved:
        expected = {key: saved[key][checkpoint] if key != 'raw' else saved[key] for key in MODES}
        mask = saved['admitted']
    times = {key: [] for key in MODES}
    for rep in range(protocol['timing_repeats'] + 1):
        offset = (index + rep) % len(MODES)
        for key in MODES[offset:] + MODES[:offset]:
            started = perf_counter()
            raw, pred, admitted = infer(key, **shared)
            elapsed = perf_counter() - started
            np.testing.assert_array_equal(raw, expected['raw'])
            np.testing.assert_array_equal(pred, expected[key])
            np.testing.assert_array_equal(admitted, mask if key == 'gated' else key == 'late')
            times[key].append(elapsed)
    row = dict(seed=seed, checkpoint=checkpoint, warmup={key: value[0] for key, value in times.items()},
               seconds={key: value[1:] for key, value in times.items()}, protocol_sha256=protocol_hash,
               accuracy_sha256=accuracy['sha256'])
    rv.save(marker, row)
    print(f'Timing map {seed} complete', flush=True)
    return row


def interval(values, level):
    values = np.asarray(values)
    mean = float(values.mean())
    half = float(t.ppf((1+level)/2, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean, interval=[mean-half, mean+half], values=values.tolist())


def summarize(accuracy, timing, protocol):
    level = protocol['level']
    gate_gain = interval([row['gated']['gain_pp'] for row in accuracy], level)
    late_gain = interval([row['late']['gain_pp'] for row in accuracy], level)
    effect = interval([row['effect']['gain_pp'] for row in accuracy], level)
    per_map_ms = {key: np.asarray([1000*np.median(row['seconds'][key]) for row in timing]) for key in MODES}
    savings = interval(per_map_ms['late']-per_map_ms['gated'], level)
    admitted = sum(row['effect']['admitted'] for row in accuracy)
    shots = sum(row['shots'] for row in accuracy)
    return dict(raw_accuracy_pp=100*sum(row['raw_correct'] for row in accuracy)/shots, g1_shots=shots,
                admitted=admitted, admission_rate=admitted/shots, gated_vs_raw_pp=gate_gain,
                late_vs_raw_pp=late_gain, gated_vs_late_pp=effect, late_minus_gated_ms=savings,
                median_ms={key: float(np.median(values)) for key, values in per_map_ms.items()},
                totals={key: {field: sum(row[key][field] for row in accuracy) for field in fields}
                        for key, fields in {'late':('improved','harmed','changed','pairs'),
                                           'gated':('improved','harmed','changed','pairs'),
                                           'effect':('blocked_harms','missed_repairs','net')}.items()},
                adopt=bool(savings['mean'] > 0 and gate_gain['interval'][0] > 0),
                accuracy=accuracy, timing=timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('develop','freeze','run'))
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate Anaconda fp first')
    assert os.getenv('MKL_THREADING_LAYER') == 'SEQUENTIAL'
    sedp.torch.set_num_threads(gen.THREADS)
    if args.mode == 'develop':
        develop()
    elif args.mode == 'freeze':
        freeze()
    else:
        path = OUT/'protocol.json'
        protocol, digest = rv.load(path), rv.digest(path)
        for source, expected in protocol['sources'].items():
            assert rv.digest(ROOT/source) == expected, source
        for checkpoint in protocol['models'].values():
            assert rv.digest(ROOT/checkpoint['checkpoint']) == checkpoint['sha256']
        assert protocol['versions'] == {name: version(name) for name in adoption.final.VERSIONS}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(map_accuracy, spec, protocol, digest) for spec in protocol['maps']]
            accuracy = [task.result() for task in tasks]
        timing = [map_timing(spec, protocol, digest, index) for index, spec in enumerate(protocol['maps'])]
        summary = summarize(accuracy, timing, protocol)
        rv.save(OUT/'summary.json', dict(protocol_sha256=digest, platform=platform.platform(),
                processor=os.environ.get('PROCESSOR_IDENTIFIER'), **summary))
        for key in ('gated_vs_raw_pp','gated_vs_late_pp','late_minus_gated_ms'):
            print(key, summary[key]['mean'], summary[key]['interval'], flush=True)


if __name__ == '__main__':
    main()
