"""Fit and independently evaluate a d7 logical-class adoption policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_candidate_generators as gen
from tools.d7_boundary_matching_costs import BoundaryCosts, checked_costs

bench, final, rv, sedp, shift = gen.bench, gen.final, gen.rv, gen.sedp, gen.shift
OUT = ROOT / 'artifacts/eval/d7_adoption/20261117'
PLAN = ROOT / 'docs/thesis/PREDECODER_D7_ADOPTION_PROTOCOL.md'
MAPS = tuple(dict(local_seed=788300 + i, g0_sampler=1794300 + 2*i, g1_sampler=1794301 + 2*i)
             for i in range(24))
THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
METHODS = ('joint', 'cost_supported', 'cost_unrestricted')
LEVEL = 0.975
L2 = 0.01
CLASSES = np.arange(4, dtype=np.uint8)


def feature_rows(base, replay=None, panel=None):
    g0, g1 = (bench.dev.data_root(base / strength) for strength in ('g0', 'g1'))
    for strength, directory in zip(('g0', 'g1'), (g0, g1)):
        case = base / strength
        record = rv.load(case/'case.json')
        for name in ('samples.npz', 'metadata.json', 'detector_error_model.dem', 'circuit.stim'):
            path = directory / name
            assert rv.digest(path) == record['sha256'][path.relative_to(case).as_posix()]
    matching = shift.open_decoders([g0])[shift.D0_MWPM].matching
    with np.load(g1/'samples.npz') as data:
        events = data['detector_events']
        layout = gen.build_rectangular_syndrome_layout(data)
        truth = bench.prior.paired.class4(data['observable_flips'])
        np.testing.assert_array_equal(truth, 2*data['logical_z_flip'] + data['logical_x_flip'])
    started = perf_counter()
    raw, costs = checked_costs(matching, BoundaryCosts(matching), events)
    shots = np.arange(len(events))
    raw_weight = costs[shots, raw]
    relative = costs[shots[:, None], raw[:, None] ^ CLASSES] - raw_weight[:, None]
    reachable = np.isfinite(relative)
    by_round = [events[:, layout.detector_time_index == step].sum(axis=1)
                for step in range(layout.time_steps)]
    raw_features = np.column_stack((raw_weight, np.eye(4)[raw][:, 1:], *by_round,
                                    np.where(reachable[:, 1:], relative[:, 1:], 0),
                                    reachable[:, 1:]))
    raw_seconds = perf_counter() - started
    if replay is not None:
        np.testing.assert_array_equal(replay['raw'], raw)
        np.testing.assert_array_equal(replay['truth'], truth)
    else:
        static, _ = sedp.volume_common._build_static_planes(layout)
        meta = rv.load(g0/'metadata.json')
        context, names = sedp.volume_common._build_context_planes(
            {**meta, 'qc_stats': {'detector_event_fraction': final.CONTEXT}}, layout,
        )
        assert np.all(context[names.index('reserved_label_context')] == 0)
    all_features, all_rows, all_classes, seconds = [], [], [], []
    for seed in gen.MODELS:
        started = perf_counter()
        name = f'learned{seed}'
        if replay is not None:
            masks = np.unpackbits(replay[name+'_masks'], axis=2, count=events.shape[1])
        else:
            model, _, kwargs, _ = rv.load_selector(panel[seed])
            started = perf_counter()
            patterns = gen.learned_patterns(layout, events, kwargs)
            masks = gen.learned_masks(events, layout, static, context, model, patterns)
        features, rows, classes = candidate_features(events, masks, matching, raw, raw_weight)
        if replay is not None:
            np.testing.assert_array_equal(classes, replay[name+'_classes'])
        all_features.append(np.column_stack((raw_features, features)))
        all_rows.append(rows)
        all_classes.append(classes)
        seconds.append(perf_counter() - started)
    return dict(raw=raw, truth=truth, raw_features=raw_features, joint_features=np.stack(all_features),
                class_rows=np.stack(all_rows), classes=np.stack(all_classes), reachable=reachable,
                raw_feature_seconds=raw_seconds, candidate_feature_seconds=seconds)


def candidate_features(events, masks, matching, raw, raw_weight):
    n, k, d = masks.shape
    assert k == gen.CAP + 1 and not masks[:, 0].any() and np.all(np.any(masks[:, 1:], axis=2))
    edited = events[:, None, :] ^ masks[:, 1:]
    decoded, weight = matching.decode_batch(edited.reshape(-1, d), return_weights=True)
    classes = bench.prior.paired.class4(decoded).reshape(n, k - 1)
    relative = (weight.reshape(n, k - 1) - raw_weight[:, None])
    assert np.isfinite(relative).all()
    transition = classes ^ raw[:, None]
    ranks = np.arange(1, k)
    features, rows = [], []
    for target in CLASSES:
        eligible = transition == target
        count = eligible.sum(axis=1)
        first = np.min(np.where(eligible, ranks, k), axis=1)
        minimum = np.min(np.where(eligible, relative, np.inf), axis=1)
        average = np.where(eligible, relative, 0).sum(axis=1) / np.maximum(1, count)
        features.extend((count / (k - 1), first / k, np.where(count > 0, minimum, 0), average))
        if target == 0:
            rows.append(np.zeros(n, dtype=np.intp))
        else:
            rows.append(np.where(first < k, first, -1))
    return np.column_stack(features), np.column_stack(rows), np.column_stack((raw, classes))


def fit_matrix(features, target):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(features)), (features-center)/scale))
    truth = np.eye(4)[target]

    def loss(flat):
        beta = flat.reshape(x.shape[1], 4)
        logits = x @ beta
        error = softmax(logits, axis=1) - truth
        value = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(x)), target])
        value += 0.5 * L2 * np.square(beta[1:]).sum()
        gradient = x.T @ error / len(x)
        gradient[1:] += L2 * beta[1:]
        return float(value), gradient.ravel()

    result = minimize(loss, np.zeros(x.shape[1]*4), method='L-BFGS-B', jac=True,
                      options={'maxiter': 350, 'ftol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Class probability fit failed: {result.message}')
    return dict(center=center.tolist(), scale=scale.tolist(), beta=result.x.reshape(x.shape[1], 4).tolist())


def fit(maps):
    return {
        'cost': fit_matrix(np.concatenate([m['raw_features'] for m in maps]),
                           np.concatenate([m['raw'] ^ m['truth'] for m in maps])),
        'joint': fit_matrix(np.concatenate([m['joint_features'].reshape(-1, m['joint_features'].shape[-1])
                                            for m in maps]),
                            np.concatenate([np.tile(m['raw'] ^ m['truth'], len(gen.MODELS)) for m in maps])),
    }


def probabilities(features, head):
    scaled = (features - np.asarray(head['center'])) / np.asarray(head['scale'])
    beta = np.asarray(head['beta'])
    return softmax(scaled @ beta[1:] + beta[0], axis=-1)


def decide(probs, eligible, threshold):
    assert eligible[..., 0].all()
    alternatives = np.where(eligible[..., 1:], probs[..., 1:], -np.inf)
    best = np.argmax(alternatives, axis=-1)
    best_prob = np.take_along_axis(alternatives, best[..., None], axis=-1)[..., 0]
    return np.where(best_prob - probs[..., 0] > threshold, best + 1, 0).astype(np.uint8)


def predict(map_data, heads):
    probs = probabilities(map_data['raw_features'], heads['cost'])
    support = map_data['class_rows'] >= 0
    return {
        'joint': (probabilities(map_data['joint_features'], heads['joint']), support),
        'cost_supported': (np.broadcast_to(probs, support.shape), support),
        'cost_unrestricted': (probs, map_data['reachable']),
    }


def counts(raw, truth, selected):
    gain = (raw != truth) & (selected == truth)
    harm = (raw == truth) & (selected != truth)
    return dict(improved=int(gain.sum()), harmed=int(harm.sum()), changed=int((selected != raw).sum()),
                pairs=selected.size, gain_pp=100 * float(int(gain.sum()) - int(harm.sum())) / selected.size)


def develop():
    source = rv.load(gen.OUT/'summary.json')
    assert source['protocol_sha256'] == rv.digest(gen.OUT/'protocol.json')
    maps, inputs = {}, {}
    for row in source['maps']:
        seed = str(row['local_seed'])
        path = gen.OUT/'evaluation'/seed/'predictions.npz'
        assert rv.digest(path) == row['sha256']
        with np.load(path) as saved:
            maps[seed] = feature_rows(path.parent, replay=saved)
        inputs[path.relative_to(ROOT).as_posix()] = row['sha256']
    assert len(maps) == 24
    cv = {}
    for seed, target in maps.items():
        heads = fit([m for key, m in maps.items() if key != seed])
        cv[seed] = {key: [counts(target['raw'], target['truth'],
                                 target['raw'] ^ decide(probs, support, threshold)) for threshold in THRESHOLDS]
                    for key, (probs, support) in predict(target, heads).items()}
        print(f'Development fold {seed} complete', flush=True)
    means = {method: np.mean([[m[method][i]['gain_pp'] for i in range(len(THRESHOLDS))]
                              for m in cv.values()], axis=0).tolist() for method in METHODS}
    chosen = {method: max(range(len(THRESHOLDS)), key=lambda i: (means[method][i], THRESHOLDS[i]))
              for method in METHODS}
    OUT.mkdir(parents=True, exist_ok=True)
    model = fit(list(maps.values()))
    rv.save(OUT/'development.json', dict(inputs_sha256=inputs, folds=cv, threshold_grid=THRESHOLDS,
            gains_by_threshold_pp=means, threshold={key: THRESHOLDS[i] for key, i in chosen.items()},
            chosen_gain_pp={key: means[key][i] for key, i in chosen.items()}, heads=model))
    print('Development selected (threshold, gain pp):',
          {key: (THRESHOLDS[i], means[key][i]) for key, i in chosen.items()}, flush=True)


def freeze():
    development = OUT/'development.json'
    dev = rv.load(development)
    used_local, used_sampler = set(), set()
    for path in (ROOT/'artifacts/eval').rglob('case.json'):
        spec = rv.load(path).get('settings', {})
        used_local.add(spec.get('local_seed'))
        used_sampler.add(spec.get('sampler_seed'))
    registry = rv.load(final.dev.OUT/'data_registry.json')
    for spec in registry['training_sample_seeds']:
        used_local.add(spec['local_seed'])
        used_sampler.add(spec['seed'])
    assert not ({m['local_seed'] for m in MAPS} & used_local)
    assert not ({m[key] for m in MAPS for key in ('g0_sampler','g1_sampler')} & used_sampler)
    reference = rv.load(gen.OUT/'protocol.json')
    sources = {name: rv.digest(ROOT/name) for name in reference['sources']}
    for path in (Path(__file__), PLAN, development, gen.OUT/'protocol.json',
                 gen.OUT/'summary.json', ROOT/'tools/d7_boundary_matching_costs.py'):
        sources[path.relative_to(ROOT).as_posix()] = rv.digest(path)
    rv.save(OUT/'protocol.json', dict(
        sources=sources, models=reference['models'], maps=MAPS, heads=dev['heads'],
        threshold=dev['threshold'], level=LEVEL, g0_shots=32, g1_shots=gen.SHOTS,
        zz_factors=reference['zz_factors'], cell=shift.D0_MWPM,
        versions={name: version(name) for name in final.VERSIONS}, threads=gen.THREADS, l2=L2,
    ))
    print(f'Frozen {len(MAPS)} maps and thresholds {dev["threshold"]}', flush=True)


def evaluate(spec, protocol, protocol_hash):
    sedp.torch.set_num_threads(protocol['threads'])
    seed = spec['local_seed']
    base = OUT/'evaluation'/str(seed)
    marker = base/'result.json'
    file = base/'predictions.npz'
    if marker.exists():
        row = rv.load(marker)
        assert row['protocol_sha256'] == protocol_hash and row['sha256'] == rv.digest(file)
        return row
    for strength, factor in zip(('g0','g1'), protocol['zz_factors']):
        rv.generate_case(base/strength, distance=7, local_seed=seed, sampler_seed=spec[strength+'_sampler'],
                         shots=protocol[strength+'_shots'], zz_factor=factor)
    assert not file.exists()
    data = feature_rows(base, panel=gen.checkpoints())
    raw, truth = data['raw'], data['truth']
    assert len(raw) == protocol['g1_shots']
    scores = predict(data, protocol['heads'])
    selected = {key: raw ^ decide(probs, support, protocol['threshold'][key])
                for key, (probs, support) in scores.items()}
    indices = np.take_along_axis(data['class_rows'], (selected['joint'] ^ raw)[..., None], axis=-1)[..., 0]
    assert np.all(indices >= 0)
    np.testing.assert_array_equal(selected['joint'],
                                  np.take_along_axis(data['classes'], indices[..., None], axis=-1)[..., 0])
    with file.open('xb') as stream:
        np.savez_compressed(stream, truth=truth, raw=raw, class_rows=data['class_rows'],
                            candidate_classes=data['classes'], reachable=data['reachable'], joint_rows=indices,
                            joint_proba=scores['joint'][0], cost_proba=scores['cost_unrestricted'][0], **selected)
    row = dict(local_seed=seed, sha256=rv.digest(file), protocol_sha256=protocol_hash,
               raw_correct=int((raw==truth).sum()), shots=len(raw),
               methods={key: counts(raw, truth, pred) for key, pred in selected.items()},
               raw_feature_seconds=data['raw_feature_seconds'], candidate_feature_seconds=data['candidate_feature_seconds'])
    rv.save(marker, row)
    print(f'Evaluation map {seed} complete', flush=True)
    return row


def summarize(rows, level):
    gain = {key: np.array([row['methods'][key]['gain_pp'] for row in rows]) for key in METHODS}

    def interval(values, coverage):
        center = values.mean()
        half = t.ppf((1+coverage)/2, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        return dict(mean_pp=float(center), interval_pp=[float(center-half), float(center+half)],
                    per_map_pp=values.tolist())

    return dict(
        raw_accuracy_pp=100 * sum(r['raw_correct'] for r in rows) / sum(r['shots'] for r in rows),
        joint_vs_raw=interval(gain['joint'], level),
        joint_vs_cost_supported=interval(gain['joint']-gain['cost_supported'], level),
        cost_supported_vs_raw=interval(gain['cost_supported'], 0.95),
        cost_unrestricted_vs_raw=interval(gain['cost_unrestricted'], 0.95),
        totals={key: {field: sum(row['methods'][key][field] for row in rows)
                      for field in ('improved','harmed','changed','pairs')} for key in METHODS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('develop', 'freeze', 'run'))
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
        protocol = rv.load(path)
        for source, digest in protocol['sources'].items():
            assert rv.digest(ROOT/source) == digest, source
        for model in protocol['models'].values():
            assert rv.digest(ROOT/model['checkpoint']) == model['sha256']
        assert protocol['versions'] == {name: version(name) for name in final.VERSIONS}
        digest = rv.digest(path)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate, spec, protocol, digest) for spec in protocol['maps']]
            rows = [future.result() for future in futures]
        report = summarize(rows, protocol['level'])
        rv.save(OUT/'summary.json', dict(protocol_sha256=digest, **report, maps=rows))
        for key in ('joint_vs_raw', 'joint_vs_cost_supported'):
            print(key, report[key]['mean_pp'], report[key]['interval_pp'], flush=True)


if __name__ == '__main__':
    main()
