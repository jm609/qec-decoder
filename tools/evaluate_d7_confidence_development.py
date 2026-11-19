"""Compare the frozen d7 selector controls on reserved development and calibration maps."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_d7_paired_selector as paired
from tools.collect_d7_confidence_candidates import OUT, rv, shift
from tools.d7_boundary_matching_costs import BoundaryCosts, checked_costs, z_gap

PAIR_CONTROL = 'paired_no_events'
PAIR = 'paired_detector'
PAIR_GAP = 'paired_detector_gap'
RAW_HEAD = 'raw_z'
RAW_GAP = 'raw_z_gap'
Z_ONLY = 'z_vs_no_edit'
MARGIN = 'margin'
EFFECT_FLOOR = 0.05
GAP_TOLERANCE = 3e-4
SOURCE = ROOT / 'artifacts/eval/noise_shift_d7/20261115_retrained/development/783000'
PROTOCOL = ROOT / 'docs/thesis/PREDECODER_D7_CONFIDENCE_DEVELOPMENT_PROTOCOL.md'


def data_root(path: Path) -> Path:
    case = rv.load(path / 'case.json')
    target = path / case['target']
    return Path(rv.load(target / 'metadata.json')['source_family_dir'])


def verify_costs() -> dict:
    started = perf_counter()
    raw_dir, strong_dir = [data_root(SOURCE / strength) for strength in ('g0', 'g1')]
    matching = shift.open_decoders([raw_dir])[shift.D0_MWPM].matching
    solver = BoundaryCosts(matching)
    with np.load(strong_dir / 'samples.npz') as data, np.load(SOURCE / 'g1/predictions_seed2.npz') as saved:
        events = data['detector_events'].astype(np.uint8)
        masks = np.unpackbits(saved['packed_masks'][saved['top1_rows'][:24]], axis=1, count=events.shape[1])
        syndromes = np.unique(np.vstack((events[:40], events[:24] ^ masks)), axis=0)
    measured = perf_counter()
    classes, costs = checked_costs(matching, solver, syndromes)
    fast_seconds = perf_counter() - measured
    measured = perf_counter()
    reference = solver.solve(syndromes, independent=True)
    independent_seconds = perf_counter() - measured
    np.testing.assert_allclose(costs, reference, atol=GAP_TOLERANCE, rtol=0)
    gap = z_gap(classes, costs)
    projected_rows = 6 * 2 * 1024 * (1 + 2 * 8)
    return {
        'validation_map': 783000, 'shots_after_deduplication': len(syndromes), 'classes': costs.shape[1],
        'boundary_groups': [list(key) for key in solver.ports], 'absolute_tolerance': GAP_TOLERANCE,
        'max_abs_error': float(np.max(np.abs(costs-reference))), 'matching_seconds': fast_seconds,
        'independent_seconds': independent_seconds, 'wall_seconds': perf_counter() - started,
        'projected_max_rows': projected_rows, 'projected_gap_seconds': fast_seconds*projected_rows/len(syndromes),
        'unreachable_z_alternatives': int(gap[:, 1].sum()), 'z_ties': int(np.all(gap == 0, axis=1).sum()),
        'demo_paths_sha256': {str(path.relative_to(ROOT)): rv.digest(path) for path in (
            raw_dir / 'detector_error_model.dem', strong_dir / 'samples.npz', SOURCE / 'g1/predictions_seed2.npz',
        )},
    }


def make_features(arrays: dict, events: np.ndarray, times: np.ndarray, lookup: np.ndarray, order: np.ndarray,
                  seed: int, matching, solver: BoundaryCosts | None, raw_gap: np.ndarray | None) -> tuple:
    case = paired.make_cases({**arrays, 'truth': np.zeros_like(arrays['d0_mwpm_raw'])}, events, lookup, order, 0)
    shots = case.pair_shots
    z_rows, keep_rows = case.top[shots], case.preserving_rows
    # The original head has four historical model slots. Replace them by the fixed eight-model panel.
    old_model_count = len(paired.probe.CHECKPOINTS)
    geometry_z, geometry_keep = np.hsplit(case.control[:, 8 + old_model_count:], 2)
    patch_z, _ = np.hsplit(case.patch_events, 2)
    ids = np.broadcast_to(np.eye(8)[seed], (len(shots), 8))
    raw_category = case.control[:, 4:8]
    control = np.column_stack((case.control[:, :8], ids, geometry_z, geometry_keep))
    raw = events[shots]
    raw_features = np.column_stack((raw.mean(axis=1), *(
        raw[:, times == step].mean(axis=1) for step in np.unique(times)
    ), raw_category, ids))
    features = {
        PAIR_CONTROL: control,
        PAIR: np.column_stack((control, case.patch_events)),
        Z_ONLY: np.column_stack((case.control[:, [1, 2]], raw_category, ids, geometry_z, patch_z)),
        RAW_HEAD: raw_features,
    }
    gap_seconds = 0.0
    if solver is not None:
        started = perf_counter()
        if raw_gap is None:
            raise ValueError('Missing raw confidence values')
        extra = []
        for rows in (z_rows, keep_rows):
            masks = np.unpackbits(arrays['packed_masks'][rows], axis=1, count=events.shape[1])
            classes, costs = checked_costs(matching, solver, raw ^ masks)
            np.testing.assert_array_equal(classes, case.prediction[rows])
            extra.append(z_gap(classes, costs))
        features[RAW_GAP] = np.column_stack((raw_features, raw_gap[shots]))
        features[PAIR_GAP] = np.column_stack((features[PAIR], raw_gap[shots, 1:], *extra))
        gap_seconds = perf_counter() - started
    assert all(np.isfinite(matrix).all() for matrix in features.values())
    return replace(case, truth=paired.class4(arrays['truth'])), features, gap_seconds


def load_map(path: Path, gap_deadline: float | None, verify: bool) -> tuple[dict, dict]:
    result = rv.load(path / 'result.json')
    raw_dirs = [data_root(path / strength) for strength in ('g0', 'g1')]
    matching = shift.open_decoders(raw_dirs[:1])[shift.D0_MWPM].matching
    solver = BoundaryCosts(matching) if gap_deadline is not None else None
    cases, stats = {}, {}
    for strength, raw in zip(('g0', 'g1'), raw_dirs, strict=True):
        with np.load(raw / 'samples.npz') as data:
            events = data['detector_events'].astype(np.uint8)
            truth = data['observable_flips']
            np.testing.assert_array_equal(truth, np.column_stack((data['logical_z_flip'], data['logical_x_flip'])))
            layout = paired.build_rectangular_syndrome_layout(data)
        times = layout.detector_time_index
        assert len(times) == events.shape[1]
        lookup, order = paired.probe.patch_lookup(layout)
        raw_gap = None
        gap_elapsed = 0.0
        if solver is not None:
            started = perf_counter()
            classes, costs = checked_costs(matching, solver, events)
            _, optimum = matching.decode_batch(events, return_weights=True)
            raw_gap = np.column_stack((optimum, z_gap(classes, costs)))
            gap_elapsed += perf_counter() - started
        roles = []
        for seed in range(8):
            record = result['data'][strength][str(seed)]
            file = path / strength / record['file']
            assert rv.digest(file) == record['sha256']
            with np.load(file) as data:
                arrays = dict(data)
            np.testing.assert_array_equal(arrays['truth'], truth)
            case, features, elapsed = make_features(arrays, events, times, lookup, order, seed,
                                                   matching, solver, raw_gap)
            if verify and strength == 'g1' and seed == 2:
                altered, altered_features, elapsed2 = make_features(
                    {**arrays, 'truth': truth ^ 1}, events, times, lookup, order, seed, matching, solver, raw_gap,
                )
                assert not np.array_equal(case.truth, altered.truth)
                for name in features:
                    np.testing.assert_array_equal(features[name], altered_features[name])
                for attr in ('raw', 'prediction', 'weight', 'top', 'margin', 'no_edit', 'pair_shots', 'preserving_rows'):
                    np.testing.assert_array_equal(getattr(case, attr), getattr(altered, attr))
                elapsed += elapsed2
            if gap_deadline is not None and perf_counter() > gap_deadline:
                raise TimeoutError('The gap calculation exceeded the 30-minute budget')
            gap_elapsed += elapsed
            roles.append((case, features))
        cases[strength] = roles
        stats[strength] = {
            'gap_seconds': gap_elapsed, 'candidate_seconds': sum(
                row['wall_seconds'] for row in result['data'][strength].values()
            ),
            'matching_rows_in_candidate_construction': sum(
                row['propose']['rows'] for row in result['data'][strength].values()
            ),
            'raw_gap': raw_gap, 'raw_correct': roles[0][0].raw == roles[0][0].truth,
        }
    return cases, stats


def fit(maps: list[list[tuple]]) -> dict:
    rows = [entry for group in maps for entry in group]
    outcome = np.concatenate([case.outcomes() for case, _ in rows])
    return {name: paired.fit_head(np.concatenate([matrix[name] for _, matrix in rows]), outcome)
            for name in rows[0][1]}


def count(case: paired.Cases, rows: np.ndarray) -> dict:
    predictions = case.prediction[rows]
    counts = paired.count_routes([case], [rows])
    changed = int(np.count_nonzero(predictions != case.raw))
    gain, loss = counts['corrections'], counts['harms']
    assert changed >= gain + loss
    return {**counts, 'raw_correct': int(np.count_nonzero(case.raw == case.truth)),
            'prediction_changes': changed, 'edit_fraction': counts['edits']/len(rows),
            'prediction_change_fraction': changed/len(rows)}


def evaluate(group: list[tuple], heads: dict, output: dict, key: str) -> dict:
    records = {}
    for seed, (case, matrices) in enumerate(group):
        rows = {'raw': (case.no_edit, case.no_edit), MARGIN: (case.margin, case.margin),
                'top1': (case.top, case.top)}
        for name, state in heads.items():
            score = paired.head_score(matrices[name], state)
            rows[name] = paired.select_rows(case, score >= 0)
        record = {}
        for name, (fixed, free) in rows.items():
            record[name] = {'fixed': count(case, fixed), 'free': count(case, free),
                            'fixed_free_prediction_differences':
                                int(np.count_nonzero(case.prediction[fixed] != case.prediction[free]))}
            output[f'{key}_seed{seed}_{name}'] = np.stack((case.prediction[fixed], case.prediction[free]))
        output[f'{key}_seed{seed}_truth'] = case.truth
        records[str(seed)] = {'paired_shots': len(case.pair_shots), 'routes': record}
    return records


def gains_by_map(cv: dict, strength: str) -> dict[str, list[float]]:
    first = next(iter(cv.values()))[strength]['0']['routes']
    return {name: [float(np.mean([row['routes'][name]['free']['gain_pp'] for row in fold[strength].values()]))
                   for fold in cv.values()] for name in first}


def select(cv: dict) -> dict:
    gains = gains_by_map(cv, 'g1')
    avg = {name: float(np.mean(rows)) for name, rows in gains.items()}
    simple = [name for name in (MARGIN, RAW_HEAD, Z_ONLY, RAW_GAP) if name in avg]
    best_simple_gain = max(avg[name] for name in simple)
    best = next(name for name in simple if best_simple_gain - avg[name] <= EFFECT_FLOOR)

    def qualifies(method: str, control: str) -> bool:
        return (avg[method] - avg[control] >= EFFECT_FLOOR
                and np.count_nonzero(np.array(gains[method]) > gains[control]) >= 3)

    chosen = PAIR if qualifies(PAIR, best) else best
    if PAIR_GAP in avg and qualifies(PAIR_GAP, PAIR) and qualifies(PAIR_GAP, best):
        chosen = PAIR_GAP
    return {'chosen': chosen, 'simple_comparator': best, 'mean_g1_free_gain_pp': avg,
            'g1_free_gain_pp_by_map': gains, 'g0_free_gain_pp_by_map': gains_by_map(cv, 'g0'),
            'selection_comparison': [chosen, best], 'effect_floor_pp': EFFECT_FLOOR,
            'wins_required': 3, 'tiebreak_ascending_cost': simple}


def calibration(stats: dict, train_roots: dict, roles: dict) -> dict:
    if next(iter(stats.values()))['g1']['raw_gap'] is None:
        return {}
    train = np.concatenate([stats[seed]['g1']['raw_gap'] for seed in train_roots])
    informative = (train[:, 2] == 0) & (train[:, 1] > 0)
    boundaries = np.quantile(train[informative, 1], [0.25, 0.50, 0.75])
    bins = {}
    for seed in roles:
        bins[seed] = {}
        for strength, details in stats[seed].items():
            gap = details['raw_gap']
            category = np.digitize(gap[:, 1], boundaries)
            category[(gap[:, 1] == 0) & (gap[:, 2] == 0)] = 4
            category[gap[:, 2] == 1] = 5
            bins[seed][strength] = [{
                'shots': int(np.sum(category == label)),
                'raw_errors': int(np.sum((category == label) & ~details['raw_correct'])),
            } for label in range(6)]
    return {'g1_training_quartiles': boundaries.tolist(), 'bins': bins, 'tie_bin': 4, 'unreachable_bin': 5}


def compare(probe: dict) -> None:
    if (OUT / 'stage2_summary.json').exists() or (OUT / 'stage2_predictions.npz').exists():
        raise FileExistsError('Stage 2 results already exist')
    registry = rv.load(OUT / 'data_registry.json')
    folders = {role: {str(spec['local_seed']): OUT / role / str(spec['local_seed']) for spec in panel}
               for role, panel in registry['reserved'].items()}
    gap_limit = 30*60 - probe['wall_seconds']
    gap_deadline = perf_counter() + gap_limit if probe['projected_gap_seconds'] < gap_limit else None
    def load_features(deadline: float | None) -> tuple[dict, dict]:
        maps, stats = {}, {}
        for role, paths in folders.items():
            for seed, path in paths.items():
                maps[seed], stats[seed] = load_map(path, deadline, verify=(seed == '787000'))
                print('Features', seed, role, flush=True)
        return maps, stats

    gap_status = 'completed' if gap_deadline is not None else 'projection_exceeds_budget'
    try:
        maps, stats = load_features(gap_deadline)
    except TimeoutError:
        gap_status = 'calculation_exceeded_budget'
        maps, stats = load_features(None)
    development = folders['development']
    cv, folds, predictions = {}, {}, {}
    for seed in development:
        heads = fit([maps[fit_seed]['g1'] for fit_seed in development if fit_seed != seed])
        folds[seed] = heads
        cv[seed] = {strength: evaluate(group, heads, predictions, f'cv_{seed}_{strength}')
                    for strength, group in maps[seed].items()}
    chosen = select(cv)
    frozen = fit([maps[seed]['g1'] for seed in development])
    calibrated = {seed: {strength: evaluate(group, frozen, predictions, f'cal_{seed}_{strength}')
                         for strength, group in maps[seed].items()} for seed in folders['calibration']}
    prediction_path = OUT / 'stage2_predictions.npz'
    with prediction_path.open('xb') as stream:
        np.savez_compressed(stream, **predictions)
    rv.save(OUT / 'stage2_summary.json', {
        'protocol_sha256': rv.digest(PROTOCOL), 'registry_sha256': rv.digest(OUT / 'data_registry.json'),
        'predictions_sha256': rv.digest(prediction_path), 'runner_sha256': rv.digest(Path(__file__)),
        'candidate_results_sha256': {seed: rv.digest(path / 'result.json')
                                     for paths in folders.values() for seed, path in paths.items()},
        'gap_validation': probe, 'gap_status': gap_status, 'features_label_independence_checked': True,
        'cross_validation': cv, 'selection': chosen, 'fold_heads': folds, 'refit_heads': frozen,
        'calibration': calibrated, 'gap_calibration': calibration(stats, development, folders['calibration']),
        'preparation_cost': {seed: {strength: {k: v for k, v in part.items() if k not in ('raw_gap','raw_correct')}
                                    for strength, part in values.items()} for seed, values in stats.items()},
    })
    print(chosen, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'compare'))
    args = parser.parse_args()
    path = OUT / 'stage2_gap_probe.json'
    if args.mode == 'probe':
        result = verify_costs()
        rv.save(path, result)
        print(result, flush=True)
    else:
        compare(rv.load(path))
