"""Evaluate a frozen d7 selector across ZZ strengths and matching priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pymatching
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circuits import build_memory_circuit
from config import ExperimentConfig
from noise_willowcore import build_correlated_edge_maps, build_local_rate_maps
from tools import revalidate_predecoder as rv

sedp = rv.sedp
REVALIDATION = ROOT / 'artifacts/eval/predecoder_revalidation_20261115'
SEED_PLAN = rv.NOISE_SHIFT_RESERVED_D7
PANEL = SEED_PLAN['model_seeds']
G0, G1 = SEED_PLAN['zz_factors']
D0_MWPM = 'd0_mwpm'
BATCH = 512
CALIBRATION_SHOTS = 154
BOOTSTRAP_SEED = 1783960
BOOTSTRAP_REPS = 20000
EFFECT_FLOOR_PP = 0.25


@dataclass
class Decoder:
    matching: pymatching.Matching
    correlated: bool
    batch_calls: int = 0
    single_calls: int = 0
    rows: int = 0

    def decode_batch(self, events: np.ndarray) -> np.ndarray:
        self.batch_calls += 1
        self.rows += len(events)
        return self.matching.decode_batch(events, enable_correlations=self.correlated)

    def decode(self, event: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        self.rows += 1
        return self.matching.decode(event, enable_correlations=self.correlated)

    def counts(self) -> dict[str, int]:
        return dict(batch_calls=self.batch_calls, single_calls=self.single_calls, rows=self.rows)


def open_decoders(raw_paths: list[Path]) -> dict[str, Decoder]:
    modes = {'mwpm': False, 'correlated': True}
    return {
        f'd{prior}_{name}': Decoder(pymatching.Matching.from_detector_error_model(
            stim.DetectorErrorModel.from_file(str(raw_path / 'detector_error_model.dem')),
            enable_correlations=enabled,
        ), enabled)
        for prior, raw_path in enumerate(raw_paths) for name, enabled in modes.items()
    }


def decode_edits(decoder: Decoder, events: np.ndarray, masks: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = raw.copy()
    failed = np.zeros(len(events), dtype=bool)
    edited = np.flatnonzero(np.any(masks, axis=1))
    if not edited.size:
        return predictions, failed
    modified = events[edited] ^ masks[edited]
    try:
        predictions[edited] = decoder.decode_batch(modified)
    except ValueError:
        for row, event in zip(edited, modified):
            try:
                predictions[row] = decoder.decode(event)
            except ValueError as error:
                if 'perfect matching' not in str(error).lower():
                    raise
                failed[row] = True
    return predictions, failed


def make_pair(path: Path, local_seed: int, shots: int) -> tuple[list[Path], list[Path], dict]:
    sampler_base = SEED_PLAN['sampler_base'] + 2 * (local_seed - SEED_PLAN['development_local_seeds'][0])
    targets = [rv.generate_case(path / f'g{i}', distance=7, local_seed=local_seed,
                                sampler_seed=sampler_base + i, shots=shots, zz_factor=zz)
               for i, zz in enumerate((G0, G1))]
    raw = [Path(rv.load(x / 'metadata.json')['source_family_dir']) for x in targets]
    circuits = [stim.Circuit.from_file(str(x / 'circuit.stim')) for x in raw]
    assert circuits[0].without_noise() == circuits[1].without_noise()
    assert circuits[0].num_observables == 2
    metadata = [rv.load(x / 'metadata.json') for x in raw]
    assert metadata[0]['dem'] == metadata[1]['dem']
    assert [m['sampling']['seed'] for m in metadata] == [sampler_base, sampler_base + 1]
    observed_rates = []
    for raw_dir in raw:
        with np.load(raw_dir / 'samples.npz') as arrays:
            assert arrays['detector_events'].shape == (shots, circuits[0].num_detectors)
            np.testing.assert_array_equal(arrays['observable_flips'], np.column_stack(
                [arrays['logical_z_flip'], arrays['logical_x_flip']],
            ))
            observed_rates.append(float(arrays['detector_events'].mean()))

    configs = [ExperimentConfig.from_dict(rv.load(x / 'config.json')['config']) for x in raw]
    assert configs[0].willow.local.seed == local_seed
    cfg0, cfg1 = configs
    assert cfg0 == replace(cfg1, willow=replace(cfg1.willow, corr=replace(cfg1.willow.corr, zz_factor=G0)))
    local_maps = [build_local_rate_maps(build_memory_circuit(cfg), cfg) for cfg in configs]
    assert local_maps[0] == local_maps[1]
    edges = [build_correlated_edge_maps(cfg, local) for cfg, local in zip(configs, local_maps)]
    assert edges[0].p_corr_swap == edges[1].p_corr_swap
    np.testing.assert_allclose(list(edges[1].p_corr_zz.values()), 5 * np.array(list(edges[0].p_corr_zz.values())), rtol=1e-12, atol=0)
    return targets, raw, {'sampler_seeds': [sampler_base, sampler_base + 1], 'detector_event_fraction': observed_rates,
                          'zz_by_edge': [{str(key): value for key, value in e.p_corr_zz.items()} for e in edges]}


def with_context(entry: sedp.PreparedEditFamily, ref_metadata: dict, event_fraction: float) -> tuple[sedp.PreparedEditFamily, dict]:
    # Provide only the predeclared detector statistic, even if QC includes labels.
    metadata = {**ref_metadata, 'qc_stats': {'detector_event_fraction': event_fraction}}
    time_steps, height, width = entry.x.shape[2:]
    planes, names = sedp.volume_common._build_context_planes(metadata, SimpleNamespace(
        valid_mask=entry.valid_mask_volume, time_steps=time_steps, height=height, width=width,
    ))
    assert entry.bundle_info['channel_names'][8:] == names
    assert np.all(planes[names.index('reserved_label_context')] == 0)
    # Preserve event and geometry channels from the actual sampled syndrome.
    entry.x[:, 8:] = planes
    return entry, dict(zip(names, map(float, planes[:, 0, 0, 0])))


def selected_rows(bundle: sedp.SelectorCandidateBundle, logits: np.ndarray, margin: float) -> np.ndarray:
    mask = sedp._select_candidate_masks_from_logits(
        bundle=bundle, selector_logits=logits, selector_emit_margin=margin,
    )['selected_edit_mask']
    rows = np.flatnonzero(np.all(bundle.candidate_edit_mask == mask[bundle.shot_indices], axis=1))
    np.testing.assert_array_equal(bundle.shot_indices[rows], np.arange(bundle.num_shots))
    return rows


def top_rows(bundle: sedp.SelectorCandidateBundle, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Candidate generation puts the unique no-edit row first for every shot.
    raw_rows = np.flatnonzero(bundle.candidate_edit_weight == 0)
    np.testing.assert_array_equal(bundle.shot_indices[raw_rows], np.arange(bundle.num_shots))
    assert np.array_equal(raw_rows, np.searchsorted(bundle.shot_indices, np.arange(bundle.num_shots)))
    assert np.isfinite(logits).all()
    maxima = np.full(bundle.num_shots, -np.inf, dtype=logits.dtype)
    np.maximum.at(maxima, bundle.shot_indices, logits)
    rows = np.flatnonzero(logits == maxima[bundle.shot_indices])
    best = np.full(bundle.num_shots, len(logits), dtype=np.int64)
    np.minimum.at(best, bundle.shot_indices[rows], rows)
    return raw_rows, best


def calibration_row(raw: np.ndarray, predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray, margin: float) -> dict:
    counts = rv.paired_counts(raw, predicted, truth)
    return {
        'edited_pymatching': sedp._hard_multiclass_metrics(
            sedp._logical_class4_from_observable_flips(predicted), sedp._logical_class4_from_observable_flips(truth),
        ),
        'change_summary': {'num_examples': len(truth), 'num_improved_over_baseline': counts['improved'],
                           'num_harmed_vs_baseline': counts['harmed'],
                           'predicted_edit_weight_histogram': {'1': int(np.count_nonzero(weights))}},
        'decision': {'selector_emit_margin': margin},
    }


def calibrate(out: Path, event_fraction: float, checkpoints: dict[int, dict]) -> dict:
    by_seed = {seed: {} for seed in PANEL}
    raw_results = []
    for local_seed in SEED_PLAN['development_local_seeds']:
        root = out / 'development' / str(local_seed) / 'g0'
        target = root / rv.load(root / 'case.json')['target']
        entry = sedp._prepare_edit_family(target, fill_value=-0.5, max_shots=CALIBRATION_SHOTS)
        entry, _ = with_context(entry, entry.metadata, event_fraction)
        raw = entry.baseline_predicted_observables
        truth = entry.observable_flips
        raw_results.append(calibration_row(raw, raw, truth, np.zeros(len(truth)), margin=0))
        for seed, checkpoint in checkpoints.items():
            predictor = rv.load_selector(checkpoint)
            bundle, logits, margin = rv.score_bundle(entry, 0, CALIBRATION_SHOTS, predictor)
            decoder = Decoder(entry.matching, False)
            predicted, _ = decode_edits(decoder, entry.detector_events[bundle.shot_indices],
                                         bundle.candidate_edit_mask, raw[bundle.shot_indices])
            grid = set(checkpoint['training_config']['selector_emit_margin_grid']) | {margin}
            for threshold in sorted(grid):
                rows = selected_rows(bundle, logits, threshold)
                result = calibration_row(raw, predicted[rows], truth, bundle.candidate_edit_weight[rows], threshold)
                by_seed[seed].setdefault(threshold, {})[str(local_seed)] = result
    outputs = {}
    for seed, checkpoint in checkpoints.items():
        margin = float(checkpoint['candidate_selector']['training_summary']['selected_selector_emit_margin'])
        by_margin = by_seed[seed]
        profile = [sedp._selector_margin_profile_record(rows) for rows in by_margin.values()]
        selected = list(by_margin[margin].values())
        selected[0]['decision']['selector_margin_profile'] = profile
        inference = checkpoint['inference']
        kwargs = inference['selector_adoption_decision']['thresholds']
        candidate_metric = sedp._mean_system_metric(selected)
        no_edit_metric = sedp._mean_system_metric(raw_results)
        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=no_edit_metric, global_metric=None, selector_metric=candidate_metric,
            selector_results=selected, requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR, **kwargs,
        )
        assert not inference['selected_no_edit_guardrail']
        outputs[str(seed)] = {'margin': margin, 'adopt': mode == sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
                              'decision': decision, 'calibration_rows': list(range(CALIBRATION_SHOTS)),
                              'original_stage_ab_mode': inference['selected_inference_mode']}
    return outputs


def write_case_predictions(path: Path, entry: sedp.PreparedEditFamily, decoders: dict[str, Decoder], predictor: tuple, adopt: bool) -> dict:
    events, truth = entry.detector_events, entry.observable_flips
    counters_before = {name: decoder.counts() for name, decoder in decoders.items()}
    raw = {name: decoder.decode_batch(events) for name, decoder in decoders.items()}
    assert all(array.shape == truth.shape for array in raw.values())
    proposal_before = decoders[D0_MWPM].counts()
    entry = replace(entry, matching=decoders[D0_MWPM], baseline_predicted_observables=raw[D0_MWPM])
    arrays = {f'{name}_raw': values for name, values in raw.items()}
    arrays['truth'] = truth
    chunks: dict[str, list[np.ndarray]] = {}
    proposal_counts = {k: 0 for k in proposal_before}
    proposal_fallbacks = 0
    offset = 0
    for start in range(0, len(truth), BATCH):
        end = min(start + BATCH, len(truth))
        before = decoders[D0_MWPM].counts()
        bundle, logits, margin = rv.score_bundle(entry, start, end, predictor)
        for key, value in decoders[D0_MWPM].counts().items():
            proposal_counts[key] += value - before[key]
        proposal_fallbacks += bundle.decode_fallback_count
        local_shot = bundle.shot_indices
        shot = start + local_shot
        raw_rows, top1 = top_rows(bundle, logits)
        branch = selected_rows(bundle, logits, margin)
        final = branch if adopt else raw_rows
        payload = {'shot_index': shot, 'packed_masks': np.packbits(bundle.candidate_edit_mask, axis=1),
                   'weights': bundle.candidate_edit_weight, 'utility': logits,
                   'top1_rows': top1 + offset, 'branch_rows': branch + offset, 'final_rows': final + offset}
        for name, decoder in decoders.items():
            predicted, failed = decode_edits(decoder, events[shot], bundle.candidate_edit_mask, raw[name][shot])
            payload[f'{name}_predicted'] = predicted
            payload[f'{name}_failure'] = failed
        for key, value in payload.items():
            chunks.setdefault(key, []).append(value)
        offset += len(logits)
    arrays.update({name: np.concatenate(parts, axis=0) for name, parts in chunks.items()})
    np.savez_compressed(path, **arrays)
    counters = {name: {key: value - counters_before[name][key] for key, value in decoder.counts().items()}
                for name, decoder in decoders.items()}
    return {'sha256': rv.digest(path), 'file': path.name, 'propose': proposal_counts, 'proposal_fallbacks': proposal_fallbacks,
            'calls_including_batch_and_retries': counters, 'cells': summarize_predictions(arrays, list(decoders))}


def summarize_predictions(arrays: dict, cells: list[str]) -> dict:
    truth, shot, weight = (arrays[key] for key in ['truth', 'shot_index', 'weights'])
    identity = np.flatnonzero(weight == 0)
    np.testing.assert_array_equal(shot[identity], np.arange(len(truth)))
    output = {}
    for name in cells:
        raw, candidates, failures = (arrays[f'{name}_{key}'] for key in ['raw', 'predicted', 'failure'])
        np.testing.assert_array_equal(candidates[identity], raw)
        assert not failures[identity].any()
        raw_correct = np.all(raw == truth, axis=1)
        possible = np.zeros(len(truth), dtype=bool)
        np.logical_or.at(possible, shot, np.all(candidates == truth[shot], axis=1))
        recovered = int(np.count_nonzero(possible & ~raw_correct))
        raw_wrong = int(np.count_nonzero(~raw_correct))
        record = {'raw_correct': int(raw_correct.sum()), 'raw_accuracy_pct': 100 * float(raw_correct.mean()),
                  'oracle_correct': int(possible.sum()), 'oracle_gain_pp': 100 * recovered / len(truth),
                  'raw_wrong_recovery': recovered / raw_wrong if raw_wrong else None,
                  'candidate_fallback_rows': int(failures.sum()),
                  'candidate_fallback_shots': int(len(np.unique(shot[failures]))),
                  'shots': len(truth), 'candidate_rows': len(shot)}
        for branch in ('top1', 'branch', 'final'):
            rows = arrays[f'{branch}_rows']
            np.testing.assert_array_equal(shot[rows], np.arange(len(truth)))
            counts = rv.paired_counts(raw, candidates[rows], truth)
            assert counts['selected_correct'] <= record['oracle_correct']
            attempted = int(np.count_nonzero(weight[rows]))
            failed = int(np.count_nonzero(failures[rows]))
            applied = attempted - failed
            record[branch] = {**counts, 'attempted_edits': attempted, 'applied_edits': applied, 'fallbacks': failed,
                              'harm_per_edit': counts['harmed'] / applied if applied else None}
        record['ranking_loss_pp'] = record['oracle_gain_pp'] - record['top1']['gain_pp']
        output[name] = record
    return output


def legacy_smoke() -> dict:
    raw_dir = next((ROOT / 'artifacts/datasets/dev_class4_d7_2k').glob('stage_c_corr__*/'))
    with np.load(raw_dir / 'samples.npz') as data:
        events, truth = data['detector_events'][:1024], data['observable_flips'][:1024]
    decoders = open_decoders([raw_dir])
    raw, correlated = [decoder.decode_batch(events) for decoder in decoders.values()]
    counts = rv.paired_counts(raw, correlated, truth)
    assert (counts['raw_correct'], counts['selected_correct'], counts['improved'], counts['harmed']) == (894, 887, 7, 14)
    return counts


def label_independence(out: Path, event_fraction: float, checkpoint: dict) -> None:
    root = out / 'development' / str(SEED_PLAN['development_local_seeds'][0]) / 'g0'
    target = root / rv.load(root / 'case.json')['target']
    original = sedp._prepare_edit_family(target, fill_value=-0.5, max_shots=64)
    predictor = rv.load_selector(checkpoint)
    left, context = with_context(replace(original, x=original.x.copy()), original.metadata, event_fraction)
    right_metadata = {**original.metadata, 'qc_stats': {'detector_event_fraction': 0.99, 'logical_flip_fraction': 1.0}}
    right, altered_context = with_context(replace(original, x=original.x.copy(),
        logical_class4=(original.logical_class4 + 1) % 4, observable_flips=1-original.observable_flips,
        metadata=right_metadata), right_metadata, event_fraction)
    assert altered_context == context
    np.testing.assert_array_equal(left.x, right.x)
    a, a_logits, margin = rv.score_bundle(left, 0, 64, predictor)
    b, b_logits, _ = rv.score_bundle(right, 0, 64, predictor)
    np.testing.assert_array_equal(a.candidate_edit_mask, b.candidate_edit_mask)
    np.testing.assert_array_equal(a_logits, b_logits)
    np.testing.assert_array_equal(selected_rows(a, a_logits, margin), selected_rows(b, b_logits, margin))

    old_root = rv.load(REVALIDATION / 'protocol.json')['distances']['7']['training_manifest']
    _, families = sedp._resolve_manifest_family_entries(ROOT / old_root, ['stage_c_corr'])
    old_entry = sedp._prepare_edit_family(families[0][1], fill_value=-0.5, max_shots=128)
    bundle, logits, margin = rv.score_bundle(old_entry, 0, 128, predictor)
    rows = selected_rows(bundle, logits, margin)
    raw = old_entry.baseline_predicted_observables
    decoded, failed = decode_edits(Decoder(old_entry.matching, False), old_entry.detector_events,
                                  bundle.candidate_edit_mask[rows], raw)
    counts = rv.paired_counts(raw, decoded, old_entry.observable_flips)
    reference = sedp.evaluate_checkpoint_on_family(
        checkpoint_path=REVALIDATION / f'd7/train/seed{PANEL[0]}/sedp_checkpoint.pt',
        family_dir=families[0][1], out_json=out / 'old_stage_c_reference.json', max_shots=128,
    )['metrics']['candidate_selector']['change_summary']
    assert (counts['improved'], counts['harmed'], int(failed.sum())) == (
        reference['num_improved_over_baseline'], reference['num_harmed_vs_baseline'],
        reference['num_decode_failures_fallback_to_baseline'],
    )


def evaluate_map(out: Path, role: str, local_seed: int, shots: int, event_fraction: float, calibration: dict) -> dict:
    sedp.torch.set_num_threads(2)
    started = perf_counter()
    path = out / role / str(local_seed)
    path.mkdir(parents=True, exist_ok=True)
    results_path = path / 'result.json'
    if results_path.exists():
        result = rv.load(results_path)
        assert result['shots_per_strength'] == shots
        for strength, record in result['data'].items():
            for seed, prediction in record.items():
                assert rv.digest(path / strength / prediction['file']) == prediction['sha256'], (strength, seed)
        return result
    targets, raw, validation = make_pair(path, local_seed, shots)
    decoders = open_decoders(raw)
    checkpoints = rv.checkpoints(REVALIDATION, 7, PANEL)
    predictors = {seed: rv.load_selector(checkpoint) for seed, checkpoint in checkpoints.items()}
    record = {'local_seed': local_seed, 'shots_per_strength': shots, 'validation': validation, 'data': {}, 'context': {}}
    reference_metadata = rv.load(raw[0] / 'metadata.json')
    for g, target in enumerate(targets):
        strength = f'g{g}'
        entry = sedp._prepare_edit_family(target, fill_value=-0.5, max_shots=None)
        entry, context = with_context(entry, reference_metadata, event_fraction)
        record['context'][strength] = context
        record['data'][strength] = {}
        for seed, predictor in predictors.items():
            output = path / strength / f'predictions_seed{seed}.npz'
            adopt = calibration[str(seed)]['adopt']
            record['data'][strength][str(seed)] = write_case_predictions(output, entry, decoders, predictor, adopt)
            print(f'{role} map {local_seed} {strength} seed {seed} complete', flush=True)
    assert record['context']['g0'] == record['context']['g1']
    record['wall_seconds'] = perf_counter() - started
    rv.save(results_path, record)
    return record


def run_maps(out: Path, role: str, map_seeds: list[int], shots: int, event_fraction: float, calibration: dict, jobs: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(evaluate_map, out, role, seed, shots, event_fraction, calibration): seed for seed in map_seeds}
        completed = {}
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            print(f'{role} map {pending[future]} complete', flush=True)
    return [completed[seed] for seed in map_seeds]


def source_hashes() -> dict:
    original = rv.load(REVALIDATION / 'protocol.json')['source_sha256']
    paths = [*original, 'tools/revalidate_predecoder.py', 'tools/run_predecoder_noise_shift.py']
    return {path: rv.digest(ROOT / path) for path in paths}


def panel_hashes() -> dict:
    rv.checkpoints(REVALIDATION, 7, PANEL)
    return {str(seed): rv.digest(REVALIDATION / f'd7/train/seed{seed}/sedp_checkpoint.pt') for seed in PANEL}


def develop(out: Path, jobs: int) -> None:
    sedp.torch.set_num_threads(2)
    dev_shots = SEED_PLAN['development_shots_per_strength']
    dev_seeds = SEED_PLAN['development_local_seeds']
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'development.json'
    if marker.exists():
        raise FileExistsError(f'Development already recorded: {marker}')
    versions = {name: version(name) for name in ('torch', 'numpy', 'stim', 'pymatching', 'scipy', 'matplotlib')}
    sources, hashes = source_hashes(), panel_hashes()
    # Store the exact uncommitted tracked diff alongside hashes of new/untracked source.
    (out / 'working_tree.diff').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    checkpoint = rv.checkpoints(REVALIDATION, 7, PANEL)
    stats = [make_pair(out / 'development' / str(seed), seed, dev_shots)[2] for seed in dev_seeds]
    event_fraction = float(np.mean([item['detector_event_fraction'][0] for item in stats]))
    legacy = legacy_smoke()
    calibration = calibrate(out, event_fraction, checkpoint)
    label_independence(out, event_fraction, checkpoint[PANEL[0]])
    print(f'Calibration complete. G0 detector event fraction {event_fraction:.6f}.', flush=True)
    for seed, row in calibration.items():
        print(f'  seed {seed}: margin {row["margin"]}, adopt {row["adopt"]}, {row["decision"]["reason"]}', flush=True)
    records = run_maps(out, 'development', dev_seeds, dev_shots, event_fraction, calibration, min(jobs, len(dev_seeds)))
    rv.save(marker, {'environment': versions, 'source_sha256': sources, 'checkpoint_sha256': hashes,
                     'legacy_raw_correlated_reproduction': legacy, 'label_independence_pass': True,
                     'context_event_fraction': event_fraction, 'context_denominator': len(dev_seeds)*dev_shots*336,
                     'calibration': calibration,
                     'development_wall_seconds_per_map': [record['wall_seconds'] for record in records]})


def evaluate(out: Path, jobs: int, budget_minutes: float) -> None:
    development = rv.load(out / 'development.json')
    assert development['source_sha256'] == source_hashes()
    assert development['checkpoint_sha256'] == panel_hashes()
    path = out / 'frozen_evaluation.json'
    seeds = SEED_PLAN['evaluation_local_seeds']
    if path.exists():
        config = rv.load(path)
        assert config['development_sha256'] == rv.digest(out / 'development.json')
    else:
        # Use runtime alone to set the sample count. Keep the eight reserved maps.
        waves = (len(seeds) + jobs - 1) // jobs
        seconds_at_dev_shots = 1.25 * max(development['development_wall_seconds_per_map']) * waves
        dev_shots = SEED_PLAN['development_shots_per_strength']
        budget_shots = int(dev_shots * budget_minutes * 60 / seconds_at_dev_shots)
        shots = max(dev_shots, min(SEED_PLAN['evaluation_shots_per_strength'], budget_shots // 256 * 256))
        config = {'development_sha256': rv.digest(out / 'development.json'), 'evaluation_local_seeds': seeds,
                  'planned_shots_per_strength': SEED_PLAN['evaluation_shots_per_strength'],
                  'shots_per_strength': shots, 'budget_minutes': budget_minutes, 'pilot_overhead_factor': 1.25,
                  'projected_wall_minutes': seconds_at_dev_shots * shots / dev_shots / 60, 'jobs': jobs,
                  'sampler_seed_formula': '1783000 + 2*(local_seed-783000) + strength_index',
                  'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_repetitions': BOOTSTRAP_REPS,
                  'interval_level': 0.975, 'effect_floor_pp': EFFECT_FLOOR_PP,
                  'zz_factors': [G0, G1], 'checkpoint_sha256': development['checkpoint_sha256']}
        rv.save(path, config)
    print(f'Frozen evaluation: {len(seeds)} maps x {config["shots_per_strength"]} shots per strength.', flush=True)
    records = run_maps(out, 'evaluation', config['evaluation_local_seeds'], config['shots_per_strength'],
                       development['context_event_fraction'], development['calibration'], jobs)
    summarize_experiment(out, records, config, development['calibration'])


def projection(row: dict) -> dict[str, float | None]:
    return {'raw_accuracy_pct': row['raw_accuracy_pct'], 'oracle_gain_pp': row['oracle_gain_pp'],
            'recovery_pct_of_raw_errors': 100 * row['raw_wrong_recovery'] if row['raw_wrong_recovery'] is not None else None,
            'ranking_loss_pp': row['ranking_loss_pp'],
            **{f'{branch}_gain_pp': row[branch]['gain_pp'] for branch in ('top1', 'branch', 'final')}}


def average_cell(rows: list[dict]) -> dict:
    scalars = [projection(row) for row in rows]
    result = {}
    for key in scalars[0]:
        values = [row[key] for row in scalars if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    for branch in ('top1', 'branch', 'final'):
        improved, harmed, requested, applied, fallbacks = (
            sum(row[branch][key] for row in rows)
            for key in ('improved', 'harmed', 'attempted_edits', 'applied_edits', 'fallbacks')
        )
        result[branch] = {'improved': improved, 'harmed': harmed, 'requested_edits': requested,
                          'applied_edits': applied, 'fallbacks': fallbacks,
                          'edits_pct': 100 * applied / sum(row['shots'] for row in rows),
                          'harm_pct_of_edits': 100 * harmed / applied if applied else None}
    result['candidate_fallback_rows'] = sum(row['candidate_fallback_rows'] for row in rows)
    result['candidate_fallback_shots'] = sum(row['candidate_fallback_shots'] for row in rows)
    result['candidate_rows_per_shot'] = sum(row['candidate_rows'] for row in rows) / sum(row['shots'] for row in rows)
    return result


def summarize_experiment(out: Path, records: list[dict], config: dict, calibration: dict) -> None:
    keys = [f'g{g}_d{d}_{method}' for method in ('mwpm', 'correlated') for g in range(2) for d in range(2)]
    map_results, checkpoint_results = [], {str(seed): [] for seed in PANEL}
    cells = {key: [] for key in keys}
    matching_calls = {name: {k: 0 for k in ('batch_calls', 'single_calls', 'rows')} for name in ('propose', 'oracle_decode')}
    for record in records:
        by_map = {key: [] for key in keys}
        for seed in PANEL:
            by_checkpoint = {}
            for strength, models in record['data'].items():
                model = models[str(seed)]
                for stat, value in model['propose'].items():
                    matching_calls['propose'][stat] += value
                for name, cell in model['cells'].items():
                    key = f'{strength}_{name}'
                    by_checkpoint[key] = cell
                    by_map[key].append(cell)
                    cells[key].append(cell)
                    for stat, value in model['calls_including_batch_and_retries'][name].items():
                        proposal = model['propose'][stat] if name == D0_MWPM else 0
                        matching_calls['oracle_decode'][stat] += value - proposal
            checkpoint_results[str(seed)].append({'local_seed': record['local_seed'], 'cells': by_checkpoint})
        map_results.append({'local_seed': record['local_seed'], 'cells': {k: average_cell(v) for k, v in by_map.items()}})
    aggregate = {k: average_cell(v) for k, v in cells.items()}
    contrasts = {
        'C1_top1_strength_pp': ('g1_d0_mwpm', 'g0_d0_mwpm', 'top1_gain_pp'),
        'C2_final_matched_pp': ('g1_d1_mwpm', 'g1_d0_mwpm', 'final_gain_pp'),
        'diagnostic_ranking_loss_strength_pp': ('g1_d0_mwpm', 'g0_d0_mwpm', 'ranking_loss_pp'),
        'diagnostic_recovery_strength_pp': ('g1_d0_mwpm', 'g0_d0_mwpm', 'recovery_pct_of_raw_errors'),
        'diagnostic_raw_matched_pp': ('g1_d1_mwpm', 'g1_d0_mwpm', 'raw_accuracy_pct'),
    }
    rng = np.random.default_rng(config['bootstrap_seed'])
    sample = rng.integers(0, len(records), size=(config['bootstrap_repetitions'], len(records)))
    comparisons = {}
    for name, (a, b, metric) in contrasts.items():
        effects = np.array([mapping['cells'][a][metric] - mapping['cells'][b][metric] for mapping in map_results])
        lb, ub = np.quantile(effects[sample].mean(axis=1), [0.0125, 0.9875]).tolist()
        floor = config['effect_floor_pp']
        decision = 'positive' if lb > floor else 'negative' if ub < -floor else 'small' if -floor < lb and ub < floor else 'uncertain'
        comparisons[name] = {'mean_pp': float(effects.mean()), 'interval_pp': [lb, ub], 'map_effects_pp': effects.tolist(),
                             'omit_one_pp': ((effects.sum()-effects)/(len(effects)-1)).tolist(),
                             'decision': decision if name.startswith(('C1', 'C2')) else 'descriptive'}
    summary = {'calibration_adopted_seeds': [int(k) for k, row in calibration.items() if row['adopt']],
               'frozen_evaluation_sha256': rv.digest(out / 'frozen_evaluation.json'), 'cells': aggregate,
               'maps': map_results, 'checkpoint_maps': checkpoint_results, 'comparisons': comparisons,
               'matching_calls': matching_calls,
               'final_policy_candidate_rows_for_adopted_model': sum(
                   row['candidate_rows'] for rows in checkpoint_results.values() for entry in rows
                   for key, row in entry['cells'].items() if key == 'g0_d0_mwpm' and row['final']['attempted_edits']
               )}
    plot_results(out / 'crossed_diagnostics.png', aggregate)
    if not (out / 'summary.json').exists():
        rv.save(out / 'summary.json', summary)
    print(f'All {len(records)} maps complete. Summary: {out / "summary.json"}', flush=True)


def plot_results(path: Path, aggregate: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    pairs = ('g0_d0', 'g0_d1', 'g1_d0', 'g1_d1')
    labels = ('G0 / D0', 'G0 / D1', 'G1 / D0', 'G1 / D1')
    x = np.arange(len(labels))
    for axs, method in zip(axes, ('mwpm', 'correlated')):
        rows = [aggregate[f'{key}_{method}'] for key in pairs]
        for i, (branch, title) in enumerate((('top1', 'Top 1'), ('branch', 'After margin'), ('final', 'Final policy'))):
            axs[0].bar(x + (i-1)*0.24, [row[f'{branch}_gain_pp'] for row in rows], width=0.23, label=title)
            harm = [row[branch]['harm_pct_of_edits'] for row in rows]
            bars_x = [int(k) + (i-1)*0.24 for k in x if harm[k] is not None]
            axs[2].bar(bars_x, [v for v in harm if v is not None], width=0.23, label=title)
        axs[1].bar(x, [row['oracle_gain_pp'] for row in rows], width=0.55, color='#317759')
        axs[0].axhline(0, color='#555555', linewidth=0.6)
        axs[0].legend(fontsize=8)
        axs[2].text(0.01, 0.99, 'No bar for a route with no applied edits', fontsize=7, va='top', transform=axs[2].transAxes)
        for ax, title, ylabel in zip(axs, ('Net gain', 'Candidate oracle', 'Harm per applied edit'), ('pp', 'pp', '%')):
            ax.set_xticks(x, labels)
            ax.set_title(f'{method.upper()} | {title}')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['develop', 'evaluate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--budget-minutes', type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or args.budget_minutes <= 0:
        parser.error('jobs must be 1..8 and budget-minutes must be positive')
    if args.phase == 'develop':
        develop(args.out.resolve(), args.jobs)
    else:
        evaluate(args.out.resolve(), args.jobs, args.budget_minutes)


if __name__ == '__main__':
    main()
