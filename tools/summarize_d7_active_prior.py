"""Verify the saved d7 prior-cross predictions and aggregate by evaluation panel."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_d7_active_prior import CELLS, OUT, POLICIES, dev, paired, rv, shift


def unpack(arrays, key, width):
    return np.unpackbits(arrays[key], axis=1, count=width)


def map_results(role, spec, source, anchor_predictions):
    seed = str(spec['local_seed'])
    base = OUT / role / seed
    paths = [dev.data_root(dev.OUT / role / seed / strength) for strength in ('g0', 'g1')]
    decoders = shift.open_decoders(paths)
    result = {}
    for strength, path in zip(('g0', 'g1'), paths, strict=True):
        with np.load(path / 'samples.npz') as samples:
            events, truth = samples['detector_events'], samples['observable_flips']
        raw = {cell: decoder.decode_batch(events) for cell, decoder in decoders.items()}
        assert len(events) == spec['shots_per_strength']
        records = {}
        for checkpoint in range(8):
            file = base / strength / f'seed{checkpoint}.npz'
            marker = rv.load(file.with_suffix('.json'))
            assert marker['sha256'] == rv.digest(file)
            with np.load(file) as data:
                np.testing.assert_array_equal(data['truth'], truth)
                anchor = {policy: unpack(data, f'{CELLS[0]}_{policy}_full_mask', events.shape[1])
                          for policy in POLICIES}
                key = f'{"cv" if role=="development" else "cal"}_{seed}_{strength}_seed{checkpoint}'
                source_case = source[key + '_truth']
                np.testing.assert_array_equal(paired.class4(truth), source_case)
                for policy in POLICIES:
                    name = dev.PAIR if policy == 'paired' else 'margin'
                    original = source[key + '_' + name][1]
                    np.testing.assert_array_equal(paired.class4(data[f'{CELLS[0]}_{policy}_full']), original)
                cells = {}
                for cell, decoder in decoders.items():
                    reference = raw[cell]
                    np.testing.assert_array_equal(reference, data[f'{cell}_raw'])
                    cells[cell] = {'raw_correct': int(np.all(reference == truth, axis=1).sum()),
                                   **marker['cells'][cell], 'policies': {}}
                    for policy in POLICIES:
                        mask = unpack(data, f'{cell}_{policy}_full_mask', events.shape[1])
                        fixed = data[f'{cell}_{policy}_fixed']
                        full = data[f'{cell}_{policy}_full']
                        stats = {}
                        for mode, chosen, prediction in (('fixed', anchor[policy], fixed), ('full', mask, full)):
                            decoded, failures = shift.decode_edits(decoder, events, chosen, reference)
                            assert not failures.any()
                            np.testing.assert_array_equal(prediction, decoded)
                            edits = np.any(chosen, axis=1)
                            np.testing.assert_array_equal(prediction[~edits], reference[~edits])
                            stats[mode] = {
                                **rv.paired_counts(reference, prediction, truth),
                                'edits': int(edits.sum()),
                                'prediction_changes': int(np.any(prediction != reference, axis=1).sum()),
                            }
                        stats['mask_changes_from_fixed'] = int(np.any(mask != anchor[policy], axis=1).sum())
                        stats['prediction_changes_from_fixed'] = int(np.any(full != fixed, axis=1).sum())
                        if cell == CELLS[0]:
                            assert stats['mask_changes_from_fixed'] == stats['prediction_changes_from_fixed'] == 0
                        cells[cell]['policies'][policy] = stats
                records[str(checkpoint)] = {'sha256': marker['sha256'], 'wall_seconds': marker['wall_seconds'],
                                            'cells': cells}
                anchor_predictions.add(key)
        result[strength] = records
    return result


def aggregate(maps):
    aggregate_results = {}
    for strength in ('g0', 'g1'):
        aggregate_results[strength] = {}
        for cell in CELLS:
            group = {seed: [row['cells'][cell] for row in record[strength].values()]
                     for seed, record in maps.items()}
            rows = [row for records in group.values() for row in records]
            n = sum(row['policies']['margin']['fixed']['shots'] for row in rows)
            raw = sum(row['raw_correct'] for row in rows)
            summary = {'model_shots': n, 'raw_correct_model_shots': raw, 'raw_accuracy_pct': 100*raw/n,
                       'paired_eligible': sum(row['paired_eligible'] for row in rows),
                       'paired_accept_z': sum(row['paired_accept_z'] for row in rows), 'policies': {}}
            for policy in POLICIES:
                out = {key: sum(row['policies'][policy][key] for row in rows)
                       for key in ('mask_changes_from_fixed', 'prediction_changes_from_fixed')}
                for mode in ('fixed', 'full'):
                    totals = {key: sum(row['policies'][policy][mode][key] for row in rows)
                              for key in ('improved', 'harmed', 'edits', 'prediction_changes')}
                    per_map = {seed: float(np.mean([row['policies'][policy][mode]['gain_pp'] for row in records]))
                               for seed, records in group.items()}
                    gain = 100*(totals['improved']-totals['harmed'])/n
                    np.testing.assert_allclose(gain, np.mean(list(per_map.values())), rtol=0, atol=1e-12)
                    out[mode] = {**totals, 'gain_pp': gain, 'gain_pp_by_map': per_map}
                summary['policies'][policy] = out
            aggregate_results[strength][cell] = summary
        raw0 = aggregate_results[strength][CELLS[0]]['raw_accuracy_pct']
        for row in aggregate_results[strength].values():
            row['raw_delta_from_d0_mwpm_pp'] = row['raw_accuracy_pct'] - raw0
    return aggregate_results


def main():
    protocol = rv.load(OUT / 'protocol.json')
    assert protocol['source_sha256'] == rv.digest(ROOT / 'tools/evaluate_d7_active_prior.py')
    summary = rv.load(dev.OUT / 'stage2_summary.json')
    assert protocol['stage2_summary_sha256'] == rv.digest(dev.OUT / 'stage2_summary.json')
    assert protocol['registry_sha256'] == rv.digest(dev.OUT / 'data_registry.json')
    anchor_file = dev.OUT / 'stage2_predictions.npz'
    assert rv.digest(anchor_file) == summary['predictions_sha256']
    checked = set()
    with np.load(anchor_file) as anchor:
        results = {role: {str(spec['local_seed']): map_results(role, spec, anchor, checked) for spec in specs}
                   for role, specs in protocol['maps'].items()}
    assert len(checked) == 96
    aggregate_results = {role: aggregate(panel) for role, panel in results.items()}
    report = {'protocol_sha256': rv.digest(OUT / 'protocol.json'),
              'reuse_replay_sha256': rv.digest(OUT / 'rescore_replay.json'),
              'summarizer_sha256': rv.digest(Path(__file__)),
              'verified_prediction_files': len(checked),
              'matching_failures_for_checked_actions': 0, 'panels': aggregate_results, 'maps': results}
    rv.save(OUT / 'summary.json', report)
    for role, panel in aggregate_results.items():
        for strength, cells in panel.items():
            for cell, row in cells.items():
                print(role, strength, cell, round(row['raw_accuracy_pct'], 4),
                      {p: [round(row['policies'][p][mode]['gain_pp'], 4) for mode in ('fixed','full')]
                       for p in POLICIES})


if __name__ == '__main__':
    main()
