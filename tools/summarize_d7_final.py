"""Replay all independent d7 predictions and compute the frozen map-level contrasts."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_final as final

rv, shift, dev, prior = final.rv, final.shift, final.dev, final.prior
CELLS, POLICIES = prior.CELLS, prior.POLICIES


def contrast(reference, prediction, truth):
    a = reference == truth
    b = prediction == truth
    improved = int((~a & b).sum())
    harmed = int((a & ~b).sum())
    assert int(b.sum()) - int(a.sum()) == improved - harmed
    return {'shots':len(truth), 'reference_correct':int(a.sum()), 'selected_correct':int(b.sum()),
            'improved':improved, 'harmed':harmed}


def replay(data, events, truth, raw, decoders):
    np.testing.assert_array_equal(truth,data['truth'])
    classes = dev.paired.class4(truth)
    shot = data['shot_index']
    masks = np.unpackbits(data['candidate_mask'],axis=1,count=events.shape[1])
    assert np.all((shot>=0) & (shot<len(events)))
    np.testing.assert_array_equal(shot[~masks.any(axis=1)],np.arange(len(events)))
    result = {}
    for cell in CELLS:
        baseline = raw[cell]
        np.testing.assert_array_equal(data[cell+'_raw'],baseline)
        array, failed = shift.decode_edits(decoders[cell],events[shot],masks,baseline[shot])
        assert not failed.any()
        candidates = dev.paired.class4(array)
        np.testing.assert_array_equal(candidates,data[cell+'_candidates'])
        oracle = np.zeros(len(events),dtype=bool)
        np.logical_or.at(oracle,shot,candidates==classes[shot])
        selected = {'raw':dev.paired.class4(baseline)}
        edits = {}
        for policy in POLICIES:
            chosen = np.unpackbits(data[cell+'_'+policy+'_mask'],axis=1,count=events.shape[1])
            predicted, fail = shift.decode_edits(decoders[cell],events,chosen,baseline)
            assert not fail.any()
            np.testing.assert_array_equal(predicted,data[cell+'_'+policy])
            selected[policy] = dev.paired.class4(predicted)
            edits[policy] = int(chosen.any(axis=1).sum())
        assert all(np.all(oracle | (values != classes)) for values in selected.values())
        result[cell] = {
            'raw_correct':int((selected['raw']==classes).sum()),
            'oracle_correct':int(oracle.sum()), 'candidate_rows':len(shot), 'edits':edits,
            'contrasts':{name:contrast(selected[b],selected[a],classes) for name,(a,b) in final.CONTRASTS.items()},
            'paired_vs_raw':contrast(selected['raw'],selected['paired'],classes),
        }
    return result


def read_map(base, spec, protocol_sha):
    dirs = [dev.data_root(base / g) for g in ('g0','g1')]
    decoders = shift.open_decoders(dirs)
    result = {}
    for strength, dataset in zip(('g0','g1'),dirs,strict=True):
        marker = rv.load(base/strength/'case.json')
        settings = marker['settings']
        assert settings['local_seed']==spec['local_seed'] and settings['sampler_seed']==spec['sampler_seeds'][strength]
        for file,sha in marker['sha256'].items():
            assert rv.digest(base/strength/file)==sha,file
        with np.load(dataset/'samples.npz') as data:
            events,truth = data['detector_events'],data['observable_flips']
        assert len(events)==final.SHOTS[strength]
        raw = {cell:decoder.decode_batch(events) for cell,decoder in decoders.items()}
        models = {}
        for model in range(8):
            file = base/strength/f'seed{model}.npz'
            record = rv.load(file.with_suffix('.json'))
            assert record['protocol_sha256']==protocol_sha and record['sha256']==rv.digest(file)
            with np.load(file) as data:
                models[str(model)] = replay(data,events,truth,raw,decoders)
        result[strength] = models
    return result


def calculate_interval(values,level):
    a = np.asarray(values,dtype=float)
    mean = float(a.mean())
    half = float(t.ppf((1+level)/2,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return mean,[mean-half,mean+half]


def aggregate(maps,protocol):
    results = {}
    for strength in ('g0','g1'):
        cells = {}
        for cell in CELLS:
            primary = strength==protocol['primary']['strength'] and cell==protocol['primary']['cell']
            level = protocol['primary' if primary else 'secondary']['level']
            panel = {seed:[row[cell] for row in data[strength].values()] for seed,data in maps.items()}
            total_model_shots=8*len(panel)*final.SHOTS[strength]
            flat = [item for rows in panel.values() for item in rows]
            row={
                'physical_shots':len(panel)*final.SHOTS[strength], 'model_shots':total_model_shots,
                'raw_accuracy_pct':100*sum(r['raw_correct'] for r in flat)/total_model_shots,
                'oracle_accuracy_pct':100*sum(r['oracle_correct'] for r in flat)/total_model_shots,
                'candidate_rows':sum(r['candidate_rows'] for r in flat),
                'edits':{k:sum(r['edits'][k] for r in flat) for k in POLICIES},
                'paired_gain_over_raw_pp':100*sum(r['paired_vs_raw']['improved']-r['paired_vs_raw']['harmed']
                                                   for r in flat)/total_model_shots,
                'comparisons':{},
            }
            for name in final.CONTRASTS:
                by_map={seed:100*sum(r['contrasts'][name]['improved']-r['contrasts'][name]['harmed'] for r in rows)
                           / (8*final.SHOTS[strength]) for seed,rows in panel.items()}
                effect,interval=calculate_interval(list(by_map.values()),level)
                totals={key:sum(r['contrasts'][name][key] for r in flat)
                        for key in ('improved','harmed','reference_correct','selected_correct')}
                assert abs(effect-100*(totals['improved']-totals['harmed'])/total_model_shots)<1e-10
                ref_errors=total_model_shots-totals['reference_correct']
                err=total_model_shots-totals['selected_correct']
                row['comparisons'][name]={
                    **totals,'gain_pp':effect,'interval_level':level,'interval_pp':interval,'by_map_pp':by_map,
                    'reference_logical_error_pct':100*ref_errors/total_model_shots,
                    'selected_logical_error_pct':100*err/total_model_shots,
                    'relative_error_reduction_pct':100*(ref_errors-err)/ref_errors if ref_errors else None,
                    'decision':('improvement' if interval[0]>0 else 'harm' if interval[1]<0 else 'inconclusive')
                               if primary else 'descriptive',
                }
            cells[cell]=row
        results[strength]=cells
    return results


def main():
    protocol = final.check_protocol()
    protocol_sha = rv.digest(final.OUT/'protocol.json')
    started = perf_counter()
    maps = {}
    for spec in protocol['maps']:
        seed = str(spec['local_seed'])
        maps[seed] = read_map(final.OUT/'evaluation'/seed,spec,protocol_sha)
        print(f'Replay verified for map {seed}',flush=True)
    results = aggregate(maps,protocol)
    marker = final.OUT/'summary.json'
    report = {'protocol_sha256':protocol_sha,'verified_prediction_files':len(maps)*16,
              'candidate_rows_redecoded':sum(row['candidate_rows'] for cells in results.values()
                                            for row in cells.values()),
              'matching_failures':0,'results':results,'maps':maps}
    if marker.exists():
        assert rv.load(marker)==report
    else:
        rv.save(marker,report)
    print(results['g1'][CELLS[0]],flush=True)
    print(f'Completed replay in {perf_counter()-started:.1f} s',flush=True)


if __name__=='__main__':
    main()
