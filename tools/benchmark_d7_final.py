"""Measure the three frozen d7 routes on the same in-memory syndrome batches."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.rotated_rect import build_rectangular_syndrome_volume
from tools import evaluate_d7_final as final
from tools.evaluate_d7_paired_selector_weak import TimedDecoder

rv, dev, prior, sedp, shift = final.rv, final.dev, final.prior, final.sedp, final.shift
OUT = ROOT / 'artifacts/eval/d7_final_benchmark/20261117'
ROUTES = ('raw', 'margin', 'paired')
MODEL = 2
SHOTS = 256
REPEATS = 3
ANCHOR = prior.CELLS[0]


def tensor(events, layout, static, context):
    volume = build_rectangular_syndrome_volume({'detector_events': events}, layout=layout, fill_value=-0.5)
    return np.concatenate((volume[:, None], np.broadcast_to(static, (len(events), *static.shape)),
                           np.broadcast_to(context, (len(events), *context.shape))), axis=1)


def inference(source, matching, route, neural, head):
    started = perf_counter()
    events = np.ascontiguousarray(source, dtype=np.uint8)
    decoder = TimedDecoder(matching, False)
    raw = decoder.decode_batch(events)
    mask = None
    candidate_mask = None
    prediction = raw
    prep_seconds = bundle_seconds = selection_seconds = 0.0
    if neural is not None:
        layout, static, context, lookup, order, predictor = neural
        mark = perf_counter()
        x = tensor(events, layout, static, context)
        entry = SimpleNamespace(x=x, detector_events=events, matching=decoder, baseline_predicted_observables=raw,
                                detector_time_index=layout.detector_time_index,
                                row_index_by_detector=layout.row_index_by_detector,
                                col_index_by_detector=layout.col_index_by_detector)
        prep_seconds = perf_counter() - mark
        mark = perf_counter()
        bundle, utility, margin = rv.score_bundle(entry, 0, len(events), predictor)
        bundle_seconds = perf_counter() - mark
        mark = perf_counter()
        predictions = prior.decoded_from_transition(bundle, raw)
        if route == 'margin':
            selected = shift.selected_rows(bundle, utility, margin)
            mask, prediction = bundle.candidate_edit_mask[selected], predictions[selected]
        else:
            masks, output, _ = prior.choose(bundle, utility, predictions, raw, events, layout,
                                             lookup, order, MODEL, head, margin)
            mask, prediction = masks[route], output[route]
        candidate_mask = bundle.candidate_edit_mask
        selection_seconds = perf_counter() - mark
    elapsed = perf_counter() - started
    return prediction, mask, candidate_mask, {
        'seconds':elapsed,'tensor_seconds':prep_seconds,'bundle_seconds':bundle_seconds,
        'selection_seconds':selection_seconds,'matching_seconds':decoder.matching_seconds,
        'matching_counts':decoder.counts(),
    }


def worker(strength, route, protocol):
    sedp.torch.set_num_threads(2)
    seed = protocol['maps'][0]['local_seed']
    root = final.OUT / 'evaluation' / str(seed)
    raw0 = dev.data_root(root / 'g0')
    data_path = dev.data_root(root / strength) / 'samples.npz'
    case = rv.load(root / strength / 'case.json')
    rel = data_path.relative_to(root / strength).as_posix()
    assert rv.digest(data_path) == case['sha256'][rel]
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    started = perf_counter()
    source = np.load(data_path)
    with source:
        events = source['detector_events'][:SHOTS]
        layout = prior.paired.build_rectangular_syndrome_layout(source) if route != 'raw' else None
    input_seconds = perf_counter() - started
    started = perf_counter()
    matching = shift.open_decoders([raw0])[ANCHOR].matching
    dem_seconds = perf_counter() - started
    model_seconds = 0.0
    neural = None
    if layout is not None:
        started = perf_counter()
        predictor = rv.load_selector(rv.checkpoints(shift.REVALIDATION, 7, [MODEL])[MODEL])
        assert predictor[2]['selector_target_mode'] == sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM
        model_seconds = perf_counter() - started
        meta = rv.load(raw0 / 'metadata.json')
        meta = {**meta, 'qc_stats': {'detector_event_fraction': protocol['context_event_fraction']}}
        static, _ = sedp.volume_common._build_static_planes(layout)
        context, names = sedp.volume_common._build_context_planes(meta, layout)
        assert np.all(context[names.index('reserved_label_context')] == 0)
        lookup, order = prior.paired.probe.patch_lookup(layout)
        neural = layout, static, context, lookup, order, predictor

    reference = root / strength / f'seed{MODEL}.npz'
    record = rv.load(reference.with_suffix('.json'))
    assert record['sha256'] == rv.digest(reference)
    assert record['protocol_sha256'] == rv.digest(final.OUT / 'protocol.json')
    with np.load(reference) as saved:
        expected = saved[f'{ANCHOR}_{route}'][:SHOTS]
        expected_mask = saved[f'{ANCHOR}_{route}_mask'][:SHOTS] if neural is not None else None
        expected_candidates = saved['candidate_mask'][saved['shot_index'] < SHOTS] if neural is not None else None
    ready_rss = process.memory_info().rss
    runs = []
    for _ in range(REPEATS + 1):
        prediction, mask, candidates, timing = inference(events, matching, route, neural, protocol['paired_head'])
        np.testing.assert_array_equal(prediction, expected)
        if neural is not None:
            np.testing.assert_array_equal(np.packbits(mask, axis=1), expected_mask)
            np.testing.assert_array_equal(np.packbits(candidates, axis=1), expected_candidates)
        del prediction, mask, candidates
        runs.append(timing)
    memory = process.memory_info()
    # The Windows process peak covers initialization, warm-up and all measured trials.
    peak = memory.peak_wset

    if neural is not None:
        prepared = sedp._prepare_edit_family(root / strength / case['target'], fill_value=-0.5, max_shots=SHOTS)
        prepared, _ = shift.with_context(prepared, rv.load(raw0 / 'metadata.json'), protocol['context_event_fraction'])
        np.testing.assert_array_equal(tensor(events, layout, static, context), prepared.x)
    samples = runs[1:]
    medians = {key:statistics.median(row[key] for row in samples) for key in (
        'seconds','tensor_seconds','bundle_seconds','selection_seconds','matching_seconds',
    )}
    assert all(x['matching_counts']==samples[0]['matching_counts'] for x in runs)
    return {'medians':medians,'shots_per_second':len(events)/medians['seconds'],'warmup':runs[0],
            'repetitions':samples,'matching_counts':samples[0]['matching_counts'],
            'setup_seconds':{'input_load_and_layout':input_seconds,'dem_load':dem_seconds,'checkpoint_load':model_seconds},
            'working_set_bytes':{'initial':initial_rss,'ready':ready_rss,'peak':peak},
            'source_sha256':rv.digest(data_path),'reference_sha256':record['sha256'],
            'exact_prediction_replay':True,'exact_tensor_and_mask_replay':neural is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('STRENGTH','ROUTE'))
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX','.')):
        parser.error('Activate the Anaconda fp environment first')
    protocol = final.check_protocol()
    if args.worker:
        strength, route = args.worker
        assert strength in ('g0','g1') and route in ROUTES
        print(json.dumps(worker(strength,route,protocol)))
        return
    results = {}
    for strength in ('g0','g1'):
        results[strength] = {}
        for route in ROUTES:
            data = subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--worker',strength,route], text=True)
            row = json.loads(data)
            raw_seconds = results[strength].get('raw',row)['medians']['seconds']
            row['time_over_raw'] = row['medians']['seconds'] / raw_seconds
            results[strength][route] = row
            print(f'{strength} {route}: {row["medians"]["seconds"]:.6f} s',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    rv.save(OUT / 'benchmark.json', {
        'protocol_sha256':rv.digest(final.OUT / 'protocol.json'), 'runner_sha256':rv.digest(Path(__file__)),
        'map':protocol['maps'][0]['local_seed'],'model':MODEL,'cell':ANCHOR,'shots':SHOTS,'warmups':1,
        'repetitions':REPEATS,'processes_per_route':1,'torch_threads':2,
        'mkl_threading_layer':os.environ.get('MKL_THREADING_LAYER'),
        'python':platform.python_version(),'platform':platform.platform(),'conda_prefix':sys.prefix,
        'cpu':os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
        'cpu_logical':psutil.cpu_count(),'cpu_physical':psutil.cpu_count(logical=False),
        'ram_bytes':psutil.virtual_memory().total,
        'versions':{name:version(name) for name in (*final.VERSIONS,'psutil')},'results':results,
    })


if __name__ == '__main__':
    main()
