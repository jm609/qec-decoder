"""Freeze and evaluate the independent d7 margin and paired policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, time

import numpy as np
from scipy.stats import t
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_active_prior as prior
from tools.collect_d7_confidence_candidates import CONTEXT

rv, shift, dev, sedp = prior.rv, prior.shift, prior.dev, prior.sedp
OUT = ROOT / 'artifacts/eval/d7_final/20261117'
PLAN = ROOT / 'docs/thesis/PREDECODER_D7_FINAL_EVALUATION_PROTOCOL.md'
MAPS = tuple(range(788100, 788124))
SHOTS = {'g0': 256, 'g1': 512}
BUDGET_MINUTES = 80
JOBS = 2
CONTRASTS = {'margin_minus_raw': ('margin', 'raw'), 'margin_minus_paired': ('margin', 'paired')}
VERSIONS = ('numpy', 'scipy', 'stim', 'pymatching', 'torch')


def planning() -> dict:
    source = prior.OUT
    by_role = {}
    for role, rows in rv.load(source / 'protocol.json')['maps'].items():
        for spec in rows:
            local_seed = str(spec['local_seed'])
            values = []
            for model in range(8):
                with np.load(source / role / local_seed / 'g1' / f'seed{model}.npz') as data:
                    truth = data['truth']
                    correct = {key: np.all(data['d0_mwpm_' + route] == truth, axis=1).astype(float)
                               for key, route in (('raw', 'raw'), ('margin', 'margin_full'), ('paired', 'paired_full'))}
                    values.append([correct[a]-correct[b] for a, b in CONTRASTS.values()])
            by_role[local_seed] = 100*np.mean(values, axis=0)
    x = np.stack(list(by_role.values()))
    map_variance = x.mean(axis=2).var(axis=0, ddof=1)
    within = x.var(axis=2, ddof=1).mean(axis=0)
    # Retain all empirical map variation, including shot noise, before adding target-length shot noise.
    conservative = t.ppf(0.9875,len(MAPS)-1)*np.sqrt((map_variance+within/SHOTS['g1'])/len(MAPS))
    return {'contrast_order': list(CONTRASTS), 'source_map_gains_pp': {
        seed: values.mean(axis=1).tolist() for seed, values in by_role.items()},
        'observed_map_variance': map_variance.tolist(), 'mean_within_map_shot_variance': within.tolist(),
        'conservative_97_5_half_width_pp': conservative.tolist(), 'precision_target_pp': 0.10,
        'reference_wall_seconds': 2137.0, 'reference_physical_shots': 12288,
        'projected_wall_seconds': 2137.0*len(MAPS)*sum(SHOTS.values())/12288}


def freeze() -> None:
    assert PLAN.is_file()
    registry_path = dev.OUT / 'data_registry.json'
    state_file = dev.OUT / 'stage2_summary.json'
    stage3 = prior.OUT / 'summary.json'
    registry = rv.load(registry_path)
    state = rv.load(state_file)
    report = rv.load(stage3)
    assert state['selection']['chosen'] == 'margin'
    assert state['registry_sha256'] == rv.digest(registry_path)
    assert report['verified_prediction_files'] == 96
    assert report['protocol_sha256'] == rv.digest(prior.OUT / 'protocol.json')
    used_local, used_sampler = set(), set()
    for path in (ROOT/'artifacts/eval').rglob('case.json'):
        settings = rv.load(path).get('settings', {})
        if 'local_seed' in settings:
            used_local.add(settings['local_seed'])
            used_sampler.add(settings['sampler_seed'])
    for row in registry['training_sample_seeds']:
        used_local.add(row['local_seed'])
        used_sampler.add(row['seed'])
    maps = [{'local_seed':seed,'sampler_seeds':{'g0':1794100 + 2*i,'g1':1794101 + 2*i}}
            for i, seed in enumerate(MAPS)]
    assert not (set(MAPS) & used_local)
    assert not ({n for row in maps for n in row['sampler_seeds'].values()} & used_sampler)
    sources = dict(shift.source_hashes())
    for path in ('tools/evaluate_d7_active_prior.py', 'tools/summarize_d7_active_prior.py',
                 'tools/evaluate_d7_confidence_development.py', 'tools/run_d7_paired_selector.py',
                 'tools/evaluate_d7_paired_selector_weak.py', 'tools/probe_d7_z_transition.py',
                 'tools/collect_d7_confidence_candidates.py', 'tools/evaluate_d7_final.py',
                 'tools/summarize_d7_final.py', 'tools/analyze_d7_logical_transitions.py',
                 'geometry/rotated_rect.py', 'logical_targets.py', 'dual_axis_manifest.py'):
        sources[path] = rv.digest(ROOT/path)
    protocol = {
        'plan_sha256':rv.digest(PLAN), 'stage2_summary_sha256':rv.digest(state_file),
        'stage3_summary_sha256':rv.digest(stage3), 'source_sha256':sources,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'models':registry['models'], 'paired_head':state['refit_heads'][dev.PAIR], 'paired_cutoff':0,
        'context_event_fraction':CONTEXT, 'zz_factors':[shift.G0,shift.G1],
        'distance':7, 'rounds':7, 'basis':'z', 'target_mode':'bell_pair_z_readout', 'p':0.0015,
        'maps':maps, 'shots_per_strength':SHOTS, 'cells':list(prior.CELLS), 'policies':['raw',*prior.POLICIES],
        'primary':{'strength':'g1','cell':prior.CELLS[0],'level':0.975,
                   'contrasts':{name:list(pair) for name,pair in CONTRASTS.items()}},
        'secondary':{'level':0.95,'contrasts':list(CONTRASTS),'interpretation':'descriptive'},
        'interval':'two-sided Student t across equal-weight map means, with equal model weights within map',
        'failure':'stop and preserve files on any unexpected matching or integrity error; no map substitution',
        'oracle':'label-assisted upper bound over all candidates including identity for each cell',
        'runtime':{'worker_count':JOBS, 'torch_threads':2, 'budget_minutes_per_execution':BUDGET_MINUTES,
                   'mkl_threading_layer':'SEQUENTIAL','versions':{name:version(name) for name in VERSIONS}},
        'planning':planning(),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    rv.save(OUT/'protocol.json', protocol)
    print('Frozen final evaluation:',protocol['planning'],flush=True)


def check_protocol() -> dict:
    p = rv.load(OUT/'protocol.json')
    assert p['plan_sha256'] == rv.digest(PLAN)
    assert p['stage2_summary_sha256'] == rv.digest(dev.OUT/'stage2_summary.json')
    assert p['stage3_summary_sha256'] == rv.digest(prior.OUT/'summary.json')
    for path, expected in p['source_sha256'].items():
        assert rv.digest(ROOT / path) == expected, path
    for checkpoint in p['models'].values():
        assert rv.digest(ROOT / checkpoint['checkpoint']) == checkpoint['sha256']
    assert p['runtime']['versions'] == {name:version(name) for name in VERSIONS}
    assert p['runtime']['mkl_threading_layer'] == os.environ.get('MKL_THREADING_LAYER')
    assert p['context_event_fraction'] == CONTEXT and p['paired_cutoff'] == 0
    assert p['shots_per_strength'] == SHOTS and [x['local_seed'] for x in p['maps']] == list(MAPS)
    return p


def write_predictions(file, entry, layout, lookup, order, raw, decoders, model, predictor, head):
    assert predictor[2]['selector_target_mode'] == sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM
    anchor = prior.CELLS[0]
    entry = replace(entry,matching=decoders[anchor], baseline_predicted_observables=raw[anchor])
    events = entry.detector_events
    truth = entry.observable_flips
    data = {'truth':truth, **{cell+'_raw':y for cell,y in raw.items()}}
    chunks = {}
    for lo in range(0,len(events),shift.BATCH):
        hi = min(lo+shift.BATCH,len(events))
        batch = events[lo:hi]
        bundle, utility, margin = rv.score_bundle(entry,lo,hi,predictor)
        predicted = prior.decoded_from_transition(bundle,raw[anchor][lo:hi])
        block = {'shot_index':lo+bundle.shot_indices, 'candidate_mask':np.packbits(bundle.candidate_edit_mask,axis=1)}
        for cell in prior.CELLS:
            baseline = raw[cell][lo:hi]
            if cell == anchor:
                scored, logits, pred = bundle, utility, predicted
            else:
                scored, logits, pred = prior.rescore(bundle,batch,baseline,decoders[cell],predictor)
            masks, predictions, _ = prior.choose(scored,logits,pred,baseline,batch,layout,lookup,order,model,head,margin)
            block[cell+'_candidates'] = dev.paired.class4(pred)
            for name in prior.POLICIES:
                block[cell+'_'+name] = predictions[name]
                block[cell+'_'+name+'_mask'] = np.packbits(masks[name],axis=1)
        for name, arr in block.items():
            chunks.setdefault(name,[]).append(arr)
    data.update({key:np.concatenate(parts) for key,parts in chunks.items()})
    with file.open('xb') as stream:
        np.savez_compressed(stream,**data)
    return {'sha256':rv.digest(file), 'candidates':len(data['shot_index']), 'margin':predictor[3]}


def evaluate_map(spec, protocol, deadline):
    sedp.torch.set_num_threads(2)
    seed = spec['local_seed']
    root = OUT / 'evaluation' / str(seed)
    for i,strength in enumerate(('g0','g1')):
        rv.generate_case(root/strength,distance=7,local_seed=seed,sampler_seed=spec['sampler_seeds'][strength],
                         shots=SHOTS[strength],zz_factor=protocol['zz_factors'][i])
    dirs = [dev.data_root(root / g) for g in ('g0','g1')]
    ideal = [stim.Circuit.from_file(str(path/'circuit.stim')).without_noise() for path in dirs]
    assert ideal[0] == ideal[1] and ideal[0].num_observables == 2
    cfg0, cfg1 = [shift.ExperimentConfig.from_dict(rv.load(path/'config.json')['config']) for path in dirs]
    assert cfg0.dataset.shots == SHOTS['g0'] and cfg1.dataset.shots == SHOTS['g1']
    assert cfg0 == replace(cfg1, dataset=replace(cfg1.dataset, shots=SHOTS['g0']),
                           willow=replace(cfg1.willow,corr=replace(cfg1.willow.corr,zz_factor=shift.G0)))
    decoders = shift.open_decoders(dirs)
    predictors = {m:rv.load_selector(ckpt) for m,ckpt in rv.checkpoints(shift.REVALIDATION,7,list(range(8))).items()}
    expected = rv.digest(OUT/'protocol.json')
    for strength in ('g0','g1'):
        entry, layout, lookup, order, _, raw = prior.inputs(root,strength,decoders)
        assert len(entry.detector_events) == SHOTS[strength]
        for model,predictor in predictors.items():
            if time() >= deadline:
                raise TimeoutError('Final evaluation reached its frozen wall-time budget')
            file = root / strength / f'seed{model}.npz'
            marker = file.with_suffix('.json')
            if marker.exists():
                record = rv.load(marker)
                assert expected == record['protocol_sha256'] and rv.digest(file) == record['sha256']
                continue
            if file.exists():
                raise FileExistsError(file)
            started = perf_counter()
            record = write_predictions(file,entry,layout,lookup,order,raw,decoders,model,predictor,protocol['paired_head'])
            rv.save(marker,{**record,'protocol_sha256':expected,'wall_seconds':perf_counter()-started})
        print(f'Map {seed} {strength} complete',flush=True)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','run'))
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX','.')):
        parser.error('Activate the Anaconda fp environment first')
    if args.mode == 'freeze':
        freeze()
        return
    p = check_protocol()
    started = perf_counter()
    deadline = time()+60*BUDGET_MINUTES
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        futures = [pool.submit(evaluate_map,spec,p,deadline) for spec in p['maps']]
        for future in as_completed(futures):
            print(f'Map {future.result()} verified ({perf_counter()-started:.1f} s elapsed)',flush=True)


if __name__ == '__main__':
    main()
