"""Regenerate the paper's d7 figures, tables, and primary contrasts in a new directory."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'artifacts/eval'
FINAL = EVAL / 'd7_final/20261117'
SOURCES = {
    'revalidation': 'predecoder_revalidation_20261115/independent_revalidation_summary.json',
    'shift': 'noise_shift_d7/20261115_retrained/summary.json',
    'diagnostic': 'd7_logical_transition_diagnostic/20261116/summary.json',
    'development': 'd7_confidence_development/20261116/stage2_summary.json',
    'active_prior': 'd7_active_prior/20261117/summary.json',
    'final': 'd7_final/20261117/summary.json',
    'cost': 'd7_final_benchmark/20261117/benchmark.json',
}
ABLAT = {
    'margin': 'Shotwise margin',
    'paired_no_events': 'Paired, no detector features',
    'paired_detector': 'Paired detector',
    'paired_detector_gap': 'Paired detector + logical gap',
    'z_vs_no_edit': 'Z candidate vs no edit',
    'raw_z': 'Raw-only Z switch',
    'raw_z_gap': 'Raw-only Z switch + gap',
}
CONTRASTS = {'margin_minus_raw': 'Raw', 'margin_minus_paired': 'Paired'}
COLORS = {'raw': '#494949', 'margin': '#1D648D', 'paired': '#B1541D'}
BLUE, ORANGE = COLORS['margin'], COLORS['paired']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, directory, name):
    for ext in ('pdf', 'png'):
        fig.savefig(directory / f'{name}.{ext}', dpi=240)
    plt.close(fig)


def write_csv(directory, name, header, rows):
    with (directory / name).open('x', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def rederive_primary(result):
    """Count paired correctness changes directly from frozen predictions."""
    p = load(FINAL / 'protocol.json')
    protocol_sha = digest(FINAL / 'protocol.json')
    assert result['protocol_sha256'] == protocol_sha
    maps = {}
    model_net = {key: np.zeros(len(p['models']), dtype=np.int64) for key in CONTRASTS}
    for spec in p['maps']:
        seed = str(spec['local_seed'])
        counts = {key: np.zeros(2, dtype=np.int64) for key in CONTRASTS}
        for model in p['models']:
            file = FINAL / 'evaluation' / seed / 'g1' / f'seed{model}.npz'
            marker = load(file.with_suffix('.json'))
            assert marker['protocol_sha256'] == protocol_sha and marker['sha256'] == digest(file)
            with np.load(file) as saved:
                truth = saved['truth']
                assert len(truth) == p['shots_per_strength']['g1']
                correct = {m: np.all(saved[f'd0_mwpm_{m}'] == truth, axis=1)
                           for m in ('margin', 'raw', 'paired')}
                for key, (_, ref) in p['primary']['contrasts'].items():
                    fixes = int(np.count_nonzero(correct['margin'] & ~correct[ref]))
                    harms = int(np.count_nonzero(~correct['margin'] & correct[ref]))
                    counts[key] += (fixes, harms)
                    model_net[key][int(model)] += fixes-harms
        maps[seed] = {key: values.tolist() for key, values in counts.items()}
    denominator = p['shots_per_strength']['g1']*len(p['models'])
    output = {}
    for key in CONTRASTS:
        by_map = {seed: 100*(v[key][0]-v[key][1])/denominator for seed, v in maps.items()}
        values = np.array(list(by_map.values()))
        mean = float(values.mean())
        half = float(t.isf(0.0125, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
        fixes, harms = np.sum([v[key] for v in maps.values()], axis=0)
        reference = result['results']['g1']['d0_mwpm']['comparisons'][key]
        assert by_map == reference['by_map_pp']
        assert (fixes, harms) == (reference['improved'], reference['harmed'])
        np.testing.assert_allclose((mean, mean-half, mean+half),
                                   [reference['gain_pp'], *reference['interval_pp']], atol=1e-12, rtol=0)
        output[key] = {'gain_pp': mean, 'interval_pp': [mean-half, mean+half], 'interval_level': 0.975,
                       'improved': int(fixes), 'harmed': int(harms), 'by_map_pp': by_map,
                       'by_checkpoint_pp': (100*model_net[key]/(len(maps)*p['shots_per_strength']['g1'])).tolist()}
    return {'protocol_sha256': protocol_sha, 'prediction_files_read': len(maps)*len(p['models']),
            'physical_shots': len(maps)*p['shots_per_strength']['g1'], 'model_shots': len(maps)*denominator,
            'contrasts': output}


def method_figure(out):
    fig, ax = plt.subplots(figsize=(7.7, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis('off')
    def card(x, y, text, fc):
        ax.text(x, y, text, ha='center', va='center',
                bbox={'boxstyle': 'round,pad=.45', 'fc': fc, 'ec': '#8492A0'})
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops={'arrowstyle': '->', 'color': '#474F57'})
    card(1.3, 2.0, 'Detector events\nSame shot', '#EDF2F5')
    card(4.3, 3.0, 'Raw PyMatching\nTwo observables', '#F1F2F3')
    card(4.3, 1.0, 'Local edit proposals\nincluding no edit', '#DFEDF4')
    card(7.0, 1.0, 'Decode candidates\nand score transitions', '#DFEDF4')
    card(8.65, 3.0, 'Policy decision\nMargin or paired', '#F8EBDD')
    arrow(2.37,2.15,3.15,2.85); arrow(2.37,1.85,3.12,1.14)
    arrow(5.48,1.0,5.85,1.0); arrow(7.63,1.42,8.34,2.56)
    arrow(5.36,3.0,7.60,3.0); arrow(4.93,2.55,6.43,1.46)
    ax.text(5, 0.12, 'Audit: compare both predictions with truth; score the candidate-set oracle separately.',
            ha='center', va='center', color='#454D54', fontsize=8)
    save(fig, out, 'fig01_same_shot_method')


def failure_figure(data,out):
    d = data['diagnostic']['cells']
    fig, (ax, counts) = plt.subplots(1, 2, figsize=(8.7, 3.7), layout='constrained')
    columns = ['gain_pp', 'fixed_edit_oracle_gain_pp', 'oracle_gain_pp']
    labels = ['Top-1', 'Fixed-shot oracle', 'Full candidate oracle']
    for i,(key,color,label) in enumerate([('g0_d0_mwpm',BLUE,'Weak ZZ'),('g1_d0_mwpm',ORANGE,'Strong ZZ')]):
        vals = [d[key]['top1'][col] for col in columns]
        bars = ax.barh(np.arange(3)+(i-.5)*.31, vals, height=.28, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:+.2f}' for v in vals], padding=3, fontsize=8)
    ax.set(yticks=np.arange(3), yticklabels=labels, xlim=(-12,15),
           xlabel='Accuracy difference vs raw (pp)', title='a  Candidate and ranking diagnostic')
    ax.invert_yaxis(); ax.axvline(0,color='.45',lw=.8)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5,-.22), ncols=2)
    cell = data['shift']['cells']['g1_d0_mwpm']
    rows = [cell[key] for key in ('top1', 'branch', 'final')]
    for i,(metric,label,color) in enumerate([('improved','Repairs',BLUE),('harmed','Harms',ORANGE)]):
        vals = [r[metric] for r in rows]
        bars = counts.barh(np.arange(3)+(i-.5)*.31, vals, height=.28, color=color, label=label)
        counts.bar_label(bars,labels=[f'{v:,}' for v in vals],padding=3,fontsize=8)
    counts.set(yticks=np.arange(3),yticklabels=['Top-1','Shotwise margin','Family gate'],xlim=(0,32600),
               xlabel='Model–shot pairs',title='b  Strong-ZZ adoption (D₀ MWPM)')
    counts.invert_yaxis()
    counts.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22),ncols=2)
    save(fig, out, 'fig02_failure_diagnostic')


def development_figure(data,out):
    b = data['development']['selection']
    fig, ax = plt.subplots(figsize=(7.3,3.8),layout='constrained')
    for i, key in enumerate(ABLAT):
        vals = b['g1_free_gain_pp_by_map'][key]
        ax.scatter(vals,i+np.linspace(-.11,.05,len(vals)),facecolors='none',edgecolors=BLUE)
        ax.scatter(np.mean(vals),i+.15,marker='D',color=ORANGE,s=23,zorder=3)
    ax.axvline(0,color='.45',lw=.8)
    ax.set(yticks=range(len(ABLAT)),yticklabels=list(ABLAT.values()),
           xlabel='Accuracy difference vs raw (pp)',title='Strong-ZZ development folds (D₀ MWPM)')
    ax.invert_yaxis()
    ax.scatter([],[],facecolors='none',edgecolors=BLUE,label='Held-out map')
    ax.scatter([],[],color=ORANGE,marker='D',label='Mean')
    ax.legend(frameon=False,loc='lower left')
    save(fig,out,'fig03_selector_development')


def final_figure(primary,out):
    fig,(maps,models) = plt.subplots(1,2,figsize=(9.0,3.1),layout='constrained',sharex=True)
    for pos,(key,compare) in enumerate(CONTRASTS.items()):
        row=primary['contrasts'][key]
        points=list(row['by_map_pp'].values())
        # A fixed vertical spread separates overlapping points without changing their x values.
        maps.scatter(points,pos+np.linspace(-.12,.12,len(points)),s=18,facecolors='none',edgecolors=BLUE,alpha=.7)
        low,high=row['interval_pp']
        maps.errorbar(row['gain_pp'],pos+.24,xerr=[[row['gain_pp']-low],[high-row['gain_pp']]],
                      fmt='D',color=ORANGE,capsize=4,zorder=4)
        models.scatter(row['by_checkpoint_pp'],pos+np.linspace(-.12,.12,len(row['by_checkpoint_pp'])),
                       s=22,facecolors='none',edgecolors=BLUE)
    for ax,title in [(maps,'a  24 maps and 97.5% intervals'),(models,'b  Eight fixed checkpoints')]:
        ax.set(yticks=[0,1],yticklabels=[f'Margin vs {name.lower()}' for name in CONTRASTS.values()],
               ylim=(1.55,-.36),xlabel='Paired accuracy difference (pp)',title=title)
        ax.axvline(0,lw=.8,color='.45')
    save(fig,out,'fig04_independent_effects')


def cost_figure(data,out):
    cost = data['cost']['results']
    final = data['final']['results']
    fig,(time,acc) = plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
    for i,key in enumerate(('raw','margin','paired')):
        for j,g in enumerate(('g0','g1')):
            pos=i+(j-.5)*.26
            marker=('o','s')[j]
            row=cost[g][key]
            seconds=row['medians']['seconds']
            measured=[trial['seconds'] for trial in row['repetitions']]
            time.errorbar(seconds,pos,xerr=[[seconds-min(measured)],[max(measured)-seconds]],
                          fmt=marker,color=COLORS[key],capsize=3)
            cell=final[g]['d0_mwpm']
            value={'raw':0.,'margin':cell['comparisons']['margin_minus_raw']['gain_pp'],
                   'paired':cell['paired_gain_over_raw_pp']}[key]
            acc.scatter(value,pos,marker=marker,color=COLORS[key],s=30)
    for ax in (time,acc):
        ax.set(yticks=range(3),yticklabels=['Raw','Margin','Paired'],ylim=(2.65,-.6))
    time.set(xscale='log', xlabel='Seconds per batch (log scale)', title='a  CPU, one map / checkpoint')
    acc.set(xlabel='Accuracy difference vs raw (pp)',title='b  Accuracy, 24 independent maps')
    acc.axvline(0,color='.45',lw=.8)
    for marker,name in [('o','Weak ZZ'),('s','Strong ZZ')]:
        time.scatter([],[],marker=marker,color='.3',label=name)
    time.legend(frameon=False,loc='lower left')
    save(fig,out,'fig05_cost_and_accuracy')


def tables(data,primary,out):
    write_csv(out,'revalidation.csv',['distance','maps','physical_shots','raw_pct','policy_pct','gain_pp',
                                     'interval_level','low_pp','high_pp','adopting_checkpoints'],
              [(d,x['maps'],x['shots'],x['raw_accuracy_pct'],x['selected_accuracy_pct'],x['gain_pp'],
                x['ci_level'],*x['ci_pp'],sum(m['mode']!='raw_no_edit' for m in x['models'].values()))
               for d,x in data['revalidation']['distances'].items()])
    write_csv(out,'noise_shift.csv',['cell','raw_pct','top1_pp','margin_pp','family_gate_pp','oracle_pp',
                                     'top1_repairs','top1_harms','margin_repairs','margin_harms'],
              [(k,x['raw_accuracy_pct'],x['top1_gain_pp'],x['branch_gain_pp'],x['final_gain_pp'],
                x['oracle_gain_pp'],x['top1']['improved'],x['top1']['harmed'],
                x['branch']['improved'],x['branch']['harmed']) for k,x in data['shift']['cells'].items()])
    write_csv(out,'active_prior.csv',['panel','strength','cell','raw_pct','policy','fixed_gain_pp','full_gain_pp'],
              [(panel,g,k,x['raw_accuracy_pct'],method,p['fixed']['gain_pp'],p['full']['gain_pp'])
               for panel, gs in data['active_prior']['panels'].items() for g,cells in gs.items()
               for k,x in cells.items() for method,p in x['policies'].items()])
    b=data['development']['selection']
    write_csv(out,'selector_ablation.csv',['arm','weak_gain_pp','strong_gain_pp','strong_winning_maps_vs_paired'],
              [(k,np.mean(b['g0_free_gain_pp_by_map'][k]),b['mean_g1_free_gain_pp'][k],
                np.count_nonzero(np.array(b['g1_free_gain_pp_by_map'][k])>b['g1_free_gain_pp_by_map']['paired_detector']))
               for k in b['mean_g1_free_gain_pp']])
    write_csv(out,'independent_cells.csv',['strength','cell','physical_shots','raw_pct','paired_gain_pp','oracle_pct',
                                          'contrast','gain_pp','interval_level','low_pp','high_pp','improved','harmed'],
              [(g,k,x['physical_shots'],x['raw_accuracy_pct'],x['paired_gain_over_raw_pp'],x['oracle_accuracy_pct'],
                name,c['gain_pp'],c['interval_level'],*c['interval_pp'],c['improved'],c['harmed'])
               for g,cells in data['final']['results'].items() for k,x in cells.items()
               for name,c in x['comparisons'].items()])
    write_csv(out,'independent_map_effects.csv',['map',*CONTRASTS],
              [(seed,*[x['by_map_pp'][seed] for x in primary['contrasts'].values()])
               for seed in next(iter(primary['contrasts'].values()))['by_map_pp']])
    write_csv(out,'cpu_cost.csv',['strength','route','median_seconds','min_seconds','max_seconds','shots_per_s',
                                 'peak_working_set_bytes','batch_matching_calls','syndrome_rows'],
              [(g,k,x['medians']['seconds'],min(r['seconds'] for r in x['repetitions']),
                max(r['seconds'] for r in x['repetitions']),x['shots_per_second'],x['working_set_bytes']['peak'],
                x['matching_counts']['batch_calls'],x['matching_counts']['rows'])
               for g,routes in data['cost']['results'].items() for k,x in routes.items()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    data={key:load(EVAL/path) for key,path in SOURCES.items()}
    assert data['cost']['protocol_sha256']==data['final']['protocol_sha256']
    assert data['diagnostic']['source_summary_sha256']==digest(EVAL/SOURCES['shift'])
    primary=rederive_primary(data['final'])
    (out/'primary_recalculation.json').write_text(json.dumps(primary,indent=2)+'\n',encoding='utf-8')
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.5,
                                'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    method_figure(out)
    failure_figure(data,out)
    development_figure(data,out)
    final_figure(primary,out)
    cost_figure(data,out)
    tables(data,primary,out)
    manifest={'sources_sha256':{('artifacts/eval/'+p):digest(EVAL/p) for p in SOURCES.values()},
              'builder_sha256':digest(Path(__file__)),'matplotlib_version':matplotlib.__version__,
              'output_sha256':{p.name:digest(p) for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Recomputed 2 primary contrasts from 192 prediction files; wrote 5 figures and 7 tables.')


if __name__=='__main__':
    main()
