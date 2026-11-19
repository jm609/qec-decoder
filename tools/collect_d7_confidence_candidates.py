"""Cache D0/MWPM candidates for the six reserved d7 development maps."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from time import perf_counter, time

import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_predecoder_noise_shift as shift

rv, sedp = shift.rv, shift.sedp
OUT = ROOT / 'artifacts/eval/d7_confidence_development/20261116'
CONTEXT = 0.042736961728050596


def collect_map(out: Path, role: str, spec: dict, registry: dict, deadline: float) -> dict:
    sedp.torch.set_num_threads(2)
    local_seed = spec['local_seed']
    base = out / role / str(local_seed)
    targets = [rv.generate_case(
        base / f'g{g}', distance=7, local_seed=local_seed, sampler_seed=spec[f'g{g}_sampler_seed'],
        shots=spec['shots_per_strength'], zz_factor=zz,
    ) for g, zz in enumerate((shift.G0, shift.G1))]
    raw = [Path(rv.load(target / 'metadata.json')['source_family_dir']) for target in targets]
    circuits = [stim.Circuit.from_file(str(path / 'circuit.stim')) for path in raw]
    assert circuits[0].without_noise() == circuits[1].without_noise() and circuits[0].num_observables == 2
    decoder = shift.open_decoders(raw[:1])[shift.D0_MWPM]
    checkpoints = rv.checkpoints(shift.REVALIDATION, 7, list(range(8)))
    for seed, model in registry['models'].items():
        assert rv.digest(ROOT / model['checkpoint']) == model['sha256'], seed
    predictors = {seed: rv.load_selector(ckpt) for seed, ckpt in checkpoints.items()}
    metadata = rv.load(raw[0] / 'metadata.json')
    data = {}
    contexts = []
    for g, target in enumerate(targets):
        entry = sedp._prepare_edit_family(target, fill_value=-0.5, max_shots=None)
        entry, context = shift.with_context(entry, metadata, CONTEXT)
        contexts.append(context)
        directory = base / f'g{g}'
        data[f'g{g}'] = {}
        for seed, predictor in predictors.items():
            if time() >= deadline:
                raise TimeoutError('Candidate collection reached its one-hour budget')
            file = directory / f'predictions_seed{seed}.npz'
            marker = file.with_suffix('.json')
            if not marker.exists():
                if file.exists():
                    raise FileExistsError(f'Prediction file without completed record: {file}')
                started = perf_counter()
                prediction = shift.write_case_predictions(file, entry, {shift.D0_MWPM: decoder},
                                                          predictor, adopt=False)
                rv.save(marker, {**prediction, 'wall_seconds': perf_counter() - started})
            record = rv.load(marker)
            assert rv.digest(file) == record['sha256']
            assert record['proposal_fallbacks'] == record['cells'][shift.D0_MWPM]['candidate_fallback_rows'] == 0
            data[f'g{g}'][str(seed)] = record
            print(f'{role} {local_seed} g{g} seed{seed}: {record["wall_seconds"]:.1f} s', flush=True)
    assert contexts[0] == contexts[1]
    result = {'spec': spec, 'data': data, 'context': contexts[0]}
    marker = base / 'result.json'
    if marker.exists():
        assert rv.load(marker) == result
    else:
        rv.save(marker, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'fp' or Path(sys.prefix) != Path(os.environ.get('CONDA_PREFIX', '.')):
        parser.error('Activate the Anaconda fp environment first')
    registry = rv.load(OUT / 'data_registry.json')
    deadline = time() + 3600
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(collect_map, OUT, role, spec, registry, deadline): spec['local_seed']
                   for role, maps in registry['reserved'].items() for spec in maps}
        for future in as_completed(pending):
            future.result()
            print(f'Map {pending[future]} complete ({perf_counter() - started:.1f} s elapsed)', flush=True)


if __name__ == '__main__':
    main()
