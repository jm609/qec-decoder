"""Freeze and evaluate the d7 eta=20 pre-decoder on independent local maps."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_d7_pauli_bias as pilot

rv, shift = pilot.rv, pilot.shift
PILOT = ROOT / "artifacts/eval/d7_pauli_bias/pilot"
OUT = ROOT / "artifacts/eval/d7_pauli_bias/independent"
ETA = 20.0
MODE = "mwpm"
SHOTS = pilot.SELECTOR_SHOTS
MAPS = tuple(range(788600, 788624))
SAMPLER_START = 1886000
MARGIN = 1.25
JOBS = 2


def freeze() -> None:
    development = rv.load(PILOT / "pilot_protocol.json")
    results = rv.load(PILOT / "baseline_summary.json")
    assert results["conditions"][str(ETA)]["best_matched_mode"] == MODE
    checkpoint = shift.REVALIDATION / "d7/train" / f"seed{pilot.MODEL_SEED}" / "sedp_checkpoint.pt"
    assert rv.digest(checkpoint) == development["checkpoint_sha256"]
    sources = dict(development["source_sha256"])
    for name in (
        "tools/evaluate_d7_pauli_bias.py", "logical_frame.py", "logical_targets.py",
        "geometry/rotated_rect.py", "tools/build_pymatching_edit_targets.py",
        "tools/collect_d7_confidence_candidates.py", "decoders/baseline_pymatching.py",
        "decoders/baseline_rectcnn.py",
    ):
        sources[name] = rv.digest(ROOT / name)
    for name, expected in sources.items():
        assert rv.digest(ROOT / name) == expected, name
    protocol = dict(
        role="independent evaluation", eta=ETA, mode=MODE, dataset=pilot.DATASET,
        model_seed=pilot.MODEL_SEED, margin=MARGIN, shots_per_map=SHOTS,
        local_seeds=list(MAPS), sampler_seed_start=SAMPLER_START,
        context_detector_fraction=pilot.CONTEXT,
        primary="paired accuracy gain of pre-decoder versus bias-matched ordinary MWPM",
        interval="two-sided 95% Student t interval across equally weighted independent maps",
        success="primary interval lower bound strictly greater than zero",
        secondary=["matched versus uniform-prior MWPM", "pre-decoder versus uniform-prior MWPM"],
        secondary_intervals="descriptive 95% map intervals",
        selector_policy="existing candidate/margin route explicitly enabled; no retraining or calibration",
        selection_reason="eta=20 is the strongest pilot bias; mode selected on pilot maps",
        precision_plan="pilot map SD 0.2282 pp; 24 maps at 2048 shots give expected 95% half-width 0.0964 pp",
        jobs=JOBS, torch_threads_per_job=2,
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=rv.digest(checkpoint),
        development_sha256={name: rv.digest(PILOT / name) for name in (
            "pilot_protocol.json", "baseline_summary.json", "selector_summary.json", "pilot_validation.json",
        )},
        source_sha256=sources,
        versions={name: version(name) for name in ("stim", "pymatching", "numpy", "torch", "scipy")},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        assert rv.load(path) == protocol, "Frozen independent protocol changed"
    else:
        rv.save(path, protocol)
    archive = OUT / "source_snapshot.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name in sources:
                zipped.write(ROOT / name, arcname=name)
    print(f"Frozen {len(MAPS)} maps, eta={ETA:g}, {MODE}, {SHOTS} shots per map", flush=True)


def check_protocol() -> dict:
    protocol = rv.load(OUT / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        assert rv.digest(ROOT / name) == expected, name
    for name, expected in protocol["development_sha256"].items():
        assert rv.digest(PILOT / name) == expected, name
    assert rv.digest(ROOT / protocol["checkpoint"]) == protocol["checkpoint_sha256"]
    for name, expected in protocol["versions"].items():
        assert version(name) == expected, name
    assert protocol["local_seeds"] == list(MAPS) and protocol["shots_per_map"] == SHOTS
    assert (protocol["eta"], protocol["mode"], protocol["margin"]) == (ETA, MODE, MARGIN)
    return protocol


def dataset(root: Path, eta: float, local_seed: int, sampler_seed: int, shots: int) -> tuple[Path, dict]:
    marker = root / "dataset.json"
    settings = dict(eta=eta, local_seed=local_seed, sampler_seed=sampler_seed, shots=shots)
    if marker.exists():
        record = rv.load(marker)
        assert record["settings"] == settings
        raw = OUT / record["path"]
        for name, expected in record["sha256"].items():
            assert rv.digest(raw / name) == expected, raw / name
        return raw, record
    raw = pilot.save_family_dataset(
        out_root=root, shots=shots, seed=sampler_seed, overwrite=False,
        local_seed=local_seed, data_pauli_z_bias=eta, **pilot.DATASET,
    )
    record = dict(
        path=raw.relative_to(OUT).as_posix(), settings=settings,
        sha256={name: rv.digest(raw / name) for name in (
            "samples.npz", "metadata.json", "config.json", "circuit.stim", "detector_error_model.dem",
        )},
    )
    rv.save(marker, record)
    return raw, record


def evaluate_map(seed: int, protocol: dict) -> dict:
    root = OUT / str(seed)
    marker = root / "result.json"
    if marker.exists():
        result = rv.load(marker)
        assert result["protocol_sha256"] == rv.digest(OUT / "protocol.json")
        for name, expected in result["artifacts_sha256"].items():
            assert rv.digest(OUT / name) == expected, name
        return result
    pilot.sedp.torch.set_num_threads(protocol["torch_threads_per_job"])
    sampler = SAMPLER_START + 2 * (seed - MAPS[0])
    control, control_record = dataset(root / "control", 0.5, seed, sampler, 1)
    biased, biased_record = dataset(root / "biased", ETA, seed, sampler + 1, SHOTS)
    circuits = [stim.Circuit.from_file(str(path / "circuit.stim")) for path in (control, biased)]
    assert circuits[0].without_noise() == circuits[1].without_noise()
    assert circuits[0].num_observables == 2
    with np.load(biased / "samples.npz") as samples:
        events, truth = samples["detector_events"], samples["observable_flips"]
    decoder = shift.open_decoders([control, biased])
    uniform = decoder["d0_mwpm"].decode_batch(events)
    matched = decoder["d1_mwpm"].decode_batch(events)
    path = biased.parent / "baseline_predictions.npz"
    if path.exists():
        with np.load(path) as saved:
            for name, data in (("truth", truth), ("mwpm_matched", matched), ("mwpm_uniform", uniform)):
                np.testing.assert_array_equal(saved[name], data)
    else:
        np.savez_compressed(path, truth=truth, mwpm_matched=matched, mwpm_uniform=uniform)
    row = dict(dataset=biased_record, predictions=dict(path=path.relative_to(OUT).as_posix(),
                                                       sha256=rv.digest(path)))
    checkpoint = rv.checkpoints(shift.REVALIDATION, 7, [pilot.MODEL_SEED])[pilot.MODEL_SEED]
    predictor = rv.load_selector(checkpoint)
    assert predictor[3] == MARGIN
    selected = pilot.selector_case(OUT, row, rv.load(control / "metadata.json"), MODE, predictor)
    with np.load(biased.parent / "selector_predictions.npz") as saved:
        predicted = saved["selected"]
        np.testing.assert_array_equal(saved["truth"], truth)
        np.testing.assert_array_equal(saved["raw"], matched)
    result = dict(
        local_seed=seed, sampler_seed=sampler + 1, protocol_sha256=rv.digest(OUT / "protocol.json"),
        primary=selected,
        matched_vs_uniform=rv.paired_counts(uniform, matched, truth),
        selected_vs_uniform=rv.paired_counts(uniform, predicted, truth),
        datasets=dict(control=control_record, biased=biased_record),
        artifacts_sha256={
            path.relative_to(OUT).as_posix(): rv.digest(path)
            for path in (control.parent / "dataset.json", biased.parent / "dataset.json",
                         biased.parent / "baseline_predictions.npz", biased.parent / "selector_predictions.npz",
                         biased.parent / "selector.json")
        },
    )
    rv.save(marker, result)
    return result


def summarize(records: list[dict]) -> dict:
    summaries = {}
    for name in ("primary", "matched_vs_uniform", "selected_vs_uniform"):
        rows = [record[name] for record in records]
        result = pilot.summarize_contrast(rows)
        result["ci95_pp"] = result.pop("descriptive_95_ci_pp")
        result["raw_ler_pp"] = 100 * sum(row["shots"] - row["raw_correct"] for row in rows) / result["shots"]
        result["selected_ler_pp"] = 100 * sum(row["shots"] - row["selected_correct"] for row in rows) / result["shots"]
        summaries[name] = result
    summaries["primary"]["supports_improvement"] = summaries["primary"]["ci95_pp"][0] > 0
    summaries["logical_changes"] = sum(record["primary"]["logical_changes"] for record in records)
    summaries["decode_fallbacks"] = sum(record["primary"]["decode_fallbacks"] for record in records)
    return summaries


def run() -> None:
    protocol = check_protocol()
    started = perf_counter()
    records = {}
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        futures = [pool.submit(evaluate_map, seed, protocol) for seed in MAPS]
        for future in as_completed(futures):
            record = future.result()
            records[record["local_seed"]] = record
            print(f"Verified map {record['local_seed']} ({len(records)}/{len(MAPS)} complete)", flush=True)
    ordered = [records[seed] for seed in MAPS]
    result = dict(
        protocol_sha256=rv.digest(OUT / "protocol.json"), contrasts=summarize(ordered),
        maps=ordered, result_sha256={str(seed): rv.digest(OUT / str(seed) / "result.json") for seed in MAPS},
    )
    path = OUT / "summary.json"
    if path.exists():
        assert rv.load(path) == result
    else:
        rv.save(path, result)
    print(f"Independent evaluation complete in {perf_counter() - started:.1f} s", flush=True)
    print(result["contrasts"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
