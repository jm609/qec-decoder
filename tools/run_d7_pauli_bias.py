"""Run the d7 fixed-total data-qubit Pauli bias development pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import t
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_dataset import save_family_dataset
from tools import run_predecoder_noise_shift as shift
from tools.collect_d7_confidence_candidates import CONTEXT

rv, sedp = shift.rv, shift.sedp
DATASET = dict(family="stage_c_corr", distance=7, rounds=7, basis="z",
               variant="stim_rotated", target_mode="bell_pair_z_readout",
               physical_error_rate=0.0015, zz_factor=0.05)
ETAS = (0.5, 5.0, 20.0)
MODES = ("mwpm", "correlated")
MAPS = tuple(range(788500, 788508))
SHOTS = 8192
SELECTOR_SHOTS = 2048
MODEL_SEED = 2  # Existing d7 revalidation recipe seed, fixed before pilot results.


def cell(eta: float) -> str:
    return str(eta)


def summarize_contrast(rows: list[dict]) -> dict:
    gains = np.array([row["gain_pp"] for row in rows])
    mean = float(gains.mean())
    sd = float(gains.std(ddof=1))
    half = float(t.ppf(0.975, len(gains) - 1) * sd / np.sqrt(len(gains)))
    return {
        "maps": len(gains), "gain_pp": mean, "map_sd_pp": sd,
        "descriptive_95_ci_pp": [mean - half, mean + half],
        "improved": sum(row["improved"] for row in rows),
        "harmed": sum(row["harmed"] for row in rows),
        "shots": sum(row["shots"] for row in rows),
    }


def error_counts(prediction: np.ndarray, truth: np.ndarray) -> dict:
    errors = prediction != truth
    return {"joint": int(errors.any(axis=1).sum()), "logical_z": int(errors[:, 0].sum()),
            "logical_x": int(errors[:, 1].sum())}


def baseline_map(out: Path, seed: int) -> dict:
    root = out / str(seed)
    marker = root / "baseline.json"
    if marker.exists():
        return rv.load(marker)
    raw_paths, records = [], {}
    for index, eta in enumerate(ETAS):
        case = root / f"b{index}"
        sampler_seed = 1885000 + 3 * (seed - MAPS[0]) + index
        case_marker = case / "dataset.json"
        if case_marker.exists():
            dataset = rv.load(case_marker)
            raw = out / dataset["path"]
            for name, expected in dataset["sha256"].items():
                assert rv.digest(raw / name) == expected, raw / name
        else:
            raw = save_family_dataset(out_root=case, shots=SHOTS, seed=sampler_seed, overwrite=False,
                                      local_seed=seed, data_pauli_z_bias=eta, **DATASET)
            dataset = {
                "path": raw.relative_to(out).as_posix(), "sampler_seed": sampler_seed,
                "sha256": {name: rv.digest(raw / name) for name in
                           ("circuit.stim", "detector_error_model.dem", "samples.npz",
                            "metadata.json", "config.json")},
            }
            rv.save(case_marker, dataset)
        raw_paths.append(raw)
        records[cell(eta)] = {"dataset": dataset}
    circuits = [stim.Circuit.from_file(str(raw / "circuit.stim")) for raw in raw_paths]
    assert circuits[0].num_observables == 2
    assert all(c.without_noise() == circuits[0].without_noise() for c in circuits)
    decoders = shift.open_decoders(raw_paths)
    for index, (eta, raw_path) in enumerate(zip(ETAS, raw_paths)):
        with np.load(raw_path / "samples.npz") as samples:
            events, truth = samples["detector_events"], samples["observable_flips"]
        assert truth.shape == (SHOTS, 2)
        row = records[cell(eta)]
        row.update(detector_event_fraction=float(events.mean()), modes={})
        path = raw_path.parent / "baseline_predictions.npz"
        predictions = {"truth": truth}
        for mode in MODES:
            uniform = decoders[f"d0_{mode}"].decode_batch(events)
            matched = decoders[f"d{index}_{mode}"].decode_batch(events)
            if index == 0:
                np.testing.assert_array_equal(uniform, matched)
            predictions.update({f"{mode}_uniform": uniform, f"{mode}_matched": matched})
            row["modes"][mode] = dict(
                uniform_errors=error_counts(uniform, truth), matched_errors=error_counts(matched, truth),
                paired=rv.paired_counts(uniform, matched, truth),
            )
        if path.exists():
            with np.load(path) as saved:
                for key, data in predictions.items():
                    np.testing.assert_array_equal(saved[key], data)
        else:
            np.savez_compressed(path, **predictions)
        row["predictions"] = {"path": path.relative_to(out).as_posix(), "sha256": rv.digest(path)}
    rv.save(marker, records)
    return records


def baseline_pilot(out: Path) -> None:
    rows = {}
    for seed in MAPS:
        rows[str(seed)] = baseline_map(out, seed)
        print(f"Baseline map {seed} complete", flush=True)
    summary = {}
    for eta in ETAS:
        levels = [row[cell(eta)] for row in rows.values()]
        modes = {}
        for mode in MODES:
            comparisons = [row["modes"][mode] for row in levels]
            rates = {f"{prior}_ler_pp": {
                target: 100 * sum(c[f"{prior}_errors"][target] for c in comparisons) / (len(MAPS) * SHOTS)
                for target in ("joint", "logical_z", "logical_x")
            } for prior in ("uniform", "matched")}
            modes[mode] = {**rates, "matched_vs_uniform": summarize_contrast([c["paired"] for c in comparisons])}
        summary[cell(eta)] = {
            "modes": modes,
            "best_matched_mode": min(MODES, key=lambda name: modes[name]["matched_ler_pp"]["joint"]),
        }
        print(f"eta={eta:g} {summary[cell(eta)]}", flush=True)
    rv.save(out / "baseline_summary.json", {"conditions": summary, "maps": rows})


def selector_case(out: Path, row: dict, control_metadata: dict, mode: str, predictor: tuple) -> dict:
    raw_path = out / row["dataset"]["path"]
    marker = raw_path.parent / "selector.json"
    path = raw_path.parent / "selector_predictions.npz"
    if marker.exists():
        record = rv.load(marker)
        assert record["mode"] == mode and record["predictions_sha256"] == rv.digest(path)
        return record
    for name, expected in row["dataset"]["sha256"].items():
        assert rv.digest(raw_path / name) == expected, raw_path / name
    base_path = out / row["predictions"]["path"]
    assert rv.digest(base_path) == row["predictions"]["sha256"]
    with np.load(base_path) as saved:
        raw, truth = saved[f"{mode}_matched"][:SELECTOR_SHOTS], saved["truth"][:SELECTOR_SHOTS]
    target_root = raw_path.parent / "targets"
    manifest_path = target_root / "manifest.json"
    manifest = rv.load(manifest_path) if manifest_path.exists() else rv.targets.build_from_family_dir(
        raw_path, out_root=target_root, search_config=rv.targets.SearchConfig(0, 0, 0, 1),
        max_shots=SELECTOR_SHOTS, allow_circuit_fallback=False,
    )
    entry = sedp._prepare_edit_family(target_root / manifest["family_dirs"][DATASET["family"]],
                                     fill_value=-0.5, max_shots=SELECTOR_SHOTS)
    entry, context = shift.with_context(entry, control_metadata, CONTEXT)
    decoder = shift.open_decoders([raw_path])[f"d0_{mode}"]
    np.testing.assert_array_equal(entry.observable_flips, truth)
    np.testing.assert_array_equal(decoder.decode_batch(entry.detector_events), raw)
    entry = replace(entry, matching=decoder, baseline_predicted_observables=raw)
    selected, oracle_parts = [], []
    fallbacks = 0
    started = perf_counter()
    for lo in range(0, len(raw), shift.BATCH):
        hi = min(lo + shift.BATCH, len(raw))
        bundle, utility, margin = rv.score_bundle(entry, lo, hi, predictor)
        chosen = shift.selected_rows(bundle, utility, margin)
        offsets = sedp._transition_feature_offsets(candidate_feature_dim=bundle.candidate_features.shape[1])
        delta = bundle.candidate_features[:, offsets["obs_delta"]]
        assert np.all((delta == 0) | (delta == 1))
        predictions = raw[lo:hi][bundle.shot_indices] ^ delta.astype(np.uint8)
        checked, failed = shift.decode_edits(decoder, entry.detector_events[lo:hi],
                                             bundle.candidate_edit_mask[chosen], raw[lo:hi])
        np.testing.assert_array_equal(checked, predictions[chosen])
        selected.append(checked)
        oracle = np.zeros(hi - lo, dtype=bool)
        correct = np.all(predictions == truth[lo:hi][bundle.shot_indices], axis=1)
        np.logical_or.at(oracle, bundle.shot_indices, correct)
        oracle_parts.append(oracle)
        fallbacks += bundle.decode_fallback_count
        assert not failed.any()
    predicted, oracle = np.concatenate(selected), np.concatenate(oracle_parts)
    wrong = (raw != truth).any(axis=1)
    assert np.all(oracle[~wrong])
    np.savez_compressed(path, truth=truth, raw=raw, selected=predicted, oracle=oracle)
    record = dict(
        mode=mode, **rv.paired_counts(raw, predicted, truth),
        raw_errors=error_counts(raw, truth), selected_errors=error_counts(predicted, truth),
        logical_changes=int((raw != predicted).any(axis=1).sum()),
        oracle_recovered=int((wrong & oracle).sum()), decode_fallbacks=fallbacks,
        selector_seconds=perf_counter() - started, context=context,
        predictions_sha256=rv.digest(path),
    )
    rv.save(marker, record)
    return record


def selector_pilot(out: Path) -> None:
    baseline = rv.load(out / "baseline_summary.json")
    checkpoint = rv.checkpoints(shift.REVALIDATION, 7, [MODEL_SEED])[MODEL_SEED]
    predictor = rv.load_selector(checkpoint)
    sedp.torch.set_num_threads(2)
    records = {cell(eta): {} for eta in ETAS}
    for seed, cases in baseline["maps"].items():
        control = rv.load(out / cases[cell(ETAS[0])]["dataset"]["path"] / "metadata.json")
        for eta in ETAS:
            key = cell(eta)
            mode = baseline["conditions"][key]["best_matched_mode"]
            row = selector_case(out, cases[key], control, mode, predictor)
            records[key][seed] = row
            print(f"Selector map {seed} eta={eta:g} {mode} {row['gain_pp']:+.4f} pp", flush=True)
    summary = {}
    for key, maps in records.items():
        rows = list(maps.values())
        errors = sum(row["raw_errors"]["joint"] for row in rows)
        recovered = sum(row["oracle_recovered"] for row in rows)
        summary[key] = {**summarize_contrast(rows), "mode": rows[0]["mode"],
                        "raw_errors": errors, "oracle_recovered": recovered,
                        "logical_changes": sum(row["logical_changes"] for row in rows),
                        "decode_fallbacks": sum(row["decode_fallbacks"] for row in rows)}
        print(f"Selector eta={key} {summary[key]}", flush=True)
    rv.save(out / "selector_summary.json", {"conditions": summary, "maps": records})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "selector"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "pilot_protocol.json"
    checkpoint = shift.REVALIDATION / "d7" / "train" / f"seed{MODEL_SEED}" / "sedp_checkpoint.pt"
    protocol = dict(
        role="development only", dataset=DATASET, swap_factor=0.02, etas=list(ETAS),
        local_seeds=list(MAPS), sampler_seed_start=1885000, shots_per_map=SHOTS,
        selector_shots_per_map=SELECTOR_SHOTS, model_seed=MODEL_SEED,
        selector_policy="A/B-trained checkpoint margin, forced evaluation against matched raw",
        context_detector_fraction=CONTEXT, checkpoint_sha256=rv.digest(checkpoint),
        matched_mode_rule="lowest joint logical error across baseline pilot maps; ties use ordinary MWPM",
        versions={name: version(name) for name in ("stim", "pymatching", "numpy", "torch", "scipy")},
        source_sha256={name: rv.digest(ROOT / name) for name in (
            "sample_dataset.py", "noise_willowcore.py", "noise_si1000.py", "config.py", "circuits.py",
            "logical_bell.py", "decoders/syndrome_edit_predecoder.py", "decoders/research_noise_aware_3d.py",
            "tools/revalidate_predecoder.py", "tools/run_predecoder_noise_shift.py",
            "tools/run_d7_pauli_bias.py",
        )},
    )
    if protocol_path.exists():
        assert rv.load(protocol_path) == protocol, "Pilot settings or source changed"
    else:
        rv.save(protocol_path, protocol)
    if args.phase == "baseline":
        if not (out / "baseline_summary.json").exists():
            baseline_pilot(out)
    elif not (out / "selector_summary.json").exists():
        selector_pilot(out)


if __name__ == "__main__":
    main()
