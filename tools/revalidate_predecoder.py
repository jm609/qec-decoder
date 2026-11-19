"""Freeze a Stage C protocol, retrain on A/B, and evaluate fresh Stage C maps."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoders import syndrome_edit_predecoder as sedp
from sample_dataset import save_family_dataset
from tools import build_pymatching_edit_targets as targets


TRAIN_FAMILIES = ["stage_a_si1000", "stage_b_local"]
DISTANCES = (3, 5, 7)
NOISE_SHIFT_RESERVED_D7 = {
    "development_local_seeds": [783000, 783001],
    "evaluation_local_seeds": list(range(783100, 783108)),
    "development_shots_per_strength": 2048,
    "evaluation_shots_per_strength": 25000,
    "sampler_base": 1783000,
    "model_seeds": [2, 11, 54, 57],
    "zz_factors": [0.05, 0.25],
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, content: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def freeze(out: Path) -> None:
    evidence = load(ROOT / "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json")
    parameters = inspect.signature(sedp.run_manifest_experiment).parameters
    protocol = {
        "input_policy": "zero_reserved",
        "versions": {name: version(name) for name in ["torch", "numpy", "stim", "pymatching", "scipy"]},
        "source_sha256": {
            name: digest(ROOT / name) for name in [
                "decoders/syndrome_edit_predecoder.py", "decoders/research_noise_aware_3d.py",
                "decoders/baseline_pymatching.py", "decoders/baseline_rectcnn.py",
                "sample_dataset.py", "config.py", "noise_willowcore.py", "noise_si1000.py",
                "circuits.py", "logical_bell.py", "tools/build_pymatching_edit_targets.py",
            ]
        },
        "distances": {},
        "primary_distances": [3, 5],
        "metric": "selected minus raw paired logical_class4 accuracy in pp, equal seed and map weights",
        "interval": "two-sided map-level Student t, 97.5% each at d3/d5 and descriptive 95% at d7",
        "decision": "positive lower bound supports improvement; negative upper bound supports harm; otherwise inconclusive",
        "precision_planning": "legacy d3 discordance 356/8192; shot-only 97.5% half-width about 0.21 pp at 49152 shots, map variation extra",
        "circuit": {"basis": "z", "variant": "stim_rotated", "target_mode": "bell_pair_z_readout", "p": 0.0015, "zz_factor": 0.05},
        "noise_shift_reserved_d7": NOISE_SHIFT_RESERVED_D7,
    }
    for record in evidence["distance_results"]:
        distance = int(record["distance"][1:])
        source_seed = 2 if distance == 7 else 4
        rows = load(ROOT / record["artifact"])["rows"]
        row = next(row for row in rows if row["seed"] == source_seed)
        source = ROOT / row["run_dir"] / "sedp_checkpoint.pt"
        checkpoint = sedp.torch.load(source, map_location="cpu", weights_only=False)
        selector = checkpoint["candidate_selector"]
        settings = {
            name: param.default for name, param in parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        settings.update({key: value for key, value in checkpoint["training_config"].items() if key in parameters})
        settings.update({key: value for key, value in checkpoint["model_kwargs"].items() if key in parameters})
        settings.update({
            f"selector_candidate_first_{key}": value
            for key, value in checkpoint["inference"]["selector_adoption_decision"]["thresholds"].items()
        })
        policy = checkpoint["inference"]["candidate_policy_grid"]
        for arg, key in [("needs_edit_threshold_grid", "needs_edit_threshold"), ("edit_threshold_grid", "edit_threshold"), ("max_edit_weight_grid", "max_predicted_edit_weight")]:
            settings[arg] = list(dict.fromkeys(item[key] for item in policy))
        settings.update(
            selector_model=selector.get("selector_model", "scalar"),
            selector_patch_head=selector["selector_kwargs"]["patch_feature_dim"] > 0,
            selector_patch_hidden_dim=selector["selector_kwargs"]["patch_hidden_dim"],
            train_families=TRAIN_FAMILIES, eval_families=TRAIN_FAMILIES,
            max_shots=1024, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        )
        settings.pop("seed")
        assert set(settings) == set(parameters) - {"manifest_path", "out_dir", "seed"}
        manifest = ROOT / f"artifacts/datasets/predecoder_targets_d{distance}_2k_router1k/manifest.json"
        _, families = sedp._resolve_manifest_family_entries(manifest, TRAIN_FAMILIES)
        source_hashes = {}
        for _, target in families:
            target_metadata = target / "metadata.json"
            for path in [target_metadata, target / "edit_targets.npz", *map(Path, load(target_metadata)["source_artifacts"].values())]:
                source_hashes[path.resolve().relative_to(ROOT).as_posix()] = digest(path)
        protocol["distances"][str(distance)] = {
            "recipe_checkpoint": source.relative_to(ROOT).as_posix(),
            "recipe_sha256": digest(source),
            "settings": settings,
            "training_manifest": manifest.relative_to(ROOT).as_posix(),
            "training_manifest_sha256": digest(manifest),
            "training_files_sha256": source_hashes,
            "model_seeds": list(range(8)),
            "evaluation_maps": [
                {"local_seed": 780000 + distance * 100 + i, "sampler_seed": 1780000 + distance * 100 + i}
                for i in range(8)
            ],
            "evaluation_shots_per_map": 4096 if distance == 7 else 6144,
        }
    out.mkdir(parents=True, exist_ok=False)
    save(out / "protocol.json", protocol)
    print(f"Frozen protocol {out / 'protocol.json'}", flush=True)


def train(out: Path, distance: int, extra_seeds: list[int] | None) -> None:
    protocol = load(out / "protocol.json")
    spec = protocol["distances"][str(distance)]
    training_sources = {path: expected for path, expected in protocol["source_sha256"].items() if path.startswith("decoders/")}
    for path, expected in {**training_sources, **spec["training_files_sha256"], spec["training_manifest"]: spec["training_manifest_sha256"]}.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    root = out / f"d{distance}" / "train"
    root.mkdir(parents=True, exist_ok=True)
    source = ROOT / spec["training_manifest"]
    manifest = load(source)
    manifest["family_dirs"] = {name: (source.parent / manifest["family_dirs"][name]).as_posix() for name in TRAIN_FAMILIES}
    filtered = root / "development_manifest.json"
    if filtered.exists():
        assert load(filtered) == manifest
    else:
        save(filtered, manifest)
    for seed in extra_seeds or spec["model_seeds"]:
        run = root / f"seed{seed}"
        if (run / "complete.json").exists():
            assert load(run / "complete.json")["checkpoint_sha256"] == digest(run / "sedp_checkpoint.pt")
            continue
        if run.exists():
            raise FileExistsError(f"Incomplete run exists: {run}")
        result = sedp.run_manifest_experiment(manifest_path=filtered, out_dir=run, seed=seed, **spec["settings"])
        checkpoint = run / "sedp_checkpoint.pt"
        saved = sedp.torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["bundle_info"]["label_context_policy"] == protocol["input_policy"]
        assert set(saved["split_indices_by_family"]) == set(TRAIN_FAMILIES)
        assert set(result["eval_families"]) == set(TRAIN_FAMILIES)
        save(run / "complete.json", {"checkpoint_sha256": digest(checkpoint), "training_mode": saved["inference"]["selected_inference_mode"]})
        print(f"d{distance} seed{seed} {saved['inference']['selected_inference_mode']}", flush=True)


def generate_case(path: Path, *, distance: int, local_seed: int, sampler_seed: int, shots: int, zz_factor: float) -> Path:
    marker = path / "case.json"
    spec = dict(distance=distance, local_seed=local_seed, sampler_seed=sampler_seed, shots=shots, zz_factor=zz_factor)
    if marker.exists():
        record = load(marker)
        assert record["settings"] == spec
        for name, expected in record["sha256"].items():
            assert digest(path / name) == expected
        return path / record["target"]
    path.mkdir(parents=True, exist_ok=False)
    raw = save_family_dataset(
        out_root=path, family="stage_c_corr", distance=distance, rounds=distance,
        basis="z", variant="stim_rotated", target_mode="bell_pair_z_readout",
        physical_error_rate=0.0015, shots=shots, seed=sampler_seed,
        overwrite=False, local_seed=local_seed, zz_factor=zz_factor,
    )
    target_root = path / "eval_targets"
    # Only the raw baseline is needed for evaluation. Skip the label-guided edit search.
    manifest = targets.build_from_family_dir(
        raw, out_root=target_root, search_config=targets.SearchConfig(0, 0, 0, 1),
        max_shots=None, allow_circuit_fallback=False,
    )
    target = target_root / manifest["family_dirs"]["stage_c_corr"]
    save(marker, {
        "settings": spec, "target": target.relative_to(path).as_posix(),
        "dataset_builder_sha256": digest(ROOT / "sample_dataset.py"),
        "sha256": {
            file.relative_to(path).as_posix(): digest(file)
            for file in [raw / "samples.npz", raw / "metadata.json", raw / "detector_error_model.dem",
                         raw / "circuit.stim", raw / "config.json", target / "metadata.json", target / "edit_targets.npz"]
        },
    })
    return target


def load_selector(checkpoint: dict) -> tuple:
    payload = checkpoint["candidate_selector"]
    summary = payload["training_summary"]
    assert payload.get("transition_prior") is None
    assert payload.get("candidate_compatibility") is None
    assert payload.get("candidate_motif_vocabulary") is None
    assert checkpoint.get("hard_shot_router") is None
    assert not summary.get("selected_selector_nonzero_bias")
    model = sedp.SyndromeEditPreDecoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    selector = sedp._make_candidate_selector_module(
        selector_model=payload.get("selector_model", sedp.SELECTOR_MODEL_SCALAR),
        selector_kwargs=payload["selector_kwargs"],
    )
    selector.load_state_dict(payload["selector_state"])
    vocab = payload.get("candidate_local_motif_vocabulary")
    vocabulary = None if vocab is None else sedp.LocalMotifVocabulary(
        offset_patterns=tuple(tuple(tuple(offset) for offset in motif) for motif in vocab["offset_patterns"]),
        counts=tuple(vocab["counts"]), detector_count=vocab["detector_count"],
    )
    kwargs = {arg: summary[arg] for arg in [
        "selector_score_edit_penalty", "selector_target_mode", "selector_harm_weight", "selector_miss_weight",
        "selector_policy_candidate_mode", "selector_candidate_geometry_features", "selector_candidate_pattern_features",
        "selector_candidate_local_evidence_features", "selector_candidate_local_patch_features",
    ]}
    kwargs.update(
        local_motif_vocabulary=vocabulary,
        local_motif_top_k=summary["selector_local_motif_top_k"],
        policy_specs=[sedp.CandidatePolicySpec(**row) for row in checkpoint["inference"]["candidate_policy_grid"]],
    )
    return model, selector, kwargs, float(summary["selected_selector_emit_margin"] or 0)


def score_bundle(entry: sedp.PreparedEditFamily, start: int, stop: int, predictor: tuple) -> tuple:
    model, selector, kwargs, margin = predictor
    # The shared training bundle builder also computes target scores. Supply dummy
    # labels here so this inference path cannot select actions from evaluation labels.
    subset = {
        "x": entry.x[start:stop],
        "detector_events": entry.detector_events[start:stop],
        "baseline_predicted_observables": entry.baseline_predicted_observables[start:stop],
        "logical_class4": np.zeros(stop - start, dtype=np.uint8),
    }
    device = sedp.torch.device("cpu")
    bundle = sedp._build_selector_candidate_bundle(
        entry=entry, subset=subset, model=model, batch_size=64, device=device, **kwargs,
    )
    logits = sedp._selector_logits_for_bundle(selector=selector, bundle=bundle, batch_size=64, device=device)
    return bundle, logits, margin


def selected_prediction(entry: sedp.PreparedEditFamily, checkpoint: dict) -> tuple[np.ndarray, int]:
    mode = checkpoint["inference"]["selected_inference_mode"]
    if mode == sedp.SELECTION_MODE_RAW_NO_EDIT:
        return entry.baseline_predicted_observables, 0
    if mode != sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR:
        raise ValueError(f"Unexpected mode in frozen protocol: {mode}")
    predictor = load_selector(checkpoint)
    chunks = []
    fallbacks = 0
    for start in range(0, len(entry.x), 256):
        stop = min(start + 256, len(entry.x))
        bundle, logits, margin = score_bundle(entry, start, stop, predictor)
        mask = sedp._select_candidate_masks_from_logits(
            bundle=bundle, selector_logits=logits, selector_emit_margin=margin,
        )["selected_edit_mask"]
        prediction, fallback = sedp._safe_decode_edited_observables(
            matching=entry.matching, edited_detector_events=entry.detector_events[start:stop] ^ mask,
            baseline_predicted_observables=entry.baseline_predicted_observables[start:stop],
        )
        chunks.append(prediction)
        fallbacks += fallback
    return np.concatenate(chunks), fallbacks


def paired_counts(raw: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> dict:
    raw_correct = np.all(raw == truth, axis=1)
    selected_correct = np.all(predicted == truth, axis=1)
    improved = int(np.count_nonzero(~raw_correct & selected_correct))
    harmed = int(np.count_nonzero(raw_correct & ~selected_correct))
    assert int(selected_correct.sum()) - int(raw_correct.sum()) == improved - harmed
    return {
        "shots": len(raw), "raw_correct": int(raw_correct.sum()),
        "selected_correct": int(selected_correct.sum()), "improved": improved, "harmed": harmed,
        "gain_pp": 100 * (improved - harmed) / len(raw),
    }


def checkpoints(out: Path, distance: int, seeds: list[int]) -> dict[int, dict]:
    results = {}
    for seed in seeds:
        path = out / f"d{distance}" / "train" / f"seed{seed}"
        checkpoint = path / "sedp_checkpoint.pt"
        assert digest(checkpoint) == load(path / "complete.json")["checkpoint_sha256"]
        results[seed] = sedp.torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert results[seed]["bundle_info"]["label_context_policy"] == "zero_reserved"
    return results


def freeze_evaluation_panel(out: Path, protocol: dict) -> dict:
    for name, expected in protocol["source_sha256"].items():
        if name != "sample_dataset.py" and digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    panel = {
        "protocol_sha256": digest(out / "protocol.json"),
        "evaluator_sha256": digest(Path(__file__)),
        "dataset_builder_sha256": digest(ROOT / "sample_dataset.py"),
        "checkpoints": {},
    }
    for distance, spec in protocol["distances"].items():
        checkpoints(out, int(distance), spec["model_seeds"])
        panel["checkpoints"][distance] = {
            str(seed): load(out / f"d{distance}/train/seed{seed}/complete.json")["checkpoint_sha256"]
            for seed in spec["model_seeds"]
        }
    path = out / "evaluation_panel.json"
    if path.exists():
        assert load(path) == panel
    else:
        save(path, panel)
    return panel


def evaluate(out: Path, distance: int) -> None:
    protocol = load(out / "protocol.json")
    panel = freeze_evaluation_panel(out, protocol)
    spec = protocol["distances"][str(distance)]
    ensemble = checkpoints(out, distance, spec["model_seeds"])
    root = out / f"d{distance}" / "independent"
    root.mkdir(parents=True, exist_ok=True)
    map_rows = []
    for seeds in spec["evaluation_maps"]:
        path = root / str(seeds["local_seed"])
        target = generate_case(path, distance=distance, shots=spec["evaluation_shots_per_map"], zz_factor=0.05, **seeds)
        result_path = path / "paired_results.json"
        predictions_path = path / "predictions.npz"
        if result_path.exists():
            row = load(result_path)
            assert row["predictions_sha256"] == digest(predictions_path)
            map_rows.append(row)
            continue
        entry = sedp._prepare_edit_family(target, fill_value=spec["settings"]["fill_value"], max_shots=None)
        raw = sedp.pym_common._decode_batch(entry.matching, entry.detector_events)
        np.testing.assert_array_equal(raw, entry.baseline_predicted_observables)
        np.testing.assert_array_equal(sedp._logical_class4_from_observable_flips(entry.observable_flips), entry.logical_class4)
        rows = []
        predictions = {"raw": raw}
        for seed, checkpoint in ensemble.items():
            predicted, fallback = selected_prediction(entry, checkpoint)
            predictions[f"seed{seed}"] = predicted
            rows.append({"seed": seed, "fallbacks": fallback, **paired_counts(raw, predicted, entry.observable_flips)})
        np.savez_compressed(predictions_path, **predictions)
        result = {**seeds, "seeds": rows, "mean_gain_pp": float(np.mean([row["gain_pp"] for row in rows])),
                  "predictions_sha256": digest(predictions_path)}
        save(result_path, result)
        map_rows.append(result)
        print(f"d{distance} completed map {seeds['local_seed']}", flush=True)
    gains = np.array([row["mean_gain_pp"] for row in map_rows])
    level = 0.95 if distance == 7 else 0.975
    half = t.ppf((1 + level) / 2, len(gains) - 1) * gains.std(ddof=1) / len(gains) ** 0.5
    output = {"maps": map_rows, "checkpoint_sha256": panel["checkpoints"][str(distance)],
              "mean_gain_pp": float(gains.mean()), "ci_level": level,
              "ci_pp": [float(gains.mean() - half), float(gains.mean() + half)]}
    if not (root / "summary.json").exists():
        save(root / "summary.json", output)
    print(f"d{distance} mean {gains.mean():+.4f} pp, CI {output['ci_pp']}", flush=True)


def wait_for_training(out: Path, hours: float) -> None:
    protocol = load(out / "protocol.json")
    required = [
        out / f"d{distance}/train/seed{seed}/complete.json"
        for distance, spec in protocol["distances"].items() for seed in spec["model_seeds"]
    ]
    deadline = time.monotonic() + hours * 3600
    remaining = len(required) + 1
    while time.monotonic() < deadline:
        pending = sum(not marker.exists() for marker in required)
        if pending != remaining:
            remaining = pending
            print(f"Primary training complete: {len(required) - remaining}/{len(required)}", flush=True)
        if not remaining:
            return
        time.sleep(15)
    raise TimeoutError(f"Training completion not observed: {remaining} run(s) lack completion markers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "train", "evaluate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distance", type=int, choices=DISTANCES)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--wait-hours", type=float, default=0, help="Wait for primary checkpoints before evaluation")
    args = parser.parse_args()
    if args.wait_hours < 0 or (args.wait_hours and args.phase != "evaluate"):
        parser.error("--wait-hours requires evaluate and a positive value")
    sedp.torch.set_num_threads(2)
    if args.phase == "freeze":
        freeze(args.out.resolve())
    elif args.phase == "train":
        if args.distance is None:
            parser.error("train requires --distance")
        train(args.out.resolve(), args.distance, args.seeds)
    else:
        if args.wait_hours:
            wait_for_training(args.out.resolve(), args.wait_hours)
        for distance in DISTANCES if args.distance is None else (args.distance,):
            evaluate(args.out.resolve(), distance)


if __name__ == "__main__":
    main()
