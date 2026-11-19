"""Train a bias-specific logical-class adoption head with a frozen candidate CNN."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pymatching
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import evaluate_d7_adoption as adoption
from tools import evaluate_d7_pauli_bias as previous

pilot, rv = previous.pilot, previous.rv
gen, sedp = adoption.gen, adoption.sedp
OUT = ROOT / "artifacts/eval/d7_pauli_bias_adoption"
SHOTS = 2048
JOBS = 2
THRESHOLDS = (*adoption.THRESHOLDS, 1.0)
ROLES = {
    "training": (788700, 32, 1887000),
    "development": (788740, 8, 1887400),
    "evaluation": (788800, 24, 1888000),
}


def specs(role):
    local, count, sampler = ROLES[role]
    return [dict(role=role, local_seed=local + i, sampler_seed=sampler + i) for i in range(count)]


def freeze_registry():
    baseline = rv.load(previous.PILOT / "baseline_summary.json")
    reference = previous.PILOT / baseline["maps"]["788500"]["0.5"]["dataset"]["path"] / "metadata.json"
    old_protocol = previous.check_protocol()
    sources = dict(old_protocol["source_sha256"])
    for name in ("tools/train_d7_pauli_bias_adoption.py", "tools/evaluate_d7_adoption.py",
                 "tools/evaluate_d7_candidate_generators.py", "tools/benchmark_d7_final.py",
                 "tools/d7_boundary_matching_costs.py"):
        sources[name] = rv.digest(ROOT / name)
    registry = dict(
        eta=20.0, mode="ordinary MWPM with each map's bias-matched DEM", dataset=pilot.DATASET,
        shots_per_map=SHOTS, roles={role: specs(role) for role in ROLES},
        candidate_checkpoint=old_protocol["checkpoint"],
        candidate_checkpoint_sha256=old_protocol["checkpoint_sha256"],
        context_reference=reference.relative_to(ROOT).as_posix(), context_sha256=rv.digest(reference),
        context_detector_fraction=pilot.CONTEXT, l2=adoption.L2, threshold_grid=list(THRESHOLDS),
        target="raw logical class XOR true logical class",
        features="matching weight, raw class, round event counts, class costs and reachability, candidate class summaries",
        selection="maximize mean development-map paired gain, breaking ties toward the higher threshold",
        refit="fit on training maps only; retain the fitted head after threshold selection",
        primary="retrained adoption versus bias-matched ordinary MWPM; 95% map t interval",
        success="primary interval lower bound strictly greater than zero",
        secondary="retrained adoption versus the frozen seed-2 margin route; descriptive 95% map interval",
        model_scope="candidate CNN and motifs frozen; four-class logistic head retrained",
        jobs=JOBS, torch_threads=2, versions=old_protocol["versions"], source_sha256=sources,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "registry.json"
    if path.exists():
        assert rv.load(path) == registry, "Frozen registry changed"
    else:
        rv.save(path, registry)
    archive = OUT / "source_snapshot.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name in sources:
                zipped.write(ROOT / name, arcname=name)
    return check_registry()


def check_registry():
    registry = rv.load(OUT / "registry.json")
    for name, expected in registry["source_sha256"].items():
        assert rv.digest(ROOT / name) == expected, name
    assert rv.digest(ROOT / registry["candidate_checkpoint"]) == registry["candidate_checkpoint_sha256"]
    assert rv.digest(ROOT / registry["context_reference"]) == registry["context_sha256"]
    assert registry["roles"] == {role: specs(role) for role in ROLES}
    assert registry["threshold_grid"] == list(THRESHOLDS)
    for name, expected in registry["versions"].items():
        assert version(name) == expected, name
    return registry


def class4(observables):
    return (2 * observables[:, 0] + observables[:, 1]).astype(np.uint8)


def observables(classes):
    return np.column_stack((classes >> 1, classes & 1)).astype(np.uint8)


def extract_features(events, layout, matching, predictor, metadata, include_margin=False):
    raw, costs = adoption.checked_costs(matching, adoption.BoundaryCosts(matching), events)
    weights = costs[np.arange(len(raw)), raw]
    relative = costs[np.arange(len(raw))[:, None], raw[:, None] ^ adoption.CLASSES] - weights[:, None]
    reachable = np.isfinite(relative)
    by_round = [events[:, layout.detector_time_index == step].sum(axis=1) for step in range(layout.time_steps)]
    raw_features = np.column_stack((weights, np.eye(4)[raw][:, 1:], *by_round,
                                   np.where(reachable[:, 1:], relative[:, 1:], 0), reachable[:, 1:]))
    static, _ = sedp.volume_common._build_static_planes(layout)
    context, names = sedp.volume_common._build_context_planes(
        {**metadata, "qc_stats": {"detector_event_fraction": pilot.CONTEXT}}, layout,
    )
    assert not context[names.index("reserved_label_context")].any()
    patterns = gen.learned_patterns(layout, events, predictor[2])
    masks, margin_predictions = [], []
    if include_margin:
        entry = SimpleNamespace(
            x=gen.bench.tensor(events, layout, static, context), detector_events=events,
            baseline_predicted_observables=observables(raw), matching=matching,
            detector_time_index=layout.detector_time_index,
            row_index_by_detector=layout.row_index_by_detector,
            col_index_by_detector=layout.col_index_by_detector,
        )
    for lo in range(0, len(events), pilot.shift.BATCH):
        hi = min(lo + pilot.shift.BATCH, len(events))
        if include_margin:
            bundle, utility, margin = rv.score_bundle(entry, lo, hi, predictor)
            assert bundle.decode_fallback_count == 0
            np.testing.assert_array_equal(bundle.shot_indices, np.repeat(np.arange(hi - lo), gen.CAP + 1))
            batch_masks = bundle.candidate_edit_mask.reshape(hi - lo, gen.CAP + 1, events.shape[1])
            selected = pilot.shift.selected_rows(bundle, utility, margin)
            offset = sedp._transition_feature_offsets(candidate_feature_dim=bundle.candidate_features.shape[1])
            delta = bundle.candidate_features[selected, offset["obs_delta"]].astype(np.uint8)
            margin_predictions.append(class4(observables(raw[lo:hi]) ^ delta))
        else:
            batch_masks = gen.learned_masks(events[lo:hi], layout, static, context, predictor[0], patterns)
        masks.append(batch_masks)
    masks = np.concatenate(masks)
    candidate_features, rows, classes = adoption.candidate_features(events, masks, matching, raw, weights)
    result = dict(features=np.column_stack((raw_features, candidate_features)), raw=raw,
                  class_rows=rows, classes=classes, packed_masks=np.packbits(masks, axis=2))
    if include_margin:
        result["margin"] = np.concatenate(margin_predictions)
    return result


def collect_case(spec, registry):
    root = OUT / spec["role"] / str(spec["local_seed"])
    marker = root / "case.json"
    if marker.exists():
        record = rv.load(marker)
        assert record["spec"] == spec and record["registry_sha256"] == rv.digest(OUT / "registry.json")
        for name, expected in record["sha256"].items():
            assert rv.digest(root / name) == expected, name
        return record
    sedp.torch.set_num_threads(registry["torch_threads"])
    dataset_marker = root / "dataset.json"
    if dataset_marker.exists():
        saved = rv.load(dataset_marker)
        assert saved["spec"] == spec
        raw_dir = root / saved["path"]
        for name, expected in saved["sha256"].items():
            assert rv.digest(raw_dir / name) == expected, name
    else:
        raw_dir = pilot.save_family_dataset(
            out_root=root, shots=SHOTS, seed=spec["sampler_seed"], overwrite=False,
            local_seed=spec["local_seed"], data_pauli_z_bias=20.0, **pilot.DATASET,
        )
        saved = dict(spec=spec, path=raw_dir.relative_to(root).as_posix(),
                     sha256={name: rv.digest(raw_dir / name) for name in
                             ("samples.npz", "metadata.json", "config.json", "circuit.stim", "detector_error_model.dem")})
        rv.save(dataset_marker, saved)
    matching = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel.from_file(str(raw_dir / "detector_error_model.dem")),
    )
    predictor = rv.load_selector(rv.checkpoints(pilot.shift.REVALIDATION, 7, [pilot.MODEL_SEED])[pilot.MODEL_SEED])
    assert predictor[3] == 1.25
    with np.load(raw_dir / "samples.npz") as samples:
        events = samples["detector_events"]
        layout = gen.build_rectangular_syndrome_layout(samples)
        data = extract_features(events, layout, matching, predictor, rv.load(ROOT / registry["context_reference"]),
                                include_margin=spec["role"] == "evaluation")
        data["truth"] = class4(samples["observable_flips"])
    file = root / "features.npz"
    np.savez_compressed(file, **data)
    record = dict(spec=spec, registry_sha256=rv.digest(OUT / "registry.json"),
                  sha256={"features.npz": rv.digest(file), "dataset.json": rv.digest(dataset_marker)})
    rv.save(marker, record)
    return record


def collect(role, registry):
    records = {}
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        futures = [pool.submit(collect_case, spec, registry) for spec in specs(role)]
        for future in as_completed(futures):
            record = future.result()
            records[record["spec"]["local_seed"]] = record
            print(f"{role}: {len(records)}/{len(specs(role))} maps verified", flush=True)
    return [records[spec["local_seed"]] for spec in specs(role)]


def load_case(role, seed):
    root = OUT / role / str(seed)
    marker = rv.load(root / "case.json")
    assert marker["registry_sha256"] == rv.digest(OUT / "registry.json")
    assert rv.digest(root / "features.npz") == marker["sha256"]["features.npz"]
    with np.load(root / "features.npz") as data:
        return dict(data)


def apply_policy(data, head, threshold):
    probabilities = adoption.probabilities(data["features"], head)
    transition = adoption.decide(probabilities, data["class_rows"] >= 0, threshold)
    rows = np.take_along_axis(data["class_rows"], transition[:, None], axis=1)[:, 0]
    assert np.all(rows >= 0)
    selected = data["raw"] ^ transition
    np.testing.assert_array_equal(selected, data["classes"][np.arange(len(rows)), rows])
    return selected, rows, probabilities


def paired(raw, predicted, truth):
    return rv.paired_counts(observables(raw), observables(predicted), observables(truth))


def select_threshold(maps, head):
    grid = []
    for threshold in THRESHOLDS:
        counts = [paired(data["raw"], apply_policy(data, head, threshold)[0], data["truth"]) for data in maps]
        grid.append(dict(threshold=threshold, mean_gain_pp=float(np.mean([r["gain_pp"] for r in counts])),
                         improved=sum(r["improved"] for r in counts), harmed=sum(r["harmed"] for r in counts)))
    selected = max(grid, key=lambda row: (row["mean_gain_pp"], row["threshold"]))
    return selected["threshold"], grid


def develop():
    registry = freeze_registry()
    collect("training", registry)
    collect("development", registry)
    path = OUT / "policy.json"
    if path.exists():
        policy = rv.load(path)
        assert policy["registry_sha256"] == rv.digest(OUT / "registry.json")
        return
    training = [load_case("training", spec["local_seed"]) for spec in specs("training")]
    development = [load_case("development", spec["local_seed"]) for spec in specs("development")]
    head = adoption.fit_matrix(np.concatenate([data["features"] for data in training]),
                               np.concatenate([data["raw"] ^ data["truth"] for data in training]))
    threshold, grid = select_threshold(development, head)
    inputs = {f"{role}/{spec['local_seed']}/features.npz": rv.digest(OUT / role / str(spec["local_seed"]) / "features.npz")
              for role in ("training", "development") for spec in specs(role)}
    policy = dict(registry_sha256=rv.digest(OUT / "registry.json"), head=head, threshold=threshold,
                  development_grid=grid, inputs_sha256=inputs, training_shots=len(training)*SHOTS,
                  development_shots=len(development)*SHOTS)
    rv.save(path, policy)
    print(f"Adoption head frozen at threshold {threshold}; development grid {grid}", flush=True)


def evaluate():
    registry = check_registry()
    policy = rv.load(OUT / "policy.json")
    assert policy["registry_sha256"] == rv.digest(OUT / "registry.json")
    for name, expected in policy["inputs_sha256"].items():
        assert rv.digest(OUT / name) == expected, name
    collect("evaluation", registry)
    records = []
    for spec in specs("evaluation"):
        root = OUT / "evaluation" / str(spec["local_seed"])
        data = load_case("evaluation", spec["local_seed"])
        selected, rows, probabilities = apply_policy(data, policy["head"], policy["threshold"])
        file = root / "predictions.npz"
        arrays = dict(raw=data["raw"], truth=data["truth"], margin=data["margin"], selected=selected,
                      selected_rows=rows, probabilities=probabilities)
        if file.exists():
            with np.load(file) as saved:
                for name, array in arrays.items():
                    np.testing.assert_array_equal(array, saved[name])
        else:
            np.savez_compressed(file, **arrays)
        records.append(dict(
            local_seed=spec["local_seed"], primary=paired(data["raw"], selected, data["truth"]),
            margin_vs_raw=paired(data["raw"], data["margin"], data["truth"]),
            new_vs_margin=paired(data["margin"], selected, data["truth"]),
            logical_changes=int(np.sum(selected != data["raw"])), predictions_sha256=rv.digest(file),
        ))
    contrasts = {}
    for name in ("primary", "margin_vs_raw", "new_vs_margin"):
        contrasts[name] = pilot.summarize_contrast([row[name] for row in records])
    contrasts["primary"]["supports_improvement"] = contrasts["primary"]["descriptive_95_ci_pp"][0] > 0
    result = dict(registry_sha256=rv.digest(OUT / "registry.json"), policy_sha256=rv.digest(OUT / "policy.json"),
                  contrasts=contrasts, maps=records)
    path = OUT / "summary.json"
    if path.exists():
        assert rv.load(path) == result
    else:
        rv.save(path, result)
    print(contrasts, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "evaluate"))
    args = parser.parse_args()
    if args.phase == "develop":
        develop()
    else:
        evaluate()


if __name__ == "__main__":
    main()
