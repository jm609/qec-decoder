"""Measure legacy label-context sensitivity without retraining or overwriting evidence.

Run from the repository root. The legacy input is restored only inside this
diagnostic; production feature builders always zero the reserved label slot.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders import syndrome_edit_predecoder as sedp


def audit_run(row: dict, entry: sedp.PreparedEditFamily, out_dir: Path) -> dict:
    checkpoint = Path(row["run_dir"]) / "sedp_checkpoint.pt"
    reference = json.loads((checkpoint.parent / "experiment_summary.json").read_text(encoding="utf-8"))
    channel = entry.bundle_info["channel_names"].index("reserved_label_context")
    legacy_value = float(entry.metadata["qc_stats"]["logical_flip_fraction"])
    results = {}
    masks = {}
    original_metrics = sedp._system_metrics_from_chosen_edit_mask

    for variant, value in [("legacy", legacy_value), ("zero", 0.0)]:
        x = entry.x.copy()
        x[:, channel] = value
        altered = replace(entry, x=x)
        captured = {}

        def capture_metrics(**kwargs):
            mode = kwargs["decision_payload"]["selection_mode"]
            captured[mode] = np.asarray(kwargs["chosen_edit_mask"]).copy()
            return original_metrics(**kwargs)

        output = out_dir / f"seed{row['seed']}_{variant}.json"
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        with patch.object(sedp, "_prepare_edit_family", return_value=altered), patch.object(
            sedp, "_system_metrics_from_chosen_edit_mask", side_effect=capture_metrics,
        ):
            result = sedp.evaluate_checkpoint_on_family(
                checkpoint_path=checkpoint, family_dir=entry.family_dir,
                out_json=output, max_shots=entry.x.shape[0], batch_size=64,
            )
        # The diagnostic override must not be mistaken for production preprocessing.
        result["audit_context_override"] = {"variant": variant, "value": value}
        result["dataset"]["label_context_policy"] = (
            "legacy_label_aggregate_diagnostic" if variant == "legacy" else "zero_reserved"
        )
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        candidate = result["metrics"]["candidate_selector"]
        assert candidate is not None, "The frozen panel requires a candidate selector"
        raw = result["metrics"]["no_edit"]["edited_pymatching"]["accuracy"]
        candidate_accuracy = candidate["edited_pymatching"]["accuracy"]
        selected_mode = row.get("selected_mode", row.get("mode"))
        assert selected_mode in {"local_motif_selector", "raw_no_edit"}, selected_mode
        selected_accuracy = candidate_accuracy if selected_mode == "local_motif_selector" else raw
        results[variant] = {
            "context_value": value,
            "raw_accuracy": raw,
            "candidate_accuracy": candidate_accuracy,
            "selected_accuracy": selected_accuracy,
            "candidate_gain_pp": 100 * (candidate_accuracy - raw),
            "selected_gain_pp": 100 * (selected_accuracy - raw),
            "candidate_changes": candidate["change_summary"],
        }
        masks[variant] = captured["local_motif_selector"]
        if variant == "legacy":
            expected = reference["eval_families_candidate_selector"][entry.family]
            assert candidate_accuracy == expected["edited_pymatching"]["accuracy"], (
                "Legacy candidate reproduction failed", checkpoint, candidate_accuracy,
            )
            assert abs((selected_accuracy - raw) - row["eval_selected_delta_over_no_edit"]) < 1e-12

    assert results["legacy"]["raw_accuracy"] == results["zero"]["raw_accuracy"]
    return {
        "seed": row["seed"], "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "shots": int(entry.x.shape[0]), "selected_mode_frozen": selected_mode,
        "legacy_reproduced": True,
        "changed_candidate_masks": int(np.count_nonzero(np.any(masks["legacy"] != masks["zero"], axis=1))),
        **results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--distances", nargs="+", choices=["d3", "d5", "d7"], default=["d3", "d5", "d7"])
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    sedp.torch.set_num_threads(4)
    evidence_path = ROOT / "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = {
        "scope": "Frozen-checkpoint sensitivity, not a corrected training experiment",
        "calibration": "Existing model, margins and final adoption modes remain fixed",
        "d7_panel": [2, 11, 54, 57],
        "rows": [],
    }
    started = time.monotonic()
    for distance in evidence["distance_results"]:
        name = distance["distance"]
        if name not in args.distances:
            continue
        compare = json.loads(Path(distance["artifact"]).read_text(encoding="utf-8"))
        manifest = ROOT / f"artifacts/datasets/predecoder_targets_{name}_2k_router1k/manifest.json"
        _, families = sedp._resolve_manifest_family_entries(manifest, ["stage_c_corr"])
        entry = sedp._prepare_edit_family(families[0][1], fill_value=-0.5, max_shots=1024)
        actual = sedp.pym_common._decode_batch(entry.matching, entry.detector_events)
        np.testing.assert_array_equal(actual, entry.baseline_predicted_observables)
        np.testing.assert_array_equal(sedp._logical_class4_from_observable_flips(entry.observable_flips), entry.logical_class4)
        for row in compare["rows"]:
            if name == "d7" and row["seed"] not in report["d7_panel"]:
                continue
            result = audit_run(row, entry, args.out_dir / name)
            result["distance"] = name
            report["rows"].append(result)
            report["elapsed_seconds"] = time.monotonic() - started
            (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{name} seed{row['seed']}: selected pp {result['legacy']['selected_gain_pp']:.6f} -> "
                  f"{result['zero']['selected_gain_pp']:.6f}; changed masks {result['changed_candidate_masks']}", flush=True)


if __name__ == "__main__":
    main()
