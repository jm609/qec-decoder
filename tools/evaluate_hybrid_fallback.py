from __future__ import annotations

"""
evaluate_hybrid_fallback.py

Confidence-aware hybrid evaluation:
- primary decoder: factorized_logical_frame_decoder
- fallback decoder: PyMatching
"""

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECODER_DIR = ROOT / "decoders"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_DIR))

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None

import baseline_rectcnn as common
import baseline_pymatching as pymatching_baseline
import factorized_logical_frame_decoder as flfd


SCHEMA_VERSION = "hybrid_fallback.eval.v1"
CLASS_LABELS = ["I", "X", "Z", "Y"]


def _class4_to_bits(class4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(class4, dtype=np.uint8).reshape(-1)
    logical_x = (labels & 1).astype(np.uint8, copy=False)
    logical_z = ((labels >> 1) & 1).astype(np.uint8, copy=False)
    return logical_x, logical_z


def _class4_to_one_hot_probs(class4: np.ndarray) -> np.ndarray:
    labels = np.asarray(class4, dtype=np.int64).reshape(-1)
    one_hot = np.zeros((labels.shape[0], 4), dtype=np.float32)
    if labels.size:
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def _confidence_score(
    probs: np.ndarray,
    *,
    mode: str,
    correctness_scores: np.ndarray | None = None,
) -> np.ndarray:
    probs_f = np.asarray(probs, dtype=np.float32)
    if mode == "max_prob":
        return np.max(probs_f, axis=1)
    if mode == "margin":
        sorted_probs = np.sort(probs_f, axis=1)
        return sorted_probs[:, -1] - sorted_probs[:, -2]
    if mode == "inv_entropy":
        entropy = -np.sum(probs_f * np.log(np.clip(probs_f, 1e-12, 1.0)), axis=1)
        max_entropy = float(np.log(probs_f.shape[1]))
        return 1.0 - (entropy / max_entropy if max_entropy > 0 else entropy)
    if mode == "correctness_head":
        if correctness_scores is None:
            raise ValueError("correctness_head score mode requires correctness_scores.")
        return np.asarray(correctness_scores, dtype=np.float32).reshape(-1)
    raise ValueError(f"Unsupported score mode: {mode!r}")


def _predict_flfd_family(
    *,
    checkpoint_path: Path,
    family_dir: Path,
    batch_size: int,
    max_shots: int | None,
    device_arg: str,
) -> dict[str, Any]:
    common._require_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for FLFD prediction.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    payload = common._load_family_payload(family_dir)
    prepared = flfd._prepare_class4_family(
        payload,
        fill_value=float(checkpoint.get("fill_value", -0.5)),
        max_shots=max_shots,
    )
    model_hparams = dict(checkpoint["model_hparams"])
    model = flfd.FactorizedLogicalFrameDecoder(
        signal_channels=int(model_hparams["signal_channels"]),
        context_channels=int(model_hparams["context_channels"]),
        hidden_channels=int(model_hparams["hidden_channels"]),
        num_blocks=int(model_hparams["num_blocks"]),
        dense_hidden_dim=int(model_hparams["dense_hidden_dim"]),
        context_hidden_dim=int(model_hparams["context_hidden_dim"]),
        dropout=float(model_hparams["dropout"]),
    )
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", ()))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", ()))
    device = common._pick_device(device_arg)
    model.to(device)
    model.eval()

    dataset = common.TensorDataset(torch.from_numpy(np.ascontiguousarray(prepared["x"])))
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    error_logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            outputs = model(xb)
            logits_chunks.append(outputs["class4_logits"].detach().cpu().numpy())
            error_logits_chunks.append(outputs["error_logits"].detach().cpu().numpy())

    logits = np.asarray(np.concatenate(logits_chunks, axis=0), dtype=np.float32)
    error_logits = np.asarray(np.concatenate(error_logits_chunks, axis=0), dtype=np.float32)
    checkpoint_calibration = checkpoint.get("calibration", {})
    temperature = float(checkpoint_calibration.get("temperature", 1.0))
    probs = common._softmax_np(flfd._apply_temperature_to_logits(logits, temperature))
    has_trained_error_head = "error_head.weight" not in missing_keys and "error_head.bias" not in missing_keys
    correctness_scores = (
        1.0 - common._sigmoid_np(error_logits)
        if has_trained_error_head
        else None
    )
    pred_class4 = np.argmax(probs, axis=1).astype(np.uint8, copy=False)
    return {
        "family_dir": family_dir.as_posix(),
        "metadata": payload.metadata,
        "prepared": prepared,
        "checkpoint": checkpoint,
        "checkpoint_compatibility": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "has_trained_error_head": bool(has_trained_error_head),
        },
        "temperature": temperature,
        "probs": probs,
        "correctness_scores": correctness_scores,
        "pred_class4": pred_class4,
    }


def _decode_pymatching_family(
    *,
    family_dir: Path,
    max_shots: int | None,
) -> dict[str, Any]:
    artifacts, metadata, arrays = pymatching_baseline._load_family_payload(family_dir)
    detector_events = pymatching_baseline._as_uint8_2d(arrays["detector_events"], name="detector_events")
    observable_flips = pymatching_baseline._as_uint8_2d(arrays["observable_flips"], name="observable_flips")
    logical_class4 = (
        pymatching_baseline._as_uint8_1d(arrays["logical_class4"], name="logical_class4")
        if "logical_class4" in arrays
        else pymatching_baseline._logical_class4_from_observable_flips(observable_flips)
    )
    if max_shots is not None:
        detector_events = detector_events[:max_shots]
        observable_flips = observable_flips[:max_shots]
        logical_class4 = logical_class4[:max_shots]

    matching, build_info = pymatching_baseline._build_matching(
        dem_path=artifacts.dem_path,
        circuit_path=artifacts.circuit_path,
        allow_circuit_fallback=False,
    )
    pred_observables = pymatching_baseline._decode_batch(matching, detector_events)
    if pred_observables.shape[1] != 2:
        raise ValueError("Hybrid fallback currently requires two-observable class4 datasets.")
    pred_class4 = pymatching_baseline._logical_class4_from_observable_flips(pred_observables)
    return {
        "metadata": metadata,
        "target_class4": logical_class4,
        "pred_class4": pred_class4,
        "pred_probs": _class4_to_one_hot_probs(pred_class4),
        "matching": build_info.to_dict(),
    }


def _threshold_grid(raw_thresholds: list[float] | None) -> list[float]:
    if raw_thresholds:
        return [float(value) for value in raw_thresholds]
    default = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.linspace(0.5, 0.95, 10, dtype=np.float64),
            np.array([0.975, 0.99, 0.995, 0.999, 1.0], dtype=np.float64),
        ]
    )
    return [float(value) for value in np.unique(default)]


def _hybrid_metrics_for_threshold(
    *,
    target_class4: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
    neural_probs: np.ndarray,
    neural_scores: np.ndarray,
    pymatching_probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    use_neural = np.asarray(neural_scores >= float(threshold), dtype=bool)
    hybrid_probs = np.where(use_neural[:, None], neural_probs, pymatching_probs)
    class4_metrics = common._multiclass_metrics_from_probs(
        hybrid_probs,
        target_class4,
        class_labels=list(CLASS_LABELS),
        loss=flfd._cross_entropy_from_probs(hybrid_probs, target_class4),
    )
    calibration = flfd._multiclass_calibration_metrics(hybrid_probs, target_class4)
    pred_class4 = np.argmax(hybrid_probs, axis=1).astype(np.uint8, copy=False)
    pred_x, pred_z = _class4_to_bits(pred_class4)
    bit_x_metrics = common._binary_metrics_from_probs(pred_x.astype(np.float32), target_x, threshold=0.5, bce_loss=None)
    bit_z_metrics = common._binary_metrics_from_probs(pred_z.astype(np.float32), target_z, threshold=0.5, bce_loss=None)
    return {
        "threshold": float(threshold),
        "neural_coverage": float(np.mean(use_neural)) if use_neural.size else None,
        "fallback_rate": float(1.0 - np.mean(use_neural)) if use_neural.size else None,
        "num_neural_shots": int(np.sum(use_neural)),
        "num_fallback_shots": int(np.sum(~use_neural)),
        "accuracy": class4_metrics["accuracy"],
        "label_error_rate": class4_metrics["label_error_rate"],
        "balanced_accuracy": class4_metrics["balanced_accuracy"],
        "macro_f1": class4_metrics["macro_f1"],
        "ece_10": calibration["ece"],
        "brier_score": calibration["brier_score"],
        "mean_predicted_confidence": class4_metrics["mean_predicted_confidence"],
        "bit_x_accuracy": bit_x_metrics["accuracy"],
        "bit_z_accuracy": bit_z_metrics["accuracy"],
        "confusion_matrix": class4_metrics["confusion_matrix"],
    }


def _select_threshold(
    *,
    sweep_by_family: dict[str, list[dict[str, Any]]],
    seen_families: list[str],
) -> dict[str, Any]:
    eligible_families = [family for family in seen_families if family in sweep_by_family]
    if not eligible_families:
        eligible_families = sorted(sweep_by_family)
    thresholds = [entry["threshold"] for entry in sweep_by_family[eligible_families[0]]]
    aggregated: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    for index, threshold in enumerate(thresholds):
        rows = [sweep_by_family[family][index] for family in eligible_families]
        mean_accuracy = float(np.mean([float(row["accuracy"]) for row in rows]))
        mean_error = float(np.mean([float(row["label_error_rate"]) for row in rows]))
        mean_fallback = float(np.mean([float(row["fallback_rate"]) for row in rows]))
        mean_ece = float(np.mean([float(row["ece_10"]) for row in rows]))
        aggregated_entry = {
            "threshold": float(threshold),
            "mean_accuracy": mean_accuracy,
            "mean_label_error_rate": mean_error,
            "mean_fallback_rate": mean_fallback,
            "mean_ece_10": mean_ece,
        }
        aggregated.append(aggregated_entry)
        if best_entry is None:
            best_entry = aggregated_entry
            continue
        best_key = (
            float(best_entry["mean_accuracy"]),
            -float(best_entry["mean_fallback_rate"]),
            -float(best_entry["threshold"]),
        )
        candidate_key = (
            mean_accuracy,
            -mean_fallback,
            -float(threshold),
        )
        if candidate_key > best_key:
            best_entry = aggregated_entry

    assert best_entry is not None
    return {
        "based_on_families": eligible_families,
        "metric": "mean_accuracy_then_neural_coverage",
        "threshold_sweep": aggregated,
        "selected_threshold": float(best_entry["threshold"]),
        "selected_summary": best_entry,
    }


def run_hybrid_fallback(
    *,
    manifest: Path,
    checkpoint: Path,
    out_json: Path,
    eval_families: list[str] | None,
    score_mode: str,
    thresholds: list[float] | None,
    max_shots: int | None,
    batch_size: int,
    device_arg: str,
) -> dict[str, Any]:
    manifest_data = common._read_json(manifest)
    family_dirs = manifest_data.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(f"Manifest does not contain family_dirs: {manifest}")
    resolved_eval_families = list(eval_families) if eval_families is not None else list(family_dirs.keys())
    threshold_grid = _threshold_grid(thresholds)

    checkpoint_data = torch.load(checkpoint, map_location="cpu") if torch is not None else {}
    seen_families = (
        list(checkpoint_data.get("train_families"))
        if checkpoint_data.get("train_families") is not None
        else [str(checkpoint_data.get("family"))]
    )
    seen_families = [family for family in seen_families if family and family != "multi_family_manifest"]

    sweep_by_family: dict[str, list[dict[str, Any]]] = {}
    baseline_by_family: dict[str, dict[str, Any]] = {}
    family_meta: dict[str, dict[str, Any]] = {}
    for family in resolved_eval_families:
        if family not in family_dirs:
            raise KeyError(f"Family {family!r} missing from manifest. Available: {sorted(family_dirs)}")
        family_dir = common._resolve_manifest_family_dir(manifest, family_dirs[family])

        flfd_result = _predict_flfd_family(
            checkpoint_path=checkpoint,
            family_dir=family_dir,
            batch_size=batch_size,
            max_shots=max_shots,
            device_arg=device_arg,
        )
        pym_result = _decode_pymatching_family(
            family_dir=family_dir,
            max_shots=max_shots,
        )
        target_class4 = np.asarray(flfd_result["prepared"]["y_class4"], dtype=np.uint8)
        target_x = np.asarray(flfd_result["prepared"]["y_x"], dtype=np.uint8)
        target_z = np.asarray(flfd_result["prepared"]["y_z"], dtype=np.uint8)
        if not np.array_equal(target_class4, np.asarray(pym_result["target_class4"], dtype=np.uint8)):
            raise ValueError(f"Target mismatch between FLFD and PyMatching paths for family={family!r}")

        neural_probs = np.asarray(flfd_result["probs"], dtype=np.float32)
        neural_scores = _confidence_score(
            neural_probs,
            mode=score_mode,
            correctness_scores=np.asarray(flfd_result.get("correctness_scores"), dtype=np.float32)
            if flfd_result.get("correctness_scores") is not None
            else None,
        )
        pym_probs = np.asarray(pym_result["pred_probs"], dtype=np.float32)

        family_sweep = [
            _hybrid_metrics_for_threshold(
                target_class4=target_class4,
                target_x=target_x,
                target_z=target_z,
                neural_probs=neural_probs,
                neural_scores=neural_scores,
                pymatching_probs=pym_probs,
                threshold=threshold,
            )
            for threshold in threshold_grid
        ]
        sweep_by_family[family] = family_sweep

        neural_metrics = common._multiclass_metrics_from_probs(
            neural_probs,
            target_class4,
            class_labels=list(CLASS_LABELS),
            loss=flfd._cross_entropy_from_probs(neural_probs, target_class4),
        )
        neural_calibration = flfd._multiclass_calibration_metrics(neural_probs, target_class4)
        pym_metrics = common._multiclass_metrics_from_probs(
            pym_probs,
            target_class4,
            class_labels=list(CLASS_LABELS),
            loss=flfd._cross_entropy_from_probs(pym_probs, target_class4),
        )
        pym_calibration = flfd._multiclass_calibration_metrics(pym_probs, target_class4)
        baseline_by_family[family] = {
            "neural": {
                "accuracy": neural_metrics["accuracy"],
                "label_error_rate": neural_metrics["label_error_rate"],
                "macro_f1": neural_metrics["macro_f1"],
                "ece_10": neural_calibration["ece"],
                "brier_score": neural_calibration["brier_score"],
                "temperature_applied": float(flfd_result["temperature"]),
            },
            "pymatching": {
                "accuracy": pym_metrics["accuracy"],
                "label_error_rate": pym_metrics["label_error_rate"],
                "macro_f1": pym_metrics["macro_f1"],
                "ece_10": pym_calibration["ece"],
                "brier_score": pym_calibration["brier_score"],
                "matching": pym_result["matching"],
            },
        }
        family_meta[family] = {
            "family_dir": family_dir.as_posix(),
            "stage": str(flfd_result["metadata"].get("stage")),
            "seen_in_training": family in seen_families,
        }

    threshold_selection = _select_threshold(
        sweep_by_family=sweep_by_family,
        seen_families=seen_families,
    )
    selected_threshold = float(threshold_selection["selected_threshold"])
    selected_per_family: list[dict[str, Any]] = []
    for family in resolved_eval_families:
        selected_row = next(
            row for row in sweep_by_family[family] if abs(float(row["threshold"]) - selected_threshold) < 1e-12
        )
        selected_per_family.append(
            {
                "family": family,
                "stage": family_meta[family]["stage"],
                "seen_in_training": family_meta[family]["seen_in_training"],
                "selected_threshold": selected_threshold,
                "hybrid_accuracy": selected_row["accuracy"],
                "hybrid_label_error_rate": selected_row["label_error_rate"],
                "hybrid_macro_f1": selected_row["macro_f1"],
                "hybrid_ece_10": selected_row["ece_10"],
                "hybrid_brier_score": selected_row["brier_score"],
                "fallback_rate": selected_row["fallback_rate"],
                "neural_coverage": selected_row["neural_coverage"],
                "pure_neural_accuracy": baseline_by_family[family]["neural"]["accuracy"],
                "pure_pymatching_accuracy": baseline_by_family[family]["pymatching"]["accuracy"],
                "delta_vs_neural_accuracy": float(selected_row["accuracy"] - baseline_by_family[family]["neural"]["accuracy"]),
                "delta_vs_pymatching_accuracy": float(selected_row["accuracy"] - baseline_by_family[family]["pymatching"]["accuracy"]),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": common._utc_now_iso(),
        "primary_decoder": "factorized_logical_frame_decoder",
        "fallback_decoder": "pymatching",
        "manifest": {
            "path": manifest.as_posix(),
            "schema_version": manifest_data.get("schema_version"),
            "families_available": list(family_dirs.keys()),
        },
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "train_families": seen_families,
            "family": checkpoint_data.get("family"),
            "stage": checkpoint_data.get("stage"),
            "calibration": checkpoint_data.get("calibration"),
        },
        "selection": {
            "score_mode": score_mode,
            "thresholds_considered": threshold_grid,
            **threshold_selection,
        },
        "per_family_baselines": baseline_by_family,
        "selected_per_family": selected_per_family,
        "threshold_sweep_by_family": sweep_by_family,
    }
    common._write_json(out_json, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate confidence-aware FLFD + PyMatching hybrid fallback.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--eval-families", nargs="+", default=None)
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["max_prob", "margin", "inv_entropy", "correctness_head"],
        default="max_prob",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hybrid_fallback(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        out_json=args.out_json,
        eval_families=(list(args.eval_families) if args.eval_families is not None else None),
        score_mode=str(args.score_mode),
        thresholds=(list(args.thresholds) if args.thresholds is not None else None),
        max_shots=args.max_shots,
        batch_size=args.batch_size,
        device_arg=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=common._json_default))


if __name__ == "__main__":
    main()
