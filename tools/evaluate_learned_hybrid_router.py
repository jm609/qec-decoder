from __future__ import annotations

"""
evaluate_learned_hybrid_router.py

Post-hoc learned hybrid evaluation:
- primary decoder: factorized_logical_frame_decoder
- fallback decoder: PyMatching
- router: frozen-feature logistic correctness model
"""

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DECODER_DIR = ROOT / "decoders"
for path in (ROOT, TOOLS_DIR, DECODER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    optim = None

import baseline_rectcnn as common
import factorized_logical_frame_decoder as flfd
import evaluate_hybrid_fallback as hybrid_common
import research_noise_aware_3d as volume_common


SCHEMA_VERSION = "hybrid_learned_router.eval.v1"
ROUTER_TARGET_CORRECTNESS = "correctness"
ROUTER_TARGET_PREFER_NEURAL = "prefer_neural"
ROUTER_TARGET_CHOICES = (ROUTER_TARGET_CORRECTNESS, ROUTER_TARGET_PREFER_NEURAL)


def _router_threshold_grid(raw_thresholds: list[float] | None) -> list[float]:
    if raw_thresholds:
        return [float(value) for value in raw_thresholds]
    default = np.concatenate(
        [
            np.linspace(0.0, 0.95, 20, dtype=np.float64),
            np.array([0.975, 0.99, 0.995, 0.999, 1.0], dtype=np.float64),
        ]
    )
    return [float(value) for value in np.unique(default)]


def _normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs_f = np.asarray(probs, dtype=np.float32)
    entropy = -np.sum(probs_f * np.log(np.clip(probs_f, 1e-12, 1.0)), axis=1)
    max_entropy = float(np.log(probs_f.shape[1])) if probs_f.shape[1] > 1 else 1.0
    if max_entropy <= 0.0:
        return np.zeros((probs_f.shape[0],), dtype=np.float32)
    return np.asarray(entropy / max_entropy, dtype=np.float32)


def _one_hot(labels: np.ndarray, *, depth: int) -> np.ndarray:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((target.shape[0], depth), dtype=np.float32)
    if target.size:
        out[np.arange(target.shape[0]), target] = 1.0
    return out


def _event_density_features(prepared: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(prepared["x"], dtype=np.float32)
    signal_channels = int(prepared["signal_channels"])
    signal = x[:, :signal_channels, :, :, :]
    event = signal[:, 0]
    valid = signal[:, 1]
    boundary = signal[:, 2]
    final_round = signal[:, 3]

    reduce_axes = (1, 2, 3)
    valid_count = np.clip(np.sum(valid, axis=reduce_axes), 1.0, None)
    boundary_count = np.clip(np.sum(boundary, axis=reduce_axes), 1.0, None)
    final_round_count = np.clip(np.sum(final_round, axis=reduce_axes), 1.0, None)
    event_count = np.sum(event, axis=reduce_axes)
    boundary_event_count = np.sum(event * boundary, axis=reduce_axes)
    final_round_event_count = np.sum(event * final_round, axis=reduce_axes)

    features = np.stack(
        [
            event_count / valid_count,
            boundary_event_count / boundary_count,
            final_round_event_count / final_round_count,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    feature_names = [
        "event_density",
        "boundary_event_density",
        "final_round_event_density",
    ]
    return features, feature_names


def _metadata_context_features(
    *,
    metadata: dict[str, Any],
    prepared: dict[str, Any],
    num_shots: int,
) -> tuple[np.ndarray, list[str]]:
    circuit = metadata.get("circuit", {}) if isinstance(metadata, dict) else {}
    qc_stats = metadata.get("qc_stats", {}) if isinstance(metadata, dict) else {}
    stage_vec = volume_common._stage_one_hot(metadata.get("stage"))
    noise_signature, noise_signature_names = volume_common._build_noise_signature_values(metadata)
    occupancy_fraction = 0.0
    x = np.asarray(prepared["x"], dtype=np.float32)
    if x.ndim == 5 and x.shape[0] > 0:
        valid_channel = x[0, 1]
        occupancy_fraction = float(np.mean(valid_channel > 0.5))
    values = np.concatenate(
        [
            stage_vec,
            np.asarray(
                [
                    float(circuit.get("distance", 0)) / 25.0,
                    float(circuit.get("rounds", 0)) / 100.0,
                    float(circuit.get("num_detectors", 0)) / 4096.0,
                    float(qc_stats.get("detector_event_fraction", 0.0)),
                    float(qc_stats.get("logical_flip_fraction", 0.0)),
                    float(qc_stats.get("logical_x_flip_fraction", 0.0)),
                    float(qc_stats.get("logical_z_flip_fraction", 0.0)),
                    float(occupancy_fraction),
                ],
                dtype=np.float32,
            ),
            noise_signature,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    names = [
        "meta_stage_ideal",
        "meta_stage_a",
        "meta_stage_b",
        "meta_stage_c",
        "meta_distance_norm",
        "meta_rounds_norm",
        "meta_num_detectors_norm",
        "meta_detector_event_fraction",
        "meta_logical_flip_fraction",
        "meta_logical_x_flip_fraction",
        "meta_logical_z_flip_fraction",
        "meta_occupancy_fraction",
    ] + [f"meta_{name}" for name in noise_signature_names]
    features = np.broadcast_to(values[None, :], (int(num_shots), values.shape[0])).astype(np.float32, copy=False)
    return features, names


def _build_router_features(
    *,
    flfd_result: dict[str, Any],
    pym_result: dict[str, Any],
    router_target_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    neural_probs = np.asarray(flfd_result["probs"], dtype=np.float32)
    neural_pred_class4 = np.asarray(flfd_result["pred_class4"], dtype=np.int64)
    target_class4 = np.asarray(flfd_result["prepared"]["y_class4"], dtype=np.int64)
    pym_pred_class4 = np.asarray(pym_result["pred_class4"], dtype=np.int64)
    neural_correct = (neural_pred_class4 == target_class4).astype(np.float32, copy=False)
    pym_correct = (pym_pred_class4 == target_class4).astype(np.float32, copy=False)

    neural_max_prob = np.max(neural_probs, axis=1, keepdims=True)
    sorted_probs = np.sort(neural_probs, axis=1)
    neural_margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    entropy = _normalized_entropy(neural_probs).reshape(-1, 1)
    bit_x_prob = (neural_probs[:, 1] + neural_probs[:, 3]).reshape(-1, 1)
    bit_z_prob = (neural_probs[:, 2] + neural_probs[:, 3]).reshape(-1, 1)

    neural_x, neural_z = hybrid_common._class4_to_bits(neural_pred_class4.astype(np.uint8, copy=False))
    pym_x, pym_z = hybrid_common._class4_to_bits(pym_pred_class4.astype(np.uint8, copy=False))
    agreement = (neural_pred_class4 == pym_pred_class4).astype(np.float32).reshape(-1, 1)
    bit_x_agreement = (neural_x == pym_x).astype(np.float32).reshape(-1, 1)
    bit_z_agreement = (neural_z == pym_z).astype(np.float32).reshape(-1, 1)
    event_features, event_feature_names = _event_density_features(flfd_result["prepared"])
    metadata_features, metadata_feature_names = _metadata_context_features(
        metadata=flfd_result["metadata"],
        prepared=flfd_result["prepared"],
        num_shots=int(neural_probs.shape[0]),
    )

    feature_blocks = [
        neural_probs,
        neural_max_prob,
        neural_margin,
        entropy,
        bit_x_prob,
        bit_z_prob,
        _one_hot(pym_pred_class4, depth=4),
        agreement,
        bit_x_agreement,
        bit_z_agreement,
        event_features,
        metadata_features,
    ]
    feature_names = [
        "neural_prob_I",
        "neural_prob_X",
        "neural_prob_Z",
        "neural_prob_Y",
        "neural_max_prob",
        "neural_margin",
        "neural_norm_entropy",
        "neural_bit_x_prob",
        "neural_bit_z_prob",
        "pym_class_I",
        "pym_class_X",
        "pym_class_Z",
        "pym_class_Y",
        "class_agreement",
        "bit_x_agreement",
        "bit_z_agreement",
        *event_feature_names,
        *metadata_feature_names,
    ]
    x_router = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    if router_target_mode == ROUTER_TARGET_CORRECTNESS:
        y_router = neural_correct
    elif router_target_mode == ROUTER_TARGET_PREFER_NEURAL:
        y_router = (neural_correct >= pym_correct).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported router_target_mode: {router_target_mode!r}")
    aux = {
        "target_class4": target_class4.astype(np.uint8, copy=False),
        "target_x": np.asarray(flfd_result["prepared"]["y_x"], dtype=np.uint8),
        "target_z": np.asarray(flfd_result["prepared"]["y_z"], dtype=np.uint8),
        "neural_probs": neural_probs,
        "pym_probs": np.asarray(pym_result["pred_probs"], dtype=np.float32),
        "neural_pred_class4": neural_pred_class4.astype(np.uint8, copy=False),
        "pym_pred_class4": pym_pred_class4.astype(np.uint8, copy=False),
        "neural_correct": neural_correct.astype(np.uint8, copy=False),
        "pym_correct": pym_correct.astype(np.uint8, copy=False),
        "router_target": y_router.astype(np.uint8, copy=False),
        "metadata": flfd_result["metadata"],
    }
    return x_router, y_router, feature_names, aux


class _CorrectnessRouter(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def _fit_correctness_router(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    device_arg: str,
) -> dict[str, Any]:
    common._require_torch()
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required for the learned hybrid router.")

    x_arr = np.asarray(x_train, dtype=np.float32)
    y_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
    feature_mean = np.mean(x_arr, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    feature_std = np.std(x_arr, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    feature_std = np.where(feature_std >= 1e-6, feature_std, 1.0).astype(np.float32, copy=False)
    x_norm = (x_arr - feature_mean[None, :]) / feature_std[None, :]

    positives = float(np.sum(y_arr >= 0.5))
    negatives = float(y_arr.shape[0] - positives)
    pos_weight_value = 1.0
    if positives > 0.0 and negatives > 0.0:
        pos_weight_value = float(np.clip(negatives / positives, 1.0, 32.0))

    device = common._pick_device(device_arg)
    model = _CorrectnessRouter(int(x_norm.shape[1])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))

    xb = torch.from_numpy(np.ascontiguousarray(x_norm)).to(device)
    yb = torch.from_numpy(np.ascontiguousarray(y_arr)).to(device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = model(xb)
        train_probs = torch.sigmoid(train_logits).detach().cpu().numpy().astype(np.float32, copy=False)

    train_metrics = common._binary_metrics_from_probs(
        train_probs,
        y_arr.astype(np.uint8, copy=False),
        threshold=0.5,
        bce_loss=float(criterion(train_logits, yb).detach().cpu().item()),
    )
    coef = model.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    intercept = float(model.linear.bias.detach().cpu().numpy().reshape(-1)[0])
    return {
        "model": model,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_probs": train_probs,
        "pos_weight": pos_weight_value,
        "train_metrics": train_metrics,
        "coefficients": coef,
        "intercept": intercept,
        "device": device,
    }


def _router_predict_scores(
    *,
    router_state: dict[str, Any],
    x_features: np.ndarray,
) -> np.ndarray:
    common._require_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for router prediction.")
    x_arr = np.asarray(x_features, dtype=np.float32)
    x_norm = (x_arr - router_state["feature_mean"][None, :]) / router_state["feature_std"][None, :]
    xb = torch.from_numpy(np.ascontiguousarray(x_norm)).to(router_state["device"])
    with torch.no_grad():
        logits = router_state["model"](xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float32).reshape(-1)


def _top_weight_features(
    *,
    feature_names: list[str],
    coefficients: np.ndarray,
    k: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    pairs = [(name, float(weight)) for name, weight in zip(feature_names, coefficients, strict=True)]
    ordered = sorted(pairs, key=lambda item: item[1], reverse=reverse)
    return [{"feature": name, "weight": weight} for name, weight in ordered[:k]]


def run_learned_hybrid_router(
    *,
    manifest: Path,
    checkpoint: Path,
    out_json: Path,
    eval_families: list[str] | None,
    thresholds: list[float] | None,
    max_shots: int | None,
    batch_size: int,
    device_arg: str,
    router_epochs: int,
    router_lr: float,
    router_weight_decay: float,
    router_target: str,
) -> dict[str, Any]:
    manifest_data = common._read_json(manifest)
    family_dirs = manifest_data.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(f"Manifest does not contain family_dirs: {manifest}")
    resolved_eval_families = list(eval_families) if eval_families is not None else list(family_dirs.keys())
    threshold_grid = _router_threshold_grid(thresholds)

    checkpoint_data = torch.load(checkpoint, map_location="cpu") if torch is not None else {}
    seen_families = (
        list(checkpoint_data.get("train_families"))
        if checkpoint_data.get("train_families") is not None
        else [str(checkpoint_data.get("family"))]
    )
    seen_families = [family for family in seen_families if family and family != "multi_family_manifest"]

    family_inputs: dict[str, dict[str, Any]] = {}
    feature_names: list[str] | None = None
    for family in resolved_eval_families:
        if family not in family_dirs:
            raise KeyError(f"Family {family!r} missing from manifest. Available: {sorted(family_dirs)}")
        family_dir = common._resolve_manifest_family_dir(manifest, family_dirs[family])
        flfd_result = hybrid_common._predict_flfd_family(
            checkpoint_path=checkpoint,
            family_dir=family_dir,
            batch_size=batch_size,
            max_shots=max_shots,
            device_arg=device_arg,
        )
        pym_result = hybrid_common._decode_pymatching_family(
            family_dir=family_dir,
            max_shots=max_shots,
        )
        x_router, y_router, names, aux = _build_router_features(
            flfd_result=flfd_result,
            pym_result=pym_result,
            router_target_mode=router_target,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Router feature layout changed across families.")
        family_inputs[family] = {
            "family_dir": family_dir.as_posix(),
            "x_router": x_router,
            "y_router": y_router,
            **aux,
        }

    assert feature_names is not None
    train_feature_blocks = [family_inputs[family]["x_router"] for family in seen_families if family in family_inputs]
    train_target_blocks = [family_inputs[family]["y_router"] for family in seen_families if family in family_inputs]
    if not train_feature_blocks or not train_target_blocks:
        raise ValueError("No seen-family data available to train the learned router.")
    x_train = np.concatenate(train_feature_blocks, axis=0)
    y_train = np.concatenate(train_target_blocks, axis=0)
    router_state = _fit_correctness_router(
        x_train=x_train,
        y_train=y_train,
        epochs=router_epochs,
        lr=router_lr,
        weight_decay=router_weight_decay,
        device_arg=device_arg,
    )

    sweep_by_family: dict[str, list[dict[str, Any]]] = {}
    baseline_by_family: dict[str, dict[str, Any]] = {}
    family_meta: dict[str, dict[str, Any]] = {}
    router_by_family: dict[str, dict[str, Any]] = {}
    for family in resolved_eval_families:
        family_input = family_inputs[family]
        router_scores = _router_predict_scores(router_state=router_state, x_features=family_input["x_router"])
        target_class4 = np.asarray(family_input["target_class4"], dtype=np.uint8)
        target_x = np.asarray(family_input["target_x"], dtype=np.uint8)
        target_z = np.asarray(family_input["target_z"], dtype=np.uint8)
        neural_probs = np.asarray(family_input["neural_probs"], dtype=np.float32)
        pym_probs = np.asarray(family_input["pym_probs"], dtype=np.float32)
        y_router = np.asarray(family_input["y_router"], dtype=np.uint8)

        family_sweep = [
            hybrid_common._hybrid_metrics_for_threshold(
                target_class4=target_class4,
                target_x=target_x,
                target_z=target_z,
                neural_probs=neural_probs,
                neural_scores=router_scores,
                pymatching_probs=pym_probs,
                threshold=threshold,
            )
            for threshold in threshold_grid
        ]
        sweep_by_family[family] = family_sweep

        neural_metrics = common._multiclass_metrics_from_probs(
            neural_probs,
            target_class4,
            class_labels=list(hybrid_common.CLASS_LABELS),
            loss=flfd._cross_entropy_from_probs(neural_probs, target_class4),
        )
        neural_calibration = flfd._multiclass_calibration_metrics(neural_probs, target_class4)
        pym_metrics = common._multiclass_metrics_from_probs(
            pym_probs,
            target_class4,
            class_labels=list(hybrid_common.CLASS_LABELS),
            loss=flfd._cross_entropy_from_probs(pym_probs, target_class4),
        )
        pym_calibration = flfd._multiclass_calibration_metrics(pym_probs, target_class4)
        router_metrics = common._binary_metrics_from_probs(router_scores, y_router, threshold=0.5, bce_loss=None)
        baseline_by_family[family] = {
            "neural": {
                "accuracy": neural_metrics["accuracy"],
                "label_error_rate": neural_metrics["label_error_rate"],
                "macro_f1": neural_metrics["macro_f1"],
                "ece_10": neural_calibration["ece"],
                "brier_score": neural_calibration["brier_score"],
            },
            "pymatching": {
                "accuracy": pym_metrics["accuracy"],
                "label_error_rate": pym_metrics["label_error_rate"],
                "macro_f1": pym_metrics["macro_f1"],
                "ece_10": pym_calibration["ece"],
                "brier_score": pym_calibration["brier_score"],
            },
        }
        router_by_family[family] = {
            "target_accuracy": router_metrics["accuracy"],
            "target_f1": router_metrics["f1"],
            "target_auroc": router_metrics["auroc"],
            "mean_router_score": float(np.mean(router_scores)) if router_scores.size else None,
            "mean_router_score_positive": float(np.mean(router_scores[y_router >= 0.5])) if np.any(y_router >= 0.5) else None,
            "mean_router_score_negative": float(np.mean(router_scores[y_router < 0.5])) if np.any(y_router < 0.5) else None,
            "neural_correct_rate": float(np.mean(np.asarray(family_input["neural_correct"], dtype=np.float32))),
            "pymatching_correct_rate": float(np.mean(np.asarray(family_input["pym_correct"], dtype=np.float32))),
        }
        family_meta[family] = {
            "family_dir": family_input["family_dir"],
            "stage": str(family_input["metadata"].get("stage")),
            "seen_in_training": family in seen_families,
        }

    threshold_selection = hybrid_common._select_threshold(
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
                "router_target_accuracy": router_by_family[family]["target_accuracy"],
                "router_target_auroc": router_by_family[family]["target_auroc"],
                "mean_router_score": router_by_family[family]["mean_router_score"],
            }
        )

    coefficients = np.asarray(router_state["coefficients"], dtype=np.float32)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": common._utc_now_iso(),
        "primary_decoder": "factorized_logical_frame_decoder",
        "fallback_decoder": "pymatching",
        "router": {
            "type": "logistic_frozen_feature_router",
            "target_mode": router_target,
            "feature_names": feature_names,
            "epochs": int(router_epochs),
            "lr": float(router_lr),
            "weight_decay": float(router_weight_decay),
            "num_train_examples": int(x_train.shape[0]),
            "num_positive_examples": int(np.sum(y_train >= 0.5)),
            "num_negative_examples": int(np.sum(y_train < 0.5)),
            "positive_rate": float(np.mean(y_train)) if y_train.size else None,
            "pos_weight": float(router_state["pos_weight"]),
            "train_metrics": router_state["train_metrics"],
            "intercept": float(router_state["intercept"]),
            "top_positive_features": _top_weight_features(
                feature_names=feature_names,
                coefficients=coefficients,
                k=5,
                reverse=True,
            ),
            "top_negative_features": _top_weight_features(
                feature_names=feature_names,
                coefficients=coefficients,
                k=5,
                reverse=False,
            ),
        },
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
            "score_mode": f"learned_router:{router_target}",
            "thresholds_considered": threshold_grid,
            **threshold_selection,
        },
        "per_family_baselines": baseline_by_family,
        "per_family_router": router_by_family,
        "selected_per_family": selected_per_family,
        "threshold_sweep_by_family": sweep_by_family,
    }
    common._write_json(out_json, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned frozen-feature hybrid router for FLFD + PyMatching.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--eval-families", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--router-epochs", type=int, default=400)
    parser.add_argument("--router-lr", type=float, default=1e-2)
    parser.add_argument("--router-weight-decay", type=float, default=1e-3)
    parser.add_argument("--router-target", type=str, choices=ROUTER_TARGET_CHOICES, default=ROUTER_TARGET_PREFER_NEURAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_learned_hybrid_router(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        out_json=args.out_json,
        eval_families=(list(args.eval_families) if args.eval_families is not None else None),
        thresholds=(list(args.thresholds) if args.thresholds is not None else None),
        max_shots=args.max_shots,
        batch_size=int(args.batch_size),
        device_arg=str(args.device),
        router_epochs=int(args.router_epochs),
        router_lr=float(args.router_lr),
        router_weight_decay=float(args.router_weight_decay),
        router_target=str(args.router_target),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=common._json_default))


if __name__ == "__main__":
    main()
