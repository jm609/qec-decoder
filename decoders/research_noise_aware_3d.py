from __future__ import annotations

"""
research_noise_aware_3d.py

Research-oriented spatiotemporal decoder on top of the rectangular syndrome
representation.

Key differences from baseline_rectcnn.py
----------------------------------------
- Uses a 3-D convolutional network over (time, height, width).
- Adds geometry channels and dataset-level noise-context channels.
- Intended for experimentation on transfer across noise families/stages.
"""

from pathlib import Path
from typing import Any
import argparse
import copy
import json
import math
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None
    optim = None

try:
    import baseline_rectcnn as common
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_rectcnn as common


SCHEMA_VERSION_TRAIN = "research_noise_aware_3d.train.v2"
SCHEMA_VERSION_EVAL = "research_noise_aware_3d.eval.v2"
SCHEMA_VERSION_EXPERIMENT = "research_noise_aware_3d.experiment.v2"


def _stage_one_hot(stage: str | None) -> np.ndarray:
    key = str(stage or "").lower()
    options = ["ideal", "a", "b", "c"]
    out = np.zeros(len(options), dtype=np.float32)
    if key in options:
        out[options.index(key)] = 1.0
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _instruction_fraction(histogram: dict[str, Any], key: str) -> float:
    total = float(sum(_safe_float(v) for v in histogram.values()))
    if total <= 0.0:
        return 0.0
    return float(_safe_float(histogram.get(key, 0.0)) / total)


def _build_noise_signature_values(metadata: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    upstream = metadata.get("upstream_summary", {}) if isinstance(metadata, dict) else {}
    histogram = metadata.get("instruction_histogram", {}) if isinstance(metadata, dict) else {}
    dem_meta = metadata.get("dem", {}) if isinstance(metadata, dict) else {}
    correlated_counts = (
        upstream.get("correlated_instruction_counts", {}) if isinstance(upstream, dict) else {}
    )

    values = np.asarray(
        [
            _safe_float(upstream.get("p", 0.0)),
            _safe_float(upstream.get("p_cz", 0.0)),
            _safe_float(upstream.get("p_1q", 0.0)),
            _safe_float(upstream.get("p_reset", 0.0)),
            _safe_float(upstream.get("p_meas", 0.0)),
            _safe_float(upstream.get("p_idle", 0.0)),
            _safe_float(upstream.get("p_ridle", 0.0)),
            float(bool(dem_meta.get("kwargs", {}).get("approximate_disjoint_errors", False))),
            float(bool(upstream.get("dem_uses_approximate_disjoint_errors", False))),
            _instruction_fraction(histogram, "DEPOLARIZE1"),
            _instruction_fraction(histogram, "DEPOLARIZE2"),
            _instruction_fraction(histogram, "X_ERROR"),
            _instruction_fraction(histogram, "PAULI_CHANNEL_2"),
            _instruction_fraction(histogram, "CORRELATED_ERROR"),
            _instruction_fraction(histogram, "ELSE_CORRELATED_ERROR"),
            _instruction_fraction(histogram, "E"),
            _instruction_fraction(correlated_counts, "CORRELATED_ERROR"),
            _instruction_fraction(correlated_counts, "PAULI_CHANNEL_2"),
        ],
        dtype=np.float32,
    )
    names = [
        "base_p",
        "p_cz",
        "p_1q",
        "p_reset",
        "p_meas",
        "p_idle",
        "p_ridle",
        "dem_approximate_disjoint_flag",
        "upstream_dem_approximate_disjoint_flag",
        "instr_fraction_depolarize1",
        "instr_fraction_depolarize2",
        "instr_fraction_x_error",
        "instr_fraction_pauli_channel_2",
        "instr_fraction_correlated_error",
        "instr_fraction_else_correlated_error",
        "instr_fraction_e",
        "corr_instr_fraction_correlated_error",
        "corr_instr_fraction_pauli_channel_2",
    ]
    return values, names


def _build_static_planes(layout: common.RectangularSyndromeLayout) -> tuple[np.ndarray, list[str]]:
    valid = layout.valid_mask.astype(np.float32, copy=False)
    checker0 = ((layout.checkerboard_volume == 0) & (layout.valid_mask == 1)).astype(np.float32)
    checker1 = ((layout.checkerboard_volume == 1) & (layout.valid_mask == 1)).astype(np.float32)
    is_x = ((layout.detector_type_volume == 1) & (layout.valid_mask == 1)).astype(np.float32)
    is_z = ((layout.detector_type_volume == 2) & (layout.valid_mask == 1)).astype(np.float32)
    planes = np.stack(
        [
            valid,
            layout.boundary_volume.astype(np.float32, copy=False) * valid,
            layout.final_round_volume.astype(np.float32, copy=False) * valid,
            checker0,
            checker1,
            is_x,
            is_z,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    names = [
        "valid_mask",
        "boundary_flag",
        "final_round_flag",
        "checkerboard_class_0",
        "checkerboard_class_1",
        "is_x_check",
        "is_z_check",
    ]
    return planes, names


def _build_context_planes(
    metadata: dict[str, Any],
    layout: common.RectangularSyndromeLayout,
) -> tuple[np.ndarray, list[str]]:
    circuit = metadata.get("circuit", {}) if isinstance(metadata, dict) else {}
    qc_stats = metadata.get("qc_stats", {}) if isinstance(metadata, dict) else {}
    stage_vec = _stage_one_hot(metadata.get("stage"))
    noise_signature, noise_signature_names = _build_noise_signature_values(metadata)
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
                    float(layout.valid_mask.mean()) if layout.valid_mask.size else 0.0,
                ],
                dtype=np.float32,
            ),
            noise_signature,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    names = [
        "stage_ideal",
        "stage_a",
        "stage_b",
        "stage_c",
        "distance_norm",
        "rounds_norm",
        "num_detectors_norm",
        "detector_event_fraction",
        "logical_flip_fraction",
        "occupancy_fraction",
    ] + noise_signature_names
    planes = np.broadcast_to(
        values[:, None, None, None],
        (values.shape[0], layout.time_steps, layout.height, layout.width),
    ).astype(np.float32, copy=False)
    return planes, names


def _prepare_loaded_family(
    payload: common.LoadedFamilyPayload,
    *,
    fill_value: float,
    max_shots: int | None,
    target_mode: str = common.TARGET_MODE_AUTO,
) -> dict[str, Any]:
    base = common._prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=target_mode,
    )
    event_volume = np.asarray(base["x"], dtype=np.float32)
    y = np.asarray(base["y"])
    layout: common.RectangularSyndromeLayout = base["layout"]

    static_planes, static_names = _build_static_planes(layout)
    context_planes, context_names = _build_context_planes(payload.metadata, layout)

    num_shots = int(event_volume.shape[0])
    static_batch = np.broadcast_to(static_planes[None, :, :, :, :], (num_shots,) + static_planes.shape)
    context_batch = np.broadcast_to(context_planes[None, :, :, :, :], (num_shots,) + context_planes.shape)
    x = np.concatenate(
        [
            event_volume[:, None, :, :, :],
            static_batch,
            context_batch,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    bundle_info = {
        "representation": "noise_aware_volume_3d",
        "shape": {
            "batch": int(x.shape[0]),
            "channels": int(x.shape[1]),
            "time_steps": int(x.shape[2]),
            "grid_height": int(x.shape[3]),
            "grid_width": int(x.shape[4]),
        },
        "channel_names": ["event"] + static_names + context_names,
        "layout_summary": base["bundle_info"]["layout_summary"],
        "fill_value": float(fill_value),
    }
    return {
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "layout": layout,
        "bundle_info": bundle_info,
        "shots_total_after_limit": base["shots_total_after_limit"],
        "target_key": base.get("target_key"),
        "task_type": base["task_type"],
        "num_classes": base["num_classes"],
        "class_labels": list(base["class_labels"]),
        "resolved_target_mode": base["resolved_target_mode"],
    }


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        groups = 4 if channels % 4 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.relu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.relu(self.norm2(h)))
        return x + h


class NoiseAwareVolumeDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dense_hidden_dim: int,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.out_dim = int(out_dim)
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [Residual3DBlock(hidden_channels, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head_norm = nn.GroupNorm(4 if hidden_channels % 4 == 0 else 1, hidden_channels)
        self.head_relu = nn.ReLU()
        self.head_drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_channels, dense_hidden_dim)
        self.fc2 = nn.Linear(dense_hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_relu(self.head_norm(h))
        h = h.mean(dim=(2, 3, 4))
        h = self.head_drop(self.head_relu(self.fc1(h)))
        out = self.fc2(h)
        return out.squeeze(1) if self.out_dim == 1 else out


def _summarize_target_distribution(
    y: np.ndarray,
    *,
    task_type: str,
    class_labels: list[str],
) -> dict[str, Any]:
    if task_type == "binary":
        target = np.asarray(y, dtype=np.float32).reshape(-1)
        return {
            "task_type": "binary",
            "num_examples": int(target.shape[0]),
            "positive_rate": float(target.mean()) if target.size else None,
        }

    target = np.asarray(y, dtype=np.int64).reshape(-1)
    num_classes = len(class_labels)
    counts = np.bincount(target, minlength=num_classes)
    return {
        "task_type": "multiclass",
        "num_examples": int(target.shape[0]),
        "class_labels": list(class_labels),
        "class_histogram": {
            (class_labels[index] if index < len(class_labels) else str(index)): int(counts[index])
            for index in range(num_classes)
        },
    }


def _resolve_manifest_family_entries(
    manifest_path: Path,
    requested_families: list[str] | None,
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    manifest_data = common._read_json(manifest_path)
    family_dirs = manifest_data.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(
            f"manifest.json does not contain a non-empty family_dirs mapping: {manifest_path}"
        )

    families = list(requested_families) if requested_families else list(family_dirs.keys())
    resolved: list[tuple[str, Path]] = []
    for family in families:
        if family not in family_dirs:
            raise KeyError(
                f"Requested family {family!r} missing from manifest. Available: {sorted(family_dirs)}"
            )
        resolved.append(
            (
                str(family),
                common._resolve_manifest_family_dir(manifest_path, family_dirs[family]),
            )
        )
    return manifest_data, resolved


def _load_prepared_family_entry(
    *,
    family_dir: Path,
    fill_value: float,
    max_shots: int | None,
    target_mode: str,
) -> dict[str, Any]:
    payload = common._load_family_payload(family_dir)
    prepared = _prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=target_mode,
    )
    y = np.asarray(prepared["y"])
    metadata = payload.metadata
    return {
        "payload": payload,
        "metadata": metadata,
        "family_dir": family_dir,
        "x": prepared["x"],
        "y": y,
        "bundle_info": prepared["bundle_info"],
        "shots_total_after_limit": int(prepared["shots_total_after_limit"]),
        "target_key": prepared["target_key"],
        "task_type": prepared["task_type"],
        "num_classes": int(prepared["num_classes"]),
        "class_labels": list(prepared["class_labels"]),
        "resolved_target_mode": prepared["resolved_target_mode"],
        "target_summary": _summarize_target_distribution(
            y,
            task_type=str(prepared["task_type"]),
            class_labels=list(prepared["class_labels"]),
        ),
    }


def _validate_compatible_family_batches(prepared_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not prepared_entries:
        raise ValueError("At least one prepared family entry is required")

    reference_shape = tuple(int(x) for x in prepared_entries[0]["x"].shape[1:])
    reference_channels = list(prepared_entries[0]["bundle_info"]["channel_names"])
    reference_layout_summary = prepared_entries[0]["bundle_info"]["layout_summary"]
    reference_task_type = str(prepared_entries[0]["task_type"])
    reference_num_classes = int(prepared_entries[0]["num_classes"])
    reference_class_labels = list(prepared_entries[0]["class_labels"])
    reference_target_mode = str(prepared_entries[0]["resolved_target_mode"])
    for entry in prepared_entries[1:]:
        shape = tuple(int(x) for x in entry["x"].shape[1:])
        if shape != reference_shape:
            raise ValueError(
                "All training families must share the same input tensor shape. "
                f"Expected {reference_shape}, got {shape} for {entry['family_dir']}"
            )
        channels = list(entry["bundle_info"]["channel_names"])
        if channels != reference_channels:
            raise ValueError(
                "All training families must share the same channel layout. "
                f"Mismatch detected for {entry['family_dir']}"
            )
        if str(entry["task_type"]) != reference_task_type:
            raise ValueError(
                "All training families must share the same supervised task type. "
                f"Expected {reference_task_type}, got {entry['task_type']} for {entry['family_dir']}"
            )
        if int(entry["num_classes"]) != reference_num_classes:
            raise ValueError(
                "All training families must share the same number of classes. "
                f"Expected {reference_num_classes}, got {entry['num_classes']} for {entry['family_dir']}"
            )
        if list(entry["class_labels"]) != reference_class_labels:
            raise ValueError(
                "All training families must share the same class label ordering. "
                f"Mismatch detected for {entry['family_dir']}"
            )
        if str(entry["resolved_target_mode"]) != reference_target_mode:
            raise ValueError(
                "All training families must resolve to the same target mode. "
                f"Expected {reference_target_mode}, got {entry['resolved_target_mode']} for {entry['family_dir']}"
            )
    return {
        "input_shape_without_batch": list(reference_shape),
        "channel_names": reference_channels,
        "layout_summary": reference_layout_summary,
        "task_type": reference_task_type,
        "num_classes": reference_num_classes,
        "class_labels": reference_class_labels,
        "resolved_target_mode": reference_target_mode,
    }


def _train_model_from_splits(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    num_blocks: int,
    dense_hidden_dim: int,
    device_arg: str,
    task_type: str,
    num_classes: int,
    class_labels: list[str],
) -> dict[str, Any]:
    model = NoiseAwareVolumeDecoder(
        in_channels=int(x_train.shape[1]),
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        dropout=dropout,
        out_dim=(1 if task_type == "binary" else num_classes),
    )
    device = common._pick_device(device_arg)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = common._make_criterion(task_type)
    train_loader = common._make_loader(
        common._make_tensor_dataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = common._train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
        )
        val_metrics = common._evaluate_arrays(
            model=model,
            x=x_val,
            y=y_val,
            batch_size=batch_size,
            device=device,
            task_type=task_type,
            threshold=0.5,
            class_labels=class_labels,
        )
        val_loss = (
            val_metrics["bce_loss"]
            if task_type == "binary"
            else val_metrics["cross_entropy_loss"]
        )
        history.append(
            (
                {
                    "epoch": epoch,
                    "train_bce_loss": train_loss,
                    "val_bce_loss": val_loss,
                    "val_accuracy_at_0_5": val_metrics["accuracy"],
                    "val_f1_at_0_5": val_metrics["f1"],
                }
                if task_type == "binary"
                else {
                    "epoch": epoch,
                    "train_cross_entropy_loss": train_loss,
                    "val_cross_entropy_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
        )
        if best_val_loss is None or (val_loss is not None and val_loss < best_val_loss):
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold_selection = common._select_decision_rule_from_validation(
        model=model,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        class_labels=class_labels,
    )
    threshold_raw = threshold_selection.get("selected_threshold")
    threshold = float(threshold_raw) if threshold_raw is not None else None
    test_metrics = common._evaluate_arrays(
        model=model,
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        threshold=threshold,
        class_labels=class_labels,
    )
    elapsed = time.perf_counter() - started
    return {
        "model": model,
        "device": device,
        "history": history,
        "threshold_selection": threshold_selection,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "task_type": task_type,
        "num_classes": num_classes,
        "class_labels": list(class_labels),
    }


def train_family_dir(
    *,
    family_dir: Path | None,
    manifest: Path | None,
    family: str | None,
    checkpoint_out: Path,
    train_json_out: Path | None,
    fill_value: float,
    target_mode: str,
    max_shots: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    num_blocks: int,
    dense_hidden_dim: int,
    device_arg: str,
) -> common.TrainResult:
    common._require_torch()
    common._set_random_seeds(seed)

    input_mode, resolved_family_dir, manifest_data = common._resolve_input_family_dir(
        family_dir=family_dir,
        manifest=manifest,
        family=family,
    )
    payload = common._load_family_payload(resolved_family_dir)
    metadata = payload.metadata

    prepared = _prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=target_mode,
    )
    x = prepared["x"]
    y = prepared["y"]
    bundle_info = prepared["bundle_info"]
    num_shots = int(y.shape[0])
    task_type = str(prepared["task_type"])
    num_classes = int(prepared["num_classes"])
    class_labels = list(prepared["class_labels"])

    split_indices = common.build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_summary = common.summarise_split_indices(split_indices, num_shots=num_shots, seed=seed)

    x_train = common._subset_rows(x, split_indices["train"])
    y_train = common._subset_rows(y, split_indices["train"])
    x_val = common._subset_rows(x, split_indices["val"])
    y_val = common._subset_rows(y, split_indices["val"])
    x_test = common._subset_rows(x, split_indices["test"])
    y_test = common._subset_rows(y, split_indices["test"])

    trained = _train_model_from_splits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        device_arg=device_arg,
        task_type=task_type,
        num_classes=num_classes,
        class_labels=class_labels,
    )
    model = trained["model"]
    device = trained["device"]
    threshold_selection = trained["threshold_selection"]
    test_metrics = trained["test_metrics"]
    elapsed = float(trained["elapsed_seconds"])
    history = trained["history"]

    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "research_noise_aware_3d",
        "created_at_utc": common._utc_now_iso(),
        "family": metadata.get("family"),
        "stage": metadata.get("stage"),
        "family_dir": str(resolved_family_dir.as_posix()),
        "fill_value": float(fill_value),
        "model_hparams": {
            "in_channels": int(x.shape[1]),
            "hidden_channels": int(hidden_channels),
            "num_blocks": int(num_blocks),
            "dense_hidden_dim": int(dense_hidden_dim),
            "dropout": float(dropout),
            "task_type": task_type,
            "num_classes": num_classes,
            "class_labels": class_labels,
        },
        "threshold_selection": threshold_selection,
        "target_mode_requested": target_mode,
        "target_mode_resolved": prepared["resolved_target_mode"],
        "target_key": prepared["target_key"],
        "split": split_summary.to_dict(),
        "state_dict": model.state_dict(),
    }
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)

    result = common.TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="research_noise_aware_3d",
        created_at_utc=common._utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family")),
        stage=str(metadata.get("stage")),
        family_dir=resolved_family_dir.as_posix(),
        model={
            "architecture": "noise_aware_volume_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(trained["num_parameters"]),
            "bundle_info": bundle_info,
            "model_hparams": checkpoint["model_hparams"],
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit"),
            "scaffold": metadata.get("scaffold"),
            "targets": metadata.get("targets"),
            "target_key_used": prepared.get("target_key"),
            "rectangular_syndrome_layout": metadata.get("rectangular_syndrome_layout"),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "num_shots_total_after_limit": prepared["shots_total_after_limit"],
            "max_shots": max_shots,
            "fill_value": float(fill_value),
            "target_mode_requested": target_mode,
            "target_mode_resolved": prepared["resolved_target_mode"],
            "manifest_requested_family": family if manifest_data is not None else None,
        },
        split=split_summary.to_dict(),
        training={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "elapsed_seconds": elapsed,
            "history": history,
        },
        metrics=test_metrics,
        artifacts={
            "checkpoint_path": checkpoint_out.as_posix(),
            "train_json_path": train_json_out.as_posix() if train_json_out is not None else None,
        },
    )
    if train_json_out is not None:
        common._write_json(train_json_out, result.to_dict())
    return result


def run_manifest_experiment(
    *,
    manifest: Path,
    train_families: list[str],
    eval_families: list[str] | None,
    out_dir: Path,
    fill_value: float,
    target_mode: str,
    max_shots: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    num_blocks: int,
    dense_hidden_dim: int,
    device_arg: str,
) -> dict[str, Any]:
    common._require_torch()
    common._set_random_seeds(seed)

    manifest_data, train_entries_raw = _resolve_manifest_family_entries(manifest, train_families)
    _, eval_entries_raw = _resolve_manifest_family_entries(manifest, eval_families)

    prepared_train_entries = [
        _load_prepared_family_entry(
            family_dir=family_dir,
            fill_value=fill_value,
            max_shots=max_shots,
            target_mode=target_mode,
        )
        for _, family_dir in train_entries_raw
    ]
    compatibility = _validate_compatible_family_batches(prepared_train_entries)

    x_train_chunks: list[np.ndarray] = []
    y_train_chunks: list[np.ndarray] = []
    x_val_chunks: list[np.ndarray] = []
    y_val_chunks: list[np.ndarray] = []
    x_test_chunks: list[np.ndarray] = []
    y_test_chunks: list[np.ndarray] = []
    split_family_summaries: list[dict[str, Any]] = []

    total_loaded = 0
    total_train = 0
    total_val = 0
    total_test = 0
    for index, entry in enumerate(prepared_train_entries):
        num_shots = int(entry["y"].shape[0])
        family_seed = int(seed + index)
        split_indices = common.build_split_indices(
            num_shots=num_shots,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=family_seed,
        )
        split_summary = common.summarise_split_indices(
            split_indices,
            num_shots=num_shots,
            seed=family_seed,
        )

        x_train_chunks.append(common._subset_rows(entry["x"], split_indices["train"]))
        y_train_chunks.append(common._subset_rows(entry["y"], split_indices["train"]))
        x_val_chunks.append(common._subset_rows(entry["x"], split_indices["val"]))
        y_val_chunks.append(common._subset_rows(entry["y"], split_indices["val"]))
        x_test_chunks.append(common._subset_rows(entry["x"], split_indices["test"]))
        y_test_chunks.append(common._subset_rows(entry["y"], split_indices["test"]))

        metadata = entry["metadata"]
        total_loaded += num_shots
        total_train += split_summary.train_count
        total_val += split_summary.val_count
        total_test += split_summary.test_count
        split_family_summaries.append(
            {
                "family": metadata.get("family"),
                "stage": metadata.get("stage"),
                "family_dir": entry["family_dir"].as_posix(),
                "shots_total_after_limit": num_shots,
                "target_key_used": entry["target_key"],
                "resolved_target_mode": entry["resolved_target_mode"],
                "target_summary": entry["target_summary"],
                "split": split_summary.to_dict(),
                "bundle_info": entry["bundle_info"],
            }
        )

    x_train = np.concatenate(x_train_chunks, axis=0)
    y_train = np.concatenate(y_train_chunks, axis=0)
    x_val = np.concatenate(x_val_chunks, axis=0)
    y_val = np.concatenate(y_val_chunks, axis=0)
    x_test = np.concatenate(x_test_chunks, axis=0)
    y_test = np.concatenate(y_test_chunks, axis=0)

    trained = _train_model_from_splits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        device_arg=device_arg,
        task_type=compatibility["task_type"],
        num_classes=int(compatibility["num_classes"]),
        class_labels=list(compatibility["class_labels"]),
    )
    model = trained["model"]
    device = trained["device"]

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    train_json_path = out_dir / "train.json"
    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "research_noise_aware_3d",
        "created_at_utc": common._utc_now_iso(),
        "family": "multi_family_manifest",
        "stage": "mixed",
        "manifest_path": manifest.as_posix(),
        "train_families": [item["family"] for item in split_family_summaries],
        "train_family_dirs": [item["family_dir"] for item in split_family_summaries],
        "fill_value": float(fill_value),
        "model_hparams": {
            "in_channels": int(x_train.shape[1]),
            "hidden_channels": int(hidden_channels),
            "num_blocks": int(num_blocks),
            "dense_hidden_dim": int(dense_hidden_dim),
            "dropout": float(dropout),
            "task_type": compatibility["task_type"],
            "num_classes": int(compatibility["num_classes"]),
            "class_labels": list(compatibility["class_labels"]),
        },
        "threshold_selection": trained["threshold_selection"],
        "target_mode_requested": target_mode,
        "target_mode_resolved": compatibility["resolved_target_mode"],
        "target_key": prepared_train_entries[0]["target_key"],
        "split": {
            "train_count": int(total_train),
            "val_count": int(total_val),
            "test_count": int(total_test),
            "train_fraction": float(total_train / total_loaded),
            "val_fraction": float(total_val / total_loaded),
            "test_fraction": float(total_test / total_loaded),
            "seed": int(seed),
            "per_family": split_family_summaries,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    train_result = common.TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="research_noise_aware_3d",
        created_at_utc=common._utc_now_iso(),
        input_mode="manifest_multi_family",
        family="multi_family_manifest",
        stage="mixed",
        family_dir=manifest.as_posix(),
        model={
            "architecture": "noise_aware_volume_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(trained["num_parameters"]),
            "bundle_info": {
                "representation": "noise_aware_volume_3d",
                "shape": {
                    "batch": int(total_loaded),
                    "channels": int(x_train.shape[1]),
                    "time_steps": int(x_train.shape[2]),
                    "grid_height": int(x_train.shape[3]),
                    "grid_width": int(x_train.shape[4]),
                },
                "channel_names": compatibility["channel_names"],
                "layout_summary": compatibility["layout_summary"],
                "fill_value": float(fill_value),
            },
            "model_hparams": checkpoint["model_hparams"],
        },
        dataset={
            "manifest_path": manifest.as_posix(),
            "manifest_schema_version": manifest_data.get("schema_version"),
            "manifest_distance": manifest_data.get("distance"),
            "manifest_rounds": manifest_data.get("rounds"),
            "manifest_basis": manifest_data.get("basis"),
            "manifest_variant": manifest_data.get("variant"),
            "training_families": split_family_summaries,
            "evaluation_families_requested": [family for family, _ in eval_entries_raw],
            "num_shots_total_after_limit": int(total_loaded),
            "max_shots_per_family": max_shots,
            "fill_value": float(fill_value),
            "target_mode_requested": target_mode,
            "target_mode_resolved": compatibility["resolved_target_mode"],
            "target_key_used": prepared_train_entries[0]["target_key"],
        },
        split={
            "train_count": int(total_train),
            "val_count": int(total_val),
            "test_count": int(total_test),
            "train_fraction": float(total_train / total_loaded),
            "val_fraction": float(total_val / total_loaded),
            "test_fraction": float(total_test / total_loaded),
            "seed": int(seed),
            "per_family": split_family_summaries,
        },
        training={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "elapsed_seconds": float(trained["elapsed_seconds"]),
            "history": trained["history"],
        },
        metrics=trained["test_metrics"],
        artifacts={
            "checkpoint_path": checkpoint_path.as_posix(),
            "train_json_path": train_json_path.as_posix(),
        },
    )
    common._write_json(train_json_path, train_result.to_dict())

    eval_summaries: list[dict[str, Any]] = []
    eval_json_paths: dict[str, str] = {}
    train_family_names = {item["family"] for item in split_family_summaries}
    for family, _ in eval_entries_raw:
        eval_json_path = out_dir / f"eval__{family}.json"
        eval_result = evaluate_checkpoint_on_family(
            family_dir=None,
            manifest=manifest,
            family=family,
            checkpoint_path=checkpoint_path,
            eval_json_out=eval_json_path,
            target_mode=target_mode,
            max_shots=max_shots,
            batch_size=batch_size,
            device_arg=device_arg,
        )
        eval_dict = eval_result.to_dict()
        eval_json_paths[family] = eval_json_path.as_posix()
        eval_metrics = eval_dict["metrics"]
        eval_task_type = str(eval_dict["model"]["model_hparams"].get("task_type", "binary"))
        summary_entry = {
            "family": family,
            "stage": eval_dict["stage"],
            "seen_in_training": family in train_family_names,
            "task_type": eval_task_type,
            "accuracy": eval_metrics.get("accuracy"),
            "label_error_rate": eval_metrics.get("label_error_rate"),
            "balanced_accuracy": eval_metrics.get("balanced_accuracy"),
            "eval_json_path": eval_json_path.as_posix(),
        }
        if eval_task_type == "binary":
            summary_entry.update(
                {
                    "f1": eval_metrics.get("f1"),
                    "auroc": eval_metrics.get("auroc"),
                    "positive_rate_target": eval_metrics.get("positive_rate_target"),
                }
            )
        else:
            summary_entry.update(
                {
                    "macro_f1": eval_metrics.get("macro_f1"),
                    "target_class_histogram": eval_metrics.get("target_class_histogram"),
                }
            )
        eval_summaries.append(summary_entry)

    experiment_summary = {
        "schema_version": SCHEMA_VERSION_EXPERIMENT,
        "decoder": "research_noise_aware_3d",
        "created_at_utc": common._utc_now_iso(),
        "manifest": {
            "path": manifest.as_posix(),
            "schema_version": manifest_data.get("schema_version"),
            "families_available": list((manifest_data.get("family_dirs") or {}).keys()),
        },
        "training": {
            "train_families": [item["family"] for item in split_family_summaries],
            "train_stages": [item["stage"] for item in split_family_summaries],
            "task_type": compatibility["task_type"],
            "target_mode_requested": target_mode,
            "target_mode_resolved": compatibility["resolved_target_mode"],
            "train_json_path": train_json_path.as_posix(),
            "checkpoint_path": checkpoint_path.as_posix(),
            "mixed_test_split_metrics": train_result.metrics,
        },
        "evaluation": {
            "eval_families": [family for family, _ in eval_entries_raw],
            "holdout_families": [family for family, _ in eval_entries_raw if family not in train_family_names],
            "per_family": eval_summaries,
            "per_family_eval_json": eval_json_paths,
        },
    }
    common._write_json(out_dir / "experiment_summary.json", experiment_summary)
    return experiment_summary


def evaluate_checkpoint_on_family(
    *,
    family_dir: Path | None,
    manifest: Path | None,
    family: str | None,
    checkpoint_path: Path,
    eval_json_out: Path | None,
    target_mode: str,
    max_shots: int | None,
    batch_size: int,
    device_arg: str,
) -> common.EvalResult:
    common._require_torch()
    checkpoint_path = checkpoint_path.resolve()

    input_mode, resolved_family_dir, _ = common._resolve_input_family_dir(
        family_dir=family_dir,
        manifest=manifest,
        family=family,
    )
    payload = common._load_family_payload(resolved_family_dir)
    metadata = payload.metadata

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_hparams = dict(checkpoint["model_hparams"])
    fill_value = float(checkpoint.get("fill_value", -0.5))
    requested_target_mode = str(checkpoint.get("target_mode_requested", target_mode))
    prepared = _prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=requested_target_mode,
    )
    x = prepared["x"]
    y = prepared["y"]
    task_type = str(model_hparams.get("task_type", prepared["task_type"]))
    num_classes = int(model_hparams.get("num_classes", prepared["num_classes"]))
    class_labels = list(model_hparams.get("class_labels", prepared["class_labels"]))

    device = common._pick_device(device_arg)
    model = NoiseAwareVolumeDecoder(
        in_channels=int(model_hparams["in_channels"]),
        hidden_channels=int(model_hparams["hidden_channels"]),
        num_blocks=int(model_hparams["num_blocks"]),
        dense_hidden_dim=int(model_hparams["dense_hidden_dim"]),
        dropout=float(model_hparams["dropout"]),
        out_dim=(1 if task_type == "binary" else num_classes),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    threshold_selection = dict(checkpoint.get("threshold_selection", {}))
    threshold_raw = threshold_selection.get("selected_threshold", 0.5)
    threshold = float(threshold_raw) if threshold_raw is not None else None
    metrics = common._evaluate_arrays(
        model=model,
        x=x,
        y=y,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        threshold=threshold,
        class_labels=class_labels,
    )

    result = common.EvalResult(
        schema_version=SCHEMA_VERSION_EVAL,
        decoder="research_noise_aware_3d",
        created_at_utc=common._utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family")),
        stage=str(metadata.get("stage")),
        family_dir=resolved_family_dir.as_posix(),
        checkpoint={
            "path": checkpoint_path.as_posix(),
            "created_at_utc": checkpoint.get("created_at_utc"),
            "family": checkpoint.get("family"),
            "stage": checkpoint.get("stage"),
            "threshold_selection": threshold_selection,
        },
        model={
            "architecture": "noise_aware_volume_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "bundle_info": prepared["bundle_info"],
            "model_hparams": model_hparams,
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit"),
            "scaffold": metadata.get("scaffold"),
            "targets": metadata.get("targets"),
            "target_key_used": prepared.get("target_key"),
            "rectangular_syndrome_layout": metadata.get("rectangular_syndrome_layout"),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "num_shots_total_after_limit": prepared["shots_total_after_limit"],
            "max_shots": max_shots,
            "fill_value": fill_value,
            "target_mode_requested": requested_target_mode,
            "target_mode_resolved": prepared["resolved_target_mode"],
        },
        split={
            "train_count": None,
            "val_count": None,
            "test_count": None,
            "train_fraction": None,
            "val_fraction": None,
            "test_fraction": None,
            "seed": checkpoint.get("split", {}).get("seed"),
            "evaluated_split": "all_loaded_shots",
            "evaluated_count": int(x.shape[0]),
        },
        metrics=metrics,
    )
    if eval_json_out is not None:
        common._write_json(eval_json_out, result.to_dict())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the research noise-aware 3-D decoder."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    common._add_common_input_args(train_parser)
    train_parser.add_argument("--checkpoint-out", type=Path, required=True)
    train_parser.add_argument("--train-json-out", type=Path, default=None)
    train_parser.add_argument("--fill-value", type=float, default=-0.5)
    train_parser.add_argument("--target-mode", choices=list(common.TARGET_MODE_CHOICES), default=common.TARGET_MODE_AUTO)
    train_parser.add_argument("--max-shots", type=int, default=None)
    train_parser.add_argument("--train-ratio", type=float, default=0.8)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=12345)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--hidden-channels", type=int, default=32)
    train_parser.add_argument("--num-blocks", type=int, default=3)
    train_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval")
    common._add_common_input_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--eval-json-out", type=Path, default=None)
    eval_parser.add_argument("--target-mode", choices=list(common.TARGET_MODE_CHOICES), default=common.TARGET_MODE_AUTO)
    eval_parser.add_argument("--max-shots", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--device", type=str, default="auto")

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument("--manifest", type=Path, required=True)
    experiment_parser.add_argument("--train-families", nargs="+", required=True)
    experiment_parser.add_argument("--eval-families", nargs="+", default=None)
    experiment_parser.add_argument("--out-dir", type=Path, required=True)
    experiment_parser.add_argument("--fill-value", type=float, default=-0.5)
    experiment_parser.add_argument("--target-mode", choices=list(common.TARGET_MODE_CHOICES), default=common.TARGET_MODE_AUTO)
    experiment_parser.add_argument("--max-shots", type=int, default=None)
    experiment_parser.add_argument("--train-ratio", type=float, default=0.8)
    experiment_parser.add_argument("--val-ratio", type=float, default=0.1)
    experiment_parser.add_argument("--test-ratio", type=float, default=0.1)
    experiment_parser.add_argument("--seed", type=int, default=12345)
    experiment_parser.add_argument("--epochs", type=int, default=20)
    experiment_parser.add_argument("--batch-size", type=int, default=128)
    experiment_parser.add_argument("--lr", type=float, default=1e-3)
    experiment_parser.add_argument("--weight-decay", type=float, default=1e-4)
    experiment_parser.add_argument("--dropout", type=float, default=0.1)
    experiment_parser.add_argument("--hidden-channels", type=int, default=32)
    experiment_parser.add_argument("--num-blocks", type=int, default=3)
    experiment_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    experiment_parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        result = train_family_dir(
            family_dir=args.family_dir,
            manifest=args.manifest,
            family=args.family,
            checkpoint_out=args.checkpoint_out,
            train_json_out=args.train_json_out,
            fill_value=args.fill_value,
            target_mode=args.target_mode,
            max_shots=args.max_shots,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            dense_hidden_dim=args.dense_hidden_dim,
            device_arg=args.device,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=common._json_default))
        return

    if args.mode == "eval":
        result = evaluate_checkpoint_on_family(
            family_dir=args.family_dir,
            manifest=args.manifest,
            family=args.family,
            checkpoint_path=args.checkpoint,
            eval_json_out=args.eval_json_out,
            target_mode=args.target_mode,
            max_shots=args.max_shots,
            batch_size=args.batch_size,
            device_arg=args.device,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=common._json_default))
        return

    if args.mode == "experiment":
        result = run_manifest_experiment(
            manifest=args.manifest,
            train_families=list(args.train_families),
            eval_families=(list(args.eval_families) if args.eval_families is not None else None),
            out_dir=args.out_dir,
            fill_value=args.fill_value,
            target_mode=args.target_mode,
            max_shots=args.max_shots,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            dense_hidden_dim=args.dense_hidden_dim,
            device_arg=args.device,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=common._json_default))
        return

    raise AssertionError(f"Unhandled mode: {args.mode!r}")


if __name__ == "__main__":
    main()
