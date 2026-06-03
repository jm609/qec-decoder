from __future__ import annotations

"""
factorized_logical_frame_decoder.py

First non-baseline decoder candidate for the rebuilt project.
"""

from pathlib import Path
from typing import Any
import argparse
import copy
import json
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None
    F = None
    optim = None

try:
    import baseline_rectcnn as common
    import research_noise_aware_3d as volume_common
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_rectcnn as common
    import research_noise_aware_3d as volume_common


SCHEMA_VERSION_TRAIN = "factorized_logical_frame_decoder.train.v1"
SCHEMA_VERSION_EVAL = "factorized_logical_frame_decoder.eval.v1"
SCHEMA_VERSION_EXPERIMENT = "factorized_logical_frame_decoder.experiment.v1"
MIN_TEMPERATURE_CALIBRATION_EXAMPLES = 16
MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES = 2
IMBALANCE_MODE_NONE = "none"
IMBALANCE_MODE_BALANCED = "balanced"
IMBALANCE_MODE_TEMPERED = "tempered"
IMBALANCE_MODE_CHOICES = (IMBALANCE_MODE_NONE, IMBALANCE_MODE_BALANCED, IMBALANCE_MODE_TEMPERED)
MAIN_CLASS4_LOSS_CROSS_ENTROPY = "cross_entropy"
MAIN_CLASS4_LOSS_FOCAL = "focal"
MAIN_CLASS4_LOSS_CHOICES = (MAIN_CLASS4_LOSS_CROSS_ENTROPY, MAIN_CLASS4_LOSS_FOCAL)
MAX_CLASS_WEIGHT = 8.0
MAX_POS_WEIGHT = 16.0
TEMPERED_CLASS_WEIGHT_EXPONENT = 0.5
TEMPERED_POS_WEIGHT_EXPONENT = 0.5
TEMPERED_MAIN_SAMPLE_EXPONENT = 0.5
DEFAULT_FOCAL_GAMMA = 2.0

SIGNAL_CHANNEL_NAMES = [
    "event",
    "valid_mask",
    "boundary_flag",
    "final_round_flag",
    "checkerboard_class_0",
    "checkerboard_class_1",
    "is_x_check",
    "is_z_check",
]


def _infer_signal_channel_count(bundle_info: dict[str, Any]) -> int:
    channel_names = list(bundle_info.get("channel_names", []))
    if channel_names[: len(SIGNAL_CHANNEL_NAMES)] != SIGNAL_CHANNEL_NAMES:
        raise ValueError(
            "Unexpected channel layout for factorized logical-frame decoder. "
            f"Expected prefix {SIGNAL_CHANNEL_NAMES}, got {channel_names[:len(SIGNAL_CHANNEL_NAMES)]}."
        )
    return len(SIGNAL_CHANNEL_NAMES)


def _slice_rows(arr: np.ndarray, max_shots: int | None) -> np.ndarray:
    if max_shots is None:
        return np.asarray(arr)
    return np.asarray(arr[:max_shots])


def _require_class4_targets(
    payload: common.LoadedFamilyPayload,
    *,
    max_shots: int | None,
) -> dict[str, np.ndarray]:
    arrays = payload.arrays
    required = ("logical_class4", "logical_x_flip", "logical_z_flip")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(
            "factorized_logical_frame_decoder requires class4 arrays, missing "
            f"{missing}"
        )
    y_class4 = common._as_uint8_1d(arrays["logical_class4"], name="logical_class4").astype(np.int64, copy=False)
    y_x = common._as_uint8_1d(arrays["logical_x_flip"], name="logical_x_flip").astype(np.float32, copy=False)
    y_z = common._as_uint8_1d(arrays["logical_z_flip"], name="logical_z_flip").astype(np.float32, copy=False)
    return {
        "logical_class4": _slice_rows(y_class4, max_shots).astype(np.int64, copy=False),
        "logical_x_flip": _slice_rows(y_x, max_shots).astype(np.float32, copy=False),
        "logical_z_flip": _slice_rows(y_z, max_shots).astype(np.float32, copy=False),
    }


def _prepare_class4_family(
    payload: common.LoadedFamilyPayload,
    *,
    fill_value: float,
    max_shots: int | None,
) -> dict[str, Any]:
    prepared = volume_common._prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=common.TARGET_MODE_LOGICAL_CLASS4,
    )
    targets = _require_class4_targets(payload, max_shots=max_shots)
    x = np.asarray(prepared["x"], dtype=np.float32)
    signal_channels = _infer_signal_channel_count(prepared["bundle_info"])
    if x.shape[0] != targets["logical_class4"].shape[0]:
        raise ValueError(
            "Class4 feature/label mismatch: "
            f"{x.shape[0]} vs {targets['logical_class4'].shape[0]}"
        )
    return {
        "x": np.ascontiguousarray(x),
        "y_class4": np.ascontiguousarray(targets["logical_class4"]),
        "y_x": np.ascontiguousarray(targets["logical_x_flip"]),
        "y_z": np.ascontiguousarray(targets["logical_z_flip"]),
        "bundle_info": prepared["bundle_info"],
        "shots_total_after_limit": int(prepared["shots_total_after_limit"]),
        "target_key": prepared["target_key"],
        "class_labels": list(prepared["class_labels"]),
        "resolved_target_mode": prepared["resolved_target_mode"],
        "signal_channels": int(signal_channels),
        "context_channels": int(x.shape[1] - signal_channels),
    }


def _prepare_axis_family(
    payload: common.LoadedFamilyPayload,
    *,
    fill_value: float,
    max_shots: int | None,
) -> dict[str, Any]:
    prepared = volume_common._prepare_loaded_family(
        payload,
        fill_value=fill_value,
        max_shots=max_shots,
        target_mode=common.TARGET_MODE_BINARY,
    )
    metadata = payload.metadata
    targets_meta = metadata.get("targets", {}) if isinstance(metadata, dict) else {}
    axis_name = str(targets_meta.get("logical_axis_flip_name", prepared["target_key"]))
    x = np.asarray(prepared["x"], dtype=np.float32)
    y = np.asarray(prepared["y"], dtype=np.float32).reshape(-1)
    signal_channels = _infer_signal_channel_count(prepared["bundle_info"])
    if axis_name not in {"logical_x_flip", "logical_z_flip"}:
        raise ValueError(f"Unsupported auxiliary axis target: {axis_name!r}")
    return {
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "axis_name": axis_name,
        "bundle_info": prepared["bundle_info"],
        "shots_total_after_limit": int(prepared["shots_total_after_limit"]),
        "resolved_target_mode": prepared["resolved_target_mode"],
        "signal_channels": int(signal_channels),
        "context_channels": int(x.shape[1] - signal_channels),
    }


def _make_main_tensor_dataset(
    x: np.ndarray,
    y_class4: np.ndarray,
    y_x: np.ndarray,
    y_z: np.ndarray,
) -> Any:
    return common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(y_class4, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(y_x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_z, dtype=np.float32)),
    )


def _class_histogram(labels: np.ndarray, *, class_labels: list[str]) -> dict[str, int]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(target, minlength=len(class_labels))
    return {str(class_labels[index]): int(counts[index]) for index in range(len(class_labels))}


def _compute_class_weight_vector(
    labels: np.ndarray,
    *,
    num_classes: int,
    max_weight: float = MAX_CLASS_WEIGHT,
) -> np.ndarray:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(target, minlength=num_classes).astype(np.float64, copy=False)
    weights = np.ones(num_classes, dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        return weights
    inv = np.zeros(num_classes, dtype=np.float64)
    inv[present] = float(np.sum(counts[present])) / (float(np.sum(present)) * counts[present])
    inv[present] = np.clip(inv[present], 1.0 / float(max_weight), float(max_weight))
    mean_present = float(np.mean(inv[present]))
    if mean_present > 0:
        inv[present] = inv[present] / mean_present
    weights = inv.astype(np.float32, copy=False)
    weights[~present] = 1.0
    return weights


def _temper_weight_vector(weights: np.ndarray, *, exponent: float) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.size == 0 or abs(float(exponent) - 1.0) <= 1e-9:
        return np.asarray(arr, dtype=np.float32)
    tempered = np.power(np.clip(arr, 1e-8, None), float(exponent)).astype(np.float32, copy=False)
    mean_weight = float(tempered.mean())
    if mean_weight > 0.0:
        tempered = tempered / mean_weight
    return np.asarray(tempered, dtype=np.float32)


def _compute_binary_pos_weight(
    labels: np.ndarray,
    *,
    max_weight: float = MAX_POS_WEIGHT,
) -> float:
    target = np.asarray(labels, dtype=np.float32).reshape(-1)
    positives = float(np.sum(target >= 0.5))
    negatives = float(target.shape[0] - positives)
    if positives <= 0.0 or negatives <= 0.0:
        return 1.0
    return float(np.clip(negatives / positives, 1.0, float(max_weight)))


def _compute_multiclass_sample_weights(
    labels: np.ndarray,
    *,
    num_classes: int,
    max_weight: float = MAX_CLASS_WEIGHT,
    exponent: float = 1.0,
) -> np.ndarray:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if target.size == 0:
        return np.zeros((0,), dtype=np.float32)
    class_weights = _compute_class_weight_vector(target, num_classes=num_classes, max_weight=max_weight)
    sample_weights = _temper_weight_vector(class_weights, exponent=float(exponent))[target]
    mean_weight = float(sample_weights.mean())
    if mean_weight > 0.0:
        sample_weights = sample_weights / mean_weight
    return np.asarray(sample_weights, dtype=np.float32)


def _compute_binary_sample_weights(
    labels: np.ndarray,
    *,
    max_weight: float = MAX_POS_WEIGHT,
    exponent: float = 1.0,
) -> np.ndarray:
    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if target.size == 0:
        return np.zeros((0,), dtype=np.float32)
    positives = int(target.sum())
    negatives = int(target.shape[0] - positives)
    if positives <= 0 or negatives <= 0:
        return np.ones(target.shape[0], dtype=np.float32)
    pos_weight = float(np.clip(float(negatives / positives), 1.0, float(max_weight)))
    sample_weights = np.where(target == 1, pos_weight, 1.0).astype(np.float32, copy=False)
    sample_weights = _temper_weight_vector(sample_weights, exponent=float(exponent))
    mean_weight = float(sample_weights.mean())
    if mean_weight > 0.0:
        sample_weights = sample_weights / mean_weight
    return np.asarray(sample_weights, dtype=np.float32)


def _make_weighted_loader(
    dataset: Any,
    *,
    sample_weights: np.ndarray,
    batch_size: int,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required to build weighted loaders")
    weights = torch.as_tensor(np.asarray(sample_weights, dtype=np.float64))
    if weights.ndim != 1:
        raise ValueError(f"sample_weights must be 1D, got shape {tuple(weights.shape)}")
    if int(weights.shape[0]) != len(dataset):
        raise ValueError(
            "sample_weights length does not match dataset length: "
            f"{int(weights.shape[0])} vs {len(dataset)}"
        )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=int(weights.shape[0]),
        replacement=True,
    )
    return common.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def _build_reweighting_config(
    *,
    y4_train: np.ndarray,
    yx_train: np.ndarray,
    yz_train: np.ndarray,
    aux_by_axis: dict[str, dict[str, Any]],
    class_labels: list[str],
    imbalance_mode: str,
) -> dict[str, Any]:
    if imbalance_mode == IMBALANCE_MODE_NONE:
        return {
            "imbalance_mode": imbalance_mode,
            "class_weight_vector": None,
            "non_identity_pos_weight": 1.0,
            "confidence_error_pos_weight": 1.0,
            "bit_x_pos_weight": 1.0,
            "bit_z_pos_weight": 1.0,
        }
    if imbalance_mode not in {IMBALANCE_MODE_BALANCED, IMBALANCE_MODE_TEMPERED}:
        raise ValueError(f"Unsupported imbalance_mode: {imbalance_mode!r}")

    yx_sources = [np.asarray(yx_train, dtype=np.float32).reshape(-1)]
    yz_sources = [np.asarray(yz_train, dtype=np.float32).reshape(-1)]
    if "logical_x_flip" in aux_by_axis:
        yx_sources.append(np.asarray(aux_by_axis["logical_x_flip"]["y"], dtype=np.float32).reshape(-1))
    if "logical_z_flip" in aux_by_axis:
        yz_sources.append(np.asarray(aux_by_axis["logical_z_flip"]["y"], dtype=np.float32).reshape(-1))
    class_weight_vector = _compute_class_weight_vector(
        y4_train,
        num_classes=len(class_labels),
        max_weight=MAX_CLASS_WEIGHT,
    )
    bit_x_pos_weight = _compute_binary_pos_weight(np.concatenate(yx_sources, axis=0), max_weight=MAX_POS_WEIGHT)
    bit_z_pos_weight = _compute_binary_pos_weight(np.concatenate(yz_sources, axis=0), max_weight=MAX_POS_WEIGHT)
    non_identity_pos_weight = _compute_binary_pos_weight(
        np.asarray(y4_train != 0, dtype=np.float32),
        max_weight=MAX_POS_WEIGHT,
    )
    confidence_error_pos_weight = float(non_identity_pos_weight)
    class_weight_exponent = 1.0
    pos_weight_exponent = 1.0
    if imbalance_mode == IMBALANCE_MODE_TEMPERED:
        class_weight_exponent = float(TEMPERED_CLASS_WEIGHT_EXPONENT)
        pos_weight_exponent = float(TEMPERED_POS_WEIGHT_EXPONENT)
    class_weight_vector = _temper_weight_vector(class_weight_vector, exponent=class_weight_exponent)
    non_identity_pos_weight = float(max(1.0, non_identity_pos_weight ** pos_weight_exponent))
    confidence_error_pos_weight = float(max(1.0, confidence_error_pos_weight ** pos_weight_exponent))
    bit_x_pos_weight = float(max(1.0, bit_x_pos_weight ** pos_weight_exponent))
    bit_z_pos_weight = float(max(1.0, bit_z_pos_weight ** pos_weight_exponent))
    return {
        "imbalance_mode": imbalance_mode,
        "class_weight_vector": class_weight_vector.tolist(),
        "non_identity_pos_weight": float(non_identity_pos_weight),
        "confidence_error_pos_weight": float(confidence_error_pos_weight),
        "bit_x_pos_weight": float(bit_x_pos_weight),
        "bit_z_pos_weight": float(bit_z_pos_weight),
    }


def _build_sampling_config(
    *,
    y4_train: np.ndarray,
    aux_by_axis: dict[str, dict[str, Any]],
    class_labels: list[str],
    imbalance_mode: str,
) -> dict[str, Any]:
    if imbalance_mode == IMBALANCE_MODE_NONE:
        return {
            "imbalance_mode": imbalance_mode,
            "main_sampler": "shuffle",
            "main_class_histogram": _class_histogram(y4_train, class_labels=class_labels),
            "main_sample_weight_stats": None,
            "auxiliary_axes": {},
        }
    if imbalance_mode not in {IMBALANCE_MODE_BALANCED, IMBALANCE_MODE_TEMPERED}:
        raise ValueError(f"Unsupported imbalance_mode: {imbalance_mode!r}")

    if imbalance_mode == IMBALANCE_MODE_BALANCED:
        main_sample_weights = _compute_multiclass_sample_weights(y4_train, num_classes=len(class_labels))
        main_sampler = "weighted_random_sampler"
        aux_sampler = "weighted_random_sampler"
        aux_sample_weight_exponent = 1.0
    else:
        main_sample_weights = _compute_multiclass_sample_weights(
            y4_train,
            num_classes=len(class_labels),
            exponent=TEMPERED_MAIN_SAMPLE_EXPONENT,
        )
        main_sampler = "tempered_weighted_random_sampler"
        aux_sampler = "shuffle"
        aux_sample_weight_exponent = float(TEMPERED_POS_WEIGHT_EXPONENT)
    auxiliary_axes: dict[str, Any] = {}
    for axis_name, aux in aux_by_axis.items():
        axis_labels = np.asarray(aux["y"], dtype=np.uint8)
        axis_weights = _compute_binary_sample_weights(axis_labels, exponent=aux_sample_weight_exponent)
        auxiliary_axes[axis_name] = {
            "sampler": aux_sampler,
            "target_histogram": {
                "0": int(axis_labels.shape[0] - int(axis_labels.sum())),
                "1": int(axis_labels.sum()),
            },
            "sample_weight_stats": {
                "min": float(axis_weights.min()) if axis_weights.size else None,
                "max": float(axis_weights.max()) if axis_weights.size else None,
                "mean": float(axis_weights.mean()) if axis_weights.size else None,
            },
        }
    return {
        "imbalance_mode": imbalance_mode,
        "main_sampler": main_sampler,
        "main_class_histogram": _class_histogram(y4_train, class_labels=class_labels),
        "main_sample_weight_stats": {
            "min": float(main_sample_weights.min()) if main_sample_weights.size else None,
            "max": float(main_sample_weights.max()) if main_sample_weights.size else None,
            "mean": float(main_sample_weights.mean()) if main_sample_weights.size else None,
        },
        "auxiliary_axes": auxiliary_axes,
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


def _validate_main_compatible(prepared_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not prepared_entries:
        raise ValueError("At least one prepared main family entry is required")
    reference_shape = tuple(int(v) for v in prepared_entries[0]["x"].shape[1:])
    reference_channels = list(prepared_entries[0]["bundle_info"]["channel_names"])
    reference_labels = list(prepared_entries[0]["class_labels"])
    reference_signal_channels = int(prepared_entries[0]["signal_channels"])
    reference_context_channels = int(prepared_entries[0]["context_channels"])
    for entry in prepared_entries[1:]:
        if tuple(int(v) for v in entry["x"].shape[1:]) != reference_shape:
            raise ValueError("All class4 training families must share the same input tensor shape.")
        if list(entry["bundle_info"]["channel_names"]) != reference_channels:
            raise ValueError("All class4 training families must share the same channel layout.")
        if list(entry["class_labels"]) != reference_labels:
            raise ValueError("All class4 training families must share the same class label ordering.")
        if int(entry["signal_channels"]) != reference_signal_channels:
            raise ValueError("Signal-channel counts must match across class4 training families.")
        if int(entry["context_channels"]) != reference_context_channels:
            raise ValueError("Context-channel counts must match across class4 training families.")
    return {
        "input_shape_without_batch": list(reference_shape),
        "channel_names": reference_channels,
        "class_labels": reference_labels,
        "signal_channels": reference_signal_channels,
        "context_channels": reference_context_channels,
        "layout_summary": prepared_entries[0]["bundle_info"]["layout_summary"],
    }


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        groups = 4 if channels % 4 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv_spatial = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv_temporal = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.dropout = nn.Dropout3d(dropout)
        self.relu = nn.ReLU()

    def _film(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]

    def forward(self, x: torch.Tensor, *, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self._film(self.norm1(x), gamma, beta)
        h = self.conv_spatial(self.relu(h))
        h = self.dropout(h)
        h = self._film(self.norm2(h), gamma, beta)
        h = self.conv_temporal(self.relu(h))
        return x + h


class FactorizedLogicalFrameDecoder(nn.Module):
    def __init__(
        self,
        *,
        signal_channels: int,
        context_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dense_hidden_dim: int,
        dropout: float,
        context_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.signal_channels = int(signal_channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.valid_mask_channel_index = 1

        self.stem = nn.Conv3d(self.signal_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_channels, dropout=dropout) for _ in range(num_blocks)]
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_channels, context_hidden_dim),
            nn.ReLU(),
            nn.Linear(context_hidden_dim, num_blocks * hidden_channels * 2),
        )
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.head_norm = nn.GroupNorm(groups, hidden_channels)
        self.head_relu = nn.ReLU()
        self.head_drop = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_channels, dense_hidden_dim)
        self.x_head = nn.Linear(dense_hidden_dim, 1)
        self.z_head = nn.Linear(dense_hidden_dim, 1)
        self.non_identity_head = nn.Linear(dense_hidden_dim, 1)
        self.error_head = nn.Linear(dense_hidden_dim, 1)
        self.residual_head = nn.Linear(dense_hidden_dim, 4)

    def _masked_mean(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(dtype=h.dtype)
        denom = weights.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return (h * weights).sum(dim=(2, 3, 4)) / denom

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        signal = x[:, : self.signal_channels, :, :, :]
        context_vec = x[:, self.signal_channels :, 0, 0, 0]
        valid_mask = signal[:, self.valid_mask_channel_index : self.valid_mask_channel_index + 1, :, :, :]

        h = self.stem(signal)
        film = self.context_mlp(context_vec).view(
            x.shape[0],
            self.num_blocks,
            2,
            self.hidden_channels,
        )
        for block_index, block in enumerate(self.blocks):
            h = block(h, gamma=film[:, block_index, 0, :], beta=film[:, block_index, 1, :])

        h = self.head_relu(self.head_norm(h))
        pooled = self._masked_mean(h, valid_mask)
        shared = self.head_drop(self.head_relu(self.shared_fc(pooled)))

        x_logits = self.x_head(shared).squeeze(1)
        z_logits = self.z_head(shared).squeeze(1)
        non_identity_logits = self.non_identity_head(shared).squeeze(1)
        error_logits = self.error_head(shared).squeeze(1)
        residual_logits = self.residual_head(shared)
        zeros = torch.zeros_like(x_logits)
        base_logits = torch.stack([zeros, x_logits, z_logits, x_logits + z_logits], dim=1)
        class4_logits = base_logits + residual_logits
        return {
            "x_logits": x_logits,
            "z_logits": z_logits,
            "non_identity_logits": non_identity_logits,
            "error_logits": error_logits,
            "base_class4_logits": base_logits,
            "residual_logits": residual_logits,
            "class4_logits": class4_logits,
        }


def _read_dual_axis_manifest(dual_axis_manifest_path: Path) -> dict[str, Any]:
    manifest = common._read_json(dual_axis_manifest_path)
    schema_version = str(manifest.get("schema_version"))
    if schema_version != "dual_axis_manifest.v1":
        raise ValueError(
            f"Expected dual_axis_manifest.v1, got {schema_version!r} for {dual_axis_manifest_path}"
        )
    family_pairs = manifest.get("family_pairs")
    if not isinstance(family_pairs, dict) or not family_pairs:
        raise ValueError(f"dual-axis manifest has no family_pairs: {dual_axis_manifest_path}")
    return manifest


def _load_auxiliary_axis_entries(
    *,
    dual_axis_manifest_path: Path,
    families: list[str],
    fill_value: float,
    max_shots: int | None,
) -> dict[str, Any]:
    manifest = _read_dual_axis_manifest(dual_axis_manifest_path)
    family_pairs = manifest["family_pairs"]
    aux_entries: list[dict[str, Any]] = []
    for family in families:
        if family not in family_pairs:
            raise KeyError(
                f"Requested auxiliary family {family!r} missing from dual-axis manifest. "
                f"Available: {sorted(family_pairs)}"
            )
        pair = family_pairs[family]
        for source_key in ("logical_x_source", "logical_z_source"):
            source = pair.get(source_key)
            if not isinstance(source, dict):
                raise ValueError(f"Missing {source_key!r} entry for family={family!r}")
            family_dir = Path(str(source["family_dir"]))
            payload = common._load_family_payload(family_dir)
            prepared = _prepare_axis_family(
                payload,
                fill_value=fill_value,
                max_shots=max_shots,
            )
            prepared["family"] = family
            prepared["stage"] = payload.metadata.get("stage")
            prepared["family_dir"] = family_dir.as_posix()
            aux_entries.append(prepared)

    by_axis: dict[str, dict[str, Any]] = {}
    by_axis_summary: dict[str, dict[str, Any]] = {}
    for axis_name in ("logical_x_flip", "logical_z_flip"):
        matching = [entry for entry in aux_entries if entry["axis_name"] == axis_name]
        if not matching:
            continue
        reference_shape = tuple(int(v) for v in matching[0]["x"].shape[1:])
        reference_channels = list(matching[0]["bundle_info"]["channel_names"])
        reference_signal_channels = int(matching[0]["signal_channels"])
        reference_context_channels = int(matching[0]["context_channels"])
        for entry in matching[1:]:
            if tuple(int(v) for v in entry["x"].shape[1:]) != reference_shape:
                raise ValueError(f"Auxiliary axis {axis_name} has inconsistent tensor shapes.")
            if list(entry["bundle_info"]["channel_names"]) != reference_channels:
                raise ValueError(f"Auxiliary axis {axis_name} has inconsistent channel layouts.")
            if int(entry["signal_channels"]) != reference_signal_channels:
                raise ValueError(f"Auxiliary axis {axis_name} has inconsistent signal-channel counts.")
            if int(entry["context_channels"]) != reference_context_channels:
                raise ValueError(f"Auxiliary axis {axis_name} has inconsistent context-channel counts.")

        x = np.concatenate([entry["x"] for entry in matching], axis=0)
        y = np.concatenate([entry["y"] for entry in matching], axis=0)
        by_axis[axis_name] = {
            "x": np.ascontiguousarray(x),
            "y": np.ascontiguousarray(y.astype(np.float32, copy=False)),
            "channel_names": reference_channels,
            "signal_channels": reference_signal_channels,
            "context_channels": reference_context_channels,
            "families": sorted({str(entry["family"]) for entry in matching}),
            "shots_total_after_limit": int(y.shape[0]),
        }
        label_hist = np.bincount(y.astype(np.int64, copy=False), minlength=2)
        by_axis_summary[axis_name] = {
            "families": sorted({str(entry["family"]) for entry in matching}),
            "shots_total_after_limit": int(y.shape[0]),
            "signal_channels": int(reference_signal_channels),
            "context_channels": int(reference_context_channels),
            "input_shape_without_batch": list(reference_shape),
            "target_histogram": {
                "0": int(label_hist[0]),
                "1": int(label_hist[1]),
            },
        }

    return {
        "dual_axis_manifest_path": dual_axis_manifest_path.as_posix(),
        "families": list(families),
        "by_axis": by_axis,
        "summary": {
            "dual_axis_manifest_path": dual_axis_manifest_path.as_posix(),
            "families": list(families),
            "axes": by_axis_summary,
        },
    }


def _compute_main_batch_loss(
    outputs: dict[str, torch.Tensor],
    *,
    y_class4: torch.Tensor,
    y_x: torch.Tensor,
    y_z: torch.Tensor,
    main_axis_loss_weight: float,
    non_identity_loss_weight: float = 0.0,
    confidence_loss_weight: float = 0.0,
    main_class4_loss: str = MAIN_CLASS4_LOSS_CROSS_ENTROPY,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    class_weight: torch.Tensor | None = None,
    non_identity_pos_weight: torch.Tensor | None = None,
    confidence_error_pos_weight: torch.Tensor | None = None,
    bit_x_pos_weight: torch.Tensor | None = None,
    bit_z_pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    y_non_identity = (y_class4 != 0).to(dtype=outputs["class4_logits"].dtype)
    y_error = (outputs["class4_logits"].detach().argmax(dim=1) != y_class4).to(dtype=outputs["class4_logits"].dtype)
    if main_class4_loss == MAIN_CLASS4_LOSS_CROSS_ENTROPY:
        ce_loss = F.cross_entropy(outputs["class4_logits"], y_class4, weight=class_weight)
    elif main_class4_loss == MAIN_CLASS4_LOSS_FOCAL:
        log_probs = F.log_softmax(outputs["class4_logits"], dim=1)
        target_log_probs = log_probs.gather(1, y_class4.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()
        focal_factor = torch.pow(1.0 - target_probs, float(focal_gamma))
        alpha = 1.0
        if class_weight is not None:
            alpha = class_weight.gather(0, y_class4)
        ce_loss = -(alpha * focal_factor * target_log_probs).mean()
    else:
        raise ValueError(f"Unsupported main_class4_loss: {main_class4_loss!r}")
    non_identity_loss = F.binary_cross_entropy_with_logits(
        outputs["non_identity_logits"],
        y_non_identity,
        pos_weight=non_identity_pos_weight,
    )
    confidence_error_loss = F.binary_cross_entropy_with_logits(
        outputs["error_logits"],
        y_error,
        pos_weight=confidence_error_pos_weight,
    )
    bit_x_loss = F.binary_cross_entropy_with_logits(outputs["x_logits"], y_x, pos_weight=bit_x_pos_weight)
    bit_z_loss = F.binary_cross_entropy_with_logits(outputs["z_logits"], y_z, pos_weight=bit_z_pos_weight)
    total_loss = (
        ce_loss
        + float(non_identity_loss_weight) * non_identity_loss
        + float(confidence_loss_weight) * confidence_error_loss
        + float(main_axis_loss_weight) * (bit_x_loss + bit_z_loss)
    )
    return {
        "total_loss": total_loss,
        "class4_loss": ce_loss,
        "class4_loss_name": str(main_class4_loss),
        "class4_cross_entropy_loss": ce_loss,
        "non_identity_bce_loss": non_identity_loss,
        "confidence_error_bce_loss": confidence_error_loss,
        "bit_x_bce_loss": bit_x_loss,
        "bit_z_bce_loss": bit_z_loss,
    }


def _compute_axis_loss(
    outputs: dict[str, torch.Tensor],
    *,
    axis_name: str,
    target: torch.Tensor,
    aux_loss_weight: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if axis_name == "logical_x_flip":
        logits = outputs["x_logits"]
    elif axis_name == "logical_z_flip":
        logits = outputs["z_logits"]
    else:
        raise ValueError(f"Unsupported axis_name: {axis_name!r}")
    return float(aux_loss_weight) * F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _cross_entropy_from_probs(probs_np: np.ndarray, target_np: np.ndarray) -> float | None:
    probs = np.asarray(probs_np, dtype=np.float64)
    target = np.asarray(target_np, dtype=np.int64).reshape(-1)
    if probs.size == 0 or target.size == 0:
        return None
    chosen = probs[np.arange(target.shape[0]), target]
    chosen = np.clip(chosen, 1e-12, 1.0)
    return float(-np.mean(np.log(chosen)))


def _multiclass_brier_score(probs_np: np.ndarray, target_np: np.ndarray) -> float | None:
    probs = np.asarray(probs_np, dtype=np.float64)
    target = np.asarray(target_np, dtype=np.int64).reshape(-1)
    if probs.size == 0 or target.size == 0:
        return None
    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(target.shape[0]), target] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _multiclass_calibration_metrics(
    probs_np: np.ndarray,
    target_np: np.ndarray,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    probs = np.asarray(probs_np, dtype=np.float64)
    target = np.asarray(target_np, dtype=np.int64).reshape(-1)
    num_examples = int(target.shape[0])
    if probs.size == 0 or num_examples == 0:
        return {
            "num_bins": int(num_bins),
            "ece": None,
            "mce": None,
            "brier_score": None,
            "nll": None,
            "reliability_bins": [],
        }

    pred = np.argmax(probs, axis=1).astype(np.int64, copy=False)
    confidence = np.max(probs, axis=1)
    correctness = (pred == target).astype(np.float64, copy=False)
    bin_edges = np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)
    ece = 0.0
    mce = 0.0
    reliability_bins: list[dict[str, Any]] = []
    for bin_index in range(int(num_bins)):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        if bin_index == int(num_bins) - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(np.sum(mask))
        if count:
            bin_accuracy = float(np.mean(correctness[mask]))
            bin_confidence = float(np.mean(confidence[mask]))
            gap = abs(bin_accuracy - bin_confidence)
            ece += float(count / num_examples) * gap
            mce = max(mce, gap)
        else:
            bin_accuracy = None
            bin_confidence = None
            gap = None
        reliability_bins.append(
            {
                "bin_index": bin_index,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
                "gap": gap,
            }
        )

    return {
        "num_bins": int(num_bins),
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": _multiclass_brier_score(probs, target),
        "nll": _cross_entropy_from_probs(probs, target),
        "reliability_bins": reliability_bins,
    }


def _apply_temperature_to_logits(logits_np: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    return np.asarray(logits_np, dtype=np.float32) / safe_temperature


def _fit_temperature_from_logits(
    logits_np: np.ndarray,
    target_np: np.ndarray,
) -> dict[str, Any]:
    logits = np.asarray(logits_np, dtype=np.float32)
    target = np.asarray(target_np, dtype=np.int64).reshape(-1)
    unique_classes = int(np.unique(target).shape[0]) if target.size else 0
    raw_probs = common._softmax_np(logits) if logits.size else np.zeros((0, 4), dtype=np.float32)
    raw_nll = _cross_entropy_from_probs(raw_probs, target)
    if logits.size == 0 or target.size == 0:
        return {
            "method": "temperature_scaling_skipped",
            "temperature": 1.0,
            "raw_nll": raw_nll,
            "calibrated_nll": raw_nll,
            "num_examples": int(target.size),
            "num_unique_classes": unique_classes,
            "reason": "no_validation_examples",
            "guardrails": {
                "min_examples": int(MIN_TEMPERATURE_CALIBRATION_EXAMPLES),
                "min_unique_classes": int(MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES),
            },
        }

    if target.shape[0] < MIN_TEMPERATURE_CALIBRATION_EXAMPLES:
        return {
            "method": "temperature_scaling_skipped",
            "temperature": 1.0,
            "raw_nll": raw_nll,
            "calibrated_nll": raw_nll,
            "num_examples": int(target.shape[0]),
            "num_unique_classes": unique_classes,
            "reason": "insufficient_validation_examples",
            "guardrails": {
                "min_examples": int(MIN_TEMPERATURE_CALIBRATION_EXAMPLES),
                "min_unique_classes": int(MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES),
            },
        }

    if unique_classes < MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES:
        return {
            "method": "temperature_scaling_skipped",
            "temperature": 1.0,
            "raw_nll": raw_nll,
            "calibrated_nll": raw_nll,
            "num_examples": int(target.shape[0]),
            "num_unique_classes": unique_classes,
            "reason": "insufficient_validation_class_diversity",
            "guardrails": {
                "min_examples": int(MIN_TEMPERATURE_CALIBRATION_EXAMPLES),
                "min_unique_classes": int(MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES),
            },
        }

    candidate_temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 33, dtype=np.float64))
    best_temperature = 1.0
    best_nll = raw_nll
    for candidate in candidate_temperatures:
        scaled_probs = common._softmax_np(_apply_temperature_to_logits(logits, float(candidate)))
        candidate_nll = _cross_entropy_from_probs(scaled_probs, target)
        if candidate_nll is not None and (best_nll is None or candidate_nll < best_nll):
            best_temperature = float(candidate)
            best_nll = float(candidate_nll)

    local_temperatures = np.exp(
        np.linspace(np.log(max(best_temperature * 0.7, 0.1)), np.log(best_temperature * 1.3), 21, dtype=np.float64)
    )
    for candidate in local_temperatures:
        scaled_probs = common._softmax_np(_apply_temperature_to_logits(logits, float(candidate)))
        candidate_nll = _cross_entropy_from_probs(scaled_probs, target)
        if candidate_nll is not None and (best_nll is None or candidate_nll < best_nll):
            best_temperature = float(candidate)
            best_nll = float(candidate_nll)

    return {
        "method": "temperature_scaling_grid_search",
        "temperature": float(best_temperature),
        "raw_nll": raw_nll,
        "calibrated_nll": best_nll,
        "num_examples": int(target.shape[0]),
        "num_unique_classes": unique_classes,
        "guardrails": {
            "min_examples": int(MIN_TEMPERATURE_CALIBRATION_EXAMPLES),
            "min_unique_classes": int(MIN_TEMPERATURE_CALIBRATION_UNIQUE_CLASSES),
        },
    }


def _collect_class4_logits_and_targets(
    *,
    model: nn.Module,
    x: np.ndarray,
    y_class4: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(y_class4, dtype=np.int64)),
    )
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, y4b in loader:
            xb = xb.to(device)
            outputs = model(xb)
            logits_chunks.append(outputs["class4_logits"].detach().cpu().numpy())
            target_chunks.append(y4b.detach().cpu().numpy())
    if not logits_chunks:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.asarray(np.concatenate(logits_chunks, axis=0), dtype=np.float32),
        np.asarray(np.concatenate(target_chunks, axis=0), dtype=np.int64),
    )


def _evaluate_main_arrays(
    *,
    model: nn.Module,
    x: np.ndarray,
    y_class4: np.ndarray,
    y_x: np.ndarray,
    y_z: np.ndarray,
    batch_size: int,
    device: torch.device,
    class_labels: list[str],
    main_axis_loss_weight: float,
    non_identity_loss_weight: float = 0.0,
    confidence_loss_weight: float = 0.0,
    main_class4_loss: str = MAIN_CLASS4_LOSS_CROSS_ENTROPY,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    class_weight: torch.Tensor | None = None,
    non_identity_pos_weight: torch.Tensor | None = None,
    confidence_error_pos_weight: torch.Tensor | None = None,
    bit_x_pos_weight: torch.Tensor | None = None,
    bit_z_pos_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    loader = common._make_loader(
        _make_main_tensor_dataset(x, y_class4, y_x, y_z),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    class4_logits_chunks: list[np.ndarray] = []
    x_logits_chunks: list[np.ndarray] = []
    z_logits_chunks: list[np.ndarray] = []
    non_identity_logits_chunks: list[np.ndarray] = []
    error_logits_chunks: list[np.ndarray] = []
    y4_chunks: list[np.ndarray] = []
    yx_chunks: list[np.ndarray] = []
    yz_chunks: list[np.ndarray] = []
    objective_loss_sum = 0.0
    non_identity_loss_sum = 0.0
    confidence_error_loss_sum = 0.0
    bx_loss_sum = 0.0
    bz_loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, y4b, yxb, yzb in loader:
            xb = xb.to(device)
            y4b = y4b.to(device)
            yxb = yxb.to(device)
            yzb = yzb.to(device)
            outputs = model(xb)
            losses = _compute_main_batch_loss(
                outputs,
                y_class4=y4b,
                y_x=yxb,
                y_z=yzb,
                main_axis_loss_weight=main_axis_loss_weight,
                non_identity_loss_weight=non_identity_loss_weight,
                confidence_loss_weight=confidence_loss_weight,
                main_class4_loss=main_class4_loss,
                focal_gamma=focal_gamma,
                class_weight=class_weight,
                non_identity_pos_weight=non_identity_pos_weight,
                confidence_error_pos_weight=confidence_error_pos_weight,
                bit_x_pos_weight=bit_x_pos_weight,
                bit_z_pos_weight=bit_z_pos_weight,
            )
            batch_count = int(xb.shape[0])
            objective_loss_sum += float(losses["total_loss"].item()) * batch_count
            non_identity_loss_sum += float(losses["non_identity_bce_loss"].item()) * batch_count
            confidence_error_loss_sum += float(losses["confidence_error_bce_loss"].item()) * batch_count
            bx_loss_sum += float(losses["bit_x_bce_loss"].item()) * batch_count
            bz_loss_sum += float(losses["bit_z_bce_loss"].item()) * batch_count
            count += batch_count
            class4_logits_chunks.append(outputs["class4_logits"].detach().cpu().numpy())
            x_logits_chunks.append(outputs["x_logits"].detach().cpu().numpy())
            z_logits_chunks.append(outputs["z_logits"].detach().cpu().numpy())
            non_identity_logits_chunks.append(outputs["non_identity_logits"].detach().cpu().numpy())
            error_logits_chunks.append(outputs["error_logits"].detach().cpu().numpy())
            y4_chunks.append(y4b.detach().cpu().numpy())
            yx_chunks.append(yxb.detach().cpu().numpy())
            yz_chunks.append(yzb.detach().cpu().numpy())

    class4_logits = np.asarray(np.concatenate(class4_logits_chunks, axis=0), dtype=np.float32)
    class4_probs_raw = common._softmax_np(class4_logits)
    class4_probs = common._softmax_np(_apply_temperature_to_logits(class4_logits, temperature))
    x_probs = common._sigmoid_np(np.asarray(np.concatenate(x_logits_chunks, axis=0), dtype=np.float32))
    z_probs = common._sigmoid_np(np.asarray(np.concatenate(z_logits_chunks, axis=0), dtype=np.float32))
    non_identity_probs = common._sigmoid_np(np.asarray(np.concatenate(non_identity_logits_chunks, axis=0), dtype=np.float32))
    error_probs = common._sigmoid_np(np.asarray(np.concatenate(error_logits_chunks, axis=0), dtype=np.float32))
    y4 = np.concatenate(y4_chunks, axis=0)
    yx = np.concatenate(yx_chunks, axis=0)
    yz = np.concatenate(yz_chunks, axis=0)
    y_non_identity = np.asarray(y4 != 0, dtype=np.uint8)
    pred_class4 = np.argmax(class4_probs, axis=1).astype(np.uint8, copy=False)
    y_error = np.asarray(pred_class4 != y4, dtype=np.uint8)
    class4_ce = _cross_entropy_from_probs(class4_probs, y4)

    class4_metrics = common._multiclass_metrics_from_probs(
        class4_probs,
        y4,
        class_labels=class_labels,
        loss=class4_ce,
    )
    x_metrics = common._binary_metrics_from_probs(
        x_probs,
        np.asarray(yx, dtype=np.uint8),
        threshold=0.5,
        bce_loss=(float(bx_loss_sum / count) if count else None),
    )
    z_metrics = common._binary_metrics_from_probs(
        z_probs,
        np.asarray(yz, dtype=np.uint8),
        threshold=0.5,
        bce_loss=(float(bz_loss_sum / count) if count else None),
    )
    non_identity_metrics = common._binary_metrics_from_probs(
        non_identity_probs,
        y_non_identity,
        threshold=0.5,
        bce_loss=(float(non_identity_loss_sum / count) if count else None),
    )
    confidence_error_metrics = common._binary_metrics_from_probs(
        error_probs,
        y_error,
        threshold=0.5,
        bce_loss=(float(confidence_error_loss_sum / count) if count else None),
    )
    raw_calibration = _multiclass_calibration_metrics(class4_probs_raw, y4)
    calibrated_calibration = _multiclass_calibration_metrics(class4_probs, y4)
    class4_metrics.update(
        {
            "total_loss": (float(objective_loss_sum / count) if count else None),
            "non_identity_bce_loss": non_identity_metrics["bce_loss"],
            "non_identity_accuracy": non_identity_metrics["accuracy"],
            "non_identity_f1": non_identity_metrics["f1"],
            "non_identity_auroc": non_identity_metrics["auroc"],
            "non_identity_confusion_matrix": non_identity_metrics["confusion_matrix_logical_label"],
            "confidence_error_bce_loss": confidence_error_metrics["bce_loss"],
            "confidence_error_accuracy": confidence_error_metrics["accuracy"],
            "confidence_error_f1": confidence_error_metrics["f1"],
            "confidence_error_auroc": confidence_error_metrics["auroc"],
            "confidence_error_confusion_matrix": confidence_error_metrics["confusion_matrix_logical_label"],
            "mean_predicted_correctness": float((1.0 - error_probs).mean()) if count else None,
            "bit_x_bce_loss": x_metrics["bce_loss"],
            "bit_z_bce_loss": z_metrics["bce_loss"],
            "bit_x_accuracy": x_metrics["accuracy"],
            "bit_z_accuracy": z_metrics["accuracy"],
            "bit_x_f1": x_metrics["f1"],
            "bit_z_f1": z_metrics["f1"],
            "bit_x_confusion_matrix": x_metrics["confusion_matrix_logical_label"],
            "bit_z_confusion_matrix": z_metrics["confusion_matrix_logical_label"],
            "temperature_applied": float(temperature),
            "raw_cross_entropy_loss": raw_calibration["nll"],
            "raw_mean_predicted_confidence": float(np.max(class4_probs_raw, axis=1).mean()) if count else None,
            "ece_10": calibrated_calibration["ece"],
            "mce_10": calibrated_calibration["mce"],
            "brier_score": calibrated_calibration["brier_score"],
            "calibration": {
                "method": "temperature_scaling" if abs(float(temperature) - 1.0) > 1e-9 else "none",
                "temperature": float(temperature),
                "num_bins": calibrated_calibration["num_bins"],
                "ece_10": calibrated_calibration["ece"],
                "mce_10": calibrated_calibration["mce"],
                "brier_score": calibrated_calibration["brier_score"],
                "nll": calibrated_calibration["nll"],
                "raw_ece_10": raw_calibration["ece"],
                "raw_mce_10": raw_calibration["mce"],
                "raw_brier_score": raw_calibration["brier_score"],
                "raw_nll": raw_calibration["nll"],
                "reliability_bins": calibrated_calibration["reliability_bins"],
            },
        }
    )
    return class4_metrics


def _train_epoch_main_and_aux(
    *,
    model: nn.Module,
    optimizer: Any,
    device: torch.device,
    main_loader: Any,
    aux_loaders: dict[str, Any],
    main_axis_loss_weight: float,
    non_identity_loss_weight: float,
    confidence_loss_weight: float,
    aux_loss_weight: float,
    main_class4_loss: str,
    focal_gamma: float,
    class_weight: torch.Tensor | None,
    non_identity_pos_weight: torch.Tensor | None,
    confidence_error_pos_weight: torch.Tensor | None,
    bit_x_pos_weight: torch.Tensor | None,
    bit_z_pos_weight: torch.Tensor | None,
    aux_pos_weight_by_axis: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    model.train()
    total_main_loss = 0.0
    total_main_class4 = 0.0
    total_main_non_identity = 0.0
    total_main_confidence = 0.0
    total_main_bx = 0.0
    total_main_bz = 0.0
    total_main_examples = 0
    total_aux_loss = 0.0
    total_aux_examples = 0

    for xb, y4b, yxb, yzb in main_loader:
        xb = xb.to(device)
        y4b = y4b.to(device)
        yxb = yxb.to(device)
        yzb = yzb.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(xb)
        losses = _compute_main_batch_loss(
            outputs,
            y_class4=y4b,
            y_x=yxb,
            y_z=yzb,
            main_axis_loss_weight=main_axis_loss_weight,
            non_identity_loss_weight=non_identity_loss_weight,
            confidence_loss_weight=confidence_loss_weight,
            main_class4_loss=main_class4_loss,
            focal_gamma=focal_gamma,
            class_weight=class_weight,
            non_identity_pos_weight=non_identity_pos_weight,
            confidence_error_pos_weight=confidence_error_pos_weight,
            bit_x_pos_weight=bit_x_pos_weight,
            bit_z_pos_weight=bit_z_pos_weight,
        )
        losses["total_loss"].backward()
        optimizer.step()
        batch_count = int(xb.shape[0])
        total_main_loss += float(losses["total_loss"].item()) * batch_count
        total_main_class4 += float(losses["class4_loss"].item()) * batch_count
        total_main_non_identity += float(losses["non_identity_bce_loss"].item()) * batch_count
        total_main_confidence += float(losses["confidence_error_bce_loss"].item()) * batch_count
        total_main_bx += float(losses["bit_x_bce_loss"].item()) * batch_count
        total_main_bz += float(losses["bit_z_bce_loss"].item()) * batch_count
        total_main_examples += batch_count

    for axis_name, loader in aux_loaders.items():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss = _compute_axis_loss(
                outputs,
                axis_name=axis_name,
                target=yb,
                aux_loss_weight=aux_loss_weight,
                pos_weight=aux_pos_weight_by_axis.get(axis_name),
            )
            loss.backward()
            optimizer.step()
            batch_count = int(xb.shape[0])
            total_aux_loss += float(loss.item()) * batch_count
            total_aux_examples += batch_count

    return {
        "train_total_loss": float(total_main_loss / total_main_examples) if total_main_examples else float("nan"),
        "train_class4_loss": float(total_main_class4 / total_main_examples) if total_main_examples else float("nan"),
        "train_non_identity_bce_loss": float(total_main_non_identity / total_main_examples) if total_main_examples else float("nan"),
        "train_confidence_error_bce_loss": float(total_main_confidence / total_main_examples) if total_main_examples else float("nan"),
        "train_bit_x_bce_loss": float(total_main_bx / total_main_examples) if total_main_examples else float("nan"),
        "train_bit_z_bce_loss": float(total_main_bz / total_main_examples) if total_main_examples else float("nan"),
        "train_aux_axis_loss": float(total_aux_loss / total_aux_examples) if total_aux_examples else None,
    }


def _train_model_from_splits(
    *,
    x_train: np.ndarray,
    y4_train: np.ndarray,
    yx_train: np.ndarray,
    yz_train: np.ndarray,
    x_val: np.ndarray,
    y4_val: np.ndarray,
    yx_val: np.ndarray,
    yz_val: np.ndarray,
    x_test: np.ndarray,
    y4_test: np.ndarray,
    yx_test: np.ndarray,
    yz_test: np.ndarray,
    aux_by_axis: dict[str, dict[str, Any]],
    class_labels: list[str],
    signal_channels: int,
    context_channels: int,
    hidden_channels: int,
    num_blocks: int,
    dense_hidden_dim: int,
    context_hidden_dim: int,
    dropout: float,
    main_axis_loss_weight: float,
    non_identity_loss_weight: float,
    confidence_loss_weight: float,
    aux_loss_weight: float,
    imbalance_mode: str,
    main_class4_loss: str,
    focal_gamma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device_arg: str,
) -> dict[str, Any]:
    model = FactorizedLogicalFrameDecoder(
        signal_channels=signal_channels,
        context_channels=context_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        dropout=dropout,
        context_hidden_dim=context_hidden_dim,
    )
    device = common._pick_device(device_arg)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    main_dataset = _make_main_tensor_dataset(x_train, y4_train, yx_train, yz_train)
    aux_loaders: dict[str, Any] = {}
    sampling = _build_sampling_config(
        y4_train=y4_train,
        aux_by_axis=aux_by_axis,
        class_labels=class_labels,
        imbalance_mode=imbalance_mode,
    )
    if imbalance_mode == IMBALANCE_MODE_BALANCED:
        main_loader = _make_weighted_loader(
            main_dataset,
            sample_weights=_compute_multiclass_sample_weights(y4_train, num_classes=len(class_labels)),
            batch_size=batch_size,
        )
    elif imbalance_mode == IMBALANCE_MODE_TEMPERED:
        main_loader = _make_weighted_loader(
            main_dataset,
            sample_weights=_compute_multiclass_sample_weights(
                y4_train,
                num_classes=len(class_labels),
                exponent=TEMPERED_MAIN_SAMPLE_EXPONENT,
            ),
            batch_size=batch_size,
        )
    else:
        main_loader = common._make_loader(
            main_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    for axis_name, aux in aux_by_axis.items():
        aux_dataset = common._make_tensor_dataset(aux["x"], aux["y"])
        if imbalance_mode == IMBALANCE_MODE_BALANCED:
            aux_loaders[axis_name] = _make_weighted_loader(
                aux_dataset,
                sample_weights=_compute_binary_sample_weights(aux["y"]),
                batch_size=batch_size,
            )
        else:
            aux_loaders[axis_name] = common._make_loader(
                aux_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
    reweighting = _build_reweighting_config(
        y4_train=y4_train,
        yx_train=yx_train,
        yz_train=yz_train,
        aux_by_axis=aux_by_axis,
        class_labels=class_labels,
        imbalance_mode=imbalance_mode,
    )
    class_weight_tensor = None
    if reweighting["class_weight_vector"] is not None:
        class_weight_tensor = torch.tensor(reweighting["class_weight_vector"], dtype=torch.float32, device=device)
    non_identity_pos_weight_tensor = torch.tensor(
        [float(reweighting["non_identity_pos_weight"])], dtype=torch.float32, device=device
    )
    confidence_error_pos_weight_tensor = torch.tensor(
        [float(reweighting["confidence_error_pos_weight"])], dtype=torch.float32, device=device
    )
    bit_x_pos_weight_tensor = torch.tensor(
        [float(reweighting["bit_x_pos_weight"])], dtype=torch.float32, device=device
    )
    bit_z_pos_weight_tensor = torch.tensor(
        [float(reweighting["bit_z_pos_weight"])], dtype=torch.float32, device=device
    )
    aux_pos_weight_by_axis = {
        "logical_x_flip": bit_x_pos_weight_tensor,
        "logical_z_flip": bit_z_pos_weight_tensor,
    }

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_losses = _train_epoch_main_and_aux(
            model=model,
            optimizer=optimizer,
            device=device,
            main_loader=main_loader,
            aux_loaders=aux_loaders,
            main_axis_loss_weight=main_axis_loss_weight,
            non_identity_loss_weight=non_identity_loss_weight,
            confidence_loss_weight=confidence_loss_weight,
            aux_loss_weight=aux_loss_weight,
            main_class4_loss=main_class4_loss,
            focal_gamma=focal_gamma,
            class_weight=class_weight_tensor,
            non_identity_pos_weight=non_identity_pos_weight_tensor,
            confidence_error_pos_weight=confidence_error_pos_weight_tensor,
            bit_x_pos_weight=bit_x_pos_weight_tensor,
            bit_z_pos_weight=bit_z_pos_weight_tensor,
            aux_pos_weight_by_axis=aux_pos_weight_by_axis,
        )
        val_metrics = _evaluate_main_arrays(
            model=model,
            x=x_val,
            y_class4=y4_val,
            y_x=yx_val,
            y_z=yz_val,
            batch_size=batch_size,
            device=device,
            class_labels=class_labels,
            main_axis_loss_weight=main_axis_loss_weight,
            non_identity_loss_weight=non_identity_loss_weight,
            confidence_loss_weight=confidence_loss_weight,
            main_class4_loss=main_class4_loss,
            focal_gamma=focal_gamma,
            class_weight=class_weight_tensor,
            non_identity_pos_weight=non_identity_pos_weight_tensor,
            confidence_error_pos_weight=confidence_error_pos_weight_tensor,
            bit_x_pos_weight=bit_x_pos_weight_tensor,
            bit_z_pos_weight=bit_z_pos_weight_tensor,
        )
        val_loss = val_metrics["total_loss"]
        history.append(
            {
                "epoch": epoch,
                **train_losses,
                "val_total_loss": val_loss,
                "val_class4_loss": val_metrics["cross_entropy_loss"],
                "val_non_identity_bce_loss": val_metrics["non_identity_bce_loss"],
                "val_confidence_error_bce_loss": val_metrics["confidence_error_bce_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_non_identity_accuracy": val_metrics["non_identity_accuracy"],
                "val_confidence_error_accuracy": val_metrics["confidence_error_accuracy"],
                "val_bit_x_accuracy": val_metrics["bit_x_accuracy"],
                "val_bit_z_accuracy": val_metrics["bit_z_accuracy"],
                "val_ece_10": val_metrics["ece_10"],
                "val_brier_score": val_metrics["brier_score"],
            }
        )
        if best_val_loss is None or (val_loss is not None and val_loss < best_val_loss):
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_targets = _collect_class4_logits_and_targets(
        model=model,
        x=x_val,
        y_class4=y4_val,
        batch_size=batch_size,
        device=device,
    )
    calibration = _fit_temperature_from_logits(val_logits, val_targets)
    calibration_temperature = float(calibration["temperature"])
    calibration["fitted_on"] = "validation_split"
    calibration["val_metrics_raw"] = _evaluate_main_arrays(
        model=model,
        x=x_val,
        y_class4=y4_val,
        y_x=yx_val,
        y_z=yz_val,
        batch_size=batch_size,
        device=device,
        class_labels=class_labels,
        main_axis_loss_weight=main_axis_loss_weight,
        non_identity_loss_weight=non_identity_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        main_class4_loss=main_class4_loss,
        focal_gamma=focal_gamma,
        class_weight=class_weight_tensor,
        non_identity_pos_weight=non_identity_pos_weight_tensor,
        confidence_error_pos_weight=confidence_error_pos_weight_tensor,
        bit_x_pos_weight=bit_x_pos_weight_tensor,
        bit_z_pos_weight=bit_z_pos_weight_tensor,
        temperature=1.0,
    ) if x_val.shape[0] else None
    calibration["val_metrics_calibrated"] = _evaluate_main_arrays(
        model=model,
        x=x_val,
        y_class4=y4_val,
        y_x=yx_val,
        y_z=yz_val,
        batch_size=batch_size,
        device=device,
        class_labels=class_labels,
        main_axis_loss_weight=main_axis_loss_weight,
        non_identity_loss_weight=non_identity_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        main_class4_loss=main_class4_loss,
        focal_gamma=focal_gamma,
        class_weight=class_weight_tensor,
        non_identity_pos_weight=non_identity_pos_weight_tensor,
        confidence_error_pos_weight=confidence_error_pos_weight_tensor,
        bit_x_pos_weight=bit_x_pos_weight_tensor,
        bit_z_pos_weight=bit_z_pos_weight_tensor,
        temperature=calibration_temperature,
    ) if x_val.shape[0] else None

    test_metrics = _evaluate_main_arrays(
        model=model,
        x=x_test,
        y_class4=y4_test,
        y_x=yx_test,
        y_z=yz_test,
        batch_size=batch_size,
        device=device,
        class_labels=class_labels,
        main_axis_loss_weight=main_axis_loss_weight,
        non_identity_loss_weight=non_identity_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        main_class4_loss=main_class4_loss,
        focal_gamma=focal_gamma,
        class_weight=class_weight_tensor,
        non_identity_pos_weight=non_identity_pos_weight_tensor,
        confidence_error_pos_weight=confidence_error_pos_weight_tensor,
        bit_x_pos_weight=bit_x_pos_weight_tensor,
        bit_z_pos_weight=bit_z_pos_weight_tensor,
        temperature=calibration_temperature,
    )
    elapsed = time.perf_counter() - started
    return {
        "model": model,
        "device": device,
        "history": history,
        "reweighting": reweighting,
        "sampling": sampling,
        "main_class4_loss": str(main_class4_loss),
        "focal_gamma": float(focal_gamma),
        "non_identity_loss_weight": float(non_identity_loss_weight),
        "confidence_loss_weight": float(confidence_loss_weight),
        "calibration": calibration,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def train_family_dir(
    *,
    family_dir: Path | None,
    manifest: Path | None,
    family: str | None,
    checkpoint_out: Path,
    train_json_out: Path | None,
    fill_value: float,
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
    context_hidden_dim: int,
    main_axis_loss_weight: float,
    non_identity_loss_weight: float,
    confidence_loss_weight: float,
    aux_loss_weight: float,
    imbalance_mode: str,
    main_class4_loss: str,
    focal_gamma: float,
    aux_dual_axis_manifest: Path | None,
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
    prepared = _prepare_class4_family(payload, fill_value=fill_value, max_shots=max_shots)

    num_shots = int(prepared["y_class4"].shape[0])
    split_indices = common.build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_summary = common.summarise_split_indices(split_indices, num_shots=num_shots, seed=seed)

    aux_info = None
    aux_by_axis: dict[str, dict[str, Any]] = {}
    family_name = str(metadata.get("family"))
    if aux_dual_axis_manifest is not None:
        aux_loaded = _load_auxiliary_axis_entries(
            dual_axis_manifest_path=aux_dual_axis_manifest,
            families=[family_name],
            fill_value=fill_value,
            max_shots=max_shots,
        )
        aux_info = dict(aux_loaded["summary"])
        aux_by_axis = dict(aux_loaded["by_axis"])

    trained = _train_model_from_splits(
        x_train=common._subset_rows(prepared["x"], split_indices["train"]),
        y4_train=common._subset_rows(prepared["y_class4"], split_indices["train"]),
        yx_train=common._subset_rows(prepared["y_x"], split_indices["train"]),
        yz_train=common._subset_rows(prepared["y_z"], split_indices["train"]),
        x_val=common._subset_rows(prepared["x"], split_indices["val"]),
        y4_val=common._subset_rows(prepared["y_class4"], split_indices["val"]),
        yx_val=common._subset_rows(prepared["y_x"], split_indices["val"]),
        yz_val=common._subset_rows(prepared["y_z"], split_indices["val"]),
        x_test=common._subset_rows(prepared["x"], split_indices["test"]),
        y4_test=common._subset_rows(prepared["y_class4"], split_indices["test"]),
        yx_test=common._subset_rows(prepared["y_x"], split_indices["test"]),
        yz_test=common._subset_rows(prepared["y_z"], split_indices["test"]),
        aux_by_axis=aux_by_axis,
        class_labels=prepared["class_labels"],
        signal_channels=prepared["signal_channels"],
        context_channels=prepared["context_channels"],
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        context_hidden_dim=context_hidden_dim,
        dropout=dropout,
        main_axis_loss_weight=main_axis_loss_weight,
        non_identity_loss_weight=non_identity_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        aux_loss_weight=aux_loss_weight,
        imbalance_mode=imbalance_mode,
        main_class4_loss=main_class4_loss,
        focal_gamma=focal_gamma,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device_arg=device_arg,
    )
    model = trained["model"]
    device = trained["device"]

    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "factorized_logical_frame_decoder",
        "created_at_utc": common._utc_now_iso(),
        "family": metadata.get("family"),
        "stage": metadata.get("stage"),
        "family_dir": resolved_family_dir.as_posix(),
        "fill_value": float(fill_value),
        "model_hparams": {
            "signal_channels": int(prepared["signal_channels"]),
            "context_channels": int(prepared["context_channels"]),
            "hidden_channels": int(hidden_channels),
            "num_blocks": int(num_blocks),
            "dense_hidden_dim": int(dense_hidden_dim),
            "context_hidden_dim": int(context_hidden_dim),
            "dropout": float(dropout),
            "class_labels": list(prepared["class_labels"]),
        },
        "selection_rule": {"class4": "argmax", "bit_threshold": 0.5},
        "main_axis_loss_weight": float(main_axis_loss_weight),
        "non_identity_loss_weight": float(non_identity_loss_weight),
        "confidence_loss_weight": float(confidence_loss_weight),
        "aux_loss_weight": float(aux_loss_weight),
        "imbalance_mode": str(imbalance_mode),
        "main_class4_loss": str(main_class4_loss),
        "focal_gamma": float(focal_gamma),
        "reweighting": trained["reweighting"],
        "sampling": trained["sampling"],
        "calibration": trained["calibration"],
        "target_mode_requested": common.TARGET_MODE_LOGICAL_CLASS4,
        "target_mode_resolved": prepared["resolved_target_mode"],
        "target_key": prepared["target_key"],
        "aux_dual_axis_manifest": aux_dual_axis_manifest.as_posix() if aux_dual_axis_manifest is not None else None,
        "split": split_summary.to_dict(),
        "state_dict": model.state_dict(),
    }
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)

    result = common.TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="factorized_logical_frame_decoder",
        created_at_utc=common._utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family")),
        stage=str(metadata.get("stage")),
        family_dir=resolved_family_dir.as_posix(),
        model={
            "architecture": "factorized_logical_frame_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(trained["num_parameters"]),
            "bundle_info": prepared["bundle_info"],
            "model_hparams": checkpoint["model_hparams"],
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit"),
            "scaffold": metadata.get("scaffold"),
            "targets": metadata.get("targets"),
            "target_key_used": prepared["target_key"],
            "rectangular_syndrome_layout": metadata.get("rectangular_syndrome_layout"),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "num_shots_total_after_limit": prepared["shots_total_after_limit"],
            "max_shots": max_shots,
            "fill_value": float(fill_value),
            "target_mode_requested": common.TARGET_MODE_LOGICAL_CLASS4,
            "target_mode_resolved": prepared["resolved_target_mode"],
            "manifest_requested_family": family if manifest_data is not None else None,
            "auxiliary_axis_supervision": aux_info,
        },
        split=split_summary.to_dict(),
        training={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "elapsed_seconds": float(trained["elapsed_seconds"]),
            "non_identity_loss_weight": float(non_identity_loss_weight),
            "confidence_loss_weight": float(confidence_loss_weight),
            "imbalance_mode": str(imbalance_mode),
            "main_class4_loss": str(main_class4_loss),
            "focal_gamma": float(focal_gamma),
            "reweighting": trained["reweighting"],
            "sampling": trained["sampling"],
            "calibration": trained["calibration"],
            "history": trained["history"],
        },
        metrics=trained["test_metrics"],
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
    context_hidden_dim: int,
    main_axis_loss_weight: float,
    non_identity_loss_weight: float,
    confidence_loss_weight: float,
    aux_loss_weight: float,
    imbalance_mode: str,
    main_class4_loss: str,
    focal_gamma: float,
    aux_dual_axis_manifest: Path | None,
    aux_train_families: list[str] | None,
    device_arg: str,
) -> dict[str, Any]:
    common._require_torch()
    common._set_random_seeds(seed)

    manifest_data, train_entries_raw = _resolve_manifest_family_entries(manifest, train_families)
    _, eval_entries_raw = _resolve_manifest_family_entries(manifest, eval_families)
    prepared_train_entries: list[dict[str, Any]] = []
    for _, family_dir in train_entries_raw:
        payload = common._load_family_payload(family_dir)
        entry = _prepare_class4_family(payload, fill_value=fill_value, max_shots=max_shots)
        entry["metadata"] = payload.metadata
        entry["family_dir"] = family_dir.as_posix()
        prepared_train_entries.append(entry)
    compatibility = _validate_main_compatible(prepared_train_entries)

    aux_info = None
    aux_by_axis: dict[str, dict[str, Any]] = {}
    if aux_dual_axis_manifest is not None:
        resolved_aux_families = list(aux_train_families) if aux_train_families is not None else list(train_families)
        aux_loaded = _load_auxiliary_axis_entries(
            dual_axis_manifest_path=aux_dual_axis_manifest,
            families=resolved_aux_families,
            fill_value=fill_value,
            max_shots=max_shots,
        )
        aux_info = dict(aux_loaded["summary"])
        aux_by_axis = dict(aux_loaded["by_axis"])
        for axis_name, aux in aux_by_axis.items():
            if list(aux["channel_names"]) != list(compatibility["channel_names"]):
                raise ValueError(f"Auxiliary axis {axis_name} channel layout does not match class4 training data.")
            if int(aux["signal_channels"]) != int(compatibility["signal_channels"]):
                raise ValueError(f"Auxiliary axis {axis_name} signal-channel count does not match class4 training data.")
            if int(aux["context_channels"]) != int(compatibility["context_channels"]):
                raise ValueError(f"Auxiliary axis {axis_name} context-channel count does not match class4 training data.")
            if tuple(int(v) for v in aux["x"].shape[1:]) != tuple(int(v) for v in compatibility["input_shape_without_batch"]):
                raise ValueError(f"Auxiliary axis {axis_name} tensor shape does not match class4 training data.")

    x_train_chunks: list[np.ndarray] = []
    y4_train_chunks: list[np.ndarray] = []
    yx_train_chunks: list[np.ndarray] = []
    yz_train_chunks: list[np.ndarray] = []
    x_val_chunks: list[np.ndarray] = []
    y4_val_chunks: list[np.ndarray] = []
    yx_val_chunks: list[np.ndarray] = []
    yz_val_chunks: list[np.ndarray] = []
    x_test_chunks: list[np.ndarray] = []
    y4_test_chunks: list[np.ndarray] = []
    yx_test_chunks: list[np.ndarray] = []
    yz_test_chunks: list[np.ndarray] = []
    split_family_summaries: list[dict[str, Any]] = []
    total_loaded = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for index, entry in enumerate(prepared_train_entries):
        num_shots = int(entry["y_class4"].shape[0])
        family_seed = int(seed + index)
        split_indices = common.build_split_indices(
            num_shots=num_shots,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=family_seed,
        )
        split_summary = common.summarise_split_indices(split_indices, num_shots=num_shots, seed=family_seed)
        x_train_chunks.append(common._subset_rows(entry["x"], split_indices["train"]))
        y4_train_chunks.append(common._subset_rows(entry["y_class4"], split_indices["train"]))
        yx_train_chunks.append(common._subset_rows(entry["y_x"], split_indices["train"]))
        yz_train_chunks.append(common._subset_rows(entry["y_z"], split_indices["train"]))
        x_val_chunks.append(common._subset_rows(entry["x"], split_indices["val"]))
        y4_val_chunks.append(common._subset_rows(entry["y_class4"], split_indices["val"]))
        yx_val_chunks.append(common._subset_rows(entry["y_x"], split_indices["val"]))
        yz_val_chunks.append(common._subset_rows(entry["y_z"], split_indices["val"]))
        x_test_chunks.append(common._subset_rows(entry["x"], split_indices["test"]))
        y4_test_chunks.append(common._subset_rows(entry["y_class4"], split_indices["test"]))
        yx_test_chunks.append(common._subset_rows(entry["y_x"], split_indices["test"]))
        yz_test_chunks.append(common._subset_rows(entry["y_z"], split_indices["test"]))

        metadata = entry["metadata"]
        total_loaded += num_shots
        total_train += split_summary.train_count
        total_val += split_summary.val_count
        total_test += split_summary.test_count
        counts = np.bincount(entry["y_class4"], minlength=len(entry["class_labels"]))
        split_family_summaries.append(
            {
                "family": metadata.get("family"),
                "stage": metadata.get("stage"),
                "family_dir": entry["family_dir"],
                "shots_total_after_limit": num_shots,
                "target_key_used": entry["target_key"],
                "resolved_target_mode": entry["resolved_target_mode"],
                "target_summary": {
                    "task_type": "multiclass",
                    "class_labels": entry["class_labels"],
                    "class_histogram": {
                        entry["class_labels"][label]: int(counts[label])
                        for label in range(len(entry["class_labels"]))
                    },
                },
                "split": split_summary.to_dict(),
                "bundle_info": entry["bundle_info"],
            }
        )

    trained = _train_model_from_splits(
        x_train=np.concatenate(x_train_chunks, axis=0),
        y4_train=np.concatenate(y4_train_chunks, axis=0),
        yx_train=np.concatenate(yx_train_chunks, axis=0),
        yz_train=np.concatenate(yz_train_chunks, axis=0),
        x_val=np.concatenate(x_val_chunks, axis=0),
        y4_val=np.concatenate(y4_val_chunks, axis=0),
        yx_val=np.concatenate(yx_val_chunks, axis=0),
        yz_val=np.concatenate(yz_val_chunks, axis=0),
        x_test=np.concatenate(x_test_chunks, axis=0),
        y4_test=np.concatenate(y4_test_chunks, axis=0),
        yx_test=np.concatenate(yx_test_chunks, axis=0),
        yz_test=np.concatenate(yz_test_chunks, axis=0),
        aux_by_axis=aux_by_axis,
        class_labels=list(compatibility["class_labels"]),
        signal_channels=int(compatibility["signal_channels"]),
        context_channels=int(compatibility["context_channels"]),
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dense_hidden_dim=dense_hidden_dim,
        context_hidden_dim=context_hidden_dim,
        dropout=dropout,
        main_axis_loss_weight=main_axis_loss_weight,
        non_identity_loss_weight=non_identity_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        aux_loss_weight=aux_loss_weight,
        imbalance_mode=imbalance_mode,
        main_class4_loss=main_class4_loss,
        focal_gamma=focal_gamma,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device_arg=device_arg,
    )
    model = trained["model"]
    device = trained["device"]

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    train_json_path = out_dir / "train.json"
    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "factorized_logical_frame_decoder",
        "created_at_utc": common._utc_now_iso(),
        "family": "multi_family_manifest",
        "stage": "mixed",
        "manifest_path": manifest.as_posix(),
        "train_families": [item["family"] for item in split_family_summaries],
        "train_family_dirs": [item["family_dir"] for item in split_family_summaries],
        "fill_value": float(fill_value),
        "model_hparams": {
            "signal_channels": int(compatibility["signal_channels"]),
            "context_channels": int(compatibility["context_channels"]),
            "hidden_channels": int(hidden_channels),
            "num_blocks": int(num_blocks),
            "dense_hidden_dim": int(dense_hidden_dim),
            "context_hidden_dim": int(context_hidden_dim),
            "dropout": float(dropout),
            "class_labels": list(compatibility["class_labels"]),
        },
        "selection_rule": {"class4": "argmax", "bit_threshold": 0.5},
        "main_axis_loss_weight": float(main_axis_loss_weight),
        "non_identity_loss_weight": float(non_identity_loss_weight),
        "confidence_loss_weight": float(confidence_loss_weight),
        "aux_loss_weight": float(aux_loss_weight),
        "imbalance_mode": str(imbalance_mode),
        "main_class4_loss": str(main_class4_loss),
        "focal_gamma": float(focal_gamma),
        "reweighting": trained["reweighting"],
        "sampling": trained["sampling"],
        "calibration": trained["calibration"],
        "target_mode_requested": common.TARGET_MODE_LOGICAL_CLASS4,
        "target_mode_resolved": common.TARGET_MODE_LOGICAL_CLASS4,
        "target_key": "logical_class4",
        "aux_dual_axis_manifest": aux_dual_axis_manifest.as_posix() if aux_dual_axis_manifest is not None else None,
        "aux_train_families": list(aux_train_families) if aux_train_families is not None else None,
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
        decoder="factorized_logical_frame_decoder",
        created_at_utc=common._utc_now_iso(),
        input_mode="manifest_multi_family",
        family="multi_family_manifest",
        stage="mixed",
        family_dir=manifest.as_posix(),
        model={
            "architecture": "factorized_logical_frame_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(trained["num_parameters"]),
            "bundle_info": {
                "representation": "noise_aware_volume_3d",
                "shape": {
                    "batch": int(total_loaded),
                    "channels": int(compatibility["input_shape_without_batch"][0]),
                    "time_steps": int(compatibility["input_shape_without_batch"][1]),
                    "grid_height": int(compatibility["input_shape_without_batch"][2]),
                    "grid_width": int(compatibility["input_shape_without_batch"][3]),
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
            "target_mode_requested": common.TARGET_MODE_LOGICAL_CLASS4,
            "target_mode_resolved": common.TARGET_MODE_LOGICAL_CLASS4,
            "target_key_used": "logical_class4",
            "auxiliary_axis_supervision": aux_info,
        },
        split=checkpoint["split"],
        training={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "elapsed_seconds": float(trained["elapsed_seconds"]),
            "non_identity_loss_weight": float(non_identity_loss_weight),
            "confidence_loss_weight": float(confidence_loss_weight),
            "imbalance_mode": str(imbalance_mode),
            "main_class4_loss": str(main_class4_loss),
            "focal_gamma": float(focal_gamma),
            "reweighting": trained["reweighting"],
            "sampling": trained["sampling"],
            "calibration": trained["calibration"],
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
            max_shots=max_shots,
            batch_size=batch_size,
            device_arg=device_arg,
        )
        eval_dict = eval_result.to_dict()
        eval_json_paths[family] = eval_json_path.as_posix()
        eval_metrics = eval_dict["metrics"]
        eval_summaries.append(
            {
                "family": family,
                "stage": eval_dict["stage"],
                "seen_in_training": family in train_family_names,
                "accuracy": eval_metrics.get("accuracy"),
                "label_error_rate": eval_metrics.get("label_error_rate"),
                "balanced_accuracy": eval_metrics.get("balanced_accuracy"),
                "macro_f1": eval_metrics.get("macro_f1"),
                "non_identity_accuracy": eval_metrics.get("non_identity_accuracy"),
                "bit_x_accuracy": eval_metrics.get("bit_x_accuracy"),
                "bit_z_accuracy": eval_metrics.get("bit_z_accuracy"),
                "ece_10": eval_metrics.get("ece_10"),
                "brier_score": eval_metrics.get("brier_score"),
                "temperature_applied": eval_metrics.get("temperature_applied"),
                "eval_json_path": eval_json_path.as_posix(),
            }
        )

    experiment_summary = {
        "schema_version": SCHEMA_VERSION_EXPERIMENT,
        "decoder": "factorized_logical_frame_decoder",
        "created_at_utc": common._utc_now_iso(),
        "manifest": {
            "path": manifest.as_posix(),
            "schema_version": manifest_data.get("schema_version"),
            "families_available": list((manifest_data.get("family_dirs") or {}).keys()),
        },
        "training": {
            "train_families": [item["family"] for item in split_family_summaries],
            "train_stages": [item["stage"] for item in split_family_summaries],
            "train_json_path": train_json_path.as_posix(),
            "checkpoint_path": checkpoint_path.as_posix(),
            "mixed_test_split_metrics": train_result.metrics,
            "auxiliary_axis_supervision": aux_info,
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
    max_shots: int | None,
    batch_size: int,
    device_arg: str,
) -> common.EvalResult:
    common._require_torch()
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_mode, resolved_family_dir, _ = common._resolve_input_family_dir(
        family_dir=family_dir,
        manifest=manifest,
        family=family,
    )
    payload = common._load_family_payload(resolved_family_dir)
    metadata = payload.metadata
    prepared = _prepare_class4_family(
        payload,
        fill_value=float(checkpoint.get("fill_value", -0.5)),
        max_shots=max_shots,
    )
    model_hparams = dict(checkpoint["model_hparams"])
    device = common._pick_device(device_arg)
    model = FactorizedLogicalFrameDecoder(
        signal_channels=int(model_hparams["signal_channels"]),
        context_channels=int(model_hparams["context_channels"]),
        hidden_channels=int(model_hparams["hidden_channels"]),
        num_blocks=int(model_hparams["num_blocks"]),
        dense_hidden_dim=int(model_hparams["dense_hidden_dim"]),
        context_hidden_dim=int(model_hparams["context_hidden_dim"]),
        dropout=float(model_hparams["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    checkpoint_calibration = checkpoint.get("calibration", {})
    temperature = float(checkpoint_calibration.get("temperature", 1.0))
    checkpoint_reweighting = dict(checkpoint.get("reweighting", {}))
    class_weight_tensor = None
    if checkpoint_reweighting.get("class_weight_vector") is not None:
        class_weight_tensor = torch.tensor(
            checkpoint_reweighting["class_weight_vector"],
            dtype=torch.float32,
            device=device,
        )
    non_identity_pos_weight_tensor = torch.tensor(
        [float(checkpoint_reweighting.get("non_identity_pos_weight", 1.0))],
        dtype=torch.float32,
        device=device,
    )
    confidence_error_pos_weight_tensor = torch.tensor(
        [float(checkpoint_reweighting.get("confidence_error_pos_weight", 1.0))],
        dtype=torch.float32,
        device=device,
    )
    bit_x_pos_weight_tensor = torch.tensor(
        [float(checkpoint_reweighting.get("bit_x_pos_weight", 1.0))],
        dtype=torch.float32,
        device=device,
    )
    bit_z_pos_weight_tensor = torch.tensor(
        [float(checkpoint_reweighting.get("bit_z_pos_weight", 1.0))],
        dtype=torch.float32,
        device=device,
    )

    metrics = _evaluate_main_arrays(
        model=model,
        x=prepared["x"],
        y_class4=prepared["y_class4"],
        y_x=prepared["y_x"],
        y_z=prepared["y_z"],
        batch_size=batch_size,
        device=device,
        class_labels=list(model_hparams["class_labels"]),
        main_axis_loss_weight=float(checkpoint.get("main_axis_loss_weight", 0.25)),
        non_identity_loss_weight=float(checkpoint.get("non_identity_loss_weight", 0.0)),
        confidence_loss_weight=float(checkpoint.get("confidence_loss_weight", 0.0)),
        main_class4_loss=str(checkpoint.get("main_class4_loss", MAIN_CLASS4_LOSS_CROSS_ENTROPY)),
        focal_gamma=float(checkpoint.get("focal_gamma", DEFAULT_FOCAL_GAMMA)),
        class_weight=class_weight_tensor,
        non_identity_pos_weight=non_identity_pos_weight_tensor,
        confidence_error_pos_weight=confidence_error_pos_weight_tensor,
        bit_x_pos_weight=bit_x_pos_weight_tensor,
        bit_z_pos_weight=bit_z_pos_weight_tensor,
        temperature=temperature,
    )

    result = common.EvalResult(
        schema_version=SCHEMA_VERSION_EVAL,
        decoder="factorized_logical_frame_decoder",
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
            "selection_rule": checkpoint.get("selection_rule"),
        },
        model={
            "architecture": "factorized_logical_frame_decoder",
            "input_representation": "noise_aware_volume_3d",
            "device": str(device),
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "bundle_info": prepared["bundle_info"],
            "model_hparams": model_hparams,
            "calibration": checkpoint_calibration,
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit"),
            "scaffold": metadata.get("scaffold"),
            "targets": metadata.get("targets"),
            "target_key_used": prepared["target_key"],
            "rectangular_syndrome_layout": metadata.get("rectangular_syndrome_layout"),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "num_shots_total_after_limit": prepared["shots_total_after_limit"],
            "max_shots": max_shots,
            "fill_value": float(checkpoint.get("fill_value", -0.5)),
            "target_mode_requested": common.TARGET_MODE_LOGICAL_CLASS4,
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
            "evaluated_count": int(prepared["x"].shape[0]),
        },
        metrics=metrics,
    )
    if eval_json_out is not None:
        common._write_json(eval_json_out, result.to_dict())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the factorized logical-frame decoder.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    common._add_common_input_args(train_parser)
    train_parser.add_argument("--checkpoint-out", type=Path, required=True)
    train_parser.add_argument("--train-json-out", type=Path, default=None)
    train_parser.add_argument("--fill-value", type=float, default=-0.5)
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
    train_parser.add_argument("--num-blocks", type=int, default=4)
    train_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    train_parser.add_argument("--context-hidden-dim", type=int, default=64)
    train_parser.add_argument("--main-axis-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--non-identity-loss-weight", type=float, default=0.0)
    train_parser.add_argument("--confidence-loss-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--imbalance-mode", type=str, choices=IMBALANCE_MODE_CHOICES, default=IMBALANCE_MODE_TEMPERED)
    train_parser.add_argument("--main-class4-loss", type=str, choices=MAIN_CLASS4_LOSS_CHOICES, default=MAIN_CLASS4_LOSS_CROSS_ENTROPY)
    train_parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    train_parser.add_argument("--aux-dual-axis-manifest", type=Path, default=None)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval")
    common._add_common_input_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--eval-json-out", type=Path, default=None)
    eval_parser.add_argument("--max-shots", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--device", type=str, default="auto")

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument("--manifest", type=Path, required=True)
    experiment_parser.add_argument("--train-families", nargs="+", required=True)
    experiment_parser.add_argument("--eval-families", nargs="+", default=None)
    experiment_parser.add_argument("--out-dir", type=Path, required=True)
    experiment_parser.add_argument("--fill-value", type=float, default=-0.5)
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
    experiment_parser.add_argument("--num-blocks", type=int, default=4)
    experiment_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    experiment_parser.add_argument("--context-hidden-dim", type=int, default=64)
    experiment_parser.add_argument("--main-axis-loss-weight", type=float, default=0.25)
    experiment_parser.add_argument("--non-identity-loss-weight", type=float, default=0.0)
    experiment_parser.add_argument("--confidence-loss-weight", type=float, default=0.0)
    experiment_parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    experiment_parser.add_argument("--imbalance-mode", type=str, choices=IMBALANCE_MODE_CHOICES, default=IMBALANCE_MODE_TEMPERED)
    experiment_parser.add_argument("--main-class4-loss", type=str, choices=MAIN_CLASS4_LOSS_CHOICES, default=MAIN_CLASS4_LOSS_CROSS_ENTROPY)
    experiment_parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    experiment_parser.add_argument("--aux-dual-axis-manifest", type=Path, default=None)
    experiment_parser.add_argument("--aux-train-families", nargs="+", default=None)
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
            context_hidden_dim=args.context_hidden_dim,
            main_axis_loss_weight=args.main_axis_loss_weight,
            non_identity_loss_weight=args.non_identity_loss_weight,
            confidence_loss_weight=args.confidence_loss_weight,
            aux_loss_weight=args.aux_loss_weight,
            imbalance_mode=args.imbalance_mode,
            main_class4_loss=args.main_class4_loss,
            focal_gamma=args.focal_gamma,
            aux_dual_axis_manifest=args.aux_dual_axis_manifest,
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
            context_hidden_dim=args.context_hidden_dim,
            main_axis_loss_weight=args.main_axis_loss_weight,
            non_identity_loss_weight=args.non_identity_loss_weight,
            confidence_loss_weight=args.confidence_loss_weight,
            aux_loss_weight=args.aux_loss_weight,
            imbalance_mode=args.imbalance_mode,
            main_class4_loss=args.main_class4_loss,
            focal_gamma=args.focal_gamma,
            aux_dual_axis_manifest=args.aux_dual_axis_manifest,
            aux_train_families=(list(args.aux_train_families) if args.aux_train_families is not None else None),
            device_arg=args.device,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=common._json_default))
        return

    raise AssertionError(f"Unhandled mode: {args.mode!r}")


if __name__ == "__main__":
    main()
