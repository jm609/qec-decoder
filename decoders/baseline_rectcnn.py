from __future__ import annotations

"""
baseline_rectcnn.py

Rectangular-lattice CNN baseline for datasets produced by sample_dataset.py.

Current scope
-------------
- Consume detector events through a space-time rectangular syndrome layout.
- Keep the lattice holes as a fixed incoherent value instead of flattening detectors.
- Train either a binary axis-wise classifier or a multiclass logical_class4 classifier
  for one geometry / one family at a time.

Migration note
--------------
The rebuilt data path can now expose either:
- axis-wise binary targets (`logical_axis_flip`)
- per-shot 4-state targets (`logical_class4`)

This decoder supports both target styles and keeps the rectangular geometry stack
as the lightweight neural baseline for comparison work.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import copy
import datetime as dt
import json
import math
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    from geometry.rotated_rect import (
        RectangularSyndromeLayout,
        build_rectangular_syndrome_layout,
        build_rectangular_syndrome_volume,
        describe_rectangular_syndrome_layout,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geometry.rotated_rect import (
        RectangularSyndromeLayout,
        build_rectangular_syndrome_layout,
        build_rectangular_syndrome_volume,
        describe_rectangular_syndrome_layout,
    )


SCHEMA_VERSION_TRAIN = "baseline_rectcnn.train.v2"
SCHEMA_VERSION_EVAL = "baseline_rectcnn.eval.v2"
TARGET_MODE_AUTO = "auto"
TARGET_MODE_BINARY = "binary"
TARGET_MODE_LOGICAL_CLASS4 = "logical_class4"
TARGET_MODE_CHOICES = (
    TARGET_MODE_AUTO,
    TARGET_MODE_BINARY,
    TARGET_MODE_LOGICAL_CLASS4,
)


class MissingTorchError(ImportError):
    """Raised when PyTorch-dependent functionality is used without torch installed."""


@dataclass(frozen=True, slots=True)
class FamilyArtifacts:
    family_dir: Path
    samples_npz: Path
    metadata_json: Path
    circuit_path: Path | None
    dem_path: Path | None


@dataclass(frozen=True, slots=True)
class LoadedFamilyPayload:
    artifacts: FamilyArtifacts
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SplitSummary:
    train_count: int
    val_count: int
    test_count: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainResult:
    schema_version: str
    decoder: str
    created_at_utc: str
    input_mode: str
    family: str
    stage: str
    family_dir: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    split: dict[str, Any]
    training: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalResult:
    schema_version: str
    decoder: str
    created_at_utc: str
    input_mode: str
    family: str
    stage: str
    family_dir: str
    checkpoint: dict[str, Any]
    model: dict[str, Any]
    dataset: dict[str, Any]
    split: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch() -> None:
    if torch is None or nn is None or optim is None or DataLoader is None or TensorDataset is None:
        raise MissingTorchError(
            "PyTorch is required for baseline_rectcnn.py but is not installed in this "
            "Python environment. Install torch in the same environment used for dataset loading."
        )


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_family_artifacts(family_dir: str | Path) -> FamilyArtifacts:
    family_dir = Path(family_dir)
    samples_npz = family_dir / "samples.npz"
    metadata_json = family_dir / "metadata.json"
    circuit_path = family_dir / "circuit.stim"
    dem_path = family_dir / "detector_error_model.dem"

    missing = [p.name for p in (samples_npz, metadata_json) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Family directory is missing required artifacts: {missing}. family_dir={family_dir}"
        )

    return FamilyArtifacts(
        family_dir=family_dir,
        samples_npz=samples_npz,
        metadata_json=metadata_json,
        circuit_path=circuit_path if circuit_path.exists() else None,
        dem_path=dem_path if dem_path.exists() else None,
    )


def _load_family_payload(family_dir: str | Path) -> LoadedFamilyPayload:
    artifacts = _resolve_family_artifacts(family_dir)
    metadata = _read_json(artifacts.metadata_json)
    with np.load(artifacts.samples_npz) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return LoadedFamilyPayload(artifacts=artifacts, metadata=metadata, arrays=arrays)


def _resolve_manifest_family_dir(manifest_path: Path, raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw

    candidate = manifest_path.parent / raw
    if candidate.exists():
        return candidate.resolve()
    if raw.exists():
        return raw.resolve()
    return candidate.resolve()


def _resolve_input_family_dir(
    *,
    family_dir: Path | None,
    manifest: Path | None,
    family: str | None,
) -> tuple[str, Path, dict[str, Any] | None]:
    if family_dir is not None:
        return "family_dir", family_dir, None

    if manifest is None:
        raise ValueError("Either family_dir or manifest must be provided")
    if not family:
        raise ValueError("When --manifest is used, --family must also be provided")

    manifest_data = _read_json(manifest)
    family_dirs = manifest_data.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(
            f"manifest.json does not contain a non-empty family_dirs mapping: {manifest}"
        )
    if family not in family_dirs:
        raise KeyError(
            f"Requested family {family!r} missing from manifest. Available: {sorted(family_dirs)}"
        )
    return "manifest", _resolve_manifest_family_dir(manifest, family_dirs[family]), manifest_data


def _as_uint8_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_uint8_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _pick_binary_target(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if "logical_axis_flip" in arrays:
        return _as_uint8_1d(arrays["logical_axis_flip"], name="logical_axis_flip"), "logical_axis_flip"
    return _as_uint8_1d(arrays["logical_label"], name="logical_label"), "logical_label"


def _class4_label_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    targets = metadata.get("targets", {}) if isinstance(metadata, dict) else {}
    sampling = metadata.get("sampling", {}) if isinstance(metadata, dict) else {}
    mapping = sampling.get("logical_class4_mapping")
    if not isinstance(mapping, dict):
        mapping = targets.get("logical_class4_mapping")
    if isinstance(mapping, dict):
        names: list[str] = []
        for index in range(4):
            names.append(str(mapping.get(str(index), index)))
        return names
    return ["I", "X", "Z", "Y"]


def _pick_target(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    target_mode: str,
) -> dict[str, Any]:
    if target_mode not in TARGET_MODE_CHOICES:
        raise ValueError(
            f"Unsupported target_mode={target_mode!r}. Expected one of {TARGET_MODE_CHOICES}."
        )

    resolved_mode = target_mode
    if resolved_mode == TARGET_MODE_AUTO:
        targets = metadata.get("targets", {}) if isinstance(metadata, dict) else {}
        if (
            "logical_class4" in arrays
            and targets.get("logical_target_kind") == "per_shot_logical_class4"
        ):
            resolved_mode = TARGET_MODE_LOGICAL_CLASS4
        else:
            resolved_mode = TARGET_MODE_BINARY

    if resolved_mode == TARGET_MODE_LOGICAL_CLASS4:
        if "logical_class4" not in arrays:
            raise KeyError(
                "Requested logical_class4 target mode but samples.npz does not contain logical_class4."
            )
        target = _as_uint8_1d(arrays["logical_class4"], name="logical_class4")
        return {
            "target": target,
            "target_key": "logical_class4",
            "task_type": "multiclass",
            "num_classes": 4,
            "class_labels": _class4_label_names_from_metadata(metadata),
            "resolved_target_mode": TARGET_MODE_LOGICAL_CLASS4,
        }

    target, target_key = _pick_binary_target(arrays)
    return {
        "target": target,
        "target_key": target_key,
        "task_type": "binary",
        "num_classes": 2,
        "class_labels": ["0", "1"],
        "resolved_target_mode": TARGET_MODE_BINARY,
    }


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _validate_split(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}={total}"
        )
    for name, value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def build_split_indices(
    *,
    num_shots: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    _validate_split(train_ratio, val_ratio, test_ratio)
    if num_shots < 3:
        raise ValueError(f"Need at least 3 shots for train/val/test split, got {num_shots}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_shots)
    train_end = int(round(num_shots * train_ratio))
    val_end = train_end + int(round(num_shots * val_ratio))

    train_end = min(max(train_end, 1), num_shots - 2)
    val_end = min(max(val_end, train_end + 1), num_shots - 1)

    return {
        "train": np.sort(perm[:train_end].astype(np.int64, copy=False)),
        "val": np.sort(perm[train_end:val_end].astype(np.int64, copy=False)),
        "test": np.sort(perm[val_end:].astype(np.int64, copy=False)),
    }


def summarise_split_indices(
    split_indices: dict[str, np.ndarray],
    *,
    num_shots: int,
    seed: int,
) -> SplitSummary:
    train_count = int(split_indices["train"].shape[0])
    val_count = int(split_indices["val"].shape[0])
    test_count = int(split_indices["test"].shape[0])
    return SplitSummary(
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_fraction=float(train_count / num_shots),
        val_fraction=float(val_count / num_shots),
        test_fraction=float(test_count / num_shots),
        seed=seed,
    )


def _default_num_filters(distance: int) -> int:
    target = max(1, int(distance * distance - 1))
    return 1 << int(math.ceil(math.log2(target)))


def build_rect_input_tensor(
    arrays: dict[str, np.ndarray],
    *,
    fill_value: float,
) -> tuple[np.ndarray, RectangularSyndromeLayout]:
    layout = build_rectangular_syndrome_layout(arrays)
    volume = build_rectangular_syndrome_volume(arrays, layout=layout, fill_value=fill_value)
    return volume.astype(np.float32, copy=False), layout


def _subset_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(arr[indices], dtype=arr.dtype)


def _make_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(y)),
    )


def _make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class RectCNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        height: int,
        width: int,
        num_filters: int,
        dense_hidden_dim: int,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if height < 4 or width < 4:
            raise ValueError(
                f"RectCNNClassifier expects grid height/width >= 4, got {(height, width)}"
            )
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.out_dim = int(out_dim)
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=(2, 2), padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        flat_dim = int(num_filters * (height - 3) * (width - 3))
        self.fc1 = nn.Linear(flat_dim, dense_hidden_dim)
        self.fc2 = nn.Linear(dense_hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = torch.flatten(h, start_dim=1)
        h = self.dropout(self.relu(self.fc1(h)))
        out = self.fc2(h)
        return out.squeeze(1) if self.out_dim == 1 else out


class RectCNNBinary(RectCNNClassifier):
    def __init__(
        self,
        *,
        in_channels: int,
        height: int,
        width: int,
        num_filters: int,
        dense_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            height=height,
            width=width,
            num_filters=num_filters,
            dense_hidden_dim=dense_hidden_dim,
            dropout=dropout,
            out_dim=1,
        )


def _binary_metrics_from_probs(
    probs_np: np.ndarray,
    target_np: np.ndarray,
    *,
    threshold: float,
    bce_loss: float | None = None,
) -> dict[str, Any]:
    pred_np = (probs_np >= threshold).astype(np.uint8)
    tp = int(np.sum((pred_np == 1) & (target_np == 1)))
    tn = int(np.sum((pred_np == 0) & (target_np == 0)))
    fp = int(np.sum((pred_np == 1) & (target_np == 0)))
    fn = int(np.sum((pred_np == 0) & (target_np == 1)))

    num_examples = int(target_np.shape[0])
    accuracy = float((tp + tn) / num_examples) if num_examples else None
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    balanced_accuracy = float(0.5 * (recall + specificity))
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    auroc = _binary_auroc(target_np, probs_np)
    pr_auc = _binary_pr_auc(target_np, probs_np)
    average_precision = _binary_average_precision(target_np, probs_np)

    return {
        "num_examples": num_examples,
        "bce_loss": bce_loss,
        "threshold_used": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "label_error_rate": (1.0 - accuracy) if accuracy is not None else None,
        "positive_rate_target": float(target_np.mean()) if target_np.size > 0 else None,
        "positive_rate_predicted": float(pred_np.mean()) if pred_np.size > 0 else None,
        "mean_predicted_probability": float(probs_np.mean()) if probs_np.size > 0 else None,
        "confusion_matrix_logical_label": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits_f = np.asarray(logits, dtype=np.float64)
    max_logits = np.max(logits_f, axis=1, keepdims=True)
    exp_logits = np.exp(logits_f - max_logits)
    denom = np.sum(exp_logits, axis=1, keepdims=True)
    return (exp_logits / denom).astype(np.float32, copy=False)


def _multiclass_metrics_from_probs(
    probs_np: np.ndarray,
    target_np: np.ndarray,
    *,
    class_labels: list[str],
    loss: float | None = None,
) -> dict[str, Any]:
    probs = np.asarray(probs_np, dtype=np.float32)
    target = np.asarray(target_np, dtype=np.int64).reshape(-1)
    num_classes = int(probs.shape[1])
    pred = np.argmax(probs, axis=1).astype(np.int64, copy=False)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class, pred_class in zip(target, pred, strict=False):
        confusion[int(true_class), int(pred_class)] += 1

    num_examples = int(target.shape[0])
    accuracy = float(np.mean(pred == target)) if num_examples else None
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    per_class: list[dict[str, Any]] = []
    for class_index in range(num_classes):
        tp = int(confusion[class_index, class_index])
        fn = int(confusion[class_index, :].sum() - tp)
        fp = int(confusion[:, class_index].sum() - tp)
        support = int(confusion[class_index, :].sum())
        predicted_support = int(confusion[:, class_index].sum())
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        per_class.append(
            {
                "class_index": class_index,
                "class_label": class_labels[class_index] if class_index < len(class_labels) else str(class_index),
                "support": support,
                "predicted_support": predicted_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    counts_target = np.bincount(target, minlength=num_classes)
    counts_pred = np.bincount(pred, minlength=num_classes)
    return {
        "num_examples": num_examples,
        "cross_entropy_loss": loss,
        "accuracy": accuracy,
        "label_error_rate": (1.0 - accuracy) if accuracy is not None else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "mean_predicted_confidence": float(np.max(probs, axis=1).mean()) if num_examples else None,
        "class_labels": list(class_labels),
        "target_class_histogram": {
            (class_labels[index] if index < len(class_labels) else str(index)): int(counts_target[index])
            for index in range(num_classes)
        },
        "predicted_class_histogram": {
            (class_labels[index] if index < len(class_labels) else str(index)): int(counts_pred[index])
            for index in range(num_classes)
        },
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _binary_auroc(target: np.ndarray, scores: np.ndarray) -> float | None:
    target_u8 = np.asarray(target, dtype=np.uint8).reshape(-1)
    scores_f = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = int(np.sum(target_u8 == 1))
    neg = int(np.sum(target_u8 == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(scores_f, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores_f[order]
    start = 0
    while start < scores_f.shape[0]:
        end = start + 1
        while end < scores_f.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[target_u8 == 1]))
    u_stat = rank_sum_pos - pos * (pos + 1) / 2.0
    return float(u_stat / (pos * neg))


def _binary_pr_curve(target: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_u8 = np.asarray(target, dtype=np.uint8).reshape(-1)
    scores_f = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target_u8.size == 0:
        return np.asarray([1.0]), np.asarray([0.0])

    order = np.argsort(-scores_f, kind="mergesort")
    y = target_u8[order]
    tp = np.cumsum(y == 1).astype(np.float64)
    fp = np.cumsum(y == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    total_pos = float(np.sum(target_u8 == 1))
    if total_pos == 0.0:
        recall = np.zeros_like(tp)
    else:
        recall = tp / total_pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return precision, recall


def _binary_pr_auc(target: np.ndarray, scores: np.ndarray) -> float | None:
    total_pos = int(np.sum(np.asarray(target, dtype=np.uint8).reshape(-1) == 1))
    if total_pos == 0:
        return None
    precision, recall = _binary_pr_curve(target, scores)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.trapz(precision, recall))


def _binary_average_precision(target: np.ndarray, scores: np.ndarray) -> float | None:
    total_pos = int(np.sum(np.asarray(target, dtype=np.uint8).reshape(-1) == 1))
    if total_pos == 0:
        return None
    precision, recall = _binary_pr_curve(target, scores)
    delta = np.diff(recall)
    return float(np.sum(delta * precision[1:]))


def _evaluate_arrays(
    *,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    task_type: str,
    threshold: float | None,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label shot mismatch: {x.shape[0]} vs {y.shape[0]}")

    model.eval()
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    loss_sum = 0.0
    loss_count = 0
    criterion = _make_criterion(task_type)

    loader = _make_loader(
        _make_tensor_dataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            logits_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(yb.detach().cpu().numpy())
            loss_sum += float(loss.item()) * int(xb.shape[0])
            loss_count += int(xb.shape[0])

    logits_np = np.concatenate(logits_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)
    mean_loss = float(loss_sum / loss_count) if loss_count > 0 else None

    if task_type == "binary":
        probs_np = _sigmoid_np(logits_np.astype(np.float32, copy=False))
        target_np = labels_np.astype(np.uint8, copy=False)
        return _binary_metrics_from_probs(
            probs_np,
            target_np,
            threshold=float(0.5 if threshold is None else threshold),
            bce_loss=mean_loss,
        )

    if task_type == "multiclass":
        probs_np = _softmax_np(np.asarray(logits_np, dtype=np.float32))
        target_np = np.asarray(labels_np, dtype=np.uint8)
        return _multiclass_metrics_from_probs(
            probs_np,
            target_np,
            class_labels=(list(class_labels) if class_labels is not None else [str(k) for k in range(probs_np.shape[1])]),
            loss=mean_loss,
        )

    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _select_decision_rule_from_validation(
    *,
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: torch.device,
    task_type: str,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    if task_type == "multiclass":
        metrics = _evaluate_arrays(
            model=model,
            x=x_val,
            y=y_val,
            batch_size=batch_size,
            device=device,
            task_type=task_type,
            threshold=None,
            class_labels=class_labels,
        )
        return {
            "mode": "argmax_multiclass",
            "selected_threshold": None,
            "metric_name": "accuracy",
            "metric_value": metrics["accuracy"],
            "reason": "fixed_argmax_for_multiclass",
        }

    model.eval()
    loader = _make_loader(_make_tensor_dataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            logits_chunks.append(logits.detach().cpu().numpy())
            labels_chunks.append(yb.detach().cpu().numpy())

    probs_np = _sigmoid_np(np.concatenate(logits_chunks, axis=0).astype(np.float32, copy=False))
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.uint8, copy=False)

    unique_labels = np.unique(labels_np)
    degenerate_labels = bool(unique_labels.size < 2)
    best: dict[str, Any] | None = None
    for k in range(1, 100):
        threshold = float(k / 100.0)
        metrics = _binary_metrics_from_probs(probs_np, labels_np, threshold=threshold)
        metric_name = "accuracy" if degenerate_labels else "f1"
        metric_value = metrics["accuracy"] if degenerate_labels else metrics["f1"]
        candidate = {
            "mode": ("val_accuracy_degenerate" if degenerate_labels else "val_f1"),
            "selected_threshold": threshold,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "grid_size": 99,
            "recall_at_selected_threshold": metrics["recall"],
            "precision_at_selected_threshold": metrics["precision"],
            "reason": "selected_from_validation_grid_search",
        }
        if best is None or float(candidate["metric_value"]) > float(best["metric_value"]):
            best = candidate
    if best is None:
        raise AssertionError("Threshold selection unexpectedly produced no candidate")
    return best


def _make_criterion(task_type: str) -> nn.Module:
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    if task_type == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch_size = int(xb.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return float(total_loss / total_examples) if total_examples else float("nan")


def _set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _prepare_loaded_family(
    payload: LoadedFamilyPayload,
    *,
    fill_value: float,
    max_shots: int | None,
    target_mode: str = TARGET_MODE_AUTO,
) -> dict[str, Any]:
    arrays = payload.arrays
    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")
    full_target_info = _pick_target(arrays, payload.metadata, target_mode=target_mode)
    full_target = np.asarray(full_target_info["target"])

    if detector_events.shape[0] != logical_label.shape[0]:
        raise ValueError(
            "detector_events and logical_label have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {logical_label.shape[0]}"
        )
    if detector_events.shape[0] != full_target.shape[0]:
        raise ValueError(
            "detector_events and target have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {full_target.shape[0]}"
        )
    if detector_events.shape[0] != observable_flips.shape[0]:
        raise ValueError(
            "detector_events and observable_flips have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {observable_flips.shape[0]}"
        )
    if observable_flips.shape[1] == 1 and not np.array_equal(observable_flips[:, 0], logical_label):
        raise ValueError(
            "logical_label does not match observable_flips[:, 0] for a single-observable dataset."
        )

    prepared_arrays = copy.deepcopy(arrays)
    if max_shots is not None:
        for key, arr in list(prepared_arrays.items()):
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and int(arr.shape[0]) == int(detector_events.shape[0]):
                prepared_arrays[key] = arr[:max_shots]

    x, layout = build_rect_input_tensor(prepared_arrays, fill_value=fill_value)
    target_info = _pick_target(prepared_arrays, payload.metadata, target_mode=target_mode)
    y = np.asarray(target_info["target"])
    if target_info["task_type"] == "binary":
        y = y.astype(np.float32, copy=False)
    else:
        y = y.astype(np.int64, copy=False)

    return {
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "layout": layout,
        "bundle_info": describe_rectangular_syndrome_layout(layout),
        "shots_total_after_limit": int(x.shape[0]),
        "target_key": target_info["target_key"],
        "task_type": target_info["task_type"],
        "num_classes": int(target_info["num_classes"]),
        "class_labels": list(target_info["class_labels"]),
        "resolved_target_mode": target_info["resolved_target_mode"],
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
    dense_hidden_dim: int,
    num_filters: int | None,
    device_arg: str,
) -> TrainResult:
    _require_torch()
    _set_random_seeds(seed)

    input_mode, resolved_family_dir, manifest_data = _resolve_input_family_dir(
        family_dir=family_dir,
        manifest=manifest,
        family=family,
    )
    payload = _load_family_payload(resolved_family_dir)
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

    split_indices = build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_summary = summarise_split_indices(split_indices, num_shots=num_shots, seed=seed)

    x_train = _subset_rows(x, split_indices["train"])
    y_train = _subset_rows(y, split_indices["train"])
    x_val = _subset_rows(x, split_indices["val"])
    y_val = _subset_rows(y, split_indices["val"])
    x_test = _subset_rows(x, split_indices["test"])
    y_test = _subset_rows(y, split_indices["test"])

    circuit_meta = metadata.get("circuit", {}) if isinstance(metadata, dict) else {}
    distance = int(circuit_meta.get("distance", 0))
    resolved_num_filters = int(num_filters) if num_filters is not None else _default_num_filters(distance)

    model = RectCNNClassifier(
        in_channels=int(x.shape[1]),
        height=int(x.shape[2]),
        width=int(x.shape[3]),
        num_filters=resolved_num_filters,
        dense_hidden_dim=dense_hidden_dim,
        dropout=dropout,
        out_dim=(1 if task_type == "binary" else num_classes),
    )
    device = _pick_device(device_arg)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = _make_criterion(task_type)

    train_loader = _make_loader(_make_tensor_dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
        )
        val_metrics = _evaluate_arrays(
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

    threshold_selection = _select_decision_rule_from_validation(
        model=model,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        class_labels=class_labels,
    )
    selected_threshold_raw = threshold_selection.get("selected_threshold")
    selected_threshold = (
        float(selected_threshold_raw)
        if selected_threshold_raw is not None
        else None
    )
    test_metrics = _evaluate_arrays(
        model=model,
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        threshold=selected_threshold,
        class_labels=class_labels,
    )
    elapsed = time.perf_counter() - started

    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "baseline_rectcnn",
        "created_at_utc": _utc_now_iso(),
        "family": metadata.get("family"),
        "stage": metadata.get("stage"),
        "family_dir": str(resolved_family_dir.as_posix()),
        "fill_value": float(fill_value),
        "model_hparams": {
            "in_channels": int(x.shape[1]),
            "height": int(x.shape[2]),
            "width": int(x.shape[3]),
            "num_filters": resolved_num_filters,
            "dense_hidden_dim": int(dense_hidden_dim),
            "dropout": float(dropout),
            "distance": distance,
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

    result = TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="baseline_rectcnn",
        created_at_utc=_utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family")),
        stage=str(metadata.get("stage")),
        family_dir=resolved_family_dir.as_posix(),
        model={
            "architecture": "rectcnn_classifier",
            "input_representation": "rectangular_syndrome_volume",
            "device": str(device),
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
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
        _write_json(train_json_out, result.to_dict())
    return result


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
) -> EvalResult:
    _require_torch()

    input_mode, resolved_family_dir, _ = _resolve_input_family_dir(
        family_dir=family_dir,
        manifest=manifest,
        family=family,
    )
    payload = _load_family_payload(resolved_family_dir)
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

    device = _pick_device(device_arg)
    model = RectCNNClassifier(
        in_channels=int(model_hparams["in_channels"]),
        height=int(model_hparams["height"]),
        width=int(model_hparams["width"]),
        num_filters=int(model_hparams["num_filters"]),
        dense_hidden_dim=int(model_hparams["dense_hidden_dim"]),
        dropout=float(model_hparams["dropout"]),
        out_dim=(1 if task_type == "binary" else num_classes),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    threshold_selection = dict(checkpoint.get("threshold_selection", {}))
    threshold_raw = threshold_selection.get("selected_threshold", 0.5)
    threshold = float(threshold_raw) if threshold_raw is not None else None
    metrics = _evaluate_arrays(
        model=model,
        x=x,
        y=y,
        batch_size=batch_size,
        device=device,
        task_type=task_type,
        threshold=threshold,
        class_labels=class_labels,
    )

    result = EvalResult(
        schema_version=SCHEMA_VERSION_EVAL,
        decoder="baseline_rectcnn",
        created_at_utc=_utc_now_iso(),
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
            "architecture": "rectcnn_classifier",
            "input_representation": "rectangular_syndrome_volume",
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
        _write_json(eval_json_out, result.to_dict())
    return result


def _add_common_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family-dir", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--family", type=str, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a rectangular-lattice CNN baseline on sample_dataset.py outputs."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    _add_common_input_args(train_parser)
    train_parser.add_argument("--checkpoint-out", type=Path, required=True)
    train_parser.add_argument("--train-json-out", type=Path, default=None)
    train_parser.add_argument("--fill-value", type=float, default=-0.5)
    train_parser.add_argument("--target-mode", choices=list(TARGET_MODE_CHOICES), default=TARGET_MODE_AUTO)
    train_parser.add_argument("--max-shots", type=int, default=None)
    train_parser.add_argument("--train-ratio", type=float, default=0.8)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=12345)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--dense-hidden-dim", type=int, default=50)
    train_parser.add_argument("--num-filters", type=int, default=None)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval")
    _add_common_input_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--eval-json-out", type=Path, default=None)
    eval_parser.add_argument("--target-mode", choices=list(TARGET_MODE_CHOICES), default=TARGET_MODE_AUTO)
    eval_parser.add_argument("--max-shots", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--device", type=str, default="auto")

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
            dense_hidden_dim=args.dense_hidden_dim,
            num_filters=args.num_filters,
            device_arg=args.device,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=_json_default))
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
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=_json_default))
        return

    raise AssertionError(f"Unhandled mode: {args.mode!r}")


if __name__ == "__main__":
    main()
