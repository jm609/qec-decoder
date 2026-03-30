from __future__ import annotations

"""
baseline_nn.py

Minimal PyTorch neural baseline for datasets produced by sample_dataset.py.

Scope of v1
-----------
- Train one binary classifier for one geometry / one family at a time.
- Use shot-level raw split indices first, then build input views.
- Support two flat input views:
    1. flat_event     : detector_events only
    2. flat_semantic  : detector_events plus detector semantic metadata masked by events
- Save structured JSON outputs for both training and evaluation.

Supported inputs
----------------
1. A single family directory containing:
   - samples.npz
   - metadata.json
2. A manifest.json produced by sample_dataset.py, plus one selected family.

Non-goals of v1
---------------
- Multi-family joint training
- Cross-geometry training
- CNN / GNN / sequence models
- Multi-observable decoding
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
import argparse
import copy
import datetime as dt
import json
import math
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


ViewName = Literal["flat_event", "flat_semantic"]
SCHEMA_VERSION_TRAIN = "baseline_nn.train.v1"
SCHEMA_VERSION_EVAL = "baseline_nn.eval.v1"


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


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if torch is None or nn is None or optim is None or DataLoader is None or TensorDataset is None:
        raise MissingTorchError(
            "PyTorch is required for baseline_nn.py but is not installed in this Python "
            "environment. Install torch in the same environment used for dataset loading."
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

    missing = [
        p.name
        for p in (samples_npz, metadata_json)
        if not p.exists()
    ]
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



def _as_int16_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int16).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)



def _num_parameters(model: nn.Module) -> int: # fuck this yellow line
    return int(sum(p.numel() for p in model.parameters()))



def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


def _validate_split(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )



def build_split_indices(
    *,
    num_shots: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Create deterministic raw shot splits.

    Important design choice:
    - Split indices are created immediately from raw shot ids, before any feature view is built.
    """
    if num_shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {num_shots}")

    _validate_split(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_shots)

    train_count = int(num_shots * train_ratio)
    val_count = int(num_shots * val_ratio)
    test_count = num_shots - train_count - val_count

    if train_count < 1:
        raise ValueError("train split is empty; increase num_shots or train_ratio")
    if val_count < 1:
        raise ValueError("val split is empty; increase num_shots or val_ratio")
    if test_count < 1:
        raise ValueError("test split is empty; increase num_shots or test_ratio")

    train_idx = np.sort(perm[:train_count])
    val_idx = np.sort(perm[train_count:train_count + val_count])
    test_idx = np.sort(perm[train_count + val_count:])

    return {
        "train": train_idx.astype(np.int64, copy=False),
        "val": val_idx.astype(np.int64, copy=False),
        "test": test_idx.astype(np.int64, copy=False),
    }



def summarise_split_indices(split_indices: dict[str, np.ndarray], *, num_shots: int, seed: int) -> SplitSummary:
    train_count = int(split_indices["train"].size)
    val_count = int(split_indices["val"].size)
    test_count = int(split_indices["test"].size)
    return SplitSummary(
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_fraction=float(train_count / num_shots),
        val_fraction=float(val_count / num_shots),
        test_fraction=float(test_count / num_shots),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Flat tensor builders
# ---------------------------------------------------------------------------


def build_flat_event_tensor(arrays: dict[str, np.ndarray]) -> np.ndarray:
    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    return detector_events.astype(np.float32, copy=False)



def _binary_one_hot(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((v.shape[0], 2), dtype=np.float32)
    out[np.arange(v.shape[0]), np.clip(v, 0, 1)] = 1.0
    return out



def build_flat_semantic_tensor(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """
    Build a masked semantic view.

    Per detector we concatenate:
      [event,
       event * time_norm,
       event * final_round_flag,
       event * boundary_flag,
       event * checkerboard_class_0,
       event * checkerboard_class_1,
       event * is_x_check,
       event * is_z_check]

    The semantic features are masked by detector_events so inactive detectors contribute zeros.
    """
    detector_events = build_flat_event_tensor(arrays)
    num_shots, num_detectors = detector_events.shape

    detector_time_index = _as_int16_1d(arrays["detector_time_index"], name="detector_time_index")
    detector_final_round_flag = _as_uint8_1d(arrays["detector_final_round_flag"], name="detector_final_round_flag")
    detector_boundary_flag = _as_uint8_1d(arrays["detector_boundary_flag"], name="detector_boundary_flag")
    detector_checkerboard_class = _as_uint8_1d(arrays["detector_checkerboard_class"], name="detector_checkerboard_class")
    detector_type = _as_uint8_1d(arrays["detector_type"], name="detector_type")

    semantic_lengths = {
        "detector_time_index": detector_time_index.shape[0],
        "detector_final_round_flag": detector_final_round_flag.shape[0],
        "detector_boundary_flag": detector_boundary_flag.shape[0],
        "detector_checkerboard_class": detector_checkerboard_class.shape[0],
        "detector_type": detector_type.shape[0],
    }
    mismatched = {k: v for k, v in semantic_lengths.items() if v != num_detectors}
    if mismatched:
        raise ValueError(
            "Semantic detector arrays have inconsistent detector dimension: "
            f"expected {num_detectors}, got {mismatched}"
        )

    max_time = int(detector_time_index.max()) if detector_time_index.size > 0 else 0
    if max_time > 0:
        time_norm = detector_time_index.astype(np.float32) / float(max_time)
    else:
        time_norm = np.zeros_like(detector_time_index, dtype=np.float32)

    checker_one_hot = _binary_one_hot(detector_checkerboard_class)
    is_x_check = (detector_type == 1).astype(np.float32)
    is_z_check = (detector_type == 2).astype(np.float32)

    semantic_per_detector = np.stack(
        [
            np.ones(num_detectors, dtype=np.float32),
            time_norm.astype(np.float32, copy=False),
            detector_final_round_flag.astype(np.float32, copy=False),
            detector_boundary_flag.astype(np.float32, copy=False),
            checker_one_hot[:, 0],
            checker_one_hot[:, 1],
            is_x_check,
            is_z_check,
        ],
        axis=1,
    )

    masked = detector_events[:, :, None] * semantic_per_detector[None, :, :]
    return masked.reshape(num_shots, num_detectors * semantic_per_detector.shape[1]).astype(
        np.float32,
        copy=False,
    )



def build_input_view(arrays: dict[str, np.ndarray], *, view: ViewName) -> np.ndarray:
    if view == "flat_event":
        return build_flat_event_tensor(arrays)
    if view == "flat_semantic":
        return build_flat_semantic_tensor(arrays)
    raise ValueError(f"Unsupported view: {view!r}")



def describe_view(*, view: ViewName, num_detectors: int) -> dict[str, Any]:
    if view == "flat_event":
        return {
            "view": view,
            "features_per_detector": 1,
            "feature_names_per_detector": ["event"],
            "input_dim": num_detectors,
        }
    if view == "flat_semantic":
        feature_names = [
            "event",
            "event_x_time_norm",
            "event_x_final_round_flag",
            "event_x_boundary_flag",
            "event_x_checkerboard_class_0",
            "event_x_checkerboard_class_1",
            "event_x_is_x_check",
            "event_x_is_z_check",
        ]
        return {
            "view": view,
            "features_per_detector": len(feature_names),
            "feature_names_per_detector": feature_names,
            "input_dim": int(num_detectors * len(feature_names)),
        }
    raise ValueError(f"Unsupported view: {view!r}")


# ---------------------------------------------------------------------------
# PyTorch minimum baseline
# ---------------------------------------------------------------------------


class MLPDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        dims = [int(input_dim), *(int(h) for h in hidden_dims)]
        layers: list[nn.Module] = [] # fuck this yellow line
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # fuck this yellow line
        logits = self.net(x)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics and evaluation helpers
# ---------------------------------------------------------------------------


def _subset_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr[indices])



def _make_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    _require_torch()
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    return TensorDataset(x_t, y_t)



def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    _require_torch()
    ds = _make_tensor_dataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



def _evaluate_arrays(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    _require_torch()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label shot mismatch: {x.shape[0]} vs {y.shape[0]}")

    loader = _make_loader(x, y, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            batch_n = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_examples += batch_n
            logits_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(yb.detach().cpu().numpy())

    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32, copy=False)
    labels_np = np.concatenate(label_chunks, axis=0).astype(np.float32, copy=False)
    probs_np = _sigmoid_np(logits_np)
    pred_np = (probs_np >= threshold).astype(np.uint8)
    target_np = labels_np.astype(np.uint8)

    tp = int(np.sum((pred_np == 1) & (target_np == 1)))
    tn = int(np.sum((pred_np == 0) & (target_np == 0)))
    fp = int(np.sum((pred_np == 1) & (target_np == 0)))
    fn = int(np.sum((pred_np == 0) & (target_np == 1)))

    accuracy = float((tp + tn) / total_examples) if total_examples > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))

    return {
        "num_examples": total_examples,
        "bce_loss": float(total_loss / total_examples) if total_examples > 0 else None,
        "accuracy": accuracy,
        "label_error_rate": float(1.0 - accuracy) if accuracy is not None else None,
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
        "f1": f1,
    }



def _train_one_epoch( 
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: optim.Optimizer,
    device: torch.device, # fuck these yellow lines
) -> float:
    _require_torch()
    criterion = nn.BCEWithLogitsLoss()
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

        batch_n = int(xb.shape[0])
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n

    return float(total_loss / total_examples) if total_examples > 0 else math.nan


# ---------------------------------------------------------------------------
# Training / evaluation core
# ---------------------------------------------------------------------------


def _set_random_seeds(seed: int) -> None:
    _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)



def _pick_device(device_arg: str) -> torch.device: # fuck this yellow line
    _require_torch()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)



def _prepare_loaded_family(
    payload: LoadedFamilyPayload,
    *,
    max_shots: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {k: np.asarray(v) for k, v in payload.arrays.items()}

    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")

    if detector_events.shape[0] != logical_label.shape[0]:
        raise ValueError(
            "detector_events and logical_label have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {logical_label.shape[0]}"
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

    num_shots_total = int(detector_events.shape[0])
    if max_shots is not None:
        if max_shots < 1:
            raise ValueError("max_shots must be >= 1 when provided")
        for key in list(arrays):
            if arrays[key].ndim >= 1 and arrays[key].shape[0] == num_shots_total:
                arrays[key] = arrays[key][:max_shots]

    prepared = {
        key: np.asarray(value)
        for key, value in arrays.items()
    }
    metadata_context = {
        "shots_total_after_limit": int(prepared["detector_events"].shape[0]),
        "max_shots": max_shots,
    }
    return prepared, metadata_context



def train_family_dir(
    family_dir: str | Path,
    *,
    input_mode: str,
    view: ViewName,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    train_seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dims: Sequence[int],
    dropout: float,
    threshold: float,
    device_arg: str,
    max_shots: int | None,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    _require_torch()
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not (0.0 < lr):
        raise ValueError(f"lr must be > 0, got {lr}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    payload = _load_family_payload(family_dir)
    arrays, prepare_info = _prepare_loaded_family(payload, max_shots=max_shots)
    labels = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    num_shots = int(labels.shape[0])

    # Raw shot split is created before any feature view is built.
    split_indices = build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    split_summary = summarise_split_indices(split_indices, num_shots=num_shots, seed=split_seed)

    x = build_input_view(arrays, view=view)
    y = labels.astype(np.float32, copy=False)

    x_train = _subset_rows(x, split_indices["train"])
    y_train = _subset_rows(y, split_indices["train"])
    x_val = _subset_rows(x, split_indices["val"])
    y_val = _subset_rows(y, split_indices["val"])
    x_test = _subset_rows(x, split_indices["test"])
    y_test = _subset_rows(y, split_indices["test"])

    device = _pick_device(device_arg)
    _set_random_seeds(train_seed)

    model = MLPDecoder(input_dim=int(x.shape[1]), hidden_dims=hidden_dims, dropout=dropout)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = _evaluate_arrays(
            model,
            x_val,
            y_val,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
        )
        val_loss = float(val_metrics["bce_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_bce_loss": float(train_loss),
                "val_bce_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    train_wall_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)

    train_metrics = _evaluate_arrays(
        model,
        x_train,
        y_train,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )
    val_metrics = _evaluate_arrays(
        model,
        x_val,
        y_val,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )
    test_metrics = _evaluate_arrays(
        model,
        x_test,
        y_test,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )

    metadata = payload.metadata
    num_detectors = int(np.asarray(arrays["detector_events"]).shape[1])
    view_info = describe_view(view=view, num_detectors=num_detectors)

    ckpt_saved = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": SCHEMA_VERSION_TRAIN,
            "decoder": "baseline_nn_mlp",
            "created_at_utc": _utc_now_iso(),
            "family": metadata.get("family", payload.artifacts.family_dir.name),
            "stage": metadata.get("stage", "unknown"),
            "family_dir": payload.artifacts.family_dir.as_posix(),
            "view": view,
            "model_hparams": {
                "input_dim": int(x.shape[1]),
                "hidden_dims": [int(v) for v in hidden_dims],
                "dropout": float(dropout),
            },
            "training_hparams": {
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                "test_ratio": float(test_ratio),
                "split_seed": int(split_seed),
                "train_seed": int(train_seed),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "threshold": float(threshold),
                "max_shots": max_shots,
            },
            "dataset_context": {
                "schema_version": metadata.get("schema_version"),
                "family": metadata.get("family"),
                "stage": metadata.get("stage"),
                "circuit": metadata.get("circuit", {}),
                "scaffold": metadata.get("scaffold", {}),
                "num_detectors": num_detectors,
                "view_info": view_info,
            },
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        ckpt_saved = checkpoint_path.as_posix()

    result = TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="baseline_nn_mlp",
        created_at_utc=_utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family", payload.artifacts.family_dir.name)),
        stage=str(metadata.get("stage", "unknown")),
        family_dir=payload.artifacts.family_dir.as_posix(),
        model={
            "architecture": "mlp",
            "view": view,
            "input_dim": int(x.shape[1]),
            "hidden_dims": [int(v) for v in hidden_dims],
            "dropout": float(dropout),
            "num_parameters": _num_parameters(model),
            "device": str(device),
            "num_detectors": num_detectors,
            "view_info": view_info,
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit", {}),
            "scaffold": metadata.get("scaffold", {}),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "circuit_stim": payload.artifacts.circuit_path.as_posix() if payload.artifacts.circuit_path is not None else None,
            "detector_error_model_dem": payload.artifacts.dem_path.as_posix() if payload.artifacts.dem_path is not None else None,
            "qc_stats_from_dataset": metadata.get("qc_stats", {}),
            "num_shots_total_after_limit": prepare_info["shots_total_after_limit"],
            "max_shots": prepare_info["max_shots"],
        },
        split={
            **split_summary.to_dict(),
            "train_ratio_requested": float(train_ratio),
            "val_ratio_requested": float(val_ratio),
            "test_ratio_requested": float(test_ratio),
        },
        training={
            "split_seed": int(split_seed),
            "train_seed": int(train_seed),
            "epochs_requested": int(epochs),
            "epochs_completed": int(epochs),
            "best_val_epoch": int(best_epoch),
            "best_val_bce_loss": float(best_val_loss),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "threshold": float(threshold),
            "train_wall_seconds": float(train_wall_seconds),
            "history": history,
        },
        metrics={
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        artifacts={
            "checkpoint_path": ckpt_saved,
        },
    )
    return result.to_dict()



def evaluate_checkpoint_on_family(
    checkpoint_path: str | Path,
    family_dir: str | Path,
    *,
    input_mode: str,
    split_name: Literal["train", "val", "test", "all"],
    batch_size: int,
    device_arg: str,
    max_shots: int | None,
) -> dict[str, Any]:
    _require_torch()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    payload = _load_family_payload(family_dir)
    arrays, prepare_info = _prepare_loaded_family(payload, max_shots=max_shots)
    labels = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    num_shots = int(labels.shape[0])

    training_hparams = checkpoint.get("training_hparams", {})
    view = str(checkpoint.get("view"))
    input_dim = int(checkpoint.get("model_hparams", {}).get("input_dim"))
    hidden_dims = [int(v) for v in checkpoint.get("model_hparams", {}).get("hidden_dims", [128, 64])]
    dropout = float(checkpoint.get("model_hparams", {}).get("dropout", 0.1))
    threshold = float(training_hparams.get("threshold", 0.5))

    split_indices = build_split_indices(
        num_shots=num_shots,
        train_ratio=float(training_hparams.get("train_ratio", 0.8)),
        val_ratio=float(training_hparams.get("val_ratio", 0.1)),
        test_ratio=float(training_hparams.get("test_ratio", 0.1)),
        seed=int(training_hparams.get("split_seed", 20260328)),
    )
    split_summary = summarise_split_indices(
        split_indices,
        num_shots=num_shots,
        seed=int(training_hparams.get("split_seed", 20260328)),
    )

    x = build_input_view(arrays, view=view)  # type: ignore[arg-type]
    if int(x.shape[1]) != input_dim:
        raise ValueError(
            "Checkpoint input_dim does not match dataset view dimension: "
            f"checkpoint={input_dim}, current={x.shape[1]}"
        )
    y = labels.astype(np.float32, copy=False)

    if split_name == "all":
        indices = np.arange(num_shots, dtype=np.int64)
    else:
        indices = split_indices[split_name]

    x_eval = _subset_rows(x, indices)
    y_eval = _subset_rows(y, indices)

    device = _pick_device(device_arg)
    model = MLPDecoder(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = _evaluate_arrays(
        model,
        x_eval,
        y_eval,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )

    metadata = payload.metadata
    num_detectors = int(np.asarray(arrays["detector_events"]).shape[1])
    view_info = describe_view(view=view, num_detectors=num_detectors)  # type: ignore[arg-type]

    result = EvalResult(
        schema_version=SCHEMA_VERSION_EVAL,
        decoder="baseline_nn_mlp",
        created_at_utc=_utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family", payload.artifacts.family_dir.name)),
        stage=str(metadata.get("stage", "unknown")),
        family_dir=payload.artifacts.family_dir.as_posix(),
        checkpoint={
            "path": checkpoint_path.as_posix(),
            "created_at_utc": checkpoint.get("created_at_utc"),
            "family": checkpoint.get("family"),
            "stage": checkpoint.get("stage"),
            "view": checkpoint.get("view"),
        },
        model={
            "architecture": "mlp",
            "view": view,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "device": str(device),
            "num_parameters": _num_parameters(model),
            "num_detectors": num_detectors,
            "view_info": view_info,
        },
        dataset={
            "dataset_schema_version": metadata.get("schema_version"),
            "family": metadata.get("family"),
            "stage": metadata.get("stage"),
            "circuit": metadata.get("circuit", {}),
            "scaffold": metadata.get("scaffold", {}),
            "metadata_json": payload.artifacts.metadata_json.as_posix(),
            "samples_npz": payload.artifacts.samples_npz.as_posix(),
            "num_shots_total_after_limit": prepare_info["shots_total_after_limit"],
            "max_shots": prepare_info["max_shots"],
        },
        split={
            **split_summary.to_dict(),
            "evaluated_split": split_name,
            "evaluated_count": int(indices.size),
        },
        metrics=metrics,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def _add_common_input_args(parser: argparse.ArgumentParser) -> None:
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--family-dir", type=Path, help="Path to one family directory")
    src.add_argument("--manifest", type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Family name to resolve when --manifest is used",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Optionally use only the first N shots for quick tests",
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate a minimal PyTorch neural baseline on sample_dataset.py outputs. "
            "v1 is one-family / one-geometry binary classification."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train baseline_nn on one family dataset")
    _add_common_input_args(train)
    train.add_argument("--view", choices=["flat_event", "flat_semantic"], default="flat_event")
    train.add_argument("--train-ratio", type=float, default=0.80)
    train.add_argument("--val-ratio", type=float, default=0.10)
    train.add_argument("--test-ratio", type=float, default=0.10)
    train.add_argument("--split-seed", type=int, default=20260328)
    train.add_argument("--train-seed", type=int, default=20260328)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--threshold", type=float, default=0.50)
    train.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    train.add_argument("--checkpoint-path", type=Path, default=None)
    train.add_argument("--out-json", type=Path, default=None)

    evaluate = subparsers.add_parser("eval", help="Evaluate a saved checkpoint on one family dataset")
    _add_common_input_args(evaluate)
    evaluate.add_argument("--checkpoint-path", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    evaluate.add_argument("--out-json", type=Path, default=None)

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    input_mode, resolved_family_dir, _manifest_data = _resolve_input_family_dir(
        family_dir=args.family_dir,
        manifest=args.manifest,
        family=args.family,
    )

    if args.command == "train":
        result = train_family_dir(
            resolved_family_dir,
            input_mode=input_mode,
            view=args.view,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            train_seed=args.train_seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            threshold=args.threshold,
            device_arg=args.device,
            max_shots=args.max_shots,
            checkpoint_path=args.checkpoint_path,
        )
    else:
        result = evaluate_checkpoint_on_family(
            args.checkpoint_path,
            resolved_family_dir,
            input_mode=input_mode,
            split_name=args.split,
            batch_size=args.batch_size,
            device_arg=args.device,
            max_shots=args.max_shots,
        )

    if args.out_json is not None:
        _write_json(args.out_json, result)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
