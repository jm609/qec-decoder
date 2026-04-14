from __future__ import annotations

"""
baseline_tracknn.py

Geometry-aware PyTorch baseline for datasets produced by sample_dataset.py.

Scope of v1
-----------
- Train one binary classifier for one geometry / one family at a time.
- Reuse the baseline_nn.py shell where possible:
    * family_dir / manifest input handling
    * raw shot split generation
    * train / eval CLI
    * structured JSON outputs
- Replace the flat tensor builder + MLP with:
    * build_track_layout
    * build_track_tensor_bundle
    * shared temporal encoder (default: GRU)
    * lightweight spatial aggregation (default: mean pooling)

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
- Multi-observable decoding
- Heavy AQ2-scale transformer stacks
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
    from track_common import (
        TrackLayout,
        TrackTensorBundle,
        build_track_layout,
        build_track_tensor_bundle,
        describe_track_bundle,
        validate_layout_against_known_geometry,
    )
except ImportError:
    from decoders.track_common import (
        TrackLayout,
        TrackTensorBundle,
        build_track_layout,
        build_track_tensor_bundle,
        describe_track_bundle,
        validate_layout_against_known_geometry,
    )


TemporalEncoderName = Literal["gru", "conv1d"]
SpatialModeName = Literal["mean", "attention"]
SCHEMA_VERSION_TRAIN = "baseline_tracknn.train.v2"
SCHEMA_VERSION_EVAL = "baseline_tracknn.eval.v2"


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
            "PyTorch is required for baseline_tracknn.py but is not installed in this Python "
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



def _as_float32_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)



def _num_parameters(model: nn.Module) -> int:
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
# Track layout + tensor builders
# ---------------------------------------------------------------------------


def _subset_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr[indices])



def _prepare_loaded_family(
    payload: LoadedFamilyPayload,
    *,
    max_shots: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {k: np.asarray(v) for k, v in payload.arrays.items()}

    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")

    required_detector_arrays = {
        "detector_coordinates": _as_float32_2d,
        "detector_time_index": _as_int16_1d,
        "detector_final_round_flag": _as_uint8_1d,
        "detector_boundary_flag": _as_uint8_1d,
        "detector_checkerboard_class": _as_uint8_1d,
        "detector_type": _as_uint8_1d,
    }
    prepared_detector_arrays: dict[str, np.ndarray] = {}
    for key, fn in required_detector_arrays.items():
        if key not in arrays:
            raise KeyError(
                f"sample_dataset.v6-style array {key!r} is required for baseline_tracknn.py"
            )
        prepared_detector_arrays[key] = fn(arrays[key], name=key)

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
        detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
        logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")
        observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")

    prepared = {
        "detector_events": detector_events,
        "logical_label": logical_label,
        "observable_flips": observable_flips,
        **prepared_detector_arrays,
    }
    metadata_context = {
        "shots_total_after_limit": int(prepared["detector_events"].shape[0]),
        "max_shots": max_shots,
    }
    return prepared, metadata_context



def _validate_layout_against_known_geometry(layout: TrackLayout, metadata: dict[str, Any]) -> None:
    validate_layout_against_known_geometry(layout, metadata)



def _slice_bundle_by_indices(bundle: TrackTensorBundle, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _subset_rows(bundle.x, indices),
        _subset_rows(bundle.mask, indices),
        _subset_rows(bundle.y, indices),
    )


# ---------------------------------------------------------------------------
# Dataset / loaders
# ---------------------------------------------------------------------------


def _make_tensor_dataset(x: np.ndarray, mask: np.ndarray, y: np.ndarray) -> TensorDataset:
    _require_torch()
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    m_t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    return TensorDataset(x_t, m_t, y_t)



def _make_loader(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    _require_torch()
    ds = _make_tensor_dataset(x, mask, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------


class TemporalGRUEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        if torch.any(lengths <= 0):
            raise ValueError("Each track must have at least one valid timestep")
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _packed_out, h_n = self.gru(packed)
        return h_n[-1]


class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, F] -> conv over T
        x = x.transpose(1, 2)
        h = self.net(x).transpose(1, 2)
        mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (h * mask_f).sum(dim=1) / denom


class TinyTrackAttention(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads: {hidden_dim} vs {num_heads}"
            )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, track_valid: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~track_valid.bool()
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(x + attn_out)


class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, track_valid: torch.Tensor) -> torch.Tensor:
        mask = track_valid.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom


class TrackNNLite(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        temporal_encoder: TemporalEncoderName = "gru",
        temporal_hidden_dim: int = 64,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.1,
        spatial_mode: SpatialModeName = "mean",
        spatial_heads: int = 2,
        mlp_hidden_dim: int = 64,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if mlp_hidden_dim < 1:
            raise ValueError(f"mlp_hidden_dim must be >= 1, got {mlp_hidden_dim}")
        if not (0.0 <= head_dropout < 1.0):
            raise ValueError(f"head_dropout must be in [0, 1), got {head_dropout}")

        self.temporal_encoder_name = temporal_encoder
        if temporal_encoder == "gru":
            self.temporal_encoder = TemporalGRUEncoder(
                input_dim=input_dim,
                hidden_dim=temporal_hidden_dim,
                num_layers=temporal_layers,
                dropout=temporal_dropout,
            )
            hidden_dim = temporal_hidden_dim
        elif temporal_encoder == "conv1d":
            self.temporal_encoder = TemporalConvEncoder(
                input_dim=input_dim,
                hidden_dim=temporal_hidden_dim,
                dropout=temporal_dropout,
            )
            hidden_dim = temporal_hidden_dim
        else:
            raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder!r}")

        self.spatial_mode = spatial_mode
        self.spatial_attention = (
            TinyTrackAttention(hidden_dim=hidden_dim, num_heads=spatial_heads)
            if spatial_mode == "attention"
            else None
        )
        self.pool = MaskedMeanPool()
        head_layers: list[nn.Module] = [nn.Linear(hidden_dim, mlp_hidden_dim), nn.ReLU()]
        if head_dropout > 0.0:
            head_layers.append(nn.Dropout(head_dropout))
        head_layers.append(nn.Linear(mlp_hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, T, F], got {tuple(x.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"mask must have shape [B, N, T], got {tuple(mask.shape)}")
        if x.shape[:3] != mask.shape:
            raise ValueError(
                f"x and mask leading dimensions must match, got x={tuple(x.shape)}, mask={tuple(mask.shape)}"
            )

        batch_size, num_tracks, time_steps, feat_dim = x.shape
        flat_x = x.reshape(batch_size * num_tracks, time_steps, feat_dim)
        flat_mask = mask.reshape(batch_size * num_tracks, time_steps)
        temporal_repr = self.temporal_encoder(flat_x, flat_mask)
        track_repr = temporal_repr.reshape(batch_size, num_tracks, -1)
        track_valid = mask.any(dim=2)

        if self.spatial_attention is not None:
            track_repr = self.spatial_attention(track_repr, track_valid)
        pooled = self.pool(track_repr, track_valid)
        logits = self.head(pooled)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics and evaluation helpers
# ---------------------------------------------------------------------------


def _safe_ratio(numer: float, denom: float) -> float | None:
    return float(numer / denom) if denom > 0 else None



def _binary_confusion_from_probs(
    probs_np: np.ndarray,
    labels_np: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    pred_np = (probs_np >= threshold).astype(np.uint8)
    target_np = labels_np.astype(np.uint8)
    return {
        "tp": int(np.sum((pred_np == 1) & (target_np == 1))),
        "tn": int(np.sum((pred_np == 0) & (target_np == 0))),
        "fp": int(np.sum((pred_np == 1) & (target_np == 0))),
        "fn": int(np.sum((pred_np == 0) & (target_np == 1))),
    }



def _roc_curve_binary(labels_np: np.ndarray, probs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    y = np.asarray(labels_np, dtype=np.uint8).reshape(-1)
    scores = np.asarray(probs_np, dtype=np.float64).reshape(-1)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0 or y.size == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]
    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idx].astype(np.float64, copy=False)
    fps = (1 + threshold_idx - tps).astype(np.float64, copy=False)

    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    return fpr, tpr



def _precision_recall_curve_binary(labels_np: np.ndarray, probs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    y = np.asarray(labels_np, dtype=np.uint8).reshape(-1)
    scores = np.asarray(probs_np, dtype=np.float64).reshape(-1)
    positives = int(np.sum(y == 1))
    if positives == 0 or y.size == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]
    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idx].astype(np.float64, copy=False)
    fps = (1 + threshold_idx - tps).astype(np.float64, copy=False)

    precision = np.r_[1.0, tps / np.maximum(tps + fps, 1.0)]
    recall = np.r_[0.0, tps / positives]
    return precision, recall



def _binary_ranking_metrics(labels_np: np.ndarray, probs_np: np.ndarray) -> dict[str, float | None]:
    roc = _roc_curve_binary(labels_np, probs_np)
    pr = _precision_recall_curve_binary(labels_np, probs_np)
    auroc = float(np.trapezoid(roc[1], roc[0])) if roc is not None else None
    pr_auc = float(np.trapezoid(pr[0], pr[1])) if pr is not None else None
    average_precision = None
    if pr is not None:
        precision, recall = pr
        average_precision = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }



def _make_bce_criterion(*, device: torch.device, pos_weight_scalar: float | None) -> nn.Module:
    _require_torch()
    if pos_weight_scalar is None or not math.isfinite(pos_weight_scalar) or pos_weight_scalar <= 0.0:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight_scalar), dtype=torch.float32, device=device)
    )



def _collect_loader_outputs(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    _require_torch()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            logits = model(xb, mb)
            loss = criterion(logits, yb)

            batch_n = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_examples += batch_n
            logits_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(yb.detach().cpu().numpy())

    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32, copy=False)
    labels_np = np.concatenate(label_chunks, axis=0).astype(np.float32, copy=False)
    return logits_np, labels_np, total_loss, total_examples



def _compute_binary_metrics(
    *,
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    threshold: float,
    total_loss: float,
    total_examples: int,
) -> dict[str, Any]:
    probs_np = _sigmoid_np(logits_np)
    target_np = labels_np.astype(np.uint8)
    conf = _binary_confusion_from_probs(probs_np, target_np, threshold)
    tp = conf["tp"]
    tn = conf["tn"]
    fp = conf["fp"]
    fn = conf["fn"]

    accuracy = _safe_ratio(tp + tn, total_examples)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = float(0.5 * (recall + specificity))

    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))

    pred_np = (probs_np >= threshold).astype(np.uint8)
    ranking = _binary_ranking_metrics(target_np, probs_np)

    return {
        "num_examples": total_examples,
        "bce_loss": float(total_loss / total_examples) if total_examples > 0 else None,
        "threshold_used": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
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
        "specificity": specificity,
        "f1": f1,
        **ranking,
    }



def _select_threshold_from_validation(
    *,
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    mode: str,
    fixed_threshold: float,
    grid_size: int,
) -> dict[str, Any]:
    if mode == "fixed":
        return {
            "mode": "fixed",
            "selected_threshold": float(fixed_threshold),
            "metric_name": None,
            "metric_value": None,
            "grid_size": 0,
            "reason": "fixed_threshold_requested",
        }

    target_np = np.asarray(labels_np, dtype=np.uint8).reshape(-1)
    if target_np.size == 0:
        return {
            "mode": mode,
            "selected_threshold": float(fixed_threshold),
            "metric_name": None,
            "metric_value": None,
            "grid_size": 0,
            "reason": "empty_validation_split",
        }
    if np.unique(target_np).size < 2:
        return {
            "mode": mode,
            "selected_threshold": float(fixed_threshold),
            "metric_name": None,
            "metric_value": None,
            "grid_size": 0,
            "reason": "validation_split_has_single_class",
        }

    grid_size = int(grid_size)
    if grid_size < 3:
        raise ValueError(f"threshold_grid_size must be >= 3, got {grid_size}")

    thresholds = np.linspace(0.01, 0.99, grid_size, dtype=np.float32)
    if 0.01 <= fixed_threshold <= 0.99:
        thresholds = np.unique(np.concatenate([thresholds, np.asarray([fixed_threshold], dtype=np.float32)]))

    metric_name = "f1" if mode == "val_f1" else "balanced_accuracy"
    best: dict[str, Any] | None = None
    for thr in thresholds.tolist():
        metrics = _compute_binary_metrics(
            logits_np=logits_np,
            labels_np=labels_np,
            threshold=float(thr),
            total_loss=0.0,
            total_examples=int(target_np.size),
        )
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue
        candidate = {
            "threshold": float(thr),
            "metric_value": float(metric_value),
            "recall": metrics.get("recall"),
            "precision": metrics.get("precision"),
            "distance_to_half": abs(float(thr) - 0.5),
        }
        if best is None:
            best = candidate
        else:
            better = False
            if candidate["metric_value"] > best["metric_value"] + 1e-12:
                better = True
            elif abs(candidate["metric_value"] - best["metric_value"]) <= 1e-12:
                cand_recall = candidate["recall"] if candidate["recall"] is not None else -1.0
                best_recall = best["recall"] if best["recall"] is not None else -1.0
                if cand_recall > best_recall + 1e-12:
                    better = True
                elif abs(cand_recall - best_recall) <= 1e-12 and candidate["distance_to_half"] < best["distance_to_half"]:
                    better = True
            if better:
                best = candidate

    if best is None:
        return {
            "mode": mode,
            "selected_threshold": float(fixed_threshold),
            "metric_name": metric_name,
            "metric_value": None,
            "grid_size": int(thresholds.size),
            "reason": "no_valid_threshold_candidate",
        }

    return {
        "mode": mode,
        "selected_threshold": float(best["threshold"]),
        "metric_name": metric_name,
        "metric_value": float(best["metric_value"]),
        "grid_size": int(thresholds.size),
        "recall_at_selected_threshold": None if best["recall"] is None else float(best["recall"]),
        "precision_at_selected_threshold": None if best["precision"] is None else float(best["precision"]),
        "reason": "selected_from_validation_grid_search",
    }



def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    criterion: nn.Module,
) -> dict[str, Any]:
    logits_np, labels_np, total_loss, total_examples = _collect_loader_outputs(
        model,
        loader,
        device=device,
        criterion=criterion,
    )
    return _compute_binary_metrics(
        logits_np=logits_np,
        labels_np=labels_np,
        threshold=threshold,
        total_loss=total_loss,
        total_examples=total_examples,
    )



def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    _require_torch()
    model.train()
    total_loss = 0.0
    total_examples = 0

    for xb, mb, yb in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, mb)
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



def _pick_device(device_arg: str) -> torch.device:
    _require_torch()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)



def train_one_family(
    family_dir: str | Path,
    *,
    input_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    train_seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    temporal_encoder: TemporalEncoderName,
    temporal_hidden_dim: int,
    temporal_layers: int,
    temporal_dropout: float,
    spatial_mode: SpatialModeName,
    spatial_heads: int,
    mlp_hidden_dim: int,
    head_dropout: float,
    threshold: float,
    threshold_mode: str,
    threshold_grid_size: int,
    pos_weight_mode: str,
    pos_weight_cap: float | None,
    device_arg: str,
    max_shots: int | None,
    checkpoint_path: Path | None,
    include_xy: bool,
    include_boundary: bool,
    include_checkerboard: bool,
    include_detector_type: bool,
    include_final_round: bool,
    include_valid_mask_feature: bool,
) -> dict[str, Any]:
    _require_torch()
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not (0.0 < lr):
        raise ValueError(f"lr must be > 0, got {lr}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if threshold_mode not in {"fixed", "val_f1", "val_balanced_accuracy"}:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode!r}")
    if pos_weight_mode not in {"none", "balanced"}:
        raise ValueError(f"Unsupported pos_weight_mode: {pos_weight_mode!r}")
    if pos_weight_cap is not None and pos_weight_cap <= 0.0:
        raise ValueError(f"pos_weight_cap must be > 0 when provided, got {pos_weight_cap}")

    payload = _load_family_payload(family_dir)
    arrays, prepare_info = _prepare_loaded_family(payload, max_shots=max_shots)
    labels = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    num_shots = int(labels.shape[0])

    split_indices = build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    split_summary = summarise_split_indices(split_indices, num_shots=num_shots, seed=split_seed)

    layout = build_track_layout(arrays)
    _validate_layout_against_known_geometry(layout, payload.metadata)
    bundle = build_track_tensor_bundle(
        arrays,
        layout=layout,
        include_xy=include_xy,
        include_boundary=include_boundary,
        include_checkerboard=include_checkerboard,
        include_detector_type=include_detector_type,
        include_final_round=include_final_round,
        include_valid_mask_feature=include_valid_mask_feature,
    )

    x_train, m_train, y_train = _slice_bundle_by_indices(bundle, split_indices["train"])
    x_val, m_val, y_val = _slice_bundle_by_indices(bundle, split_indices["val"])
    x_test, m_test, y_test = _slice_bundle_by_indices(bundle, split_indices["test"])

    device = _pick_device(device_arg)
    _set_random_seeds(train_seed)

    model = TrackNNLite(
        input_dim=int(bundle.x.shape[-1]),
        temporal_encoder=temporal_encoder,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_layers=temporal_layers,
        temporal_dropout=temporal_dropout,
        spatial_mode=spatial_mode,
        spatial_heads=spatial_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        head_dropout=head_dropout,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = _make_loader(x_train, m_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(x_val, m_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(x_test, m_test, y_test, batch_size=batch_size, shuffle=False)

    train_pos_count = int(np.sum(y_train >= 0.5))
    train_neg_count = int(y_train.size - train_pos_count)
    if pos_weight_mode == "balanced" and train_pos_count > 0 and train_neg_count > 0:
        pos_weight_scalar = float(train_neg_count / train_pos_count)
        if pos_weight_cap is not None:
            pos_weight_scalar = float(min(pos_weight_scalar, pos_weight_cap))
    else:
        pos_weight_scalar = 1.0
    criterion = _make_bce_criterion(device=device, pos_weight_scalar=pos_weight_scalar)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer=optimizer, device=device, criterion=criterion)
        val_metrics = _evaluate_loader(model, val_loader, device=device, threshold=threshold, criterion=criterion)
        val_loss = float(val_metrics["bce_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_bce_loss": float(train_loss),
                "val_bce_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
                "val_average_precision": val_metrics["average_precision"],
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    train_wall_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)

    val_logits_np, val_labels_np, val_total_loss, val_total_examples = _collect_loader_outputs(
        model,
        val_loader,
        device=device,
        criterion=criterion,
    )
    threshold_selection = _select_threshold_from_validation(
        logits_np=val_logits_np,
        labels_np=val_labels_np,
        mode=threshold_mode,
        fixed_threshold=threshold,
        grid_size=threshold_grid_size,
    )
    selected_threshold = float(threshold_selection["selected_threshold"])

    train_metrics = _evaluate_loader(
        model,
        _make_loader(x_train, m_train, y_train, batch_size=batch_size, shuffle=False),
        device=device,
        threshold=selected_threshold,
        criterion=criterion,
    )
    val_metrics = _compute_binary_metrics(
        logits_np=val_logits_np,
        labels_np=val_labels_np,
        threshold=selected_threshold,
        total_loss=val_total_loss,
        total_examples=val_total_examples,
    )
    test_metrics = _evaluate_loader(model, test_loader, device=device, threshold=selected_threshold, criterion=criterion)

    metadata = payload.metadata
    bundle_info = describe_track_bundle(bundle)

    ckpt_saved = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": SCHEMA_VERSION_TRAIN,
            "decoder": "baseline_tracknn",
            "created_at_utc": _utc_now_iso(),
            "family": metadata.get("family", payload.artifacts.family_dir.name),
            "stage": metadata.get("stage", "unknown"),
            "family_dir": payload.artifacts.family_dir.as_posix(),
            "model_hparams": {
                "input_dim": int(bundle.x.shape[-1]),
                "temporal_encoder": temporal_encoder,
                "temporal_hidden_dim": int(temporal_hidden_dim),
                "temporal_layers": int(temporal_layers),
                "temporal_dropout": float(temporal_dropout),
                "spatial_mode": spatial_mode,
                "spatial_heads": int(spatial_heads),
                "mlp_hidden_dim": int(mlp_hidden_dim),
                "head_dropout": float(head_dropout),
            },
            "feature_flags": {
                "include_xy": bool(include_xy),
                "include_boundary": bool(include_boundary),
                "include_checkerboard": bool(include_checkerboard),
                "include_detector_type": bool(include_detector_type),
                "include_final_round": bool(include_final_round),
                "include_valid_mask_feature": bool(include_valid_mask_feature),
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
                "threshold": float(selected_threshold),
                "threshold_fixed_input": float(threshold),
                "threshold_mode": str(threshold_mode),
                "threshold_grid_size": int(threshold_grid_size),
                "pos_weight_mode": str(pos_weight_mode),
                "pos_weight_cap": pos_weight_cap,
                "pos_weight_used": float(pos_weight_scalar),
                "max_shots": max_shots,
            },
            "dataset_context": {
                "schema_version": metadata.get("schema_version"),
                "family": metadata.get("family"),
                "stage": metadata.get("stage"),
                "circuit": metadata.get("circuit", {}),
                "scaffold": metadata.get("scaffold", {}),
                "bundle_info": bundle_info,
            },
            "threshold_selection": threshold_selection,
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        ckpt_saved = checkpoint_path.as_posix()

    result = TrainResult(
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder="baseline_tracknn",
        created_at_utc=_utc_now_iso(),
        input_mode=input_mode,
        family=str(metadata.get("family", payload.artifacts.family_dir.name)),
        stage=str(metadata.get("stage", "unknown")),
        family_dir=payload.artifacts.family_dir.as_posix(),
        model={
            "architecture": "tracknn_lite",
            "input_representation": "track_tensor",
            "temporal_encoder": temporal_encoder,
            "temporal_hidden_dim": int(temporal_hidden_dim),
            "temporal_layers": int(temporal_layers),
            "temporal_dropout": float(temporal_dropout),
            "spatial_mode": spatial_mode,
            "spatial_heads": int(spatial_heads),
            "mlp_hidden_dim": int(mlp_hidden_dim),
            "head_dropout": float(head_dropout),
            "num_parameters": _num_parameters(model),
            "device": str(device),
            "bundle_info": bundle_info,
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
            "feature_flags": {
                "include_xy": bool(include_xy),
                "include_boundary": bool(include_boundary),
                "include_checkerboard": bool(include_checkerboard),
                "include_detector_type": bool(include_detector_type),
                "include_final_round": bool(include_final_round),
                "include_valid_mask_feature": bool(include_valid_mask_feature),
            },
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
            "threshold": float(selected_threshold),
            "threshold_fixed_input": float(threshold),
            "threshold_mode": str(threshold_mode),
            "threshold_grid_size": int(threshold_grid_size),
            "threshold_selection": threshold_selection,
            "pos_weight_mode": str(pos_weight_mode),
            "pos_weight_cap": pos_weight_cap,
            "pos_weight_used": float(pos_weight_scalar),
            "train_positive_count": int(train_pos_count),
            "train_negative_count": int(train_neg_count),
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



def eval_one_family(
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
    model_hparams = checkpoint.get("model_hparams", {})
    feature_flags = checkpoint.get("feature_flags", {})
    threshold_selection = checkpoint.get("threshold_selection", {})
    threshold = float(threshold_selection.get("selected_threshold", training_hparams.get("threshold", 0.5)))
    pos_weight_used = training_hparams.get("pos_weight_used", 1.0)

    split_indices = build_split_indices(
        num_shots=num_shots,
        train_ratio=float(training_hparams.get("train_ratio", 0.8)),
        val_ratio=float(training_hparams.get("val_ratio", 0.1)),
        test_ratio=float(training_hparams.get("test_ratio", 0.1)),
        seed=int(training_hparams.get("split_seed", 20260330)),
    )
    split_summary = summarise_split_indices(
        split_indices,
        num_shots=num_shots,
        seed=int(training_hparams.get("split_seed", 20260330)),
    )

    layout = build_track_layout(arrays)
    _validate_layout_against_known_geometry(layout, payload.metadata)
    bundle = build_track_tensor_bundle(
        arrays,
        layout=layout,
        include_xy=bool(feature_flags.get("include_xy", True)),
        include_boundary=bool(feature_flags.get("include_boundary", True)),
        include_checkerboard=bool(feature_flags.get("include_checkerboard", True)),
        include_detector_type=bool(feature_flags.get("include_detector_type", True)),
        include_final_round=bool(feature_flags.get("include_final_round", True)),
        include_valid_mask_feature=bool(feature_flags.get("include_valid_mask_feature", True)),
    )

    expected_input_dim = int(model_hparams.get("input_dim", -1))
    if expected_input_dim != int(bundle.x.shape[-1]):
        raise ValueError(
            "Checkpoint input_dim does not match rebuilt track tensor dimension: "
            f"checkpoint={expected_input_dim}, current={bundle.x.shape[-1]}"
        )

    if split_name == "all":
        indices = np.arange(num_shots, dtype=np.int64)
    else:
        indices = split_indices[split_name]

    x_eval, m_eval, y_eval = _slice_bundle_by_indices(bundle, indices)
    eval_loader = _make_loader(x_eval, m_eval, y_eval, batch_size=batch_size, shuffle=False)

    device = _pick_device(device_arg)
    criterion = _make_bce_criterion(device=device, pos_weight_scalar=float(pos_weight_used) if pos_weight_used is not None else None)
    model = TrackNNLite(
        input_dim=expected_input_dim,
        temporal_encoder=str(model_hparams.get("temporal_encoder", "gru")),
        temporal_hidden_dim=int(model_hparams.get("temporal_hidden_dim", 64)),
        temporal_layers=int(model_hparams.get("temporal_layers", 1)),
        temporal_dropout=float(model_hparams.get("temporal_dropout", 0.1)),
        spatial_mode=str(model_hparams.get("spatial_mode", "mean")),
        spatial_heads=int(model_hparams.get("spatial_heads", 2)),
        mlp_hidden_dim=int(model_hparams.get("mlp_hidden_dim", 64)),
        head_dropout=float(model_hparams.get("head_dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = _evaluate_loader(model, eval_loader, device=device, threshold=threshold, criterion=criterion)
    metadata = payload.metadata
    bundle_info = describe_track_bundle(bundle)

    result = EvalResult(
        schema_version=SCHEMA_VERSION_EVAL,
        decoder="baseline_tracknn",
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
            "threshold_selection": threshold_selection,
        },
        model={
            "architecture": "tracknn_lite",
            "input_representation": "track_tensor",
            "device": str(device),
            "num_parameters": _num_parameters(model),
            "bundle_info": bundle_info,
            "model_hparams": model_hparams,
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
            "feature_flags": feature_flags,
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
            "Train or evaluate a geometry-aware neural baseline on sample_dataset.py outputs. "
            "v1 is one-family / one-geometry binary classification using stabilizer-track tensors."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train baseline_tracknn on one family dataset")
    _add_common_input_args(train)
    train.add_argument("--train-ratio", type=float, default=0.80)
    train.add_argument("--val-ratio", type=float, default=0.10)
    train.add_argument("--test-ratio", type=float, default=0.10)
    train.add_argument("--split-seed", type=int, default=20260330)
    train.add_argument("--train-seed", type=int, default=20260330)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--temporal-encoder", choices=["gru", "conv1d"], default="gru")
    train.add_argument("--temporal-hidden-dim", type=int, default=64)
    train.add_argument("--temporal-layers", type=int, default=1)
    train.add_argument("--temporal-dropout", type=float, default=0.10)
    train.add_argument("--spatial-mode", choices=["mean", "attention"], default="mean")
    train.add_argument("--spatial-heads", type=int, default=2)
    train.add_argument("--mlp-hidden-dim", type=int, default=64)
    train.add_argument("--head-dropout", type=float, default=0.10)
    train.add_argument("--threshold", type=float, default=0.50)
    train.add_argument("--threshold-mode", choices=["fixed", "val_f1", "val_balanced_accuracy"], default="val_f1")
    train.add_argument("--threshold-grid-size", type=int, default=99)
    train.add_argument("--pos-weight-mode", choices=["none", "balanced"], default="balanced")
    train.add_argument("--pos-weight-cap", type=float, default=20.0)
    train.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    train.add_argument("--checkpoint-path", type=Path, default=None)
    train.add_argument("--out-json", type=Path, default=None)
    train.add_argument("--disable-xy", action="store_true")
    train.add_argument("--disable-boundary", action="store_true")
    train.add_argument("--disable-checkerboard", action="store_true")
    train.add_argument("--disable-detector-type", action="store_true")
    train.add_argument("--disable-final-round", action="store_true")
    train.add_argument("--disable-valid-mask-feature", action="store_true")

    evaluate = subparsers.add_parser("eval", help="Evaluate a saved checkpoint on one family dataset")
    _add_common_input_args(evaluate)
    evaluate.add_argument("--checkpoint-path", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=128)
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
        result = train_one_family(
            resolved_family_dir,
            input_mode=input_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            train_seed=args.train_seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temporal_encoder=args.temporal_encoder,
            temporal_hidden_dim=args.temporal_hidden_dim,
            temporal_layers=args.temporal_layers,
            temporal_dropout=args.temporal_dropout,
            spatial_mode=args.spatial_mode,
            spatial_heads=args.spatial_heads,
            mlp_hidden_dim=args.mlp_hidden_dim,
            head_dropout=args.head_dropout,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            threshold_grid_size=args.threshold_grid_size,
            pos_weight_mode=args.pos_weight_mode,
            pos_weight_cap=args.pos_weight_cap,
            device_arg=args.device,
            max_shots=args.max_shots,
            checkpoint_path=args.checkpoint_path,
            include_xy=not args.disable_xy,
            include_boundary=not args.disable_boundary,
            include_checkerboard=not args.disable_checkerboard,
            include_detector_type=not args.disable_detector_type,
            include_final_round=not args.disable_final_round,
            include_valid_mask_feature=not args.disable_valid_mask_feature,
        )
    else:
        result = eval_one_family(
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
