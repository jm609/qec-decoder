from __future__ import annotations

"""
syndrome_edit_predecoder.py

First syndrome-edit pre-decoder for the rebuilt repository.

The model predicts a sparse detector-bit edit mask and an auxiliary
shot-level `needs_edit` score. The predicted edited syndrome is then passed to
unchanged PyMatching for the final logical-frame decision.
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
    import baseline_pymatching as pym_common
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_rectcnn as common
    import research_noise_aware_3d as volume_common
    import baseline_pymatching as pym_common


SCHEMA_VERSION_TRAIN = "syndrome_edit_predecoder.train.v1"
SCHEMA_VERSION_EVAL = "syndrome_edit_predecoder.eval.v1"
SCHEMA_VERSION_EXPERIMENT = "syndrome_edit_predecoder.experiment.v1"
CLASS4_LABELS = ["I", "X", "Z", "Y"]
DEFAULT_HARD_SHOT_SOLVED_WEIGHT = 6.0
DEFAULT_HARD_SHOT_UNSOLVED_WEIGHT = 2.0
DEFAULT_SELECTOR_HIDDEN_DIM = 64
DEFAULT_SELECTOR_PATCH_HEAD = False
DEFAULT_SELECTOR_PATCH_HIDDEN_DIM = 32
DEFAULT_SELECTOR_EPOCHS = 6
DEFAULT_SELECTOR_LR = 5e-4
DEFAULT_SELECTOR_SCORE_EDIT_PENALTY = 0.1
DEFAULT_SELECTOR_HARD_SHOT_WEIGHT = 4.0
DEFAULT_SELECTOR_CANDIDATE_MOTIF_MAX_CLASSES = 0
DEFAULT_SELECTOR_LOCAL_MOTIF_MAX_CLASSES = 0
DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K = 32
DEFAULT_SELECTOR_IDENTITY_MARGIN_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_IDENTITY_MARGIN = 1.0
DEFAULT_SELECTOR_HARM_MARGIN_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_HARM_MARGIN = 1.0
DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN = 1.0
DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_MARGIN = 0.0
DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_MARGIN = 0.0
DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_LOSS_WEIGHT = 0.0
DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_MARGIN = 0.0
SELECTOR_MODEL_SCALAR = "scalar"
SELECTOR_MODEL_RISK_AWARE = "risk_aware"
SELECTOR_MODEL_RISK_GUARD = "risk_guard"
SELECTOR_MODEL_CHOICES = (
    SELECTOR_MODEL_SCALAR,
    SELECTOR_MODEL_RISK_AWARE,
    SELECTOR_MODEL_RISK_GUARD,
)
DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT = 1.0
DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_LOSS_WEIGHT = 1.0
DEFAULT_SELECTOR_RISK_AWARE_HARM_LOSS_WEIGHT = 1.0
DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_POS_WEIGHT = 0.0
DEFAULT_SELECTOR_RISK_AWARE_HARM_POS_WEIGHT = 0.0
SELECTOR_EPOCH_SELECTION_PROXY = "proxy"
SELECTOR_EPOCH_SELECTION_DIAGNOSTIC_SYSTEM = "diagnostic_system"
SELECTOR_EPOCH_SELECTION_CHOICES = (
    SELECTOR_EPOCH_SELECTION_PROXY,
    SELECTOR_EPOCH_SELECTION_DIAGNOSTIC_SYSTEM,
)
DEFAULT_SELECTOR_EMIT_MARGIN_GRID = (0.0,)
DEFAULT_SELECTOR_NONZERO_BIAS_GRID = (0.0,)
DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID = (0.0,)
DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID = (0,)
DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID = (0.0,)
DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID = (0,)
DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES = False
DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES = False
DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES = False
DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES = False
DEFAULT_SELECTOR_ADOPTION_MIN_DELTA = 0.0
SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR = "global_noninferiority"
SELECTOR_ADOPTION_POLICY_CANDIDATE_FIRST_SAFETY = "candidate_first_safety"
SELECTOR_ADOPTION_POLICY_CHOICES = (
    SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR,
    SELECTOR_ADOPTION_POLICY_CANDIDATE_FIRST_SAFETY,
)
DEFAULT_SELECTOR_CANDIDATE_FIRST_STRONG_DELTA = 0.02
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_DELTA = 0.005
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MARGIN_FLOOR = 0.5
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED = 2
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_MARGIN = 1.5
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MIN_NONZERO = 0
DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_PLATEAU_GUARD = False
DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_DELTA = -1e-9
DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MARGIN_FLOOR = 1.0
DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_NONZERO = 6
DEFAULT_SELECTOR_CANDIDATE_FIRST_ALLOW_GLOBAL = False
DEFAULT_SELECTOR_CANDIDATE_FIRST_GLOBAL_MIN_DELTA = 0.01
DEFAULT_SELECTED_NO_EDIT_GUARDRAIL = False
DEFAULT_SELECTED_NO_EDIT_MIN_DELTA = 0.0
SELECTOR_CANDIDATE_BASE_FEATURE_DIM = 13
SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM = 9
SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM = 6
SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM = 10
SELECTOR_CANDIDATE_LOCAL_PATCH_RADIUS = 1
SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM = ((2 * SELECTOR_CANDIDATE_LOCAL_PATCH_RADIUS + 1) ** 3) * 2
DEFAULT_CANDIDATE_COMPAT_EPOCHS = 6
DEFAULT_CANDIDATE_COMPAT_LR = 5e-4
DEFAULT_CANDIDATE_COMPAT_HIDDEN_DIM = 64
DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO = 1.0
DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT = 1
CANDIDATE_COMPAT_OBJECTIVE_BCE = "bce"
CANDIDATE_COMPAT_OBJECTIVE_GROUP_BALANCED = "group_balanced"
CANDIDATE_COMPAT_OBJECTIVE_PAIRWISE_RANK = "pairwise_rank"
CANDIDATE_COMPAT_OBJECTIVE_CHOICES = (
    CANDIDATE_COMPAT_OBJECTIVE_BCE,
    CANDIDATE_COMPAT_OBJECTIVE_GROUP_BALANCED,
    CANDIDATE_COMPAT_OBJECTIVE_PAIRWISE_RANK,
)
DEFAULT_TRANSITION_PRIOR_EPOCHS = 6
DEFAULT_TRANSITION_PRIOR_LR = 5e-4
DEFAULT_TRANSITION_PRIOR_HIDDEN_DIM = 64
DEFAULT_SELECTOR_HARM_WEIGHT = 2.0
DEFAULT_SELECTOR_MISS_WEIGHT = 0.25
DEFAULT_ROUTER_HIDDEN_DIM = 64
DEFAULT_ROUTER_EPOCHS = 12
DEFAULT_ROUTER_LR = 5e-4
DEFAULT_ROUTER_POS_WEIGHT = 8.0
DEFAULT_ROUTER_THRESHOLD_GRID = (0.3, 0.5, 0.7, 0.9)
ROUTER_LABEL_IDENTITY_VS_NONZERO = "identity_vs_nonzero"
ROUTER_LABEL_BASELINE_FAILURE = "baseline_failure"
ROUTER_LABEL_ORACLE_SOLVABLE = "oracle_solvable"
ROUTER_LABEL_CHOICES = (
    ROUTER_LABEL_IDENTITY_VS_NONZERO,
    ROUTER_LABEL_BASELINE_FAILURE,
    ROUTER_LABEL_ORACLE_SOLVABLE,
)
ROUTER_PRETRAIN_TARGET_NONE = "none"
ROUTER_PRETRAIN_TARGET_CHOICES = (ROUTER_PRETRAIN_TARGET_NONE,) + ROUTER_LABEL_CHOICES
DEFAULT_ACTION_MOTIF_MAX_CLASSES = 0
DEFAULT_ACTION_MOTIF_LOSS_WEIGHT = 0.0
DEFAULT_ACTION_MOTIF_IDENTITY_MARGIN = 1.0
DEFAULT_ACTION_MOTIF_EMIT_MARGIN_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_LOCAL_MOTIF_MAX_CLASSES = 0
DEFAULT_LOCAL_MOTIF_EMIT_MARGIN_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_LOCAL_MOTIF_MIN_BIT_LOGIT_GRID = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_DECISION_AWARE_LOSS_WEIGHT = 0.25
DEFAULT_DECISION_AWARE_MARGIN = 1.0
DEFAULT_MOTIF_MAX_CLASSES = 16
DEFAULT_MOTIF_EPOCHS = 8
DEFAULT_MOTIF_LR = 5e-4
DEFAULT_MOTIF_HARD_SHOT_WEIGHT = 8.0
EDIT_SUPERVISION_MODE_ALL_KNOWN = "all_known"
EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY = "hard_shots_only"
EDIT_SUPERVISION_MODE_CHOICES = (
    EDIT_SUPERVISION_MODE_ALL_KNOWN,
    EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY,
)
SELECTION_MODE_GLOBAL_POLICY = "global_policy"
SELECTION_MODE_RAW_NO_EDIT = "raw_no_edit"
SELECTION_MODE_CANDIDATE_SELECTOR = "candidate_selector"
SELECTION_MODE_MOTIF_VOCAB = "motif_vocab"
SELECTION_MODE_ACTION_MOTIF = "action_motif"
SELECTION_MODE_LOCAL_MOTIF = "local_motif"
SELECTION_MODE_LOCAL_MOTIF_SELECTOR = "local_motif_selector"
SELECTION_MODE_LOCAL_MOTIF_ROUTER = "local_motif_router"
SELECTION_MODE_CHOICES = (
    SELECTION_MODE_GLOBAL_POLICY,
    SELECTION_MODE_CANDIDATE_SELECTOR,
    SELECTION_MODE_MOTIF_VOCAB,
    SELECTION_MODE_ACTION_MOTIF,
    SELECTION_MODE_LOCAL_MOTIF,
    SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
    SELECTION_MODE_LOCAL_MOTIF_ROUTER,
)
SELECTOR_OBJECTIVE_BCE = "bce"
SELECTOR_OBJECTIVE_GROUP_RANK = "group_rank"
SELECTOR_OBJECTIVE_CHOICES = (
    SELECTOR_OBJECTIVE_BCE,
    SELECTOR_OBJECTIVE_GROUP_RANK,
)
SELECTOR_TARGET_MODE_CORRECTNESS = "correctness"
SELECTOR_TARGET_MODE_BENEFIT_HARM = "benefit_harm"
SELECTOR_TARGET_MODE_CHOICES = (
    SELECTOR_TARGET_MODE_CORRECTNESS,
    SELECTOR_TARGET_MODE_BENEFIT_HARM,
)
SELECTOR_POLICY_CANDIDATE_MODE_ALL = "all"
SELECTOR_POLICY_CANDIDATE_MODE_NONE = "none"
SELECTOR_POLICY_CANDIDATE_MODE_CHOICES = (
    SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    SELECTOR_POLICY_CANDIDATE_MODE_NONE,
)


@dataclass(frozen=True, slots=True)
class CandidatePolicySpec:
    needs_edit_threshold: float
    edit_threshold: float
    max_predicted_edit_weight: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "needs_edit_threshold": float(self.needs_edit_threshold),
            "edit_threshold": float(self.edit_threshold),
            "max_predicted_edit_weight": int(self.max_predicted_edit_weight),
        }


@dataclass(frozen=True, slots=True)
class EditTargetArtifacts:
    family_dir: Path
    metadata_json: Path
    edit_targets_npz: Path


@dataclass(frozen=True, slots=True)
class PreparedEditFamily:
    family: str
    stage: str
    family_dir: Path
    source_family_dir: Path
    x: np.ndarray
    detector_events: np.ndarray
    observable_flips: np.ndarray
    logical_class4: np.ndarray
    edit_target_volume: np.ndarray
    detector_edit_target_mask: np.ndarray
    needs_edit: np.ndarray
    edit_target_known: np.ndarray
    baseline_correct: np.ndarray
    baseline_predicted_observables: np.ndarray
    valid_mask_volume: np.ndarray
    detector_time_index: np.ndarray
    row_index_by_detector: np.ndarray
    col_index_by_detector: np.ndarray
    bundle_info: dict[str, Any]
    metadata: dict[str, Any]
    edit_metadata: dict[str, Any]
    matching: Any
    matching_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SplitBundle:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True, slots=True)
class SelectorCandidateBundle:
    shot_features: np.ndarray
    candidate_features: np.ndarray
    target_scores: np.ndarray
    candidate_is_correct: np.ndarray
    shot_indices: np.ndarray
    candidate_edit_mask: np.ndarray
    candidate_edit_weight: np.ndarray
    target_transition_class: np.ndarray
    num_shots: int
    num_detectors: int
    decode_fallback_count: int


@dataclass(frozen=True, slots=True)
class MotifVocabulary:
    mask_table: np.ndarray
    detector_index_lists: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    detector_count: int


@dataclass(frozen=True, slots=True)
class LocalMotifVocabulary:
    offset_patterns: tuple[tuple[tuple[int, int, int], ...], ...]
    counts: tuple[int, ...]
    detector_count: int


@dataclass(frozen=True, slots=True)
class LocalMotifPlacements:
    mask_table: np.ndarray
    detector_index_lists: tuple[tuple[int, ...], ...]
    pattern_indices: np.ndarray
    counts: np.ndarray
    detector_count: int
    num_patterns: int


@dataclass(frozen=True, slots=True)
class MotifTargetBundle:
    shot_features: np.ndarray
    labels: np.ndarray
    shot_weights: np.ndarray
    active_mask: np.ndarray
    hard_shot_fraction: float | None


def _require_torch() -> None:
    if torch is None or nn is None or F is None or optim is None:
        raise common.MissingTorchError(
            "PyTorch is required for syndrome_edit_predecoder.py but is not installed "
            "in this Python environment."
        )


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _as_uint8_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_uint8_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_float32_4d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _logical_class4_from_observable_flips(observable_flips: np.ndarray) -> np.ndarray:
    obs = _as_uint8_2d(observable_flips, name="observable_flips")
    if obs.shape[1] != 2:
        raise ValueError(
            "logical_class4 requires exactly two observables ordered as [logical_z_flip, logical_x_flip]."
        )
    logical_z_flip = obs[:, 0].astype(np.uint8, copy=False)
    logical_x_flip = obs[:, 1].astype(np.uint8, copy=False)
    return (
        logical_x_flip.astype(np.uint8, copy=False)
        + (logical_z_flip.astype(np.uint8, copy=False) << 1)
    ).astype(np.uint8, copy=False)


def _hard_multiclass_metrics(pred_class: np.ndarray, target_class: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred_class, dtype=np.int64).reshape(-1)
    target = np.asarray(target_class, dtype=np.int64).reshape(-1)
    probs = np.zeros((target.shape[0], 4), dtype=np.float32)
    if target.shape[0]:
        probs[np.arange(target.shape[0]), pred] = 1.0
    return common._multiclass_metrics_from_probs(
        probs,
        target,
        class_labels=list(CLASS4_LABELS),
        loss=None,
    )


def _binary_metrics_from_probs(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    return common._binary_metrics_from_probs(
        np.asarray(probs, dtype=np.float32).reshape(-1),
        np.asarray(target, dtype=np.uint8).reshape(-1),
        threshold=float(threshold),
        bce_loss=None,
    )


def _resolve_edit_target_artifacts(family_dir: str | Path) -> EditTargetArtifacts:
    family_dir = Path(family_dir)
    metadata_json = family_dir / "metadata.json"
    edit_targets_npz = family_dir / "edit_targets.npz"
    missing = [p.name for p in (metadata_json, edit_targets_npz) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Edit-target family directory is missing required artifacts {missing}: {family_dir}"
        )
    return EditTargetArtifacts(
        family_dir=family_dir,
        metadata_json=metadata_json,
        edit_targets_npz=edit_targets_npz,
    )


def _load_edit_target_payload(
    family_dir: str | Path,
) -> tuple[EditTargetArtifacts, dict[str, Any], dict[str, np.ndarray], common.LoadedFamilyPayload]:
    artifacts = _resolve_edit_target_artifacts(family_dir)
    metadata = _read_json(artifacts.metadata_json)
    source_family_dir = Path(str(metadata["source_family_dir"]))
    source_payload = common._load_family_payload(source_family_dir)
    with np.load(artifacts.edit_targets_npz) as data:
        edit_arrays = {key: np.asarray(data[key]) for key in data.files}
    return artifacts, metadata, edit_arrays, source_payload


def _prepare_edit_family(
    family_dir: str | Path,
    *,
    fill_value: float,
    max_shots: int | None,
    allow_circuit_fallback: bool = False,
) -> PreparedEditFamily:
    artifacts, edit_metadata, edit_arrays, source_payload = _load_edit_target_payload(family_dir)
    edit_target_num_shots = int(np.asarray(edit_arrays["detector_edit_target_mask"]).shape[0])
    effective_max_shots = (
        min(int(max_shots), edit_target_num_shots) if max_shots is not None else edit_target_num_shots
    )
    prepared = volume_common._prepare_loaded_family(
        source_payload,
        fill_value=fill_value,
        max_shots=effective_max_shots,
        target_mode=common.TARGET_MODE_LOGICAL_CLASS4,
    )
    x = np.asarray(prepared["x"], dtype=np.float32)
    layout = prepared["layout"]
    arrays = source_payload.arrays
    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")
    logical_class4 = _as_uint8_1d(arrays["logical_class4"], name="logical_class4")
    detector_edit_target_mask = _as_uint8_2d(
        edit_arrays["detector_edit_target_mask"],
        name="detector_edit_target_mask",
    )
    needs_edit = _as_uint8_1d(edit_arrays["needs_edit"], name="needs_edit").astype(np.float32, copy=False)
    edit_target_known = _as_uint8_1d(edit_arrays["edit_target_known"], name="edit_target_known").astype(
        np.float32,
        copy=False,
    )
    baseline_correct = _as_uint8_1d(
        edit_arrays["baseline_pymatching_correct"],
        name="baseline_pymatching_correct",
    ).astype(np.float32, copy=False)
    baseline_predicted_observables = _as_uint8_2d(
        edit_arrays["baseline_predicted_observables"],
        name="baseline_predicted_observables",
    )

    detector_events = detector_events[:effective_max_shots]
    observable_flips = observable_flips[:effective_max_shots]
    logical_class4 = logical_class4[:effective_max_shots]
    detector_edit_target_mask = detector_edit_target_mask[:effective_max_shots]
    needs_edit = needs_edit[:effective_max_shots]
    edit_target_known = edit_target_known[:effective_max_shots]
    baseline_correct = baseline_correct[:effective_max_shots]
    baseline_predicted_observables = baseline_predicted_observables[:effective_max_shots]

    if detector_events.shape[0] != x.shape[0]:
        raise ValueError(
            f"Feature/shot mismatch for {family_dir}: x={x.shape[0]} detector_events={detector_events.shape[0]}"
        )

    num_shots = int(x.shape[0])
    num_detectors = int(detector_events.shape[1])
    t = layout.detector_time_index.astype(np.intp, copy=False)
    r = layout.row_index_by_detector.astype(np.intp, copy=False)
    c = layout.col_index_by_detector.astype(np.intp, copy=False)
    edit_target_volume = np.zeros(
        (num_shots, int(layout.time_steps), int(layout.height), int(layout.width)),
        dtype=np.float32,
    )
    edit_target_volume[:, t, r, c] = detector_edit_target_mask.astype(np.float32, copy=False)
    valid_mask_volume = layout.valid_mask.astype(np.float32, copy=False)

    matching, matching_info = pym_common._build_matching(
        dem_path=source_payload.artifacts.dem_path,
        circuit_path=source_payload.artifacts.circuit_path,
        allow_circuit_fallback=allow_circuit_fallback,
    )
    return PreparedEditFamily(
        family=str(edit_metadata.get("family", artifacts.family_dir.name)),
        stage=str(edit_metadata.get("stage", source_payload.metadata.get("stage", "unknown"))),
        family_dir=artifacts.family_dir,
        source_family_dir=Path(str(edit_metadata["source_family_dir"])),
        x=np.ascontiguousarray(x),
        detector_events=np.ascontiguousarray(detector_events),
        observable_flips=np.ascontiguousarray(observable_flips),
        logical_class4=np.ascontiguousarray(logical_class4),
        edit_target_volume=np.ascontiguousarray(edit_target_volume),
        detector_edit_target_mask=np.ascontiguousarray(detector_edit_target_mask),
        needs_edit=np.ascontiguousarray(needs_edit),
        edit_target_known=np.ascontiguousarray(edit_target_known),
        baseline_correct=np.ascontiguousarray(baseline_correct),
        baseline_predicted_observables=np.ascontiguousarray(baseline_predicted_observables),
        valid_mask_volume=np.ascontiguousarray(valid_mask_volume),
        detector_time_index=np.ascontiguousarray(layout.detector_time_index.astype(np.int16, copy=False)),
        row_index_by_detector=np.ascontiguousarray(layout.row_index_by_detector.astype(np.int16, copy=False)),
        col_index_by_detector=np.ascontiguousarray(layout.col_index_by_detector.astype(np.int16, copy=False)),
        bundle_info=dict(prepared["bundle_info"]),
        metadata=dict(source_payload.metadata),
        edit_metadata=dict(edit_metadata),
        matching=matching,
        matching_info=dict(matching_info.to_dict()),
    )


def _build_split_bundle(
    *,
    num_shots: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitBundle:
    split = common.build_split_indices(
        num_shots=num_shots,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return SplitBundle(train=split["train"], val=split["val"], test=split["test"])


def _subset_family(entry: PreparedEditFamily, indices: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "x": np.ascontiguousarray(entry.x[idx]),
        "edit_target_volume": np.ascontiguousarray(entry.edit_target_volume[idx]),
        "needs_edit": np.ascontiguousarray(entry.needs_edit[idx]),
        "edit_target_known": np.ascontiguousarray(entry.edit_target_known[idx]),
        "detector_events": np.ascontiguousarray(entry.detector_events[idx]),
        "observable_flips": np.ascontiguousarray(entry.observable_flips[idx]),
        "logical_class4": np.ascontiguousarray(entry.logical_class4[idx]),
        "baseline_correct": np.ascontiguousarray(entry.baseline_correct[idx]),
        "baseline_predicted_observables": np.ascontiguousarray(entry.baseline_predicted_observables[idx]),
    }


def _concatenate_subsets(subsets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not subsets:
        raise ValueError("At least one subset is required")
    keys = subsets[0].keys()
    return {
        key: np.ascontiguousarray(np.concatenate([subset[key] for subset in subsets], axis=0))
        for key in keys
    }


def _validate_compatible_entries(entries: list[PreparedEditFamily]) -> dict[str, Any]:
    if not entries:
        raise ValueError("At least one prepared family entry is required")
    reference_shape = tuple(int(x) for x in entries[0].x.shape[1:])
    reference_channels = list(entries[0].bundle_info.get("channel_names", []))
    reference_valid_mask = np.asarray(entries[0].valid_mask_volume, dtype=np.uint8)
    for entry in entries[1:]:
        if tuple(int(x) for x in entry.x.shape[1:]) != reference_shape:
            raise ValueError(
                "All pre-decoder training families must share the same input shape. "
                f"Expected {reference_shape}, got {tuple(int(x) for x in entry.x.shape[1:])} for {entry.family}"
            )
        if list(entry.bundle_info.get("channel_names", [])) != reference_channels:
            raise ValueError(f"Channel layout mismatch for family {entry.family}")
        if not np.array_equal(np.asarray(entry.valid_mask_volume, dtype=np.uint8), reference_valid_mask):
            raise ValueError(f"Valid-mask layout mismatch for family {entry.family}")
    return {
        "input_shape": list(reference_shape),
        "channel_names": reference_channels,
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


class SyndromeEditPreDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dense_hidden_dim: int,
        dropout: float,
        valid_mask_channel_index: int = 1,
    ) -> None:
        super().__init__()
        self.valid_mask_channel_index = int(valid_mask_channel_index)
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [Residual3DBlock(hidden_channels, dropout=dropout) for _ in range(num_blocks)]
        )
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.head_norm = nn.GroupNorm(groups, hidden_channels)
        self.head_relu = nn.ReLU()
        self.edit_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        self.needs_fc1 = nn.Linear(hidden_channels, dense_hidden_dim)
        self.needs_fc2 = nn.Linear(dense_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_relu(self.head_norm(h))
        edit_logits = self.edit_head(h).squeeze(1)
        valid_mask = x[:, self.valid_mask_channel_index : self.valid_mask_channel_index + 1]
        denom = torch.clamp(valid_mask.sum(dim=(2, 3, 4)), min=1.0)
        pooled = (h * valid_mask).sum(dim=(2, 3, 4)) / denom
        pooled = self.dropout(self.head_relu(self.needs_fc1(pooled)))
        needs_edit_logits = self.needs_fc2(pooled).squeeze(1)
        return {
            "edit_logits": edit_logits,
            "needs_edit_logits": needs_edit_logits,
            "pooled_features": pooled,
        }


class CandidateEditSelector(nn.Module):
    def __init__(
        self,
        *,
        shot_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        patch_feature_offset: int | None = None,
        patch_feature_dim: int = 0,
        patch_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_feature_offset = None if patch_feature_offset is None else int(patch_feature_offset)
        self.patch_feature_dim = int(patch_feature_dim)
        self.patch_hidden_dim = int(patch_hidden_dim or DEFAULT_SELECTOR_PATCH_HIDDEN_DIM)
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            if self.patch_feature_offset < 0:
                raise ValueError("patch_feature_offset must be non-negative")
            if self.patch_feature_offset + self.patch_feature_dim > int(candidate_feature_dim):
                raise ValueError(
                    "Patch feature slice exceeds candidate feature dimension. "
                    f"offset={self.patch_feature_offset} dim={self.patch_feature_dim} "
                    f"candidate_feature_dim={candidate_feature_dim}"
                )
            self.patch_fc1 = nn.Linear(self.patch_feature_dim, self.patch_hidden_dim)
            self.patch_fc2 = nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim)
            candidate_input_dim = int(candidate_feature_dim) - self.patch_feature_dim + self.patch_hidden_dim
        else:
            self.patch_feature_offset = None
            self.patch_feature_dim = 0
            candidate_input_dim = int(candidate_feature_dim)
        self.fc1 = nn.Linear(int(shot_feature_dim) + candidate_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, shot_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            start = int(self.patch_feature_offset)
            end = start + int(self.patch_feature_dim)
            patch = candidate_features[:, start:end]
            patch_embed = self.dropout(self.relu(self.patch_fc1(patch)))
            patch_embed = self.dropout(self.relu(self.patch_fc2(patch_embed)))
            candidate_features = torch.cat(
                [candidate_features[:, :start], candidate_features[:, end:], patch_embed],
                dim=1,
            )
        x = torch.cat([shot_features, candidate_features], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x).squeeze(1)


class RiskAwareCandidateEditSelector(nn.Module):
    def __init__(
        self,
        *,
        shot_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        patch_feature_offset: int | None = None,
        patch_feature_dim: int = 0,
        patch_hidden_dim: int | None = None,
        utility_harm_logit_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
    ) -> None:
        super().__init__()
        self.patch_feature_offset = None if patch_feature_offset is None else int(patch_feature_offset)
        self.patch_feature_dim = int(patch_feature_dim)
        self.patch_hidden_dim = int(patch_hidden_dim or DEFAULT_SELECTOR_PATCH_HIDDEN_DIM)
        self.utility_harm_logit_weight = float(utility_harm_logit_weight)
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            if self.patch_feature_offset < 0:
                raise ValueError("patch_feature_offset must be non-negative")
            if self.patch_feature_offset + self.patch_feature_dim > int(candidate_feature_dim):
                raise ValueError(
                    "Patch feature slice exceeds candidate feature dimension. "
                    f"offset={self.patch_feature_offset} dim={self.patch_feature_dim} "
                    f"candidate_feature_dim={candidate_feature_dim}"
                )
            self.patch_fc1 = nn.Linear(self.patch_feature_dim, self.patch_hidden_dim)
            self.patch_fc2 = nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim)
            candidate_input_dim = int(candidate_feature_dim) - self.patch_feature_dim + self.patch_hidden_dim
        else:
            self.patch_feature_offset = None
            self.patch_feature_dim = 0
            candidate_input_dim = int(candidate_feature_dim)
        self.fc1 = nn.Linear(int(shot_feature_dim) + candidate_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.benefit_out = nn.Linear(hidden_dim, 1)
        self.harm_out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _prepare_candidate_features(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            start = int(self.patch_feature_offset)
            end = start + int(self.patch_feature_dim)
            patch = candidate_features[:, start:end]
            patch_embed = self.dropout(self.relu(self.patch_fc1(patch)))
            patch_embed = self.dropout(self.relu(self.patch_fc2(patch_embed)))
            candidate_features = torch.cat(
                [candidate_features[:, :start], candidate_features[:, end:], patch_embed],
                dim=1,
            )
        return candidate_features

    def component_logits(
        self,
        shot_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_features = self._prepare_candidate_features(candidate_features)
        x = torch.cat([shot_features, candidate_features], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        benefit_logits = self.benefit_out(x).squeeze(1)
        harm_logits = self.harm_out(x).squeeze(1)
        return benefit_logits, harm_logits

    def utility_from_components(
        self,
        benefit_logits: torch.Tensor,
        harm_logits: torch.Tensor,
    ) -> torch.Tensor:
        return benefit_logits - float(self.utility_harm_logit_weight) * harm_logits

    def forward(self, shot_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        benefit_logits, harm_logits = self.component_logits(shot_features, candidate_features)
        return self.utility_from_components(benefit_logits, harm_logits)


class RiskGuardCandidateEditSelector(nn.Module):
    def __init__(
        self,
        *,
        shot_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        patch_feature_offset: int | None = None,
        patch_feature_dim: int = 0,
        patch_hidden_dim: int | None = None,
        utility_harm_logit_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
    ) -> None:
        super().__init__()
        self.patch_feature_offset = None if patch_feature_offset is None else int(patch_feature_offset)
        self.patch_feature_dim = int(patch_feature_dim)
        self.patch_hidden_dim = int(patch_hidden_dim or DEFAULT_SELECTOR_PATCH_HIDDEN_DIM)
        self.utility_harm_logit_weight = float(utility_harm_logit_weight)
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            if self.patch_feature_offset < 0:
                raise ValueError("patch_feature_offset must be non-negative")
            if self.patch_feature_offset + self.patch_feature_dim > int(candidate_feature_dim):
                raise ValueError(
                    "Patch feature slice exceeds candidate feature dimension. "
                    f"offset={self.patch_feature_offset} dim={self.patch_feature_dim} "
                    f"candidate_feature_dim={candidate_feature_dim}"
                )
            self.patch_fc1 = nn.Linear(self.patch_feature_dim, self.patch_hidden_dim)
            self.patch_fc2 = nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim)
            candidate_input_dim = int(candidate_feature_dim) - self.patch_feature_dim + self.patch_hidden_dim
        else:
            self.patch_feature_offset = None
            self.patch_feature_dim = 0
            candidate_input_dim = int(candidate_feature_dim)
        self.fc1 = nn.Linear(int(shot_feature_dim) + candidate_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rank_out = nn.Linear(hidden_dim, 1)
        self.harm_out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _prepare_candidate_features(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if self.patch_feature_offset is not None and self.patch_feature_dim > 0:
            start = int(self.patch_feature_offset)
            end = start + int(self.patch_feature_dim)
            patch = candidate_features[:, start:end]
            patch_embed = self.dropout(self.relu(self.patch_fc1(patch)))
            patch_embed = self.dropout(self.relu(self.patch_fc2(patch_embed)))
            candidate_features = torch.cat(
                [candidate_features[:, :start], candidate_features[:, end:], patch_embed],
                dim=1,
            )
        return candidate_features

    def component_logits(
        self,
        shot_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_features = self._prepare_candidate_features(candidate_features)
        x = torch.cat([shot_features, candidate_features], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        rank_logits = self.rank_out(x).squeeze(1)
        harm_logits = self.harm_out(x).squeeze(1)
        return rank_logits, harm_logits

    def utility_from_components(
        self,
        rank_logits: torch.Tensor,
        harm_logits: torch.Tensor,
    ) -> torch.Tensor:
        return rank_logits - float(self.utility_harm_logit_weight) * harm_logits

    def forward(self, shot_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        rank_logits, harm_logits = self.component_logits(shot_features, candidate_features)
        return self.utility_from_components(rank_logits, harm_logits)


def _make_candidate_selector_module(
    *,
    selector_model: str,
    selector_kwargs: dict[str, Any],
) -> nn.Module:
    model_name = str(selector_model)
    if model_name == SELECTOR_MODEL_SCALAR:
        return CandidateEditSelector(**selector_kwargs)
    if model_name == SELECTOR_MODEL_RISK_AWARE:
        return RiskAwareCandidateEditSelector(**selector_kwargs)
    if model_name == SELECTOR_MODEL_RISK_GUARD:
        return RiskGuardCandidateEditSelector(**selector_kwargs)
    raise ValueError(f"Unsupported selector_model: {selector_model!r}")


def _selector_model_name(selector: nn.Module) -> str:
    if isinstance(selector, RiskAwareCandidateEditSelector):
        return SELECTOR_MODEL_RISK_AWARE
    if isinstance(selector, RiskGuardCandidateEditSelector):
        return SELECTOR_MODEL_RISK_GUARD
    return SELECTOR_MODEL_SCALAR


class HardShotRouter(nn.Module):
    def __init__(
        self,
        *,
        shot_feature_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(shot_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, shot_features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(shot_features)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x).squeeze(1)


class MotifVocabularyHead(nn.Module):
    def __init__(
        self,
        *,
        shot_feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(shot_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, shot_features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(shot_features)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x)


def _make_tensor_dataset(
    x: np.ndarray,
    edit_target_volume: np.ndarray,
    needs_edit: np.ndarray,
    edit_target_known: np.ndarray,
    action_motif_label: np.ndarray,
    action_motif_active: np.ndarray,
) -> Any:
    return common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(edit_target_volume, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(needs_edit, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(edit_target_known, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(action_motif_label, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(action_motif_active, dtype=np.float32)),
    )


def _compute_shot_sample_weights(
    *,
    needs_edit: np.ndarray,
    edit_target_known: np.ndarray,
    hard_shot_solved_weight: float,
    hard_shot_unsolved_weight: float,
) -> np.ndarray:
    needs = np.asarray(needs_edit, dtype=np.float32).reshape(-1)
    known = np.asarray(edit_target_known, dtype=np.float32).reshape(-1)
    solved_hard = (needs >= 0.5) & (known >= 0.5)
    unsolved_hard = (needs >= 0.5) & (known < 0.5)
    weights = np.ones(needs.shape[0], dtype=np.float32)
    weights[solved_hard] = float(max(hard_shot_solved_weight, 1.0))
    weights[unsolved_hard] = float(max(hard_shot_unsolved_weight, 1.0))
    mean_weight = float(weights.mean())
    if mean_weight > 0.0:
        weights = weights / mean_weight
    return np.ascontiguousarray(weights)


def _build_edit_supervision_mask(
    *,
    known_mask: np.ndarray,
    needs_edit: np.ndarray,
    mode: str,
) -> np.ndarray:
    known = np.asarray(known_mask, dtype=np.float32).reshape(-1)
    needs = np.asarray(needs_edit, dtype=np.float32).reshape(-1)
    if mode == EDIT_SUPERVISION_MODE_ALL_KNOWN:
        return np.ascontiguousarray(known)
    if mode == EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY:
        return np.ascontiguousarray(known * (needs >= 0.5).astype(np.float32))
    raise ValueError(f"Unsupported edit_supervision_mode: {mode!r}")


def _infer_valid_mask_channel_index(bundle_info: dict[str, Any]) -> int:
    channel_names = list(bundle_info.get("channel_names", []))
    if len(channel_names) < 2 or channel_names[1] != "valid_mask":
        raise ValueError(
            f"Unexpected channel layout for syndrome_edit_predecoder: {channel_names[:4]}"
        )
    return 1


def _compute_pos_weight_from_targets(target: np.ndarray, weight_mask: np.ndarray, *, max_weight: float = 64.0) -> float:
    target_f = np.asarray(target, dtype=np.float32)
    weights = np.asarray(weight_mask, dtype=np.float32)
    positives = float(np.sum(target_f * weights))
    total = float(np.sum(weights))
    negatives = max(total - positives, 0.0)
    if positives <= 0.0 or negatives <= 0.0:
        return 1.0
    return float(np.clip(negatives / positives, 1.0, max_weight))


def _masked_edit_loss(
    edit_logits: torch.Tensor,
    target_volume: torch.Tensor,
    *,
    valid_mask_volume: torch.Tensor,
    known_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    per_cell = F.binary_cross_entropy_with_logits(
        edit_logits,
        target_volume,
        reduction="none",
        pos_weight=pos_weight,
    )
    weight = known_mask[:, None, None, None] * valid_mask_volume[None, :, :, :]
    denom = torch.clamp(weight.sum(), min=1.0)
    return torch.sum(per_cell * weight) / denom


def _decision_aware_ranking_loss(
    *,
    edit_logits: torch.Tensor,
    needs_edit_logits: torch.Tensor,
    target_volume: torch.Tensor,
    known_mask: torch.Tensor,
    needs_edit_target: torch.Tensor,
    valid_mask_volume: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_known = (
        (known_mask >= 0.5)
        & (needs_edit_target >= 0.5)
    ).to(dtype=edit_logits.dtype)
    target_mask = target_volume * valid_mask_volume[None, :, :, :]
    target_weight = target_mask.sum(dim=(1, 2, 3))
    active_mask = hard_known * (target_weight > 0.0).to(dtype=edit_logits.dtype)
    # Against the identity no-edit candidate, the Bernoulli log-prob ratio
    # collapses to the needs_edit logit plus the logits of the target edit bits.
    advantage = needs_edit_logits + (edit_logits * target_mask).sum(dim=(1, 2, 3))
    per_shot_loss = F.relu(float(margin) - advantage) * active_mask
    denom = torch.clamp(active_mask.sum(), min=1.0)
    loss = per_shot_loss.sum() / denom
    return loss, advantage.detach(), active_mask.detach()


def _action_motif_competition_loss(
    *,
    edit_logits: torch.Tensor,
    needs_edit_logits: torch.Tensor,
    action_motif_label: torch.Tensor,
    action_motif_active: torch.Tensor,
    motif_mask_table: torch.Tensor,
    detector_time_index: torch.Tensor,
    row_index_by_detector: torch.Tensor,
    col_index_by_detector: torch.Tensor,
    identity_margin: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    active = (
        (action_motif_active >= 0.5)
        & (action_motif_label >= 0)
    )
    zero = torch.tensor(0.0, dtype=torch.float32, device=edit_logits.device)
    if not bool(torch.any(active)):
        return zero, {
            "action_motif_ce_loss": None,
            "action_motif_identity_margin_loss": None,
            "action_motif_accuracy": None,
            "action_motif_active_count": 0.0,
            "action_motif_nonzero_active_fraction": None,
            "action_motif_mean_identity_gap": None,
        }
    detector_logits = edit_logits[:, detector_time_index, row_index_by_detector, col_index_by_detector]
    detector_logits = detector_logits[active]
    labels = action_motif_label[active].to(dtype=torch.int64)
    needs_logits = needs_edit_logits[active]
    mask = motif_mask_table.to(dtype=detector_logits.dtype)[None, :, :]
    bit_logits = detector_logits[:, None, :]
    bit_logprob = (
        mask * F.logsigmoid(bit_logits)
        + (1.0 - mask) * F.logsigmoid(-bit_logits)
    ).sum(dim=2)
    nonzero_mask = (motif_mask_table.sum(dim=1) > 0).to(dtype=detector_logits.dtype)[None, :]
    action_scores = bit_logprob + (
        nonzero_mask * F.logsigmoid(needs_logits)[:, None]
        + (1.0 - nonzero_mask) * F.logsigmoid(-needs_logits)[:, None]
    )
    ce_loss = F.cross_entropy(action_scores, labels)
    pred = torch.argmax(action_scores, dim=1)
    nonzero_active = labels > 0
    margin_loss = zero
    mean_identity_gap: float | None = None
    if bool(torch.any(nonzero_active)):
        row_index = torch.arange(action_scores.shape[0], device=action_scores.device)
        target_scores = action_scores[row_index, labels]
        identity_scores = action_scores[:, 0]
        gaps = target_scores - identity_scores
        margin_loss = F.relu(float(identity_margin) - gaps[nonzero_active]).mean()
        mean_identity_gap = float(gaps[nonzero_active].mean().item())
        nonzero_fraction = float(nonzero_active.to(dtype=torch.float32).mean().item())
    else:
        nonzero_fraction = 0.0
    total_loss = ce_loss + margin_loss
    return total_loss, {
        "action_motif_ce_loss": float(ce_loss.item()),
        "action_motif_identity_margin_loss": float(margin_loss.item()),
        "action_motif_accuracy": float((pred == labels).to(dtype=torch.float32).mean().item()),
        "action_motif_active_count": float(labels.shape[0]),
        "action_motif_nonzero_active_fraction": float(nonzero_fraction),
        "action_motif_mean_identity_gap": mean_identity_gap,
    }


def _masked_edit_binary_metrics(
    edit_probs_volume: np.ndarray,
    target_volume: np.ndarray,
    *,
    valid_mask_volume: np.ndarray,
    known_mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    weight = (
        np.asarray(known_mask, dtype=np.float32)[:, None, None, None]
        * np.asarray(valid_mask_volume, dtype=np.float32)[None, :, :, :]
    )
    active = weight > 0.0
    if not np.any(active):
        return {
            "num_examples": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
            "positive_rate_target": None,
            "positive_rate_predicted": None,
            "mean_predicted_probability": None,
            "threshold_used": float(threshold),
        }
    return _binary_metrics_from_probs(
        np.asarray(edit_probs_volume, dtype=np.float32)[active],
        np.asarray(target_volume, dtype=np.uint8)[active],
        threshold=threshold,
    )


def _collect_outputs(
    *,
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = common.TensorDataset(torch.from_numpy(np.ascontiguousarray(x)))
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    edit_chunks: list[np.ndarray] = []
    needs_chunks: list[np.ndarray] = []
    pooled_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            outputs = model(xb)
            edit_chunks.append(outputs["edit_logits"].detach().cpu().numpy())
            needs_chunks.append(outputs["needs_edit_logits"].detach().cpu().numpy())
            pooled_chunks.append(outputs["pooled_features"].detach().cpu().numpy())
    if not edit_chunks:
        return {
            "edit_logits": np.zeros((0, 0, 0, 0), dtype=np.float32),
            "needs_edit_logits": np.zeros((0,), dtype=np.float32),
            "pooled_features": np.zeros((0, 0), dtype=np.float32),
        }
    return {
        "edit_logits": np.asarray(np.concatenate(edit_chunks, axis=0), dtype=np.float32),
        "needs_edit_logits": np.asarray(np.concatenate(needs_chunks, axis=0), dtype=np.float32),
        "pooled_features": np.asarray(np.concatenate(pooled_chunks, axis=0), dtype=np.float32),
    }


def _selector_candidate_feature_dim(
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    *,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
) -> int:
    base_dim = SELECTOR_CANDIDATE_BASE_FEATURE_DIM
    if bool(candidate_geometry_features):
        base_dim += SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM
    if bool(candidate_pattern_features):
        base_dim += SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM
    if bool(candidate_local_evidence_features):
        base_dim += SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM
    if bool(candidate_local_patch_features):
        base_dim += SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM
    if str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM:
        # obs delta (2) + class-changed flag (1) + baseline one-hot (4)
        # + edited one-hot (4) + baseline-to-edited transition one-hot (16)
        return base_dim + 27
    return base_dim


def _candidate_local_patch_feature_slice(
    *,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
) -> slice | None:
    if not bool(candidate_local_patch_features):
        return None
    offset = SELECTOR_CANDIDATE_BASE_FEATURE_DIM
    if bool(candidate_geometry_features):
        offset += SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM
    if bool(candidate_pattern_features):
        offset += SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM
    if bool(candidate_local_evidence_features):
        offset += SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM
    return slice(offset, offset + SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM)


def _transition_feature_offsets(
    *,
    candidate_feature_dim: int | None = None,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
) -> dict[str, slice]:
    if candidate_feature_dim is not None:
        base_dim = int(candidate_feature_dim) - 27
        if base_dim < SELECTOR_CANDIDATE_BASE_FEATURE_DIM:
            raise ValueError(
                "Candidate feature dimension is too small to contain benefit/harm transition features. "
                f"Got {candidate_feature_dim}"
            )
    else:
        base_dim = _selector_candidate_feature_dim(
            SELECTOR_TARGET_MODE_CORRECTNESS,
            candidate_geometry_features=bool(candidate_geometry_features),
            candidate_pattern_features=bool(candidate_pattern_features),
            candidate_local_evidence_features=bool(candidate_local_evidence_features),
            candidate_local_patch_features=bool(candidate_local_patch_features),
        )
    obs_delta = slice(base_dim, base_dim + 2)
    class_changed = slice(base_dim + 2, base_dim + 3)
    baseline_class = slice(base_dim + 3, base_dim + 7)
    edited_class = slice(base_dim + 7, base_dim + 11)
    transition = slice(base_dim + 11, base_dim + 27)
    return {
        "obs_delta": obs_delta,
        "class_changed": class_changed,
        "baseline_class": baseline_class,
        "edited_class": edited_class,
        "transition": transition,
    }


def _build_candidate_policy_specs(
    *,
    needs_edit_threshold_grid: list[float],
    edit_threshold_grid: list[float],
    max_edit_weight_grid: list[int],
) -> list[CandidatePolicySpec]:
    return [
        CandidatePolicySpec(
            needs_edit_threshold=float(needs_thr),
            edit_threshold=float(edit_thr),
            max_predicted_edit_weight=int(max_weight),
        )
        for needs_thr in needs_edit_threshold_grid
        for edit_thr in edit_threshold_grid
        for max_weight in max_edit_weight_grid
    ]


def _choose_candidate_indices_for_policy(
    shot_probs: np.ndarray,
    needs_edit_prob: float,
    *,
    needs_edit_threshold: float,
    edit_threshold: float,
    max_predicted_edit_weight: int,
) -> np.ndarray:
    if max_predicted_edit_weight <= 0:
        return np.zeros((0,), dtype=np.int64)
    if float(needs_edit_prob) < float(needs_edit_threshold):
        return np.zeros((0,), dtype=np.int64)
    probs = np.asarray(shot_probs, dtype=np.float32).reshape(-1)
    chosen = np.flatnonzero(probs >= float(edit_threshold))
    if chosen.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if chosen.size > int(max_predicted_edit_weight):
        order = np.argsort(-probs[chosen], kind="mergesort")
        chosen = chosen[order[: int(max_predicted_edit_weight)]]
    return np.asarray(chosen, dtype=np.int64)


def _candidate_geometry_feature_vector(
    *,
    edit_indices: tuple[int, ...],
    detector_time_index: np.ndarray | None,
    row_index_by_detector: np.ndarray | None,
    col_index_by_detector: np.ndarray | None,
) -> np.ndarray:
    if (
        not edit_indices
        or detector_time_index is None
        or row_index_by_detector is None
        or col_index_by_detector is None
    ):
        return np.zeros((SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM,), dtype=np.float32)
    indices = np.asarray(edit_indices, dtype=np.intp)
    t_all = np.asarray(detector_time_index, dtype=np.float32).reshape(-1)
    r_all = np.asarray(row_index_by_detector, dtype=np.float32).reshape(-1)
    c_all = np.asarray(col_index_by_detector, dtype=np.float32).reshape(-1)
    if indices.size == 0:
        return np.zeros((SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM,), dtype=np.float32)
    t_norm = t_all[indices] / max(float(np.max(t_all)), 1.0)
    r_norm = r_all[indices] / max(float(np.max(r_all)), 1.0)
    c_norm = c_all[indices] / max(float(np.max(c_all)), 1.0)
    return np.asarray(
        [
            float(np.mean(t_norm)),
            float(np.mean(r_norm)),
            float(np.mean(c_norm)),
            float(np.std(t_norm)),
            float(np.std(r_norm)),
            float(np.std(c_norm)),
            float(np.max(t_norm) - np.min(t_norm)),
            float(np.max(r_norm) - np.min(r_norm)),
            float(np.max(c_norm) - np.min(c_norm)),
        ],
        dtype=np.float32,
    )


def _candidate_anchor_coordinate_features(
    *,
    edit_indices: tuple[int, ...],
    detector_time_index: np.ndarray | None,
    row_index_by_detector: np.ndarray | None,
    col_index_by_detector: np.ndarray | None,
) -> tuple[float, float, float]:
    if (
        not edit_indices
        or detector_time_index is None
        or row_index_by_detector is None
        or col_index_by_detector is None
    ):
        return 0.0, 0.0, 0.0
    t_all = np.asarray(detector_time_index, dtype=np.float32).reshape(-1)
    r_all = np.asarray(row_index_by_detector, dtype=np.float32).reshape(-1)
    c_all = np.asarray(col_index_by_detector, dtype=np.float32).reshape(-1)
    coords = sorted(
        (float(t_all[int(idx)]), float(r_all[int(idx)]), float(c_all[int(idx)]))
        for idx in edit_indices
    )
    if not coords:
        return 0.0, 0.0, 0.0
    anchor_t, anchor_r, anchor_c = coords[0]
    return (
        float(anchor_t / max(float(np.max(t_all)), 1.0)),
        float(anchor_r / max(float(np.max(r_all)), 1.0)),
        float(anchor_c / max(float(np.max(c_all)), 1.0)),
    )


def _candidate_pattern_feature_vector(
    *,
    edit_indices: tuple[int, ...],
    local_motif_pattern_index: int | None,
    local_motif_pattern_count: int,
    local_motif_num_patterns: int,
    detector_time_index: np.ndarray | None,
    row_index_by_detector: np.ndarray | None,
    col_index_by_detector: np.ndarray | None,
) -> np.ndarray:
    if local_motif_pattern_index is None or int(local_motif_pattern_index) < 0:
        return np.zeros((SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM,), dtype=np.float32)
    pattern_idx = int(local_motif_pattern_index)
    denom = max(int(local_motif_num_patterns), pattern_idx + 1, 1)
    anchor_t, anchor_r, anchor_c = _candidate_anchor_coordinate_features(
        edit_indices=edit_indices,
        detector_time_index=detector_time_index,
        row_index_by_detector=row_index_by_detector,
        col_index_by_detector=col_index_by_detector,
    )
    return np.asarray(
        [
            1.0,
            float((pattern_idx + 1) / denom),
            float(math.log1p(max(int(local_motif_pattern_count), 0))),
            anchor_t,
            anchor_r,
            anchor_c,
        ],
        dtype=np.float32,
    )


def _candidate_local_evidence_feature_vector(
    shot_probs: np.ndarray,
    *,
    edit_indices: tuple[int, ...],
    shot_detector_events: np.ndarray | None,
    detector_time_index: np.ndarray | None,
    row_index_by_detector: np.ndarray | None,
    col_index_by_detector: np.ndarray | None,
) -> np.ndarray:
    if (
        not edit_indices
        or shot_detector_events is None
        or detector_time_index is None
        or row_index_by_detector is None
        or col_index_by_detector is None
    ):
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM,), dtype=np.float32)
    probs = np.asarray(shot_probs, dtype=np.float32).reshape(-1)
    events = np.asarray(shot_detector_events, dtype=np.float32).reshape(-1)
    t_all = np.asarray(detector_time_index, dtype=np.intp).reshape(-1)
    r_all = np.asarray(row_index_by_detector, dtype=np.intp).reshape(-1)
    c_all = np.asarray(col_index_by_detector, dtype=np.intp).reshape(-1)
    indices = np.asarray(edit_indices, dtype=np.intp)
    if indices.size == 0 or probs.shape[0] != events.shape[0]:
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM,), dtype=np.float32)

    selected_events = events[indices]
    selected_probs = probs[indices]
    coords = sorted(
        (int(t_all[int(idx)]), int(r_all[int(idx)]), int(c_all[int(idx)]))
        for idx in indices.tolist()
    )
    if not coords:
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM,), dtype=np.float32)
    anchor_t, anchor_r, anchor_c = coords[0]
    neighborhood = (
        (np.abs(t_all - int(anchor_t)) <= 1)
        & (np.abs(r_all - int(anchor_r)) <= 1)
        & (np.abs(c_all - int(anchor_c)) <= 1)
    )
    if not np.any(neighborhood):
        neighborhood[indices] = True
    neigh_events = events[neighborhood]
    neigh_probs = probs[neighborhood]
    neigh_count = max(int(neigh_events.shape[0]), 1)
    selected_count = max(int(selected_events.shape[0]), 1)
    return np.asarray(
        [
            float(np.mean(selected_events)),
            float(np.max(selected_events)),
            float(np.sum(selected_events) / selected_count),
            float(np.mean(selected_events * selected_probs)),
            float(np.mean((1.0 - selected_events) * selected_probs)),
            float(np.mean(neigh_events)),
            float(np.max(neigh_events)),
            float(math.log1p(float(np.sum(neigh_events))) / math.log1p(float(neigh_count))),
            float(np.mean(neigh_probs)),
            float(np.max(neigh_probs)),
        ],
        dtype=np.float32,
    )


def _candidate_local_patch_feature_vector(
    shot_probs: np.ndarray,
    *,
    edit_indices: tuple[int, ...],
    shot_detector_events: np.ndarray | None,
    detector_time_index: np.ndarray | None,
    row_index_by_detector: np.ndarray | None,
    col_index_by_detector: np.ndarray | None,
) -> np.ndarray:
    if (
        not edit_indices
        or shot_detector_events is None
        or detector_time_index is None
        or row_index_by_detector is None
        or col_index_by_detector is None
    ):
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM,), dtype=np.float32)
    probs = np.asarray(shot_probs, dtype=np.float32).reshape(-1)
    events = np.asarray(shot_detector_events, dtype=np.float32).reshape(-1)
    t_all = np.asarray(detector_time_index, dtype=np.intp).reshape(-1)
    r_all = np.asarray(row_index_by_detector, dtype=np.intp).reshape(-1)
    c_all = np.asarray(col_index_by_detector, dtype=np.intp).reshape(-1)
    indices = np.asarray(edit_indices, dtype=np.intp)
    if indices.size == 0 or probs.shape[0] != events.shape[0]:
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM,), dtype=np.float32)

    coords = sorted(
        (int(t_all[int(idx)]), int(r_all[int(idx)]), int(c_all[int(idx)]))
        for idx in indices.tolist()
    )
    if not coords:
        return np.zeros((SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM,), dtype=np.float32)
    anchor_t, anchor_r, anchor_c = coords[0]
    coord_to_detector: dict[tuple[int, int, int], int] = {}
    for det_idx, coord in enumerate(zip(t_all.tolist(), r_all.tolist(), c_all.tolist())):
        coord_to_detector.setdefault((int(coord[0]), int(coord[1]), int(coord[2])), int(det_idx))

    radius = int(SELECTOR_CANDIDATE_LOCAL_PATCH_RADIUS)
    features: list[float] = []
    for dt in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                det_idx = coord_to_detector.get((anchor_t + dt, anchor_r + dr, anchor_c + dc))
                if det_idx is None:
                    features.extend([0.0, 0.0])
                else:
                    features.extend([float(events[det_idx]), float(probs[det_idx])])
    return np.asarray(features, dtype=np.float32)


def _candidate_feature_vector(
    shot_probs: np.ndarray,
    *,
    needs_edit_prob: float,
    edit_indices: tuple[int, ...],
    policy: CandidatePolicySpec | None,
    source: str,
    motif_count: int = 0,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    local_motif_pattern_index: int | None = None,
    local_motif_pattern_count: int = 0,
    local_motif_num_patterns: int = 0,
    detector_time_index: np.ndarray | None = None,
    row_index_by_detector: np.ndarray | None = None,
    col_index_by_detector: np.ndarray | None = None,
    shot_detector_events: np.ndarray | None = None,
) -> np.ndarray:
    if edit_indices:
        selected_probs = np.asarray(shot_probs, dtype=np.float32)[list(edit_indices)]
        max_prob = float(selected_probs.max())
        mean_prob = float(selected_probs.mean())
        min_prob = float(selected_probs.min())
        sum_prob = float(selected_probs.sum())
    else:
        max_prob = 0.0
        mean_prob = 0.0
        min_prob = 0.0
        sum_prob = 0.0
    edit_weight = float(len(edit_indices))
    if policy is not None:
        policy_needs_edit_threshold = float(policy.needs_edit_threshold)
        policy_edit_threshold = float(policy.edit_threshold)
        policy_max_predicted_edit_weight = float(policy.max_predicted_edit_weight)
    else:
        policy_needs_edit_threshold = -1.0
        policy_edit_threshold = -1.0
        policy_max_predicted_edit_weight = -1.0
    features = [
        float(needs_edit_prob),
        policy_needs_edit_threshold,
        policy_edit_threshold,
        policy_max_predicted_edit_weight,
        edit_weight,
        1.0 if edit_weight <= 0.0 else 0.0,
        max_prob,
        mean_prob,
        min_prob,
        sum_prob,
        1.0 if source == "policy" else 0.0,
        1.0 if source == "motif" else 0.0,
        float(math.log1p(max(int(motif_count), 0))),
    ]
    if bool(candidate_geometry_features):
        features.extend(
            _candidate_geometry_feature_vector(
                edit_indices=edit_indices,
                detector_time_index=detector_time_index,
                row_index_by_detector=row_index_by_detector,
                col_index_by_detector=col_index_by_detector,
            ).tolist()
        )
    if bool(candidate_pattern_features):
        features.extend(
            _candidate_pattern_feature_vector(
                edit_indices=edit_indices,
                local_motif_pattern_index=local_motif_pattern_index,
                local_motif_pattern_count=int(local_motif_pattern_count),
                local_motif_num_patterns=int(local_motif_num_patterns),
                detector_time_index=detector_time_index,
                row_index_by_detector=row_index_by_detector,
                col_index_by_detector=col_index_by_detector,
            ).tolist()
        )
    if bool(candidate_local_evidence_features):
        features.extend(
            _candidate_local_evidence_feature_vector(
                shot_probs,
                edit_indices=edit_indices,
                shot_detector_events=shot_detector_events,
                detector_time_index=detector_time_index,
                row_index_by_detector=row_index_by_detector,
                col_index_by_detector=col_index_by_detector,
            ).tolist()
        )
    if bool(candidate_local_patch_features):
        features.extend(
            _candidate_local_patch_feature_vector(
                shot_probs,
                edit_indices=edit_indices,
                shot_detector_events=shot_detector_events,
                detector_time_index=detector_time_index,
                row_index_by_detector=row_index_by_detector,
                col_index_by_detector=col_index_by_detector,
            ).tolist()
        )
    return np.asarray(features, dtype=np.float32)


def _merge_motif_evidence_into_candidate_feature(
    feature: np.ndarray,
    *,
    motif_count: int,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    edit_indices: tuple[int, ...] = (),
    local_motif_pattern_index: int | None = None,
    local_motif_pattern_count: int = 0,
    local_motif_num_patterns: int = 0,
    detector_time_index: np.ndarray | None = None,
    row_index_by_detector: np.ndarray | None = None,
    col_index_by_detector: np.ndarray | None = None,
) -> np.ndarray:
    merged = np.asarray(feature, dtype=np.float32).copy()
    if merged.shape[0] < SELECTOR_CANDIDATE_BASE_FEATURE_DIM:
        return merged
    merged[11] = 1.0
    merged[12] = max(float(merged[12]), float(math.log1p(max(int(motif_count), 0))))
    if bool(candidate_pattern_features):
        pattern_offset = SELECTOR_CANDIDATE_BASE_FEATURE_DIM
        if bool(candidate_geometry_features):
            pattern_offset += SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM
        pattern_end = pattern_offset + SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM
        if merged.shape[0] >= pattern_end:
            pattern_features = _candidate_pattern_feature_vector(
                edit_indices=edit_indices,
                local_motif_pattern_index=local_motif_pattern_index,
                local_motif_pattern_count=int(local_motif_pattern_count),
                local_motif_num_patterns=int(local_motif_num_patterns),
                detector_time_index=detector_time_index,
                row_index_by_detector=row_index_by_detector,
                col_index_by_detector=col_index_by_detector,
            )
            if float(pattern_features[0]) > 0.0:
                merged[pattern_offset:pattern_end] = pattern_features
    return merged


def _enumerate_shot_candidates(
    shot_probs: np.ndarray,
    *,
    needs_edit_prob: float,
    policy_specs: list[CandidatePolicySpec],
    policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    detector_time_index: np.ndarray | None = None,
    row_index_by_detector: np.ndarray | None = None,
    col_index_by_detector: np.ndarray | None = None,
    shot_detector_events: np.ndarray | None = None,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_placements: LocalMotifPlacements | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_detectors = int(np.asarray(shot_probs).reshape(-1).shape[0])
    geometry_kwargs = {
        "candidate_geometry_features": bool(candidate_geometry_features),
        "candidate_pattern_features": bool(candidate_pattern_features),
        "candidate_local_evidence_features": bool(candidate_local_evidence_features),
        "candidate_local_patch_features": bool(candidate_local_patch_features),
        "detector_time_index": detector_time_index,
        "row_index_by_detector": row_index_by_detector,
        "col_index_by_detector": col_index_by_detector,
        "shot_detector_events": shot_detector_events,
    }
    seen: dict[tuple[int, ...], np.ndarray] = {}
    seen[()] = _candidate_feature_vector(
        shot_probs,
        needs_edit_prob=needs_edit_prob,
        edit_indices=(),
        policy=None,
        source="identity",
        **geometry_kwargs,
    )
    if str(policy_candidate_mode) == SELECTOR_POLICY_CANDIDATE_MODE_ALL:
        for policy in policy_specs:
            chosen = _choose_candidate_indices_for_policy(
                shot_probs,
                needs_edit_prob,
                needs_edit_threshold=policy.needs_edit_threshold,
                edit_threshold=policy.edit_threshold,
                max_predicted_edit_weight=policy.max_predicted_edit_weight,
            )
            key = tuple(int(idx) for idx in chosen.tolist())
            if key in seen:
                continue
            seen[key] = _candidate_feature_vector(
                shot_probs,
                needs_edit_prob=needs_edit_prob,
                edit_indices=key,
                policy=policy,
                source="policy",
                **geometry_kwargs,
            )
    elif str(policy_candidate_mode) != SELECTOR_POLICY_CANDIDATE_MODE_NONE:
        raise ValueError(f"Unsupported selector policy candidate mode: {policy_candidate_mode!r}")
    if motif_vocabulary is not None:
        if int(motif_vocabulary.detector_count) != num_detectors:
            raise ValueError(
                "motif_vocabulary detector count must match shot probability width. "
                f"Expected {num_detectors}, got {motif_vocabulary.detector_count}"
            )
        for class_idx, key in enumerate(motif_vocabulary.detector_index_lists):
            if class_idx == 0 or not key:
                continue
            if key in seen:
                seen[key] = _merge_motif_evidence_into_candidate_feature(
                    seen[key],
                    motif_count=int(motif_vocabulary.counts[class_idx]),
                )
                continue
            seen[key] = _candidate_feature_vector(
                shot_probs,
                needs_edit_prob=needs_edit_prob,
                edit_indices=key,
                policy=None,
                source="motif",
                motif_count=int(motif_vocabulary.counts[class_idx]),
                **geometry_kwargs,
            )
    if local_motif_placements is not None:
        if int(local_motif_placements.detector_count) != num_detectors:
            raise ValueError(
                "local_motif_placements detector count must match shot probability width. "
                f"Expected {num_detectors}, got {local_motif_placements.detector_count}"
            )
        nonzero_rows = [
            row_idx
            for row_idx, key in enumerate(local_motif_placements.detector_index_lists)
            if row_idx > 0 and key
        ]
        if nonzero_rows:
            probs = np.asarray(shot_probs, dtype=np.float32).reshape(-1)
            scored_rows: list[tuple[float, int, int]] = []
            for row_idx in nonzero_rows:
                key = local_motif_placements.detector_index_lists[row_idx]
                selected_probs = probs[list(key)]
                score = float(selected_probs.sum())
                scored_rows.append((-score, len(key), int(row_idx)))
            scored_rows.sort()
            limit = min(len(scored_rows), max(int(local_motif_top_k), 0))
            for _neg_score, _weight, row_idx in scored_rows[:limit]:
                key = tuple(int(x) for x in local_motif_placements.detector_index_lists[row_idx])
                if key in seen:
                    seen[key] = _merge_motif_evidence_into_candidate_feature(
                        seen[key],
                        motif_count=int(local_motif_placements.counts[row_idx]),
                        candidate_geometry_features=bool(candidate_geometry_features),
                        candidate_pattern_features=bool(candidate_pattern_features),
                        edit_indices=key,
                        local_motif_pattern_index=int(local_motif_placements.pattern_indices[row_idx]),
                        local_motif_pattern_count=int(local_motif_placements.counts[row_idx]),
                        local_motif_num_patterns=int(local_motif_placements.num_patterns),
                        detector_time_index=detector_time_index,
                        row_index_by_detector=row_index_by_detector,
                        col_index_by_detector=col_index_by_detector,
                    )
                    continue
                seen[key] = _candidate_feature_vector(
                    shot_probs,
                    needs_edit_prob=needs_edit_prob,
                    edit_indices=key,
                    policy=None,
                    source="motif",
                    motif_count=int(local_motif_placements.counts[row_idx]),
                    local_motif_pattern_index=int(local_motif_placements.pattern_indices[row_idx]),
                    local_motif_pattern_count=int(local_motif_placements.counts[row_idx]),
                    local_motif_num_patterns=int(local_motif_placements.num_patterns),
                    **geometry_kwargs,
                )
    keys = list(seen.keys())
    candidate_features = np.asarray([seen[key] for key in keys], dtype=np.float32)
    candidate_edit_mask = np.zeros((len(keys), num_detectors), dtype=np.uint8)
    candidate_edit_weight = np.zeros((len(keys),), dtype=np.int16)
    for row_idx, key in enumerate(keys):
        if key:
            candidate_edit_mask[row_idx, list(key)] = np.uint8(1)
            candidate_edit_weight[row_idx] = np.int16(len(key))
    return candidate_features, candidate_edit_mask, candidate_edit_weight


def _candidate_selector_target_scores(
    *,
    candidate_is_correct: np.ndarray,
    candidate_edit_weight: np.ndarray,
    selector_target_mode: str,
    selector_score_edit_penalty: float,
    selector_harm_weight: float,
    selector_miss_weight: float,
) -> np.ndarray:
    correct = np.asarray(candidate_is_correct, dtype=np.uint8).reshape(-1) > 0
    edit_weight = np.asarray(candidate_edit_weight, dtype=np.int16).reshape(-1)
    if correct.shape[0] != edit_weight.shape[0]:
        raise ValueError("Candidate correctness and edit-weight arrays must have the same length")

    mode = str(selector_target_mode)
    if mode == SELECTOR_TARGET_MODE_CORRECTNESS:
        target_scores = correct.astype(np.float32, copy=False)
        if selector_score_edit_penalty > 0.0:
            target_scores = np.clip(
                target_scores - float(selector_score_edit_penalty) * edit_weight.astype(np.float32),
                0.0,
                1.0,
            )
        return np.asarray(target_scores, dtype=np.float32)

    if mode != SELECTOR_TARGET_MODE_BENEFIT_HARM:
        raise ValueError(f"Unsupported selector_target_mode: {selector_target_mode!r}")

    target_scores = np.zeros((correct.shape[0],), dtype=np.float32)
    identity_rows = np.flatnonzero(edit_weight == 0)
    baseline_correct = bool(correct[int(identity_rows[0])]) if identity_rows.size else False
    nonzero = edit_weight > 0
    if baseline_correct:
        # No nonzero edit can improve a shot that raw PyMatching already solved.
        target_scores[~correct] = -float(selector_harm_weight)
    else:
        target_scores[correct] = 1.0
        target_scores[np.logical_and(~correct, nonzero)] = -float(selector_miss_weight)
    if selector_score_edit_penalty > 0.0:
        target_scores[nonzero] -= (
            float(selector_score_edit_penalty) * edit_weight[nonzero].astype(np.float32)
        )
    return np.asarray(target_scores, dtype=np.float32)


def _candidate_transition_feature_matrix(
    *,
    baseline_predicted_observables: np.ndarray,
    edited_predicted_observables: np.ndarray,
) -> np.ndarray:
    baseline_obs = _as_uint8_2d(
        baseline_predicted_observables,
        name="baseline_predicted_observables",
    )
    edited_obs = _as_uint8_2d(
        edited_predicted_observables,
        name="edited_predicted_observables",
    )
    if baseline_obs.shape != edited_obs.shape:
        raise ValueError(
            "baseline and edited predicted observables must have the same shape. "
            f"Expected {baseline_obs.shape}, got {edited_obs.shape}"
        )
    baseline_class4 = _logical_class4_from_observable_flips(baseline_obs)
    edited_class4 = _logical_class4_from_observable_flips(edited_obs)
    eye = np.eye(len(CLASS4_LABELS), dtype=np.float32)
    baseline_one_hot = eye[baseline_class4]
    edited_one_hot = eye[edited_class4]
    transition_one_hot = np.zeros((baseline_class4.shape[0], len(CLASS4_LABELS) ** 2), dtype=np.float32)
    transition_index = baseline_class4.astype(np.int64) * len(CLASS4_LABELS) + edited_class4.astype(np.int64)
    transition_one_hot[np.arange(transition_index.shape[0]), transition_index] = 1.0
    obs_delta = np.bitwise_xor(baseline_obs, edited_obs).astype(np.float32, copy=False)
    class_changed = (baseline_class4 != edited_class4).astype(np.float32, copy=False).reshape(-1, 1)
    return np.ascontiguousarray(
        np.concatenate(
            [obs_delta, class_changed, baseline_one_hot, edited_one_hot, transition_one_hot],
            axis=1,
        ),
        dtype=np.float32,
    )


def _build_selector_candidate_bundle(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    policy_specs: list[CandidatePolicySpec],
    selector_score_edit_penalty: float,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_vocabulary: LocalMotifVocabulary | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
) -> SelectorCandidateBundle:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    edit_probs = common._sigmoid_np(outputs["edit_logits"])
    needs_edit_probs = common._sigmoid_np(outputs["needs_edit_logits"]).reshape(-1)
    pooled_features = np.asarray(outputs["pooled_features"], dtype=np.float32)
    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    detector_probs = np.asarray(edit_probs[:, t, r, c], dtype=np.float32)
    local_motif_placements = (
        _build_local_motif_placements(entry=entry, vocabulary=local_motif_vocabulary)
        if local_motif_vocabulary is not None
        else None
    )
    baseline_class4 = _logical_class4_from_observable_flips(subset["baseline_predicted_observables"])
    target_class4 = np.asarray(subset["logical_class4"], dtype=np.uint8).reshape(-1)
    target_transition_class = (
        baseline_class4.astype(np.int64, copy=False) * len(CLASS4_LABELS)
        + target_class4.astype(np.int64, copy=False)
    ).astype(np.int64, copy=False)

    shot_feature_rows: list[np.ndarray] = []
    candidate_feature_rows: list[np.ndarray] = []
    target_score_rows: list[np.ndarray] = []
    candidate_correct_rows: list[np.ndarray] = []
    shot_index_rows: list[np.ndarray] = []
    candidate_mask_rows: list[np.ndarray] = []
    candidate_weight_rows: list[np.ndarray] = []
    decode_fallback_count = 0
    for shot_idx in range(detector_probs.shape[0]):
        candidate_features, candidate_edit_mask, candidate_edit_weight = _enumerate_shot_candidates(
            detector_probs[shot_idx],
            needs_edit_prob=float(needs_edit_probs[shot_idx]),
            policy_specs=policy_specs,
            policy_candidate_mode=str(selector_policy_candidate_mode),
            candidate_geometry_features=bool(selector_candidate_geometry_features),
            candidate_pattern_features=bool(selector_candidate_pattern_features),
            candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            detector_time_index=t,
            row_index_by_detector=r,
            col_index_by_detector=c,
            shot_detector_events=subset["detector_events"][shot_idx],
            motif_vocabulary=motif_vocabulary,
            local_motif_placements=local_motif_placements,
            local_motif_top_k=int(local_motif_top_k),
        )
        edited_detector_events = np.repeat(
            subset["detector_events"][shot_idx : shot_idx + 1],
            candidate_edit_mask.shape[0],
            axis=0,
        )
        edited_detector_events ^= candidate_edit_mask
        baseline_predicted = np.repeat(
            subset["baseline_predicted_observables"][shot_idx : shot_idx + 1],
            candidate_edit_mask.shape[0],
            axis=0,
        )
        edited_predicted_observables, shot_decode_fallback = _safe_decode_edited_observables(
            matching=entry.matching,
            edited_detector_events=edited_detector_events,
            baseline_predicted_observables=baseline_predicted,
        )
        if str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM:
            transition_features = _candidate_transition_feature_matrix(
                baseline_predicted_observables=baseline_predicted,
                edited_predicted_observables=edited_predicted_observables,
            )
            candidate_features = np.concatenate(
                [np.asarray(candidate_features, dtype=np.float32), transition_features],
                axis=1,
            )
        decode_fallback_count += int(shot_decode_fallback)
        edited_class4 = _logical_class4_from_observable_flips(edited_predicted_observables)
        candidate_is_correct = (edited_class4 == subset["logical_class4"][shot_idx]).astype(np.uint8, copy=False)
        target_scores = _candidate_selector_target_scores(
            candidate_is_correct=candidate_is_correct,
            candidate_edit_weight=candidate_edit_weight,
            selector_target_mode=str(selector_target_mode),
            selector_score_edit_penalty=float(selector_score_edit_penalty),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
        )
        shot_feature_rows.append(
            np.repeat(pooled_features[shot_idx : shot_idx + 1], candidate_edit_mask.shape[0], axis=0)
        )
        candidate_feature_rows.append(np.asarray(candidate_features, dtype=np.float32))
        target_score_rows.append(np.asarray(target_scores, dtype=np.float32))
        candidate_correct_rows.append(np.asarray(candidate_is_correct, dtype=np.uint8))
        shot_index_rows.append(
            np.full((candidate_edit_mask.shape[0],), int(shot_idx), dtype=np.int64)
        )
        candidate_mask_rows.append(np.asarray(candidate_edit_mask, dtype=np.uint8))
        candidate_weight_rows.append(np.asarray(candidate_edit_weight, dtype=np.int16))

    if not shot_feature_rows:
        num_detectors = int(subset["detector_events"].shape[1]) if subset["detector_events"].ndim == 2 else 0
        return SelectorCandidateBundle(
            shot_features=np.zeros((0, pooled_features.shape[1] if pooled_features.ndim == 2 else 0), dtype=np.float32),
            candidate_features=np.zeros(
                (
                    0,
                    _selector_candidate_feature_dim(
                        str(selector_target_mode),
                        candidate_geometry_features=bool(selector_candidate_geometry_features),
                        candidate_pattern_features=bool(selector_candidate_pattern_features),
                        candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                        candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                    ),
                ),
                dtype=np.float32,
            ),
            target_scores=np.zeros((0,), dtype=np.float32),
            candidate_is_correct=np.zeros((0,), dtype=np.uint8),
            shot_indices=np.zeros((0,), dtype=np.int64),
            candidate_edit_mask=np.zeros((0, num_detectors), dtype=np.uint8),
            candidate_edit_weight=np.zeros((0,), dtype=np.int16),
            target_transition_class=np.ascontiguousarray(target_transition_class, dtype=np.int64),
            num_shots=int(subset["x"].shape[0]),
            num_detectors=num_detectors,
            decode_fallback_count=0,
        )

    return SelectorCandidateBundle(
        shot_features=np.ascontiguousarray(np.concatenate(shot_feature_rows, axis=0), dtype=np.float32),
        candidate_features=np.ascontiguousarray(np.concatenate(candidate_feature_rows, axis=0), dtype=np.float32),
        target_scores=np.ascontiguousarray(np.concatenate(target_score_rows, axis=0), dtype=np.float32),
        candidate_is_correct=np.ascontiguousarray(np.concatenate(candidate_correct_rows, axis=0), dtype=np.uint8),
        shot_indices=np.ascontiguousarray(np.concatenate(shot_index_rows, axis=0), dtype=np.int64),
        candidate_edit_mask=np.ascontiguousarray(np.concatenate(candidate_mask_rows, axis=0), dtype=np.uint8),
        candidate_edit_weight=np.ascontiguousarray(np.concatenate(candidate_weight_rows, axis=0), dtype=np.int16),
        target_transition_class=np.ascontiguousarray(target_transition_class, dtype=np.int64),
        num_shots=int(subset["x"].shape[0]),
        num_detectors=int(subset["detector_events"].shape[1]),
        decode_fallback_count=int(decode_fallback_count),
    )


def _concatenate_selector_training_arrays(
    bundles: list[SelectorCandidateBundle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    non_empty = [bundle for bundle in bundles if bundle.shot_features.shape[0] > 0]
    if not non_empty:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, _selector_candidate_feature_dim()), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.ascontiguousarray(np.concatenate([bundle.shot_features for bundle in non_empty], axis=0), dtype=np.float32),
        np.ascontiguousarray(
            np.concatenate([bundle.candidate_features for bundle in non_empty], axis=0),
            dtype=np.float32,
        ),
        np.ascontiguousarray(np.concatenate([bundle.target_scores for bundle in non_empty], axis=0), dtype=np.float32),
    )


def _apply_predicted_edits(
    *,
    detector_events: np.ndarray,
    edit_probs_volume: np.ndarray,
    needs_edit_probs: np.ndarray,
    detector_time_index: np.ndarray,
    row_index_by_detector: np.ndarray,
    col_index_by_detector: np.ndarray,
    needs_edit_threshold: float,
    edit_threshold: float,
    max_predicted_edit_weight: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shots = _as_uint8_2d(detector_events, name="detector_events").copy()
    probs = np.asarray(edit_probs_volume, dtype=np.float32)
    needs = np.asarray(needs_edit_probs, dtype=np.float32).reshape(-1)
    t = np.asarray(detector_time_index, dtype=np.intp)
    r = np.asarray(row_index_by_detector, dtype=np.intp)
    c = np.asarray(col_index_by_detector, dtype=np.intp)
    detector_probs = probs[:, t, r, c]
    chosen_mask = np.zeros_like(detector_probs, dtype=np.uint8)
    edit_weight = np.zeros(detector_probs.shape[0], dtype=np.int16)

    for shot_idx in range(detector_probs.shape[0]):
        chosen = _choose_candidate_indices_for_policy(
            detector_probs[shot_idx],
            float(needs[shot_idx]),
            needs_edit_threshold=needs_edit_threshold,
            edit_threshold=edit_threshold,
            max_predicted_edit_weight=max_predicted_edit_weight,
        )
        if chosen.size == 0:
            continue
        chosen_mask[shot_idx, chosen] = np.uint8(1)
        edit_weight[shot_idx] = np.int16(chosen.size)
        shots[shot_idx, chosen] ^= np.uint8(1)

    return shots, chosen_mask.astype(np.uint8, copy=False), edit_weight


def _safe_decode_edited_observables(
    *,
    matching: Any,
    edited_detector_events: np.ndarray,
    baseline_predicted_observables: np.ndarray,
) -> tuple[np.ndarray, int]:
    try:
        return pym_common._decode_batch(matching, edited_detector_events), 0
    except Exception:
        pass

    decode = getattr(matching, "decode", None)
    if not callable(decode):
        raise RuntimeError("PyMatching object exposes no per-shot decode fallback")

    shots = _as_uint8_2d(edited_detector_events, name="edited_detector_events")
    baseline = _as_uint8_2d(
        baseline_predicted_observables,
        name="baseline_predicted_observables",
    )
    pred_rows: list[np.ndarray] = []
    fallback_count = 0
    for shot_idx in range(shots.shape[0]):
        try:
            pred = np.asarray(decode(shots[shot_idx]), dtype=np.uint8)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            pred_rows.append(np.asarray(pred[0], dtype=np.uint8))
        except Exception:
            pred_rows.append(np.asarray(baseline[shot_idx], dtype=np.uint8))
            fallback_count += 1
    return np.asarray(pred_rows, dtype=np.uint8), fallback_count


def _system_metrics_from_chosen_edit_mask(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    chosen_edit_mask: np.ndarray,
    decision_payload: dict[str, Any],
    needs_metrics: dict[str, Any],
    edit_metrics: dict[str, Any],
    extra_change_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chosen_mask = _as_uint8_2d(chosen_edit_mask, name="chosen_edit_mask")
    if chosen_mask.shape != subset["detector_events"].shape:
        raise ValueError(
            "chosen_edit_mask shape must match detector_events. "
            f"Expected {subset['detector_events'].shape}, got {chosen_mask.shape}"
        )
    edited_detector_events = subset["detector_events"].copy()
    edited_detector_events ^= chosen_mask
    predicted_edit_weight = np.asarray(chosen_mask.sum(axis=1), dtype=np.int16)
    edited_predicted_observables, decode_fallback_count = _safe_decode_edited_observables(
        matching=entry.matching,
        edited_detector_events=edited_detector_events,
        baseline_predicted_observables=subset["baseline_predicted_observables"],
    )
    edited_class4 = _logical_class4_from_observable_flips(edited_predicted_observables)
    baseline_class4 = _logical_class4_from_observable_flips(subset["baseline_predicted_observables"])
    target_class4 = subset["logical_class4"]

    baseline_metrics = _hard_multiclass_metrics(baseline_class4, target_class4)
    edited_metrics = _hard_multiclass_metrics(edited_class4, target_class4)
    improved = np.logical_and(edited_class4 == target_class4, baseline_class4 != target_class4)
    harmed = np.logical_and(edited_class4 != target_class4, baseline_class4 == target_class4)
    unchanged = ~(improved | harmed)
    predicted_weight_hist = {
        str(int(weight)): int(count)
        for weight, count in zip(*np.unique(predicted_edit_weight, return_counts=True), strict=False)
    }
    change_summary = {
        "num_examples": int(target_class4.shape[0]),
        "num_improved_over_baseline": int(improved.sum()),
        "num_harmed_vs_baseline": int(harmed.sum()),
        "num_unchanged_vs_baseline": int(unchanged.sum()),
        "num_decode_failures_fallback_to_baseline": int(decode_fallback_count),
        "mean_predicted_edit_weight": float(predicted_edit_weight.mean()) if predicted_edit_weight.size else None,
        "predicted_edit_weight_histogram": predicted_weight_hist,
        "fraction_with_any_predicted_edit": (
            float(np.mean(predicted_edit_weight > 0)) if predicted_edit_weight.size else None
        ),
    }
    if extra_change_summary:
        change_summary.update(extra_change_summary)
    return {
        "decision": dict(decision_payload),
        "baseline_pymatching": baseline_metrics,
        "edited_pymatching": edited_metrics,
        "change_summary": change_summary,
        "needs_edit_head": needs_metrics,
        "edit_head": edit_metrics,
    }


def _no_edit_system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
) -> dict[str, Any]:
    num_shots = int(subset["detector_events"].shape[0])
    num_detectors = int(subset["detector_events"].shape[1])
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=np.zeros((num_shots, num_detectors), dtype=np.uint8),
        decision_payload={
            "selection_mode": SELECTION_MODE_RAW_NO_EDIT,
        },
        needs_metrics={
            "num_examples": num_shots,
            "threshold_used": None,
        },
        edit_metrics={
            "num_examples": num_shots * num_detectors,
            "threshold_used": None,
        },
    )


def _make_selector_tensor_dataset(
    shot_features: np.ndarray,
    candidate_features: np.ndarray,
    target_scores: np.ndarray,
) -> Any:
    return common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(shot_features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(candidate_features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(target_scores, dtype=np.float32)),
    )


def _selector_group_slices(bundle: SelectorCandidateBundle) -> list[slice]:
    if bundle.shot_indices.size == 0:
        return []
    slices: list[slice] = []
    start = 0
    current = int(bundle.shot_indices[0])
    for idx in range(1, int(bundle.shot_indices.shape[0])):
        shot_idx = int(bundle.shot_indices[idx])
        if shot_idx != current:
            slices.append(slice(start, idx))
            start = idx
            current = shot_idx
    slices.append(slice(start, int(bundle.shot_indices.shape[0])))
    return slices


def _selector_group_target_index(bundle: SelectorCandidateBundle, group_slice: slice) -> int:
    target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
    candidate_edit_weight = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
    if target_scores.size == 0:
        raise ValueError("Selector group cannot be empty")
    max_score = float(target_scores.max())
    if max_score <= 0.0:
        zero_weight = np.flatnonzero(candidate_edit_weight == 0)
        if zero_weight.size:
            return int(zero_weight[0])
        return 0
    best_rows = np.flatnonzero(np.isclose(target_scores, max_score))
    if best_rows.size == 1:
        return int(best_rows[0])
    best_weights = candidate_edit_weight[best_rows]
    return int(best_rows[int(np.argmin(best_weights))])


def _selector_group_weight(
    bundle: SelectorCandidateBundle,
    group_slice: slice,
    *,
    hard_shot_weight: float,
) -> float:
    target_row = _selector_group_target_index(bundle, group_slice)
    target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
    candidate_edit_weight = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
    if float(target_scores[target_row]) > 0.0 and int(candidate_edit_weight[target_row]) > 0:
        return float(max(hard_shot_weight, 1.0))
    return 1.0


def _selector_benefit_harm_pos_weights(
    bundles: list[SelectorCandidateBundle],
) -> tuple[float, float]:
    scores: list[np.ndarray] = []
    for bundle in bundles:
        if bundle.target_scores.size:
            scores.append(np.asarray(bundle.target_scores, dtype=np.float32).reshape(-1))
    if not scores:
        return 1.0, 1.0
    target_scores = np.concatenate(scores, axis=0)
    benefit_positive = target_scores > 0.0
    harm_positive = target_scores < 0.0

    def _pos_weight(mask: np.ndarray) -> float:
        positive = float(np.sum(mask))
        negative = float(mask.size - positive)
        if positive <= 0.0:
            return 1.0
        return float(max(negative / positive, 1.0))

    return _pos_weight(benefit_positive), _pos_weight(harm_positive)


def _selector_identity_competition_indices(
    bundle: SelectorCandidateBundle,
    group_slice: slice,
) -> tuple[int, int] | None:
    target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
    candidate_edit_weight = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
    if target_scores.size == 0:
        return None
    identity_rows = np.flatnonzero(candidate_edit_weight == 0)
    nonzero_rows = np.flatnonzero(candidate_edit_weight > 0)
    if identity_rows.size == 0 or nonzero_rows.size == 0:
        return None
    identity_row = int(identity_rows[0])
    identity_score = float(target_scores[identity_row])
    best_nonzero_score = float(np.max(target_scores[nonzero_rows]))
    if not best_nonzero_score > identity_score:
        return None
    best_nonzero_rows = nonzero_rows[np.isclose(target_scores[nonzero_rows], best_nonzero_score)]
    if best_nonzero_rows.size > 1:
        nonzero_weights = candidate_edit_weight[best_nonzero_rows]
        best_nonzero_row = int(best_nonzero_rows[int(np.argmin(nonzero_weights))])
    else:
        best_nonzero_row = int(best_nonzero_rows[0])
    return identity_row, best_nonzero_row


def _selector_identity_harm_competition_indices(
    bundle: SelectorCandidateBundle,
    group_slice: slice,
) -> tuple[int, int] | None:
    target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
    candidate_edit_weight = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
    if target_scores.size == 0:
        return None
    identity_rows = np.flatnonzero(candidate_edit_weight == 0)
    nonzero_rows = np.flatnonzero(candidate_edit_weight > 0)
    if identity_rows.size == 0 or nonzero_rows.size == 0:
        return None
    identity_row = int(identity_rows[0])
    identity_score = float(target_scores[identity_row])
    best_nonzero_score = float(np.max(target_scores[nonzero_rows]))
    if not identity_score > best_nonzero_score:
        return None
    best_nonzero_rows = nonzero_rows[np.isclose(target_scores[nonzero_rows], best_nonzero_score)]
    if best_nonzero_rows.size > 1:
        nonzero_weights = candidate_edit_weight[best_nonzero_rows]
        best_nonzero_row = int(best_nonzero_rows[int(np.argmin(nonzero_weights))])
    else:
        best_nonzero_row = int(best_nonzero_rows[0])
    return identity_row, best_nonzero_row


def _build_motif_vocabulary(
    bundles: list[tuple[PreparedEditFamily, dict[str, np.ndarray]]],
    *,
    max_classes: int,
) -> MotifVocabulary:
    if max_classes < 2:
        raise ValueError("motif vocabulary requires at least 2 classes including identity")
    counter: dict[tuple[int, ...], int] = {}
    detector_count: int | None = None
    for entry, subset in bundles:
        num_detectors = int(subset["detector_events"].shape[1])
        detector_count = num_detectors if detector_count is None else detector_count
        if detector_count != num_detectors:
            raise ValueError("All motif-vocabulary subsets must share detector count")
        t = np.asarray(entry.detector_time_index, dtype=np.intp)
        r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
        c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
        target_volume = np.asarray(subset["edit_target_volume"], dtype=np.float32)
        needs_edit = np.asarray(subset["needs_edit"], dtype=np.float32).reshape(-1)
        known = np.asarray(subset["edit_target_known"], dtype=np.float32).reshape(-1)
        detector_target_mask = (target_volume[:, t, r, c] > 0.5).astype(np.uint8)
        for shot_idx in range(target_volume.shape[0]):
            if known[shot_idx] < 0.5 or needs_edit[shot_idx] < 0.5:
                continue
            idx = tuple(np.flatnonzero(detector_target_mask[shot_idx]).tolist())
            if not idx:
                continue
            counter[idx] = counter.get(idx, 0) + 1
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[: max_classes - 1]
    detector_index_lists: list[tuple[int, ...]] = [()]
    counts: list[int] = [0]
    if detector_count is None:
        detector_count = 0
    for indices, count in ordered:
        detector_index_lists.append(tuple(int(x) for x in indices))
        counts.append(int(count))
    mask_table = np.zeros((len(detector_index_lists), detector_count), dtype=np.uint8)
    for class_idx, indices in enumerate(detector_index_lists):
        if indices:
            mask_table[class_idx, list(indices)] = np.uint8(1)
    return MotifVocabulary(
        mask_table=np.ascontiguousarray(mask_table),
        detector_index_lists=tuple(detector_index_lists),
        counts=tuple(counts),
        detector_count=int(detector_count),
    )


def _build_local_motif_vocabulary(
    bundles: list[tuple[PreparedEditFamily, dict[str, np.ndarray]]],
    *,
    max_classes: int,
) -> LocalMotifVocabulary:
    if max_classes < 2:
        raise ValueError("local motif vocabulary requires at least 2 classes including identity")
    counter: dict[tuple[tuple[int, int, int], ...], int] = {}
    detector_count: int | None = None
    for entry, subset in bundles:
        num_detectors = int(subset["detector_events"].shape[1])
        detector_count = num_detectors if detector_count is None else detector_count
        if detector_count != num_detectors:
            raise ValueError("All local-motif training subsets must share detector count")
        t = np.asarray(entry.detector_time_index, dtype=np.intp)
        r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
        c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
        target_volume = np.asarray(subset["edit_target_volume"], dtype=np.float32)
        needs_edit = np.asarray(subset["needs_edit"], dtype=np.float32).reshape(-1)
        known = np.asarray(subset["edit_target_known"], dtype=np.float32).reshape(-1)
        detector_target_mask = (target_volume[:, t, r, c] > 0.5).astype(np.uint8)
        for shot_idx in range(target_volume.shape[0]):
            if known[shot_idx] < 0.5 or needs_edit[shot_idx] < 0.5:
                continue
            indices = np.flatnonzero(detector_target_mask[shot_idx])
            if indices.size == 0:
                continue
            coords = sorted((int(t[idx]), int(r[idx]), int(c[idx])) for idx in indices.tolist())
            anchor_t, anchor_r, anchor_c = coords[0]
            pattern = tuple(
                sorted(
                    (
                        int(coord_t - anchor_t),
                        int(coord_r - anchor_r),
                        int(coord_c - anchor_c),
                    )
                    for coord_t, coord_r, coord_c in coords
                )
            )
            if not pattern:
                continue
            counter[pattern] = counter.get(pattern, 0) + 1
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[: max_classes - 1]
    if detector_count is None:
        detector_count = 0
    return LocalMotifVocabulary(
        offset_patterns=tuple(tuple(tuple(int(v) for v in offset) for offset in pattern) for pattern, _count in ordered),
        counts=tuple(int(count) for _pattern, count in ordered),
        detector_count=int(detector_count),
    )


def _build_local_motif_placements(
    *,
    entry: PreparedEditFamily,
    vocabulary: LocalMotifVocabulary,
) -> LocalMotifPlacements:
    detector_count = int(entry.detector_events.shape[1])
    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    coord_to_detector: dict[tuple[int, int, int], int] = {
        (int(t[idx]), int(r[idx]), int(c[idx])): int(idx)
        for idx in range(detector_count)
    }
    anchor_coords = sorted(coord_to_detector)
    seen: dict[tuple[int, ...], tuple[int, int]] = {(): (-1, 0)}
    for pattern_idx, pattern in enumerate(vocabulary.offset_patterns):
        pattern_count = int(vocabulary.counts[pattern_idx])
        for anchor_t, anchor_r, anchor_c in anchor_coords:
            detectors: list[int] = []
            valid = True
            for dt, dr, dc in pattern:
                detector_idx = coord_to_detector.get((
                    int(anchor_t + dt),
                    int(anchor_r + dr),
                    int(anchor_c + dc),
                ))
                if detector_idx is None:
                    valid = False
                    break
                detectors.append(int(detector_idx))
            if not valid:
                continue
            key = tuple(sorted(detectors))
            if not key or len(key) != len(detectors) or key in seen:
                continue
            seen[key] = (int(pattern_idx), int(pattern_count))
    detector_index_lists = tuple(seen.keys())
    mask_table = np.zeros((len(detector_index_lists), detector_count), dtype=np.uint8)
    pattern_indices = np.zeros((len(detector_index_lists),), dtype=np.int16)
    counts = np.zeros((len(detector_index_lists),), dtype=np.int32)
    for row_idx, key in enumerate(detector_index_lists):
        pattern_idx, pattern_count = seen[key]
        pattern_indices[row_idx] = np.int16(pattern_idx)
        counts[row_idx] = np.int32(pattern_count)
        if key:
            mask_table[row_idx, list(key)] = np.uint8(1)
    return LocalMotifPlacements(
        mask_table=np.ascontiguousarray(mask_table),
        detector_index_lists=tuple(tuple(int(x) for x in row) for row in detector_index_lists),
        pattern_indices=np.ascontiguousarray(pattern_indices),
        counts=np.ascontiguousarray(counts),
        detector_count=int(detector_count),
        num_patterns=int(len(vocabulary.offset_patterns)),
    )


def _build_motif_target_bundle(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    vocabulary: MotifVocabulary,
    hard_shot_weight: float,
) -> MotifTargetBundle:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    shot_features = np.asarray(outputs["pooled_features"], dtype=np.float32)
    labels = np.full((shot_features.shape[0],), -1, dtype=np.int64)
    shot_weights = np.ones((shot_features.shape[0],), dtype=np.float32)
    active_mask = np.zeros((shot_features.shape[0],), dtype=np.float32)

    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    target_volume = np.asarray(subset["edit_target_volume"], dtype=np.float32)
    detector_target_mask = np.zeros((target_volume.shape[0], vocabulary.detector_count), dtype=np.uint8)
    detector_target_mask[:, :] = (target_volume[:, t, r, c] > 0.5).astype(np.uint8)
    vocab_lookup = {indices: class_idx for class_idx, indices in enumerate(vocabulary.detector_index_lists)}
    hard_count = 0
    active_count = 0
    for shot_idx in range(shot_features.shape[0]):
        needs = float(subset["needs_edit"][shot_idx])
        known = float(subset["edit_target_known"][shot_idx])
        target_indices = tuple(np.flatnonzero(detector_target_mask[shot_idx]).tolist())
        if needs < 0.5:
            labels[shot_idx] = 0
            active_mask[shot_idx] = 1.0
            active_count += 1
            continue
        if known < 0.5:
            continue
        label = vocab_lookup.get(target_indices)
        if label is None:
            continue
        labels[shot_idx] = int(label)
        active_mask[shot_idx] = 1.0
        active_count += 1
        if label > 0:
            shot_weights[shot_idx] = float(max(hard_shot_weight, 1.0))
            hard_count += 1
    return MotifTargetBundle(
        shot_features=np.ascontiguousarray(shot_features),
        labels=np.ascontiguousarray(labels),
        shot_weights=np.ascontiguousarray(shot_weights),
        active_mask=np.ascontiguousarray(active_mask),
        hard_shot_fraction=(float(hard_count / active_count) if active_count else None),
    )


def _build_action_motif_supervision_arrays(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    vocabulary: MotifVocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full((int(subset["x"].shape[0]),), -1, dtype=np.int64)
    active_mask = np.zeros((int(subset["x"].shape[0]),), dtype=np.float32)
    if int(vocabulary.mask_table.shape[0]) <= 1:
        return labels, active_mask
    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    target_volume = np.asarray(subset["edit_target_volume"], dtype=np.float32)
    needs_edit = np.asarray(subset["needs_edit"], dtype=np.float32).reshape(-1)
    known = np.asarray(subset["edit_target_known"], dtype=np.float32).reshape(-1)
    detector_target_mask = (target_volume[:, t, r, c] > 0.5).astype(np.uint8)
    vocab_lookup = {indices: class_idx for class_idx, indices in enumerate(vocabulary.detector_index_lists)}
    for shot_idx in range(target_volume.shape[0]):
        if known[shot_idx] < 0.5:
            continue
        if needs_edit[shot_idx] < 0.5:
            labels[shot_idx] = 0
            active_mask[shot_idx] = 1.0
            continue
        target_indices = tuple(np.flatnonzero(detector_target_mask[shot_idx]).tolist())
        label = vocab_lookup.get(target_indices)
        if label is None:
            continue
        labels[shot_idx] = int(label)
        active_mask[shot_idx] = 1.0
    return np.ascontiguousarray(labels), np.ascontiguousarray(active_mask)


def _train_motif_head_epoch(
    *,
    motif_head: nn.Module,
    bundle: MotifTargetBundle,
    optimizer: Any,
    device: torch.device,
) -> dict[str, float]:
    motif_head.train()
    active = np.flatnonzero(bundle.active_mask >= 0.5)
    if active.size == 0:
        return {
            "motif_loss": None,
            "motif_accuracy": None,
            "motif_hard_shot_fraction": bundle.hard_shot_fraction,
        }
    order = np.random.permutation(active)
    loss_sum = 0.0
    correct = 0
    count = 0
    weight_sum = 0.0
    for shot_idx in order.tolist():
        features = torch.from_numpy(
            np.ascontiguousarray(bundle.shot_features[shot_idx : shot_idx + 1], dtype=np.float32)
        ).to(device)
        label = torch.tensor([int(bundle.labels[shot_idx])], dtype=torch.int64, device=device)
        weight = float(bundle.shot_weights[shot_idx])
        optimizer.zero_grad()
        logits = motif_head(features)
        loss = F.cross_entropy(logits, label) * weight
        loss.backward()
        optimizer.step()
        pred = int(torch.argmax(logits, dim=1).item())
        correct += int(pred == int(bundle.labels[shot_idx]))
        count += 1
        loss_sum += float(loss.item())
        weight_sum += weight
    return {
        "motif_loss": float(loss_sum / weight_sum) if weight_sum > 0.0 else None,
        "motif_accuracy": float(correct / count) if count else None,
        "motif_hard_shot_fraction": bundle.hard_shot_fraction,
    }


def _predict_motif_labels(
    *,
    motif_head: nn.Module,
    shot_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if shot_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    dataset = common.TensorDataset(torch.from_numpy(np.ascontiguousarray(shot_features, dtype=np.float32)))
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    pred_chunks: list[np.ndarray] = []
    motif_head.eval()
    with torch.no_grad():
        for (feat,) in loader:
            feat = feat.to(device)
            logits = motif_head(feat)
            pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.asarray(np.concatenate(pred_chunks, axis=0), dtype=np.int64)


def _motif_system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    motif_head: nn.Module,
    vocabulary: MotifVocabulary,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    shot_features = np.asarray(outputs["pooled_features"], dtype=np.float32)
    pred_labels = _predict_motif_labels(
        motif_head=motif_head,
        shot_features=shot_features,
        batch_size=batch_size,
        device=device,
    )
    chosen_edit_mask = np.asarray(vocabulary.mask_table[pred_labels], dtype=np.uint8)
    predicted_edit_weight = np.asarray(chosen_edit_mask.sum(axis=1), dtype=np.int16)
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=chosen_edit_mask,
        decision_payload={
            "selection_mode": SELECTION_MODE_MOTIF_VOCAB,
            "motif_num_classes": int(vocabulary.mask_table.shape[0]),
        },
        needs_metrics={
            "num_examples": int(pred_labels.shape[0]),
            "motif_predicted_identity_fraction": float(np.mean(pred_labels == 0)) if pred_labels.size else None,
        },
        edit_metrics={
            "motif_vocab_num_classes": int(vocabulary.mask_table.shape[0]),
            "motif_vocab_nonzero_classes": int(max(vocabulary.mask_table.shape[0] - 1, 0)),
        },
        extra_change_summary={
            "motif_fraction_with_any_selected_edit": (
                float(np.mean(predicted_edit_weight > 0)) if predicted_edit_weight.size else None
            ),
            "motif_mean_selected_edit_weight": (
                float(predicted_edit_weight.mean()) if predicted_edit_weight.size else None
            ),
        },
    )


def _logsigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return -np.logaddexp(0.0, -x)


def _action_motif_scores_from_logits(
    *,
    edit_logits_volume: np.ndarray,
    needs_edit_logits: np.ndarray,
    entry: PreparedEditFamily,
    vocabulary: MotifVocabulary,
) -> np.ndarray:
    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    detector_logits = np.asarray(edit_logits_volume, dtype=np.float32)[:, t, r, c]
    needs_logits = np.asarray(needs_edit_logits, dtype=np.float32).reshape(-1)
    motif_masks = np.asarray(vocabulary.mask_table, dtype=np.float32)
    bit_logprob = (
        motif_masks[None, :, :] * _logsigmoid_np(detector_logits[:, None, :])
        + (1.0 - motif_masks[None, :, :]) * _logsigmoid_np(-detector_logits[:, None, :])
    ).sum(axis=2)
    nonzero_mask = (motif_masks.sum(axis=1) > 0).astype(np.float32)
    needs_logprob = (
        nonzero_mask[None, :] * _logsigmoid_np(needs_logits)[:, None]
        + (1.0 - nonzero_mask[None, :]) * _logsigmoid_np(-needs_logits)[:, None]
    )
    return np.asarray(bit_logprob + needs_logprob, dtype=np.float32)


def _action_motif_system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    vocabulary: MotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin: float,
) -> dict[str, Any]:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    edit_logits = outputs["edit_logits"]
    needs_edit_logits = outputs["needs_edit_logits"]
    edit_probs = common._sigmoid_np(edit_logits)
    needs_edit_probs = common._sigmoid_np(needs_edit_logits)
    action_scores = _action_motif_scores_from_logits(
        edit_logits_volume=edit_logits,
        needs_edit_logits=needs_edit_logits,
        entry=entry,
        vocabulary=vocabulary,
    )
    if action_scores.shape[1] > 1:
        nonzero_scores = action_scores[:, 1:]
        best_nonzero_offset = np.asarray(np.argmax(nonzero_scores, axis=1), dtype=np.int64)
        best_nonzero_labels = best_nonzero_offset + 1
        best_nonzero_scores = nonzero_scores[np.arange(nonzero_scores.shape[0]), best_nonzero_offset]
        identity_scores = action_scores[:, 0]
        best_nonzero_margin = best_nonzero_scores - identity_scores
        pred_labels = np.where(
            best_nonzero_margin >= float(emit_margin),
            best_nonzero_labels,
            0,
        ).astype(np.int64, copy=False)
    else:
        identity_scores = action_scores[:, 0] if action_scores.size else np.zeros((0,), dtype=np.float32)
        best_nonzero_margin = np.zeros_like(identity_scores, dtype=np.float32)
        pred_labels = np.zeros((action_scores.shape[0],), dtype=np.int64)
    chosen_edit_mask = np.asarray(vocabulary.mask_table[pred_labels], dtype=np.uint8)
    predicted_edit_weight = np.asarray(chosen_edit_mask.sum(axis=1), dtype=np.int16)
    selected_scores = action_scores[np.arange(action_scores.shape[0]), pred_labels] if action_scores.size else np.zeros((0,), dtype=np.float32)
    needs_metrics = _binary_metrics_from_probs(
        needs_edit_probs,
        subset["needs_edit"],
        threshold=0.5,
    )
    edit_metrics = _masked_edit_binary_metrics(
        edit_probs,
        subset["edit_target_volume"],
        valid_mask_volume=entry.valid_mask_volume,
        known_mask=subset["edit_target_known"],
        threshold=0.5,
    )
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=chosen_edit_mask,
        decision_payload={
            "selection_mode": SELECTION_MODE_ACTION_MOTIF,
            "action_motif_num_classes": int(vocabulary.mask_table.shape[0]),
            "action_motif_emit_margin": float(emit_margin),
        },
        needs_metrics=needs_metrics,
        edit_metrics=edit_metrics,
        extra_change_summary={
            "action_motif_vocab_num_classes": int(vocabulary.mask_table.shape[0]),
            "action_motif_predicted_identity_fraction": (
                float(np.mean(pred_labels == 0)) if pred_labels.size else None
            ),
            "action_motif_fraction_with_any_selected_edit": (
                float(np.mean(predicted_edit_weight > 0)) if predicted_edit_weight.size else None
            ),
            "action_motif_mean_selected_edit_weight": (
                float(predicted_edit_weight.mean()) if predicted_edit_weight.size else None
            ),
            "action_motif_mean_selected_score_margin_vs_identity": (
                float(np.mean(selected_scores - identity_scores)) if selected_scores.size else None
            ),
            "action_motif_mean_best_nonzero_margin_vs_identity": (
                float(np.mean(best_nonzero_margin)) if best_nonzero_margin.size else None
            ),
        },
    )


def _local_motif_scores_from_logits(
    *,
    edit_logits_volume: np.ndarray,
    needs_edit_logits: np.ndarray,
    entry: PreparedEditFamily,
    placements: LocalMotifPlacements,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(entry.detector_time_index, dtype=np.intp)
    r = np.asarray(entry.row_index_by_detector, dtype=np.intp)
    c = np.asarray(entry.col_index_by_detector, dtype=np.intp)
    detector_logits = np.asarray(edit_logits_volume, dtype=np.float32)[:, t, r, c]
    needs_logits = np.asarray(needs_edit_logits, dtype=np.float32).reshape(-1)
    num_shots = int(detector_logits.shape[0])
    num_actions = int(placements.mask_table.shape[0])
    scores = np.full((num_shots, num_actions), -np.inf, dtype=np.float32)
    min_bit_logits = np.full((num_shots, num_actions), np.inf, dtype=np.float32)
    if num_actions == 0:
        return scores, min_bit_logits
    scores[:, 0] = _logsigmoid_np(-needs_logits)
    nonzero_base = _logsigmoid_np(needs_logits)
    for action_idx, detector_indices in enumerate(placements.detector_index_lists):
        if action_idx == 0 or not detector_indices:
            continue
        selected_logits = detector_logits[:, list(detector_indices)]
        scores[:, action_idx] = nonzero_base + selected_logits.sum(axis=1)
        min_bit_logits[:, action_idx] = selected_logits.min(axis=1)
    return np.asarray(scores, dtype=np.float32), np.asarray(min_bit_logits, dtype=np.float32)


def _local_motif_system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    vocabulary: LocalMotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin: float,
    min_bit_logit: float,
) -> dict[str, Any]:
    placements = _build_local_motif_placements(entry=entry, vocabulary=vocabulary)
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    edit_logits = outputs["edit_logits"]
    needs_edit_logits = outputs["needs_edit_logits"]
    edit_probs = common._sigmoid_np(edit_logits)
    needs_edit_probs = common._sigmoid_np(needs_edit_logits)
    local_scores, min_bit_logits = _local_motif_scores_from_logits(
        edit_logits_volume=edit_logits,
        needs_edit_logits=needs_edit_logits,
        entry=entry,
        placements=placements,
    )
    if local_scores.shape[1] > 1:
        nonzero_min_bit_logits = min_bit_logits[:, 1:]
        valid_nonzero = nonzero_min_bit_logits >= float(min_bit_logit)
        nonzero_scores = np.where(valid_nonzero, local_scores[:, 1:], -np.inf)
        best_nonzero_offset = np.asarray(np.argmax(nonzero_scores, axis=1), dtype=np.int64)
        best_nonzero_labels = best_nonzero_offset + 1
        best_nonzero_scores = nonzero_scores[np.arange(nonzero_scores.shape[0]), best_nonzero_offset]
        best_nonzero_valid = np.isfinite(best_nonzero_scores)
        best_nonzero_min_bit_logits = nonzero_min_bit_logits[
            np.arange(nonzero_min_bit_logits.shape[0]),
            best_nonzero_offset,
        ]
        identity_scores = local_scores[:, 0]
        best_nonzero_margin = best_nonzero_scores - identity_scores
        pred_labels = np.where(
            best_nonzero_valid & (best_nonzero_margin >= float(emit_margin)),
            best_nonzero_labels,
            0,
        ).astype(np.int64, copy=False)
    else:
        identity_scores = local_scores[:, 0] if local_scores.size else np.zeros((0,), dtype=np.float32)
        best_nonzero_margin = np.zeros_like(identity_scores, dtype=np.float32)
        best_nonzero_valid = np.zeros_like(identity_scores, dtype=bool)
        best_nonzero_min_bit_logits = np.zeros_like(identity_scores, dtype=np.float32)
        pred_labels = np.zeros((local_scores.shape[0],), dtype=np.int64)
    chosen_edit_mask = np.asarray(placements.mask_table[pred_labels], dtype=np.uint8)
    predicted_edit_weight = np.asarray(chosen_edit_mask.sum(axis=1), dtype=np.int16)
    selected_scores = (
        local_scores[np.arange(local_scores.shape[0]), pred_labels]
        if local_scores.size
        else np.zeros((0,), dtype=np.float32)
    )
    needs_metrics = _binary_metrics_from_probs(
        needs_edit_probs,
        subset["needs_edit"],
        threshold=0.5,
    )
    edit_metrics = _masked_edit_binary_metrics(
        edit_probs,
        subset["edit_target_volume"],
        valid_mask_volume=entry.valid_mask_volume,
        known_mask=subset["edit_target_known"],
        threshold=0.5,
    )
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=chosen_edit_mask,
        decision_payload={
            "selection_mode": SELECTION_MODE_LOCAL_MOTIF,
            "local_motif_num_patterns": int(len(vocabulary.offset_patterns)),
            "local_motif_num_actions": int(placements.mask_table.shape[0]),
            "local_motif_emit_margin": float(emit_margin),
            "local_motif_min_bit_logit": float(min_bit_logit),
        },
        needs_metrics=needs_metrics,
        edit_metrics=edit_metrics,
        extra_change_summary={
            "local_motif_predicted_identity_fraction": (
                float(np.mean(pred_labels == 0)) if pred_labels.size else None
            ),
            "local_motif_fraction_with_any_selected_edit": (
                float(np.mean(predicted_edit_weight > 0)) if predicted_edit_weight.size else None
            ),
            "local_motif_mean_selected_edit_weight": (
                float(predicted_edit_weight.mean()) if predicted_edit_weight.size else None
            ),
            "local_motif_mean_selected_score_margin_vs_identity": (
                float(np.mean(selected_scores - identity_scores)) if selected_scores.size else None
            ),
            "local_motif_fraction_with_valid_nonzero_candidate": (
                float(np.mean(best_nonzero_valid)) if best_nonzero_valid.size else None
            ),
            "local_motif_mean_best_nonzero_margin_vs_identity": (
                float(np.mean(best_nonzero_margin[best_nonzero_valid]))
                if best_nonzero_margin.size and np.any(best_nonzero_valid)
                else None
            ),
            "local_motif_mean_best_nonzero_min_bit_logit": (
                float(np.mean(best_nonzero_min_bit_logits[best_nonzero_valid]))
                if best_nonzero_min_bit_logits.size and np.any(best_nonzero_valid)
                else None
            ),
            "local_motif_num_actions": int(placements.mask_table.shape[0]),
        },
    )


def _system_policy_score(result: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    edited = result["edited_pymatching"]
    change = result["change_summary"]
    return (
        float(edited.get("accuracy") or 0.0),
        float(edited.get("macro_f1") or 0.0),
        float(change.get("num_improved_over_baseline") or 0.0)
        - float(change.get("num_harmed_vs_baseline") or 0.0),
        -float(change.get("num_harmed_vs_baseline") or 0.0),
        -float(change.get("mean_predicted_edit_weight") or 0.0),
        -float(change.get("fraction_with_any_predicted_edit") or 0.0),
    )


def _grid_search_action_motif_policy(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    vocabulary: MotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin_grid: list[float],
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    for emit_margin in emit_margin_grid:
        result = _action_motif_system_metrics_for_subset(
            entry=entry,
            subset=subset,
            model=model,
            vocabulary=vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin=float(emit_margin),
        )
        score = _system_policy_score(result)
        if best_key is None or score > best_key:
            best_key = score
            best_result = result
    if best_result is None:
        raise RuntimeError("Action motif grid search produced no candidates")
    return best_result


def _grid_search_action_motif_policy_by_family(
    *,
    entries: list[PreparedEditFamily],
    split_by_family: dict[str, SplitBundle],
    model: nn.Module,
    vocabulary: MotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin_grid: list[float],
) -> dict[str, Any]:
    best_by_family: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for emit_margin in emit_margin_grid:
        by_family = {
            entry.family: _action_motif_system_metrics_for_subset(
                entry=entry,
                subset=_subset_family(entry, split_by_family[entry.family].val),
                model=model,
                vocabulary=vocabulary,
                batch_size=batch_size,
                device=device,
                emit_margin=float(emit_margin),
            )
            for entry in entries
        }
        mean_metric = _mean_system_metric(list(by_family.values()))
        mean_harmed = float(
            np.mean([
                float(result["change_summary"].get("num_harmed_vs_baseline") or 0.0)
                for result in by_family.values()
            ])
        )
        mean_weight = float(
            np.mean([
                float(result["change_summary"].get("mean_predicted_edit_weight") or 0.0)
                for result in by_family.values()
            ])
        )
        key = (float(mean_metric), -mean_harmed, -mean_weight)
        if best_key is None or key > best_key:
            best_key = key
            best_by_family = by_family
    if best_by_family is None:
        raise RuntimeError("Action motif grid search produced no family results")
    return best_by_family


def _grid_search_local_motif_policy(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    vocabulary: LocalMotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin_grid: list[float],
    min_bit_logit_grid: list[float],
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    for emit_margin in emit_margin_grid:
        for min_bit_logit in min_bit_logit_grid:
            result = _local_motif_system_metrics_for_subset(
                entry=entry,
                subset=subset,
                model=model,
                vocabulary=vocabulary,
                batch_size=batch_size,
                device=device,
                emit_margin=float(emit_margin),
                min_bit_logit=float(min_bit_logit),
            )
            score = _system_policy_score(result)
            if best_key is None or score > best_key:
                best_key = score
                best_result = result
    if best_result is None:
        raise RuntimeError("Local motif grid search produced no candidates")
    return best_result


def _grid_search_local_motif_policy_by_family(
    *,
    entries: list[PreparedEditFamily],
    split_by_family: dict[str, SplitBundle],
    model: nn.Module,
    vocabulary: LocalMotifVocabulary,
    batch_size: int,
    device: torch.device,
    emit_margin_grid: list[float],
    min_bit_logit_grid: list[float],
) -> dict[str, Any]:
    best_by_family: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for emit_margin in emit_margin_grid:
        for min_bit_logit in min_bit_logit_grid:
            by_family = {
                entry.family: _local_motif_system_metrics_for_subset(
                    entry=entry,
                    subset=_subset_family(entry, split_by_family[entry.family].val),
                    model=model,
                    vocabulary=vocabulary,
                    batch_size=batch_size,
                    device=device,
                    emit_margin=float(emit_margin),
                    min_bit_logit=float(min_bit_logit),
                )
                for entry in entries
            }
            mean_metric = _mean_system_metric(list(by_family.values()))
            mean_harmed = float(
                np.mean([
                    float(result["change_summary"].get("num_harmed_vs_baseline") or 0.0)
                    for result in by_family.values()
                ])
            )
            mean_weight = float(
                np.mean([
                    float(result["change_summary"].get("mean_predicted_edit_weight") or 0.0)
                    for result in by_family.values()
                ])
            )
            key = (float(mean_metric), -mean_harmed, -mean_weight)
            if best_key is None or key > best_key:
                best_key = key
                best_by_family = by_family
    if best_by_family is None:
        raise RuntimeError("Local motif grid search produced no family results")
    return best_by_family


def _train_motif_vocabulary_head(
    *,
    model: nn.Module,
    train_bundles: list[tuple[PreparedEditFamily, dict[str, np.ndarray]]],
    val_bundles_by_family: dict[str, tuple[PreparedEditFamily, dict[str, np.ndarray]]],
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    max_classes: int,
    lr: float,
    epochs: int,
    hard_shot_weight: float,
) -> dict[str, Any] | None:
    vocabulary = _build_motif_vocabulary(train_bundles, max_classes=max_classes)
    if vocabulary.mask_table.shape[0] <= 1:
        return None
    train_target_bundles = [
        _build_motif_target_bundle(
            entry=entry,
            subset=subset,
            model=model,
            batch_size=batch_size,
            device=device,
            vocabulary=vocabulary,
            hard_shot_weight=hard_shot_weight,
        )
        for entry, subset in train_bundles
    ]
    if not any(np.any(bundle.active_mask >= 0.5) for bundle in train_target_bundles):
        return None
    shot_feature_dim = int(train_target_bundles[0].shot_features.shape[1])
    motif_kwargs = {
        "shot_feature_dim": shot_feature_dim,
        "hidden_dim": int(hidden_dim),
        "num_classes": int(vocabulary.mask_table.shape[0]),
        "dropout": float(dropout),
    }
    motif_head = MotifVocabularyHead(**motif_kwargs).to(device)
    optimizer = optim.Adam(motif_head.parameters(), lr=lr)
    best_state: dict[str, Any] | None = None
    best_val_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics_per_bundle = [
            _train_motif_head_epoch(
                motif_head=motif_head,
                bundle=bundle,
                optimizer=optimizer,
                device=device,
            )
            for bundle in train_target_bundles
        ]
        train_loss_values = [x["motif_loss"] for x in train_metrics_per_bundle if x["motif_loss"] is not None]
        train_acc_values = [x["motif_accuracy"] for x in train_metrics_per_bundle if x["motif_accuracy"] is not None]
        val_by_family: dict[str, Any] = {}
        motif_eval_by_family: dict[str, Any] = {}
        for family, (entry, subset) in val_bundles_by_family.items():
            metrics = _motif_system_metrics_for_subset(
                entry=entry,
                subset=subset,
                model=model,
                motif_head=motif_head,
                vocabulary=vocabulary,
                batch_size=batch_size,
                device=device,
            )
            motif_eval_by_family[family] = metrics
            val_by_family[family] = {
                "accuracy": metrics["edited_pymatching"].get("accuracy"),
                "macro_f1": metrics["edited_pymatching"].get("macro_f1"),
                "motif_fraction_with_any_selected_edit": metrics["change_summary"].get("motif_fraction_with_any_selected_edit"),
            }
        val_metric = _mean_system_metric(list(motif_eval_by_family.values()))
        epoch_history.append(
            {
                "epoch": epoch,
                "train": {
                    "motif_loss": (float(np.mean(train_loss_values)) if train_loss_values else None),
                    "motif_accuracy": (float(np.mean(train_acc_values)) if train_acc_values else None),
                },
                "val_by_family": val_by_family,
                "val_selection_metric": float(val_metric),
            }
        )
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(motif_head.state_dict()),
                "val_by_family": copy.deepcopy(val_by_family),
                "eval_by_family": copy.deepcopy(motif_eval_by_family),
            }
    if best_state is None:
        return None
    motif_head.load_state_dict(best_state["model_state"])
    return {
        "motif_head": motif_head,
        "motif_kwargs": motif_kwargs,
        "vocabulary": vocabulary,
        "epoch_history": epoch_history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_selection_metric": float(best_val_metric),
        "best_val_by_family": best_state["val_by_family"],
        "eval_by_family": best_state["eval_by_family"],
        "hard_shot_weight": float(hard_shot_weight),
    }


def _selector_logits_for_bundle(
    *,
    selector: nn.Module,
    bundle: SelectorCandidateBundle,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if bundle.shot_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    dataset = common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(bundle.shot_features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(bundle.candidate_features, dtype=np.float32)),
    )
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    selector.eval()
    with torch.no_grad():
        for shot_feat, cand_feat in loader:
            shot_feat = shot_feat.to(device)
            cand_feat = cand_feat.to(device)
            logits_chunks.append(selector(shot_feat, cand_feat).detach().cpu().numpy())
    return np.asarray(np.concatenate(logits_chunks, axis=0), dtype=np.float32)


def _router_logits_for_shot_features(
    *,
    router: nn.Module,
    shot_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if shot_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    dataset = common.TensorDataset(torch.from_numpy(np.ascontiguousarray(shot_features, dtype=np.float32)))
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    router.eval()
    with torch.no_grad():
        for (shot_feat,) in loader:
            shot_feat = shot_feat.to(device)
            logits_chunks.append(router(shot_feat).detach().cpu().numpy())
    return np.asarray(np.concatenate(logits_chunks, axis=0), dtype=np.float32)


def _router_target_from_group(
    *,
    identity_score: float,
    best_nonzero_score: float,
    label_mode: str,
) -> float:
    if label_mode == ROUTER_LABEL_IDENTITY_VS_NONZERO:
        return 1.0 if float(best_nonzero_score) > float(identity_score) else 0.0
    if label_mode == ROUTER_LABEL_BASELINE_FAILURE:
        return 1.0 if float(identity_score) < 0.5 else 0.0
    if label_mode == ROUTER_LABEL_ORACLE_SOLVABLE:
        return 1.0 if float(identity_score) < 0.5 and float(best_nonzero_score) >= 0.5 else 0.0
    raise ValueError(f"Unsupported router label mode: {label_mode!r}")


def _build_router_training_arrays(
    bundles: list[SelectorCandidateBundle],
    *,
    label_mode: str = ROUTER_LABEL_IDENTITY_VS_NONZERO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shot_feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    candidate_oracle_rows: list[float] = []
    for bundle in bundles:
        router_features = _router_feature_rows_from_bundle(bundle)
        for group_idx, group_slice in enumerate(_selector_group_slices(bundle)):
            group_features = np.asarray(bundle.shot_features[group_slice], dtype=np.float32)
            if group_features.shape[0] == 0:
                continue
            target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            candidate_correct = np.asarray(bundle.candidate_is_correct[group_slice], dtype=np.uint8)
            candidate_weights = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
            identity_rows = np.flatnonzero(candidate_weights == 0)
            nonzero_rows = np.flatnonzero(candidate_weights > 0)
            identity_score = float(target_scores[int(identity_rows[0])]) if identity_rows.size else 0.0
            best_nonzero_score = float(np.max(target_scores[nonzero_rows])) if nonzero_rows.size else 0.0
            identity_correct = bool(candidate_correct[int(identity_rows[0])]) if identity_rows.size else False
            nonzero_correct_exists = (
                bool(np.any(candidate_correct[nonzero_rows] > 0)) if nonzero_rows.size else False
            )
            shot_feature_rows.append(np.asarray(router_features[group_idx], dtype=np.float32))
            if str(label_mode) == ROUTER_LABEL_BASELINE_FAILURE:
                target_rows.append(0.0 if identity_correct else 1.0)
            elif str(label_mode) == ROUTER_LABEL_ORACLE_SOLVABLE:
                target_rows.append(1.0 if (not identity_correct and nonzero_correct_exists) else 0.0)
            else:
                target_rows.append(
                    _router_target_from_group(
                        identity_score=identity_score,
                        best_nonzero_score=best_nonzero_score,
                        label_mode=str(label_mode),
                    )
                )
            candidate_oracle_rows.append(float(max(identity_score, best_nonzero_score)))
    if not shot_feature_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.ascontiguousarray(np.stack(shot_feature_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.asarray(target_rows, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(candidate_oracle_rows, dtype=np.float32)),
    )


def _router_feature_rows_from_bundle(bundle: SelectorCandidateBundle) -> np.ndarray:
    rows: list[np.ndarray] = []
    for group_slice in _selector_group_slices(bundle):
        shot_features = np.asarray(bundle.shot_features[group_slice], dtype=np.float32)
        candidate_features = np.asarray(bundle.candidate_features[group_slice], dtype=np.float32)
        candidate_weights = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
        if shot_features.shape[0] == 0:
            continue
        nonzero = np.flatnonzero(candidate_weights > 0)
        if nonzero.size:
            nonzero_features = candidate_features[nonzero]
            nonzero_weights = candidate_weights[nonzero].astype(np.float32, copy=False)
            summary = np.asarray(
                [
                    float(candidate_features.shape[0]),
                    float(nonzero.size),
                    float(nonzero_features[:, 6].max()),
                    float(nonzero_features[:, 7].max()),
                    float(nonzero_features[:, 8].max()),
                    float(nonzero_features[:, 9].max()),
                    float(nonzero_weights.min()),
                    float(nonzero_weights.mean()),
                    float(nonzero_weights.max()),
                    float(nonzero_features[:, 12].max()),
                ],
                dtype=np.float32,
            )
        else:
            summary = np.asarray(
                [
                    float(candidate_features.shape[0]),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        rows.append(np.concatenate([np.asarray(shot_features[0], dtype=np.float32), summary], axis=0))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


def _shot_feature_rows_from_bundle(bundle: SelectorCandidateBundle) -> np.ndarray:
    rows: list[np.ndarray] = []
    for group_slice in _selector_group_slices(bundle):
        shot_features = np.asarray(bundle.shot_features[group_slice], dtype=np.float32)
        if shot_features.shape[0] == 0:
            continue
        rows.append(np.asarray(shot_features[0], dtype=np.float32))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


def _transition_prior_training_arrays(
    bundles: list[SelectorCandidateBundle],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for bundle in bundles:
        features = _shot_feature_rows_from_bundle(bundle)
        if features.shape[0] == 0:
            continue
        labels = np.asarray(bundle.target_transition_class, dtype=np.int64).reshape(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "Transition-prior feature/label mismatch: "
                f"features={features.shape[0]}, labels={labels.shape[0]}"
            )
        feature_rows.append(features)
        label_rows.append(labels)
    if not feature_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.ascontiguousarray(np.concatenate(feature_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(label_rows, axis=0), dtype=np.int64),
    )


def _train_transition_prior_one_epoch(
    *,
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    optimizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    head.train()
    order = np.random.permutation(int(labels.shape[0]))
    loss_sum = 0.0
    correct_sum = 0
    count = 0
    for start in range(0, int(order.shape[0]), int(batch_size)):
        idx = order[start : start + int(batch_size)]
        xb = torch.from_numpy(np.ascontiguousarray(features[idx], dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.ascontiguousarray(labels[idx], dtype=np.int64)).to(device)
        optimizer.zero_grad()
        logits = head(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        batch_count = int(yb.shape[0])
        loss_sum += float(loss.item()) * batch_count
        correct_sum += int((preds == yb).sum().item())
        count += batch_count
    return {
        "transition_prior_ce_loss": float(loss_sum / count) if count else None,
        "transition_prior_accuracy": float(correct_sum / count) if count else None,
    }


def _transition_prior_metrics(
    *,
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if features.shape[0] == 0:
        return {
            "transition_prior_ce_loss": None,
            "transition_prior_accuracy": None,
        }
    dataset = common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
    )
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    loss_sum = 0.0
    correct_sum = 0
    count = 0
    head.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            preds = torch.argmax(logits, dim=1)
            batch_count = int(yb.shape[0])
            loss_sum += float(loss.item()) * batch_count
            correct_sum += int((preds == yb).sum().item())
            count += batch_count
    return {
        "transition_prior_ce_loss": float(loss_sum / count) if count else None,
        "transition_prior_accuracy": float(correct_sum / count) if count else None,
    }


def _train_transition_prior_head(
    *,
    train_bundles: list[SelectorCandidateBundle],
    val_bundles_by_family: dict[str, SelectorCandidateBundle],
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    train_features, train_labels = _transition_prior_training_arrays(train_bundles)
    if train_features.shape[0] == 0:
        return None
    head_kwargs = {
        "shot_feature_dim": int(train_features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "num_classes": len(CLASS4_LABELS) ** 2,
        "dropout": float(dropout),
    }
    head = MotifVocabularyHead(**head_kwargs).to(device)
    optimizer = optim.Adam(head.parameters(), lr=float(lr))
    val_arrays_by_family = {
        family: _transition_prior_training_arrays([bundle])
        for family, bundle in val_bundles_by_family.items()
    }
    best_state: dict[str, Any] | None = None
    best_val_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _train_transition_prior_one_epoch(
            head=head,
            features=train_features,
            labels=train_labels,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
        )
        val_by_family = {
            family: _transition_prior_metrics(
                head=head,
                features=features,
                labels=labels,
                batch_size=batch_size,
                device=device,
            )
            for family, (features, labels) in val_arrays_by_family.items()
        }
        val_metric = (
            float(np.mean([
                float(metrics["transition_prior_accuracy"] or 0.0)
                for metrics in val_by_family.values()
            ]))
            if val_by_family
            else 0.0
        )
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train": train_metrics,
                "val_by_family": val_by_family,
                "val_selection_metric": float(val_metric),
            }
        )
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_state = {
                "epoch": int(epoch),
                "model_state": copy.deepcopy(head.state_dict()),
                "val_by_family": copy.deepcopy(val_by_family),
            }
    if best_state is None:
        return None
    head.load_state_dict(best_state["model_state"])
    return {
        "head": head,
        "head_kwargs": head_kwargs,
        "epoch_history": epoch_history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_selection_metric": float(best_val_metric),
        "best_val_by_family": best_state["val_by_family"],
        "lr": float(lr),
        "epochs": int(epochs),
    }


def _transition_prior_logits_for_bundle(
    *,
    head: nn.Module,
    bundle: SelectorCandidateBundle,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features = _shot_feature_rows_from_bundle(bundle)
    if features.shape[0] == 0:
        return np.zeros((bundle.num_shots, len(CLASS4_LABELS) ** 2), dtype=np.float32)
    dataset = common.TensorDataset(torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)))
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for (xb,) in loader:
            chunks.append(head(xb.to(device)).detach().cpu().numpy())
    logits = np.asarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    if logits.shape[0] != bundle.num_shots:
        raise ValueError(
            "Transition-prior logits must match bundle shot count. "
            f"Expected {bundle.num_shots}, got {logits.shape[0]}"
        )
    return logits


def _candidate_compatibility_training_arrays(
    bundles: list[SelectorCandidateBundle],
    *,
    balanced: bool = False,
    negative_ratio: float = DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO,
    no_positive_negative_count: int = DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shot_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for bundle in bundles:
        if not balanced:
            nonzero = np.flatnonzero(np.asarray(bundle.candidate_edit_weight, dtype=np.int16) > 0)
            if nonzero.size == 0:
                continue
            shot_rows.append(np.asarray(bundle.shot_features[nonzero], dtype=np.float32))
            candidate_rows.append(np.asarray(bundle.candidate_features[nonzero], dtype=np.float32))
            label_rows.append((np.asarray(bundle.target_scores[nonzero], dtype=np.float32) > 0.0).astype(np.float32))
            continue
        for group_slice in _selector_group_slices(bundle):
            rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
            if rows.size == 0:
                continue
            weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
            nonzero_local = np.flatnonzero(weights > 0)
            if nonzero_local.size == 0:
                continue
            nonzero_rows = rows[nonzero_local]
            labels = (np.asarray(bundle.target_scores[nonzero_rows], dtype=np.float32) > 0.0).astype(np.float32)
            positive_local = np.flatnonzero(labels >= 0.5)
            negative_local = np.flatnonzero(labels < 0.5)
            if positive_local.size:
                negative_count = min(
                    int(negative_local.size),
                    int(math.ceil(float(negative_ratio) * float(positive_local.size))),
                )
                selected_negative = (
                    np.random.choice(negative_local, size=negative_count, replace=False)
                    if negative_count > 0
                    else np.zeros((0,), dtype=np.int64)
                )
                selected_local = np.concatenate([positive_local, selected_negative], axis=0)
            else:
                negative_count = min(int(negative_local.size), max(int(no_positive_negative_count), 0))
                selected_local = (
                    np.random.choice(negative_local, size=negative_count, replace=False)
                    if negative_count > 0
                    else np.zeros((0,), dtype=np.int64)
                )
            if selected_local.size == 0:
                continue
            selected_rows = nonzero_rows[selected_local]
            shot_rows.append(np.asarray(bundle.shot_features[selected_rows], dtype=np.float32))
            candidate_rows.append(np.asarray(bundle.candidate_features[selected_rows], dtype=np.float32))
            label_rows.append(np.asarray(labels[selected_local], dtype=np.float32))
    if not shot_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.ascontiguousarray(np.concatenate(shot_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(candidate_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(label_rows, axis=0), dtype=np.float32),
    )


def _train_candidate_compatibility_one_epoch(
    *,
    head: nn.Module,
    shot_features: np.ndarray,
    candidate_features: np.ndarray,
    labels: np.ndarray,
    optimizer: Any,
    device: torch.device,
    batch_size: int,
    positive_weight: float | None = None,
) -> dict[str, float]:
    head.train()
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    order = np.random.permutation(int(labels_np.shape[0]))
    positive_count = float(np.sum(labels_np >= 0.5))
    negative_count = float(labels_np.shape[0] - positive_count)
    pos_weight = (
        float(negative_count / max(positive_count, 1.0))
        if positive_weight is None
        else float(positive_weight)
    )
    loss_sum = 0.0
    correct_sum = 0
    count = 0
    for start in range(0, int(order.shape[0]), int(batch_size)):
        idx = order[start : start + int(batch_size)]
        shot = torch.from_numpy(np.ascontiguousarray(shot_features[idx], dtype=np.float32)).to(device)
        cand = torch.from_numpy(np.ascontiguousarray(candidate_features[idx], dtype=np.float32)).to(device)
        y = torch.from_numpy(np.ascontiguousarray(labels_np[idx], dtype=np.float32)).to(device)
        optimizer.zero_grad()
        logits = head(shot, cand)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device),
        )
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
        batch_count = int(y.shape[0])
        loss_sum += float(loss.item()) * batch_count
        correct_sum += int((preds == y).sum().item())
        count += batch_count
    return {
        "candidate_compat_bce_loss": float(loss_sum / count) if count else None,
        "candidate_compat_accuracy": float(correct_sum / count) if count else None,
        "candidate_compat_positive_fraction": float(np.mean(labels_np >= 0.5)) if labels_np.size else None,
        "candidate_compat_pos_weight": float(pos_weight),
    }


def _train_candidate_compatibility_pairwise_epoch(
    *,
    head: nn.Module,
    bundles: list[SelectorCandidateBundle],
    optimizer: Any,
    device: torch.device,
) -> dict[str, float | None]:
    head.train()
    group_refs: list[tuple[SelectorCandidateBundle, slice]] = []
    for bundle in bundles:
        for group_slice in _selector_group_slices(bundle):
            rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
            if rows.size == 0:
                continue
            nonzero = rows[np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16) > 0]
            if nonzero.size == 0:
                continue
            positive = nonzero[np.asarray(bundle.target_scores[nonzero], dtype=np.float32) > 0.0]
            negative = nonzero[np.asarray(bundle.target_scores[nonzero], dtype=np.float32) <= 0.0]
            if positive.size and negative.size:
                group_refs.append((bundle, group_slice))
    if not group_refs:
        return {
            "candidate_compat_pairwise_loss": None,
            "candidate_compat_pairwise_groups": 0.0,
            "candidate_compat_pairwise_accuracy": None,
        }

    order = np.random.permutation(len(group_refs))
    loss_sum = 0.0
    pair_count_sum = 0.0
    correct_pair_sum = 0.0
    for order_idx in order.tolist():
        bundle, group_slice = group_refs[order_idx]
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        nonzero = rows[np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16) > 0]
        labels = np.asarray(bundle.target_scores[nonzero], dtype=np.float32) > 0.0
        positive = nonzero[labels]
        negative = nonzero[~labels]
        if positive.size == 0 or negative.size == 0:
            continue
        selected_rows = np.concatenate([positive, negative], axis=0)
        shot = torch.from_numpy(np.ascontiguousarray(bundle.shot_features[selected_rows], dtype=np.float32)).to(device)
        cand = torch.from_numpy(np.ascontiguousarray(bundle.candidate_features[selected_rows], dtype=np.float32)).to(device)
        optimizer.zero_grad()
        logits = head(shot, cand).reshape(-1)
        pos_logits = logits[: int(positive.size)]
        neg_logits = logits[int(positive.size) :]
        diffs = pos_logits[:, None] - neg_logits[None, :]
        loss = F.softplus(-diffs).mean()
        loss.backward()
        optimizer.step()
        pair_count = float(diffs.numel())
        loss_sum += float(loss.item()) * pair_count
        correct_pair_sum += float((diffs.detach() > 0.0).to(dtype=torch.float32).sum().item())
        pair_count_sum += pair_count
    return {
        "candidate_compat_pairwise_loss": (
            float(loss_sum / pair_count_sum) if pair_count_sum > 0.0 else None
        ),
        "candidate_compat_pairwise_groups": float(len(group_refs)),
        "candidate_compat_pairwise_accuracy": (
            float(correct_pair_sum / pair_count_sum) if pair_count_sum > 0.0 else None
        ),
    }


def _candidate_compatibility_metrics(
    *,
    head: nn.Module,
    shot_features: np.ndarray,
    candidate_features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | None]:
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    if labels_np.size == 0:
        return {
            "candidate_compat_accuracy": None,
            "candidate_compat_positive_fraction": None,
            "candidate_compat_predicted_positive_fraction": None,
        }
    logits = _selector_logits_for_arrays(
        selector=head,
        shot_features=shot_features,
        candidate_features=candidate_features,
        batch_size=batch_size,
        device=device,
    )
    probs = common._sigmoid_np(logits).reshape(-1)
    preds = probs >= 0.5
    labels_bool = labels_np >= 0.5
    return {
        "candidate_compat_accuracy": float(np.mean(preds == labels_bool)),
        "candidate_compat_positive_fraction": float(np.mean(labels_bool)),
        "candidate_compat_predicted_positive_fraction": float(np.mean(preds)),
    }


def _selector_logits_for_arrays(
    *,
    selector: nn.Module,
    shot_features: np.ndarray,
    candidate_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if shot_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    dataset = common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(shot_features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(candidate_features, dtype=np.float32)),
    )
    loader = common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    selector.eval()
    with torch.no_grad():
        for shot_feat, cand_feat in loader:
            chunks.append(selector(shot_feat.to(device), cand_feat.to(device)).detach().cpu().numpy())
    return np.asarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def _train_candidate_compatibility_head(
    *,
    train_bundles: list[SelectorCandidateBundle],
    val_bundles_by_family: dict[str, SelectorCandidateBundle],
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    objective: str = CANDIDATE_COMPAT_OBJECTIVE_BCE,
    negative_ratio: float = DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO,
    no_positive_negative_count: int = DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT,
) -> dict[str, Any] | None:
    objective_name = str(objective)
    if objective_name not in CANDIDATE_COMPAT_OBJECTIVE_CHOICES:
        raise ValueError(f"Unsupported candidate compatibility objective: {objective!r}")
    train_shot, train_candidate, train_labels = _candidate_compatibility_training_arrays(
        train_bundles,
        balanced=(objective_name == CANDIDATE_COMPAT_OBJECTIVE_GROUP_BALANCED),
        negative_ratio=float(negative_ratio),
        no_positive_negative_count=int(no_positive_negative_count),
    )
    if objective_name == CANDIDATE_COMPAT_OBJECTIVE_PAIRWISE_RANK:
        train_shot, train_candidate, train_labels = _candidate_compatibility_training_arrays(train_bundles)
    if train_shot.shape[0] == 0:
        return None
    head_kwargs = {
        "shot_feature_dim": int(train_shot.shape[1]),
        "candidate_feature_dim": int(train_candidate.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
    }
    head = CandidateEditSelector(**head_kwargs).to(device)
    optimizer = optim.Adam(head.parameters(), lr=float(lr))
    val_arrays = {
        family: _candidate_compatibility_training_arrays([bundle])
        for family, bundle in val_bundles_by_family.items()
    }
    best_state: dict[str, Any] | None = None
    best_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        if objective_name == CANDIDATE_COMPAT_OBJECTIVE_PAIRWISE_RANK:
            train_metrics = _train_candidate_compatibility_pairwise_epoch(
                head=head,
                bundles=train_bundles,
                optimizer=optimizer,
                device=device,
            )
        else:
            train_metrics = _train_candidate_compatibility_one_epoch(
                head=head,
                shot_features=train_shot,
                candidate_features=train_candidate,
                labels=train_labels,
                optimizer=optimizer,
                device=device,
                batch_size=batch_size,
                positive_weight=(1.0 if objective_name == CANDIDATE_COMPAT_OBJECTIVE_GROUP_BALANCED else None),
            )
        val_by_family = {
            family: _candidate_compatibility_metrics(
                head=head,
                shot_features=arrays[0],
                candidate_features=arrays[1],
                labels=arrays[2],
                device=device,
                batch_size=batch_size,
            )
            for family, arrays in val_arrays.items()
        }
        metric = (
            float(np.mean([
                float(metrics["candidate_compat_accuracy"] or 0.0)
                for metrics in val_by_family.values()
            ]))
            if val_by_family
            else 0.0
        )
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train": train_metrics,
                "val_by_family": val_by_family,
                "val_selection_metric": float(metric),
            }
        )
        if metric > best_metric:
            best_metric = float(metric)
            best_state = {
                "epoch": int(epoch),
                "head_state": copy.deepcopy(head.state_dict()),
                "val_by_family": copy.deepcopy(val_by_family),
            }
    if best_state is None:
        return None
    head.load_state_dict(best_state["head_state"])
    return {
        "head": head,
        "head_kwargs": head_kwargs,
        "epoch_history": epoch_history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_selection_metric": float(best_metric),
        "best_val_by_family": best_state["val_by_family"],
        "lr": float(lr),
        "epochs": int(epochs),
        "objective": str(objective_name),
        "negative_ratio": float(negative_ratio),
        "no_positive_negative_count": int(no_positive_negative_count),
    }


def _candidate_compatibility_logits_for_bundle(
    *,
    head: nn.Module,
    bundle: SelectorCandidateBundle,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return _selector_logits_for_arrays(
        selector=head,
        shot_features=bundle.shot_features,
        candidate_features=bundle.candidate_features,
        batch_size=batch_size,
        device=device,
    )


def _apply_transition_prior_to_selector_logits(
    *,
    bundle: SelectorCandidateBundle,
    selector_logits: np.ndarray,
    transition_prior_logits: np.ndarray | None,
    transition_prior_weight: float,
) -> np.ndarray:
    logits = np.asarray(selector_logits, dtype=np.float32).reshape(-1).copy()
    if transition_prior_logits is None or float(transition_prior_weight) == 0.0:
        return logits
    if bundle.candidate_features.shape[1] < _selector_candidate_feature_dim(SELECTOR_TARGET_MODE_BENEFIT_HARM):
        return logits
    prior = np.asarray(transition_prior_logits, dtype=np.float32)
    if prior.shape != (bundle.num_shots, len(CLASS4_LABELS) ** 2):
        raise ValueError(
            "Transition-prior logits shape mismatch. "
            f"Expected {(bundle.num_shots, len(CLASS4_LABELS) ** 2)}, got {prior.shape}"
        )
    prior = prior - np.max(prior, axis=1, keepdims=True)
    log_probs = prior - np.log(np.sum(np.exp(prior), axis=1, keepdims=True))
    transition_slice = _transition_feature_offsets(
        candidate_feature_dim=int(bundle.candidate_features.shape[1]),
    )["transition"]
    candidate_transition = np.argmax(bundle.candidate_features[:, transition_slice], axis=1).astype(np.int64)
    shot_indices = np.asarray(bundle.shot_indices, dtype=np.int64)
    logits += float(transition_prior_weight) * log_probs[shot_indices, candidate_transition]
    return np.ascontiguousarray(logits, dtype=np.float32)


def _select_candidate_masks_from_logits(
    *,
    bundle: SelectorCandidateBundle,
    selector_logits: np.ndarray,
    selector_emit_margin: float = 0.0,
    selector_nonzero_bias: float = 0.0,
    transition_prior_logits: np.ndarray | None = None,
    selector_transition_compat_top_k: int = 0,
    candidate_compatibility_logits: np.ndarray | None = None,
    selector_candidate_compat_threshold: float = 0.0,
    selector_candidate_compat_top_k: int = 0,
    router_probs: np.ndarray | None = None,
    router_threshold: float = 0.5,
    routed_nonzero_only: bool = False,
) -> dict[str, np.ndarray | float | int]:
    if bundle.shot_features.shape[0] != selector_logits.shape[0]:
        raise ValueError(
            "Selector logit count must match candidate rows. "
            f"Expected {bundle.shot_features.shape[0]}, got {selector_logits.shape[0]}"
        )
    selected_edit_mask = np.zeros((bundle.num_shots, bundle.num_detectors), dtype=np.uint8)
    selected_edit_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    selected_target_score = np.zeros((bundle.num_shots,), dtype=np.float32)
    selected_correct = np.zeros((bundle.num_shots,), dtype=np.uint8)
    selected_score = np.full((bundle.num_shots,), -np.inf, dtype=np.float32)
    selected_candidate_count = np.zeros((bundle.num_shots,), dtype=np.int16)
    selected_routed = np.zeros((bundle.num_shots,), dtype=np.uint8)
    routed_probs = (
        np.asarray(router_probs, dtype=np.float32).reshape(-1)
        if router_probs is not None
        else None
    )
    if routed_probs is not None and routed_probs.shape[0] != bundle.num_shots:
        raise ValueError(
            "Router probability count must match number of shots. "
            f"Expected {bundle.num_shots}, got {routed_probs.shape[0]}"
        )

    logits = np.asarray(selector_logits, dtype=np.float32).reshape(-1)
    prior_logits = (
        np.asarray(transition_prior_logits, dtype=np.float32)
        if transition_prior_logits is not None
        else None
    )
    compat_logits = (
        np.asarray(candidate_compatibility_logits, dtype=np.float32).reshape(-1)
        if candidate_compatibility_logits is not None
        else None
    )
    if compat_logits is not None and compat_logits.shape[0] != bundle.shot_features.shape[0]:
        raise ValueError(
            "Candidate compatibility logit count must match candidate rows. "
            f"Expected {bundle.shot_features.shape[0]}, got {compat_logits.shape[0]}"
        )
    compat_top_k = max(int(selector_transition_compat_top_k), 0)
    if prior_logits is not None and prior_logits.shape != (bundle.num_shots, len(CLASS4_LABELS) ** 2):
        raise ValueError(
            "Transition compatibility logits shape mismatch. "
            f"Expected {(bundle.num_shots, len(CLASS4_LABELS) ** 2)}, got {prior_logits.shape}"
        )
    transition_slice = _transition_feature_offsets(
        candidate_feature_dim=int(bundle.candidate_features.shape[1]),
    )["transition"]
    for group_slice in _selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        shot_idx = int(bundle.shot_indices[int(rows[0])])
        group_logits = logits[rows].copy()
        group_weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
        if (
            prior_logits is not None
            and compat_top_k > 0
            and bundle.candidate_features.shape[1] >= _selector_candidate_feature_dim(SELECTOR_TARGET_MODE_BENEFIT_HARM)
        ):
            k = min(int(compat_top_k), len(CLASS4_LABELS) ** 2)
            top_transitions = set(
                int(idx)
                for idx in np.argsort(-prior_logits[shot_idx], kind="mergesort")[:k].tolist()
            )
            candidate_transition = np.argmax(
                bundle.candidate_features[rows, transition_slice],
                axis=1,
            ).astype(np.int64, copy=False)
            incompatible = np.asarray(
                [
                    bool(weight > 0 and int(transition) not in top_transitions)
                    for weight, transition in zip(group_weights, candidate_transition)
                ],
                dtype=bool,
            )
            group_logits[incompatible] = -np.inf
        if compat_logits is not None and float(selector_candidate_compat_threshold) > 0.0:
            compat_probs = common._sigmoid_np(compat_logits[rows]).reshape(-1)
            incompatible = np.logical_and(
                group_weights > 0,
                compat_probs < float(selector_candidate_compat_threshold),
            )
            group_logits[incompatible] = -np.inf
        if compat_logits is not None and int(selector_candidate_compat_top_k) > 0:
            nonzero_local = np.flatnonzero(group_weights > 0)
            if nonzero_local.size > int(selector_candidate_compat_top_k):
                compat_group_logits = compat_logits[rows]
                keep_local = nonzero_local[
                    np.argsort(-compat_group_logits[nonzero_local], kind="mergesort")[
                        : int(selector_candidate_compat_top_k)
                    ]
                ]
                keep = np.zeros((group_logits.shape[0],), dtype=bool)
                keep[keep_local] = True
                incompatible = np.logical_and(group_weights > 0, ~keep)
                group_logits[incompatible] = -np.inf
        if float(selector_nonzero_bias) != 0.0:
            group_logits = group_logits + (
                (group_weights > 0).astype(np.float32, copy=False) * float(selector_nonzero_bias)
            )
        identity_candidates = np.flatnonzero(group_weights == 0)
        nonzero_candidates = np.flatnonzero(group_weights > 0)
        routed = routed_probs is None or float(routed_probs[shot_idx]) >= float(router_threshold)
        selected_routed[shot_idx] = np.uint8(1 if routed else 0)
        if not routed and identity_candidates.size:
            chosen_row = int(rows[int(identity_candidates[0])])
            selected_score[shot_idx] = np.float32(logits[chosen_row])
            selected_edit_mask[shot_idx] = bundle.candidate_edit_mask[chosen_row]
            selected_edit_weight[shot_idx] = np.int16(bundle.candidate_edit_weight[chosen_row])
            selected_target_score[shot_idx] = np.float32(bundle.target_scores[chosen_row])
            selected_correct[shot_idx] = np.uint8(bundle.candidate_is_correct[chosen_row])
            selected_candidate_count[shot_idx] = np.int16(rows.size)
            continue
        selectable = nonzero_candidates if bool(routed_nonzero_only) and nonzero_candidates.size else None
        if selectable is not None:
            selectable_logits = group_logits[selectable]
            best_offset = int(np.argmax(selectable_logits))
            best_local = int(selectable[best_offset])
        else:
            best_local = int(np.argmax(group_logits))
        tie_pool = selectable if selectable is not None else np.arange(group_logits.shape[0], dtype=np.int64)
        best_candidates = tie_pool[np.isclose(group_logits[tie_pool], group_logits[best_local])]
        if best_candidates.size > 1:
            best_local = int(best_candidates[int(np.argmin(group_weights[best_candidates]))])
        chosen_row = int(rows[best_local])
        if identity_candidates.size:
            identity_local = int(identity_candidates[0])
            chosen_is_nonzero = int(bundle.candidate_edit_weight[chosen_row]) > 0
            if chosen_is_nonzero:
                score_gap = float(group_logits[best_local] - group_logits[identity_local])
                if score_gap < float(selector_emit_margin):
                    chosen_row = int(rows[identity_local])
        chosen_local = int(np.flatnonzero(rows == chosen_row)[0]) if rows.size else 0
        selected_score[shot_idx] = np.float32(group_logits[chosen_local])
        selected_edit_mask[shot_idx] = bundle.candidate_edit_mask[chosen_row]
        selected_edit_weight[shot_idx] = np.int16(bundle.candidate_edit_weight[chosen_row])
        selected_target_score[shot_idx] = np.float32(bundle.target_scores[chosen_row])
        selected_correct[shot_idx] = np.uint8(bundle.candidate_is_correct[chosen_row])
        selected_candidate_count[shot_idx] = np.int16(rows.size)

    return {
        "selected_edit_mask": selected_edit_mask,
        "selected_edit_weight": selected_edit_weight,
        "selected_target_score": selected_target_score,
        "selected_correct": selected_correct,
        "selected_score": selected_score,
        "selected_candidate_count": selected_candidate_count,
        "selected_routed": selected_routed,
    }


def _selector_selection_metric(
    *,
    bundle: SelectorCandidateBundle,
    selector_logits: np.ndarray,
    selector_emit_margin: float = 0.0,
    selector_nonzero_bias: float = 0.0,
    transition_prior_logits: np.ndarray | None = None,
    selector_transition_prior_weight: float = 0.0,
    selector_transition_compat_top_k: int = 0,
    candidate_compatibility_logits: np.ndarray | None = None,
    selector_candidate_compat_threshold: float = 0.0,
    selector_candidate_compat_top_k: int = 0,
    router_probs: np.ndarray | None = None,
    router_threshold: float = 0.5,
    routed_nonzero_only: bool = False,
) -> dict[str, Any]:
    adjusted_logits = _apply_transition_prior_to_selector_logits(
        bundle=bundle,
        selector_logits=selector_logits,
        transition_prior_logits=transition_prior_logits,
        transition_prior_weight=float(selector_transition_prior_weight),
    )
    selected = _select_candidate_masks_from_logits(
        bundle=bundle,
        selector_logits=adjusted_logits,
        selector_emit_margin=float(selector_emit_margin),
        selector_nonzero_bias=float(selector_nonzero_bias),
        transition_prior_logits=transition_prior_logits,
        selector_transition_compat_top_k=int(selector_transition_compat_top_k),
        candidate_compatibility_logits=candidate_compatibility_logits,
        selector_candidate_compat_threshold=float(selector_candidate_compat_threshold),
        selector_candidate_compat_top_k=int(selector_candidate_compat_top_k),
        router_probs=router_probs,
        router_threshold=float(router_threshold),
        routed_nonzero_only=bool(routed_nonzero_only),
    )
    selected_correct = np.asarray(selected["selected_correct"], dtype=np.uint8)
    selected_target_score = np.asarray(selected["selected_target_score"], dtype=np.float32)
    selected_edit_weight = np.asarray(selected["selected_edit_weight"], dtype=np.int16)
    selected_routed = np.asarray(selected["selected_routed"], dtype=np.uint8)
    oracle_per_shot = np.zeros((bundle.num_shots,), dtype=np.uint8)
    if bundle.shot_indices.size:
        np.maximum.at(oracle_per_shot, bundle.shot_indices, bundle.candidate_is_correct.astype(np.uint8, copy=False))
    return {
        "selector_accuracy": float(np.mean(selected_correct)) if selected_correct.size else None,
        "mean_selected_target_score": float(selected_target_score.mean()) if selected_target_score.size else None,
        "mean_selected_edit_weight": float(selected_edit_weight.mean()) if selected_edit_weight.size else None,
        "fraction_with_any_selected_edit": (
            float(np.mean(selected_edit_weight > 0)) if selected_edit_weight.size else None
        ),
        "candidate_oracle_accuracy": float(np.mean(oracle_per_shot)) if oracle_per_shot.size else None,
        "mean_candidates_per_shot": (
            float(bundle.shot_indices.shape[0] / bundle.num_shots) if bundle.num_shots else None
        ),
        "selector_emit_margin": float(selector_emit_margin),
        "selector_nonzero_bias": float(selector_nonzero_bias),
        "selector_transition_prior_weight": float(selector_transition_prior_weight),
        "selector_transition_compat_top_k": int(selector_transition_compat_top_k),
        "selector_candidate_compat_threshold": float(selector_candidate_compat_threshold),
        "selector_candidate_compat_top_k": int(selector_candidate_compat_top_k),
        "router_threshold": float(router_threshold),
        "router_fraction_routed": float(np.mean(selected_routed > 0)) if selected_routed.size else None,
        "selection": selected,
    }


def _selector_identity_correct_by_shot(bundle: SelectorCandidateBundle) -> np.ndarray:
    identity_correct = np.zeros((bundle.num_shots,), dtype=np.uint8)
    for group_slice in _selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        shot_idx = int(bundle.shot_indices[int(rows[0])])
        weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
        identity = np.flatnonzero(weights == 0)
        if identity.size:
            row = int(rows[int(identity[0])])
            identity_correct[shot_idx] = np.uint8(bundle.candidate_is_correct[row])
    return identity_correct


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
    }


def _selector_epoch_margin_diagnostics(
    *,
    bundle: SelectorCandidateBundle,
    selector_logits: np.ndarray,
    emit_margin_grid: list[float],
) -> dict[str, Any]:
    logits = np.asarray(selector_logits, dtype=np.float32).reshape(-1)
    identity_correct = _selector_identity_correct_by_shot(bundle).astype(bool, copy=False)
    baseline_accuracy = float(np.mean(identity_correct)) if identity_correct.size else None
    best_nonzero_gap = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    best_nonzero_target_score = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    for group_slice in _selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        shot_idx = int(bundle.shot_indices[int(rows[0])])
        weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
        identity = np.flatnonzero(weights == 0)
        nonzero = np.flatnonzero(weights > 0)
        if identity.size == 0 or nonzero.size == 0:
            continue
        group_logits = logits[rows]
        identity_logit = float(group_logits[int(identity[0])])
        best_local = int(nonzero[int(np.argmax(group_logits[nonzero]))])
        best_row = int(rows[best_local])
        best_nonzero_gap[shot_idx] = float(group_logits[best_local] - identity_logit)
        best_nonzero_target_score[shot_idx] = float(bundle.target_scores[best_row])

    by_margin: dict[str, Any] = {}
    for emit_margin in emit_margin_grid:
        selection_metric = _selector_selection_metric(
            bundle=bundle,
            selector_logits=logits,
            selector_emit_margin=float(emit_margin),
        )
        selected = selection_metric["selection"]
        selected_correct = np.asarray(selected["selected_correct"], dtype=np.uint8).reshape(-1) > 0
        selected_weight = np.asarray(selected["selected_edit_weight"], dtype=np.int16).reshape(-1)
        selected_target_score = np.asarray(selected["selected_target_score"], dtype=np.float32).reshape(-1)
        nonzero = selected_weight > 0
        improved = np.logical_and(nonzero, np.logical_and(~identity_correct, selected_correct))
        harmed = np.logical_and(nonzero, np.logical_and(identity_correct, ~selected_correct))
        selected_accuracy = float(np.mean(selected_correct)) if selected_correct.size else None
        delta = (
            float(selected_accuracy - baseline_accuracy)
            if selected_accuracy is not None and baseline_accuracy is not None
            else None
        )
        by_margin[str(float(emit_margin))] = {
            "selector_emit_margin": float(emit_margin),
            "baseline_accuracy": baseline_accuracy,
            "selector_accuracy": selected_accuracy,
            "delta_over_no_edit": delta,
            "selected_nonzero": int(nonzero.sum()),
            "improved": int(improved.sum()),
            "harmed": int(harmed.sum()),
            "selected_positive_target": int(np.logical_and(nonzero, selected_target_score > 0.0).sum()),
            "selected_zero_target": int(np.logical_and(nonzero, np.isclose(selected_target_score, 0.0)).sum()),
            "selected_negative_target": int(np.logical_and(nonzero, selected_target_score < 0.0).sum()),
            "best_nonzero_gap_quantiles": _finite_quantiles(best_nonzero_gap),
            "selected_gap_quantiles": _finite_quantiles(best_nonzero_gap[nonzero]),
            "mean_best_nonzero_target_score": (
                float(np.nanmean(best_nonzero_target_score))
                if np.isfinite(best_nonzero_target_score).any()
                else None
            ),
        }
    return {
        "baseline_accuracy": baseline_accuracy,
        "num_examples": int(bundle.num_shots),
        "by_margin": by_margin,
    }


def _selector_epoch_diagnostic_system_selection(
    diagnostics_by_family: dict[str, Any],
) -> dict[str, Any] | None:
    margins = sorted(
        {
            str(margin)
            for family_diag in diagnostics_by_family.values()
            for margin in ((family_diag.get("by_margin") or {}).keys())
        },
        key=lambda text: float(text),
    )
    best_key: tuple[float, float, float, float] | None = None
    best_summary: dict[str, Any] | None = None
    for margin_text in margins:
        accuracies: list[float] = []
        deltas: list[float] = []
        selected_nonzero = 0
        improved = 0
        harmed = 0
        positive_target = 0
        negative_target = 0
        by_family: dict[str, Any] = {}
        for family, family_diag in diagnostics_by_family.items():
            margin_diag = (family_diag.get("by_margin") or {}).get(margin_text) or {}
            accuracy = margin_diag.get("selector_accuracy")
            delta = margin_diag.get("delta_over_no_edit")
            if accuracy is not None:
                accuracies.append(float(accuracy))
            if delta is not None:
                deltas.append(float(delta))
            nz = int(margin_diag.get("selected_nonzero") or 0)
            imp = int(margin_diag.get("improved") or 0)
            harm = int(margin_diag.get("harmed") or 0)
            pos = int(margin_diag.get("selected_positive_target") or 0)
            neg = int(margin_diag.get("selected_negative_target") or 0)
            selected_nonzero += nz
            improved += imp
            harmed += harm
            positive_target += pos
            negative_target += neg
            by_family[str(family)] = {
                "selector_accuracy": (None if accuracy is None else float(accuracy)),
                "delta_over_no_edit": (None if delta is None else float(delta)),
                "selected_nonzero": int(nz),
                "improved": int(imp),
                "harmed": int(harm),
                "selected_positive_target": int(pos),
                "selected_negative_target": int(neg),
            }
        if not accuracies:
            continue
        mean_accuracy = float(np.mean(accuracies))
        mean_delta = float(np.mean(deltas)) if deltas else None
        key = (
            mean_accuracy,
            float(positive_target - negative_target),
            float(improved - harmed),
            -float(selected_nonzero),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_summary = {
                "selector_epoch_selection_mode": SELECTOR_EPOCH_SELECTION_DIAGNOSTIC_SYSTEM,
                "selected_margin": float(margin_text),
                "mean_selector_accuracy": mean_accuracy,
                "mean_delta_over_no_edit": mean_delta,
                "selected_nonzero": int(selected_nonzero),
                "improved": int(improved),
                "harmed": int(harmed),
                "selected_positive_target": int(positive_target),
                "selected_negative_target": int(negative_target),
                "by_family": by_family,
            }
    return best_summary


def _train_selector_one_epoch(
    *,
    selector: nn.Module,
    loader: Any,
    optimizer: Any,
    device: torch.device,
) -> dict[str, float]:
    selector.train()
    loss_sum = 0.0
    count = 0
    for shot_feat, cand_feat, target_score in loader:
        shot_feat = shot_feat.to(device)
        cand_feat = cand_feat.to(device)
        target_score = target_score.to(device)
        optimizer.zero_grad()
        logits = selector(shot_feat, cand_feat)
        loss = F.binary_cross_entropy_with_logits(logits, target_score)
        loss.backward()
        optimizer.step()
        batch_count = int(target_score.shape[0])
        loss_sum += float(loss.item()) * batch_count
        count += batch_count
    return {
        "selector_bce_loss": float(loss_sum / count) if count else None,
    }


def _train_selector_group_rank_epoch(
    *,
    selector: nn.Module,
    bundles: list[SelectorCandidateBundle],
    optimizer: Any,
    device: torch.device,
    hard_shot_weight: float,
    identity_margin_loss_weight: float,
    identity_margin: float,
    harm_margin_loss_weight: float,
    harm_margin: float,
    negative_identity_margin_loss_weight: float,
    negative_identity_margin: float,
    benefit_harm_pairwise_loss_weight: float,
    benefit_harm_pairwise_margin: float,
    positive_negative_hard_loss_weight: float,
    positive_negative_hard_margin: float,
    cross_family_positive_negative_loss_weight: float,
    cross_family_positive_negative_margin: float,
    risk_aware_benefit_loss_weight: float = 0.0,
    risk_aware_harm_loss_weight: float = 0.0,
    risk_aware_benefit_pos_weight: float = 1.0,
    risk_aware_harm_pos_weight: float = 1.0,
) -> dict[str, float]:
    selector.train()
    total_loss_sum = 0.0
    total_ce_loss_sum = 0.0
    total_margin_loss_sum = 0.0
    total_harm_margin_loss_sum = 0.0
    total_negative_identity_margin_loss_sum = 0.0
    total_benefit_harm_pairwise_loss_sum = 0.0
    total_positive_negative_hard_loss_sum = 0.0
    total_cross_family_positive_negative_loss_sum = 0.0
    total_risk_aware_benefit_loss_sum = 0.0
    total_risk_aware_harm_loss_sum = 0.0
    total_weight_sum = 0.0
    hard_group_count = 0.0
    total_group_count = 0.0
    identity_competition_count = 0.0
    harm_competition_count = 0.0
    negative_identity_competition_count = 0.0
    benefit_harm_pairwise_count = 0.0
    positive_negative_hard_count = 0.0
    cross_family_positive_negative_count = 0.0
    risk_aware_benefit_count = 0.0
    risk_aware_harm_count = 0.0
    risk_aware_enabled = (
        float(risk_aware_benefit_loss_weight) > 0.0
        or float(risk_aware_harm_loss_weight) > 0.0
    )
    component_logits_fn = getattr(selector, "component_logits", None)
    if risk_aware_enabled and not callable(component_logits_fn):
        raise ValueError("risk-aware selector losses require selector_model='risk_aware'")
    group_refs: list[tuple[int, SelectorCandidateBundle, slice]] = []
    negative_refs_by_family: dict[int, list[tuple[SelectorCandidateBundle, slice]]] = {}
    for family_idx, bundle in enumerate(bundles):
        for group_slice in _selector_group_slices(bundle):
            group_refs.append((family_idx, bundle, group_slice))
            if float(cross_family_positive_negative_loss_weight) > 0.0:
                group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
                group_weights_np = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
                negative_nonzero = np.flatnonzero(
                    np.logical_and(group_weights_np > 0, group_target_scores < 0.0)
                )
                if negative_nonzero.size:
                    negative_refs_by_family.setdefault(family_idx, []).append((bundle, group_slice))
    if not group_refs:
        return {
            "selector_group_rank_loss": None,
            "selector_group_rank_ce_loss": None,
            "selector_identity_margin_loss": None,
            "selector_harm_margin_loss": None,
            "selector_negative_identity_margin_loss": None,
            "selector_hard_group_fraction": None,
            "selector_identity_competition_fraction": None,
            "selector_harm_competition_fraction": None,
            "selector_negative_identity_competition_fraction": None,
            "selector_benefit_harm_pairwise_loss": None,
            "selector_benefit_harm_pairwise_fraction": None,
            "selector_positive_negative_hard_loss": None,
            "selector_positive_negative_hard_fraction": None,
            "selector_cross_family_positive_negative_loss": None,
            "selector_cross_family_positive_negative_fraction": None,
            "selector_risk_aware_benefit_loss": None,
            "selector_risk_aware_harm_loss": None,
            "selector_risk_aware_benefit_fraction": None,
            "selector_risk_aware_harm_fraction": None,
        }
    order = np.random.permutation(len(group_refs))
    for order_idx in order.tolist():
        family_idx, bundle, group_slice = group_refs[order_idx]
        shot_feat = torch.from_numpy(
            np.ascontiguousarray(bundle.shot_features[group_slice], dtype=np.float32)
        ).to(device)
        cand_feat = torch.from_numpy(
            np.ascontiguousarray(bundle.candidate_features[group_slice], dtype=np.float32)
        ).to(device)
        target_idx = _selector_group_target_index(bundle, group_slice)
        group_weight = _selector_group_weight(
            bundle,
            group_slice,
            hard_shot_weight=hard_shot_weight,
        )
        optimizer.zero_grad()
        benefit_logits: torch.Tensor | None = None
        harm_logits: torch.Tensor | None = None
        if callable(component_logits_fn):
            benefit_logits, harm_logits = component_logits_fn(shot_feat, cand_feat)
            utility_from_components = getattr(selector, "utility_from_components", None)
            if callable(utility_from_components):
                logits_1d = utility_from_components(benefit_logits, harm_logits).reshape(-1)
            else:
                logits_1d = (benefit_logits - harm_logits).reshape(-1)
        else:
            logits_1d = selector(shot_feat, cand_feat).reshape(-1)
        logits = logits_1d.unsqueeze(0)
        target = torch.tensor([target_idx], dtype=torch.int64, device=device)
        ce_loss = F.cross_entropy(logits, target)
        margin_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        harm_margin_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        negative_identity_margin_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        benefit_harm_pairwise_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        positive_negative_hard_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        cross_family_positive_negative_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        risk_aware_benefit_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        risk_aware_harm_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        if risk_aware_enabled:
            if benefit_logits is None or harm_logits is None:
                raise RuntimeError("risk-aware selector did not produce component logits")
            group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            benefit_targets = torch.from_numpy(
                np.ascontiguousarray((group_target_scores > 0.0).astype(np.float32))
            ).to(device)
            harm_targets = torch.from_numpy(
                np.ascontiguousarray((group_target_scores < 0.0).astype(np.float32))
            ).to(device)
            if float(risk_aware_benefit_loss_weight) > 0.0:
                risk_aware_benefit_loss = F.binary_cross_entropy_with_logits(
                    benefit_logits.reshape(-1),
                    benefit_targets,
                    pos_weight=torch.tensor(
                        float(risk_aware_benefit_pos_weight),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                risk_aware_benefit_count += 1.0
            if float(risk_aware_harm_loss_weight) > 0.0:
                risk_aware_harm_loss = F.binary_cross_entropy_with_logits(
                    harm_logits.reshape(-1),
                    harm_targets,
                    pos_weight=torch.tensor(
                        float(risk_aware_harm_pos_weight),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                risk_aware_harm_count += 1.0
        if identity_margin_loss_weight > 0.0:
            competition = _selector_identity_competition_indices(bundle, group_slice)
            if competition is not None:
                identity_idx, best_nonzero_idx = competition
                identity_logit = logits_1d[int(identity_idx)]
                nonzero_logit = logits_1d[int(best_nonzero_idx)]
                margin_loss = F.relu(
                    torch.tensor(float(identity_margin), dtype=torch.float32, device=device)
                    - (nonzero_logit - identity_logit)
                )
                identity_competition_count += 1.0
        if harm_margin_loss_weight > 0.0:
            harm_competition = _selector_identity_harm_competition_indices(bundle, group_slice)
            if harm_competition is not None:
                identity_idx, best_nonzero_idx = harm_competition
                identity_logit = logits_1d[int(identity_idx)]
                nonzero_logit = logits_1d[int(best_nonzero_idx)]
                harm_margin_loss = F.relu(
                    torch.tensor(float(harm_margin), dtype=torch.float32, device=device)
                    - (identity_logit - nonzero_logit)
                )
                harm_competition_count += 1.0
        if negative_identity_margin_loss_weight > 0.0:
            group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            group_weights_np = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
            identity_rows = np.flatnonzero(group_weights_np == 0)
            negative_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores < 0.0)
            )
            if identity_rows.size and negative_nonzero.size:
                identity_logit = logits_1d[int(identity_rows[0])]
                negative_logits = logits_1d[
                    torch.as_tensor(negative_nonzero, dtype=torch.int64, device=device)
                ]
                max_negative_logit = torch.max(negative_logits)
                negative_identity_margin_loss = F.relu(
                    torch.tensor(
                        float(negative_identity_margin),
                        dtype=torch.float32,
                        device=device,
                    )
                    - (identity_logit - max_negative_logit)
                )
                negative_identity_competition_count += 1.0
        if benefit_harm_pairwise_loss_weight > 0.0:
            group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            group_weights_np = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
            positive_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores > 0.0)
            )
            negative_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores <= 0.0)
            )
            if positive_nonzero.size and negative_nonzero.size:
                pos_logits = logits_1d[
                    torch.as_tensor(positive_nonzero, dtype=torch.int64, device=device)
                ]
                neg_logits = logits_1d[
                    torch.as_tensor(negative_nonzero, dtype=torch.int64, device=device)
                ]
                diffs = pos_logits[:, None] - neg_logits[None, :]
                if float(benefit_harm_pairwise_margin) > 0.0:
                    benefit_harm_pairwise_loss = F.relu(
                        torch.tensor(
                            float(benefit_harm_pairwise_margin),
                            dtype=torch.float32,
                            device=device,
                        )
                        - diffs
                    ).mean()
                else:
                    benefit_harm_pairwise_loss = F.softplus(-diffs).mean()
                benefit_harm_pairwise_count += 1.0
        if positive_negative_hard_loss_weight > 0.0:
            group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            group_weights_np = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
            positive_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores > 0.0)
            )
            negative_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores < 0.0)
            )
            if positive_nonzero.size and negative_nonzero.size:
                pos_logits = logits_1d[
                    torch.as_tensor(positive_nonzero, dtype=torch.int64, device=device)
                ]
                neg_logits = logits_1d[
                    torch.as_tensor(negative_nonzero, dtype=torch.int64, device=device)
                ]
                diff = torch.max(pos_logits) - torch.max(neg_logits)
                if float(positive_negative_hard_margin) > 0.0:
                    positive_negative_hard_loss = F.relu(
                        torch.tensor(
                            float(positive_negative_hard_margin),
                            dtype=torch.float32,
                            device=device,
                        )
                        - diff
                    )
                else:
                    positive_negative_hard_loss = F.softplus(-diff)
                positive_negative_hard_count += 1.0
        if cross_family_positive_negative_loss_weight > 0.0 and len(negative_refs_by_family) > 1:
            group_target_scores = np.asarray(bundle.target_scores[group_slice], dtype=np.float32)
            group_weights_np = np.asarray(bundle.candidate_edit_weight[group_slice], dtype=np.int16)
            positive_nonzero = np.flatnonzero(
                np.logical_and(group_weights_np > 0, group_target_scores > 0.0)
            )
            other_negative_refs: list[tuple[SelectorCandidateBundle, slice]] = []
            for other_family_idx, family_refs in negative_refs_by_family.items():
                if int(other_family_idx) != int(family_idx):
                    other_negative_refs.extend(family_refs)
            if positive_nonzero.size and other_negative_refs:
                neg_bundle, neg_slice = other_negative_refs[
                    int(np.random.randint(0, len(other_negative_refs)))
                ]
                neg_target_scores = np.asarray(neg_bundle.target_scores[neg_slice], dtype=np.float32)
                neg_weights_np = np.asarray(neg_bundle.candidate_edit_weight[neg_slice], dtype=np.int16)
                negative_nonzero = np.flatnonzero(
                    np.logical_and(neg_weights_np > 0, neg_target_scores < 0.0)
                )
                if negative_nonzero.size:
                    neg_shot_feat = torch.from_numpy(
                        np.ascontiguousarray(neg_bundle.shot_features[neg_slice], dtype=np.float32)
                    ).to(device)
                    neg_cand_feat = torch.from_numpy(
                        np.ascontiguousarray(neg_bundle.candidate_features[neg_slice], dtype=np.float32)
                    ).to(device)
                    neg_logits_1d = selector(neg_shot_feat, neg_cand_feat).reshape(-1)
                    pos_logits = logits_1d[
                        torch.as_tensor(positive_nonzero, dtype=torch.int64, device=device)
                    ]
                    neg_logits = neg_logits_1d[
                        torch.as_tensor(negative_nonzero, dtype=torch.int64, device=device)
                    ]
                    diff = torch.max(pos_logits) - torch.max(neg_logits)
                    if float(cross_family_positive_negative_margin) > 0.0:
                        cross_family_positive_negative_loss = F.relu(
                            torch.tensor(
                                float(cross_family_positive_negative_margin),
                                dtype=torch.float32,
                                device=device,
                            )
                            - diff
                        )
                    else:
                        cross_family_positive_negative_loss = F.softplus(-diff)
                    cross_family_positive_negative_count += 1.0
        loss = (
            ce_loss
            + float(identity_margin_loss_weight) * margin_loss
            + float(harm_margin_loss_weight) * harm_margin_loss
            + float(negative_identity_margin_loss_weight) * negative_identity_margin_loss
            + float(benefit_harm_pairwise_loss_weight) * benefit_harm_pairwise_loss
            + float(positive_negative_hard_loss_weight) * positive_negative_hard_loss
            + float(cross_family_positive_negative_loss_weight) * cross_family_positive_negative_loss
            + float(risk_aware_benefit_loss_weight) * risk_aware_benefit_loss
            + float(risk_aware_harm_loss_weight) * risk_aware_harm_loss
        ) * float(group_weight)
        loss.backward()
        optimizer.step()
        total_loss_sum += float(loss.item())
        total_ce_loss_sum += float(ce_loss.item()) * float(group_weight)
        total_margin_loss_sum += float(margin_loss.item()) * float(group_weight)
        total_harm_margin_loss_sum += float(harm_margin_loss.item()) * float(group_weight)
        total_negative_identity_margin_loss_sum += (
            float(negative_identity_margin_loss.item()) * float(group_weight)
        )
        total_benefit_harm_pairwise_loss_sum += (
            float(benefit_harm_pairwise_loss.item()) * float(group_weight)
        )
        total_positive_negative_hard_loss_sum += (
            float(positive_negative_hard_loss.item()) * float(group_weight)
        )
        total_cross_family_positive_negative_loss_sum += (
            float(cross_family_positive_negative_loss.item()) * float(group_weight)
        )
        total_risk_aware_benefit_loss_sum += (
            float(risk_aware_benefit_loss.item()) * float(group_weight)
        )
        total_risk_aware_harm_loss_sum += (
            float(risk_aware_harm_loss.item()) * float(group_weight)
        )
        total_weight_sum += float(group_weight)
        total_group_count += 1.0
        if float(group_weight) > 1.0:
            hard_group_count += 1.0
    return {
        "selector_group_rank_loss": (
            float(total_loss_sum / total_weight_sum) if total_weight_sum > 0.0 else None
        ),
        "selector_group_rank_ce_loss": (
            float(total_ce_loss_sum / total_weight_sum) if total_weight_sum > 0.0 else None
        ),
        "selector_identity_margin_loss": (
            float(total_margin_loss_sum / total_weight_sum) if total_weight_sum > 0.0 else None
        ),
        "selector_harm_margin_loss": (
            float(total_harm_margin_loss_sum / total_weight_sum) if total_weight_sum > 0.0 else None
        ),
        "selector_negative_identity_margin_loss": (
            float(total_negative_identity_margin_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0
            else None
        ),
        "selector_benefit_harm_pairwise_loss": (
            float(total_benefit_harm_pairwise_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0
            else None
        ),
        "selector_positive_negative_hard_loss": (
            float(total_positive_negative_hard_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0
            else None
        ),
        "selector_cross_family_positive_negative_loss": (
            float(total_cross_family_positive_negative_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0
            else None
        ),
        "selector_risk_aware_benefit_loss": (
            float(total_risk_aware_benefit_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0 and float(risk_aware_benefit_loss_weight) > 0.0
            else None
        ),
        "selector_risk_aware_harm_loss": (
            float(total_risk_aware_harm_loss_sum / total_weight_sum)
            if total_weight_sum > 0.0 and float(risk_aware_harm_loss_weight) > 0.0
            else None
        ),
        "selector_hard_group_fraction": (
            float(hard_group_count / total_group_count) if total_group_count > 0.0 else None
        ),
        "selector_identity_competition_fraction": (
            float(identity_competition_count / total_group_count) if total_group_count > 0.0 else None
        ),
        "selector_harm_competition_fraction": (
            float(harm_competition_count / total_group_count) if total_group_count > 0.0 else None
        ),
        "selector_negative_identity_competition_fraction": (
            float(negative_identity_competition_count / total_group_count)
            if total_group_count > 0.0
            else None
        ),
        "selector_benefit_harm_pairwise_fraction": (
            float(benefit_harm_pairwise_count / total_group_count)
            if total_group_count > 0.0
            else None
        ),
        "selector_positive_negative_hard_fraction": (
            float(positive_negative_hard_count / total_group_count)
            if total_group_count > 0.0
            else None
        ),
        "selector_cross_family_positive_negative_fraction": (
            float(cross_family_positive_negative_count / total_group_count)
            if total_group_count > 0.0
            else None
        ),
        "selector_risk_aware_benefit_fraction": (
            float(risk_aware_benefit_count / total_group_count)
            if total_group_count > 0.0 and float(risk_aware_benefit_loss_weight) > 0.0
            else None
        ),
        "selector_risk_aware_harm_fraction": (
            float(risk_aware_harm_count / total_group_count)
            if total_group_count > 0.0 and float(risk_aware_harm_loss_weight) > 0.0
            else None
        ),
    }


def _train_router_one_epoch(
    *,
    router: nn.Module,
    shot_features: np.ndarray,
    targets: np.ndarray,
    optimizer: Any,
    device: torch.device,
    batch_size: int,
    positive_weight: float,
    negative_ratio: float | None = None,
) -> dict[str, float]:
    router.train()
    features = np.asarray(shot_features, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.float32).reshape(-1)
    if negative_ratio is None or float(negative_ratio) < 0.0:
        order = np.random.permutation(labels.shape[0])
    else:
        positive_idx = np.flatnonzero(labels >= 0.5)
        negative_idx = np.flatnonzero(labels < 0.5)
        if positive_idx.size and negative_idx.size:
            negative_count = min(
                int(negative_idx.size),
                int(math.ceil(float(negative_ratio) * float(positive_idx.size))),
            )
            sampled_negative = (
                np.random.choice(negative_idx, size=negative_count, replace=False)
                if negative_count > 0
                else np.zeros((0,), dtype=np.int64)
            )
            order = np.random.permutation(np.concatenate([positive_idx, sampled_negative], axis=0))
        else:
            order = np.random.permutation(labels.shape[0])
    sampled_labels = labels[order] if order.size else labels
    loss_sum = 0.0
    correct_sum = 0.0
    weight_sum = 0.0
    for start in range(0, int(order.shape[0]), int(batch_size)):
        idx = order[start : start + int(batch_size)]
        xb = torch.from_numpy(np.ascontiguousarray(features[idx], dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.ascontiguousarray(labels[idx], dtype=np.float32)).to(device)
        weights = torch.ones_like(yb)
        weights = torch.where(yb >= 0.5, weights * float(positive_weight), weights)
        optimizer.zero_grad()
        logits = router(xb)
        per_item = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
        loss = torch.sum(per_item * weights) / torch.clamp(weights.sum(), min=1.0)
        loss.backward()
        optimizer.step()
        batch_weight = float(weights.sum().item())
        preds = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
        loss_sum += float(loss.item()) * batch_weight
        correct_sum += float(((preds == yb).to(dtype=torch.float32) * weights).sum().item())
        weight_sum += batch_weight
    return {
        "router_bce_loss": float(loss_sum / weight_sum) if weight_sum > 0.0 else None,
        "router_weighted_accuracy": float(correct_sum / weight_sum) if weight_sum > 0.0 else None,
        "router_positive_fraction": float(np.mean(labels >= 0.5)) if labels.size else None,
        "router_sampled_positive_fraction": (
            float(np.mean(sampled_labels >= 0.5)) if sampled_labels.size else None
        ),
        "router_negative_ratio": (float(negative_ratio) if negative_ratio is not None else None),
    }


def _train_hard_shot_router(
    *,
    train_bundles: list[SelectorCandidateBundle],
    val_bundles_by_family: dict[str, SelectorCandidateBundle],
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    positive_weight: float,
    supervision_target: str = ROUTER_LABEL_IDENTITY_VS_NONZERO,
    pretrain_target: str = ROUTER_PRETRAIN_TARGET_NONE,
    pretrain_epochs: int = 0,
    pretrain_positive_weight: float | None = None,
    negative_ratio: float | None = None,
) -> dict[str, Any] | None:
    train_features, train_targets, _train_oracle = _build_router_training_arrays(
        train_bundles,
        label_mode=str(supervision_target),
    )
    if train_features.shape[0] == 0:
        return None
    router_kwargs = {
        "shot_feature_dim": int(train_features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
    }
    router = HardShotRouter(**router_kwargs).to(device)
    val_arrays_by_family = {
        family: _build_router_training_arrays([bundle], label_mode=str(supervision_target))
        for family, bundle in val_bundles_by_family.items()
    }
    pretrain_epoch_history: list[dict[str, Any]] = []
    if str(pretrain_target) != ROUTER_PRETRAIN_TARGET_NONE and int(pretrain_epochs) > 0:
        pretrain_features, pretrain_targets, _pretrain_oracle = _build_router_training_arrays(
            train_bundles,
            label_mode=str(pretrain_target),
        )
        pretrain_val_arrays_by_family = {
            family: _build_router_training_arrays([bundle], label_mode=str(pretrain_target))
            for family, bundle in val_bundles_by_family.items()
        }
        pretrain_optimizer = optim.Adam(router.parameters(), lr=lr)
        for epoch in range(1, int(pretrain_epochs) + 1):
            train_metrics = _train_router_one_epoch(
                router=router,
                shot_features=pretrain_features,
                targets=pretrain_targets,
                optimizer=pretrain_optimizer,
                device=device,
                batch_size=batch_size,
                positive_weight=(
                    float(pretrain_positive_weight)
                    if pretrain_positive_weight is not None
                    else float(positive_weight)
                ),
                negative_ratio=negative_ratio,
            )
            val_by_family: dict[str, Any] = {}
            accuracy_values: list[float] = []
            positive_rate_values: list[float] = []
            for family, (features, targets, _oracle) in pretrain_val_arrays_by_family.items():
                logits = _router_logits_for_shot_features(
                    router=router,
                    shot_features=features,
                    batch_size=batch_size,
                    device=device,
                )
                probs = common._sigmoid_np(logits)
                pred = (probs >= 0.5).astype(np.uint8)
                y = (np.asarray(targets, dtype=np.float32) >= 0.5).astype(np.uint8)
                accuracy = float(np.mean(pred == y)) if y.size else None
                positive_rate = float(np.mean(pred > 0)) if pred.size else None
                target_positive_rate = float(np.mean(y > 0)) if y.size else None
                if accuracy is not None:
                    accuracy_values.append(float(accuracy))
                if positive_rate is not None:
                    positive_rate_values.append(float(positive_rate))
                val_by_family[family] = {
                    "router_accuracy": accuracy,
                    "router_predicted_positive_rate": positive_rate,
                    "router_target_positive_rate": target_positive_rate,
                    "router_mean_probability": float(np.mean(probs)) if probs.size else None,
                }
            pretrain_epoch_history.append(
                {
                    "epoch": int(epoch),
                    "train": train_metrics,
                    "val_by_family": val_by_family,
                    "val_selection_metric": (
                        float(np.mean(accuracy_values)) if accuracy_values else 0.0
                    ),
                    "val_predicted_positive_rate": (
                        float(np.mean(positive_rate_values)) if positive_rate_values else 0.0
                    ),
                }
            )
    optimizer = optim.Adam(router.parameters(), lr=lr)
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _train_router_one_epoch(
            router=router,
            shot_features=train_features,
            targets=train_targets,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            positive_weight=positive_weight,
            negative_ratio=negative_ratio,
        )
        val_by_family: dict[str, Any] = {}
        accuracy_values: list[float] = []
        positive_rate_values: list[float] = []
        for family, (features, targets, _oracle) in val_arrays_by_family.items():
            logits = _router_logits_for_shot_features(
                router=router,
                shot_features=features,
                batch_size=batch_size,
                device=device,
            )
            probs = common._sigmoid_np(logits)
            pred = (probs >= 0.5).astype(np.uint8)
            y = (np.asarray(targets, dtype=np.float32) >= 0.5).astype(np.uint8)
            accuracy = float(np.mean(pred == y)) if y.size else None
            positive_rate = float(np.mean(pred > 0)) if pred.size else None
            target_positive_rate = float(np.mean(y > 0)) if y.size else None
            if accuracy is not None:
                accuracy_values.append(float(accuracy))
            if positive_rate is not None:
                positive_rate_values.append(float(positive_rate))
            val_by_family[family] = {
                "router_accuracy": accuracy,
                "router_predicted_positive_rate": positive_rate,
                "router_target_positive_rate": target_positive_rate,
                "router_mean_probability": float(np.mean(probs)) if probs.size else None,
            }
        mean_acc = float(np.mean(accuracy_values)) if accuracy_values else 0.0
        mean_pos = float(np.mean(positive_rate_values)) if positive_rate_values else 0.0
        key = (mean_acc, -mean_pos)
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train": train_metrics,
                "val_by_family": val_by_family,
                "val_selection_metric": float(mean_acc),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                "epoch": int(epoch),
                "router_state": copy.deepcopy(router.state_dict()),
                "val_by_family": copy.deepcopy(val_by_family),
                "best_val_selection_metric": float(mean_acc),
            }
    if best_state is None:
        return None
    router.load_state_dict(best_state["router_state"])
    return {
        "router": router,
        "router_kwargs": router_kwargs,
        "epoch_history": epoch_history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_selection_metric": float(best_state["best_val_selection_metric"]),
        "best_val_by_family": best_state["val_by_family"],
        "positive_weight": float(positive_weight),
        "supervision_target": str(supervision_target),
        "pretrain_target": str(pretrain_target),
        "pretrain_epochs": int(pretrain_epochs),
        "pretrain_positive_weight": (
            float(pretrain_positive_weight) if pretrain_positive_weight is not None else None
        ),
        "pretrain_epoch_history": pretrain_epoch_history,
        "negative_ratio": (float(negative_ratio) if negative_ratio is not None else None),
    }


def _selector_system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    selector: nn.Module,
    batch_size: int,
    device: torch.device,
    policy_specs: list[CandidatePolicySpec],
    selector_score_edit_penalty: float,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_vocabulary: LocalMotifVocabulary | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    selector_selection_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    selector_emit_margin: float = 0.0,
    selector_nonzero_bias: float = 0.0,
    transition_prior_head: nn.Module | None = None,
    selector_transition_prior_weight: float = 0.0,
    selector_transition_compat_top_k: int = 0,
    candidate_compatibility_head: nn.Module | None = None,
    selector_candidate_compat_threshold: float = 0.0,
    selector_candidate_compat_top_k: int = 0,
    router: nn.Module | None = None,
    router_threshold: float = 0.5,
    routed_nonzero_only: bool = False,
) -> dict[str, Any]:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    edit_logits = outputs["edit_logits"]
    needs_edit_logits = outputs["needs_edit_logits"]
    edit_probs = common._sigmoid_np(edit_logits)
    needs_edit_probs = common._sigmoid_np(needs_edit_logits)
    bundle = _build_selector_candidate_bundle(
        entry=entry,
        subset=subset,
        model=model,
        batch_size=batch_size,
        device=device,
        policy_specs=policy_specs,
        selector_score_edit_penalty=selector_score_edit_penalty,
        selector_target_mode=str(selector_target_mode),
        selector_harm_weight=float(selector_harm_weight),
        selector_miss_weight=float(selector_miss_weight),
        selector_policy_candidate_mode=str(selector_policy_candidate_mode),
        selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
        selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
        selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
        selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
        motif_vocabulary=motif_vocabulary,
        local_motif_vocabulary=local_motif_vocabulary,
        local_motif_top_k=int(local_motif_top_k),
    )
    selector_logits = _selector_logits_for_bundle(
        selector=selector,
        bundle=bundle,
        batch_size=batch_size,
        device=device,
    )
    router_probs: np.ndarray | None = None
    transition_prior_logits: np.ndarray | None = None
    candidate_compatibility_logits: np.ndarray | None = None
    if (
        transition_prior_head is not None
        and (
            float(selector_transition_prior_weight) != 0.0
            or int(selector_transition_compat_top_k) > 0
        )
    ):
        transition_prior_logits = _transition_prior_logits_for_bundle(
            head=transition_prior_head,
            bundle=bundle,
            batch_size=batch_size,
            device=device,
        )
    if router is not None:
        router_features = _router_feature_rows_from_bundle(bundle)
        router_logits = _router_logits_for_shot_features(
            router=router,
            shot_features=router_features,
            batch_size=batch_size,
            device=device,
        )
        router_probs = common._sigmoid_np(router_logits).reshape(-1)
    if (
        candidate_compatibility_head is not None
        and (
            float(selector_candidate_compat_threshold) > 0.0
            or int(selector_candidate_compat_top_k) > 0
        )
    ):
        candidate_compatibility_logits = _candidate_compatibility_logits_for_bundle(
            head=candidate_compatibility_head,
            bundle=bundle,
            batch_size=batch_size,
            device=device,
        )
    selection_metric = _selector_selection_metric(
        bundle=bundle,
        selector_logits=selector_logits,
        selector_emit_margin=float(selector_emit_margin),
        selector_nonzero_bias=float(selector_nonzero_bias),
        transition_prior_logits=transition_prior_logits,
        selector_transition_prior_weight=float(selector_transition_prior_weight),
        selector_transition_compat_top_k=int(selector_transition_compat_top_k),
        candidate_compatibility_logits=candidate_compatibility_logits,
        selector_candidate_compat_threshold=float(selector_candidate_compat_threshold),
        selector_candidate_compat_top_k=int(selector_candidate_compat_top_k),
        router_probs=router_probs,
        router_threshold=float(router_threshold),
        routed_nonzero_only=bool(routed_nonzero_only),
    )
    needs_metrics = _binary_metrics_from_probs(
        needs_edit_probs,
        subset["needs_edit"],
        threshold=0.5,
    )
    edit_metrics = _masked_edit_binary_metrics(
        edit_probs,
        subset["edit_target_volume"],
        valid_mask_volume=entry.valid_mask_volume,
        known_mask=subset["edit_target_known"],
        threshold=0.5,
    )
    selected_edit_mask = np.asarray(selection_metric["selection"]["selected_edit_mask"], dtype=np.uint8)
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=selected_edit_mask,
        decision_payload={
            "selection_mode": str(selector_selection_mode),
            "selector_model": _selector_model_name(selector),
            "candidate_policy_grid": [spec.to_dict() for spec in policy_specs],
            "selector_score_edit_penalty": float(selector_score_edit_penalty),
            "selector_target_mode": str(selector_target_mode),
            "selector_harm_weight": float(selector_harm_weight),
            "selector_miss_weight": float(selector_miss_weight),
            "selector_policy_candidate_mode": str(selector_policy_candidate_mode),
            "selector_candidate_geometry_features": bool(selector_candidate_geometry_features),
            "selector_candidate_pattern_features": bool(selector_candidate_pattern_features),
            "selector_candidate_local_evidence_features": bool(selector_candidate_local_evidence_features),
            "selector_candidate_local_patch_features": bool(selector_candidate_local_patch_features),
            "candidate_motif_vocab_num_classes": (
                int(motif_vocabulary.mask_table.shape[0]) if motif_vocabulary is not None else 0
            ),
            "candidate_local_motif_num_patterns": (
                int(len(local_motif_vocabulary.offset_patterns))
                if local_motif_vocabulary is not None
                else 0
            ),
            "candidate_local_motif_top_k": int(local_motif_top_k),
            "selector_emit_margin": float(selector_emit_margin),
            "selector_nonzero_bias": float(selector_nonzero_bias),
            "selector_transition_prior_weight": float(selector_transition_prior_weight),
            "selector_transition_compat_top_k": int(selector_transition_compat_top_k),
            "selector_candidate_compat_threshold": float(selector_candidate_compat_threshold),
            "selector_candidate_compat_top_k": int(selector_candidate_compat_top_k),
            "router_threshold": (float(router_threshold) if router is not None else None),
            "routed_nonzero_only": bool(routed_nonzero_only),
        },
        needs_metrics=needs_metrics,
        edit_metrics=edit_metrics,
        extra_change_summary={
            "selector_accuracy": selection_metric["selector_accuracy"],
            "selector_mean_selected_target_score": selection_metric["mean_selected_target_score"],
            "selector_candidate_oracle_accuracy": selection_metric["candidate_oracle_accuracy"],
            "selector_mean_candidates_per_shot": selection_metric["mean_candidates_per_shot"],
            "selector_mean_selected_edit_weight": selection_metric["mean_selected_edit_weight"],
            "selector_fraction_with_any_selected_edit": selection_metric["fraction_with_any_selected_edit"],
            "router_fraction_routed": selection_metric["router_fraction_routed"],
            "selector_decode_failures_fallback_to_baseline": int(bundle.decode_fallback_count),
        },
    )


def _train_candidate_selector(
    *,
    model: nn.Module,
    train_bundles: list[SelectorCandidateBundle],
    val_bundles_by_family: dict[str, SelectorCandidateBundle],
    entry_by_family: dict[str, PreparedEditFamily],
    subset_by_family: dict[str, dict[str, np.ndarray]],
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    selector_objective: str,
    selector_hard_shot_weight: float,
    selector_identity_margin_loss_weight: float,
    selector_identity_margin: float,
    selector_harm_margin_loss_weight: float,
    selector_harm_margin: float,
    selector_negative_identity_margin_loss_weight: float,
    selector_negative_identity_margin: float,
    selector_benefit_harm_pairwise_loss_weight: float,
    selector_benefit_harm_pairwise_margin: float,
    selector_positive_negative_hard_loss_weight: float,
    selector_positive_negative_hard_margin: float,
    selector_cross_family_positive_negative_loss_weight: float,
    selector_cross_family_positive_negative_margin: float,
    selector_model: str = SELECTOR_MODEL_SCALAR,
    selector_risk_aware_harm_logit_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
    selector_risk_aware_benefit_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_LOSS_WEIGHT,
    selector_risk_aware_harm_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOSS_WEIGHT,
    selector_risk_aware_benefit_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_POS_WEIGHT,
    selector_risk_aware_harm_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_POS_WEIGHT,
    selector_patch_head: bool = DEFAULT_SELECTOR_PATCH_HEAD,
    selector_patch_hidden_dim: int = DEFAULT_SELECTOR_PATCH_HIDDEN_DIM,
    policy_specs: list[CandidatePolicySpec],
    selector_score_edit_penalty: float,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_vocabulary: LocalMotifVocabulary | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    selector_selection_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    selector_epoch_diagnostic_margin_grid: list[float] | None = None,
    selector_epoch_selection_mode: str = SELECTOR_EPOCH_SELECTION_PROXY,
) -> dict[str, Any] | None:
    train_shot_features, train_candidate_features, train_target_scores = _concatenate_selector_training_arrays(
        train_bundles
    )
    if train_shot_features.shape[0] == 0:
        return None
    if str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM and selector_objective == SELECTOR_OBJECTIVE_BCE:
        raise ValueError("benefit_harm selector targets require group_rank selector objective")
    selector_model_name = str(selector_model)
    if selector_model_name not in SELECTOR_MODEL_CHOICES:
        raise ValueError(f"Unsupported selector_model: {selector_model!r}")
    if (
        selector_model_name in (SELECTOR_MODEL_RISK_AWARE, SELECTOR_MODEL_RISK_GUARD)
        and str(selector_target_mode) != SELECTOR_TARGET_MODE_BENEFIT_HARM
    ):
        raise ValueError("risk-aware selector models require selector_target_mode='benefit_harm'")
    selector_kwargs = {
        "shot_feature_dim": int(train_shot_features.shape[1]),
        "candidate_feature_dim": int(train_candidate_features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
    }
    if selector_model_name == SELECTOR_MODEL_RISK_AWARE:
        selector_kwargs["utility_harm_logit_weight"] = float(selector_risk_aware_harm_logit_weight)
    if bool(selector_patch_head):
        patch_slice = _candidate_local_patch_feature_slice(
            candidate_geometry_features=bool(selector_candidate_geometry_features),
            candidate_pattern_features=bool(selector_candidate_pattern_features),
            candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            candidate_local_patch_features=bool(selector_candidate_local_patch_features),
        )
        if patch_slice is None:
            raise ValueError("selector_patch_head requires --selector-candidate-local-patch-features")
        selector_kwargs.update(
            {
                "patch_feature_offset": int(patch_slice.start or 0),
                "patch_feature_dim": int(SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM),
                "patch_hidden_dim": int(selector_patch_hidden_dim),
            }
        )
    selector = _make_candidate_selector_module(
        selector_model=selector_model_name,
        selector_kwargs=selector_kwargs,
    ).to(device)
    optimizer = optim.Adam(selector.parameters(), lr=lr)
    train_loader = None
    if selector_objective == SELECTOR_OBJECTIVE_BCE:
        train_dataset = _make_selector_tensor_dataset(
            train_shot_features,
            train_candidate_features,
            train_target_scores,
        )
        train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    elif selector_objective != SELECTOR_OBJECTIVE_GROUP_RANK:
        raise ValueError(f"Unsupported selector_objective: {selector_objective!r}")
    auto_benefit_pos_weight, auto_harm_pos_weight = _selector_benefit_harm_pos_weights(train_bundles)
    effective_risk_aware_benefit_loss_weight = (
        float(selector_risk_aware_benefit_loss_weight)
        if selector_model_name == SELECTOR_MODEL_RISK_AWARE
        else 0.0
    )
    effective_risk_aware_harm_loss_weight = (
        float(selector_risk_aware_harm_loss_weight)
        if selector_model_name in (SELECTOR_MODEL_RISK_AWARE, SELECTOR_MODEL_RISK_GUARD)
        else 0.0
    )
    effective_risk_aware_benefit_pos_weight = (
        float(selector_risk_aware_benefit_pos_weight)
        if float(selector_risk_aware_benefit_pos_weight) > 0.0
        else float(auto_benefit_pos_weight)
    )
    effective_risk_aware_harm_pos_weight = (
        float(selector_risk_aware_harm_pos_weight)
        if float(selector_risk_aware_harm_pos_weight) > 0.0
        else float(auto_harm_pos_weight)
    )

    best_state: dict[str, Any] | None = None
    best_val_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    diagnostic_margin_grid = [
        float(x) for x in (selector_epoch_diagnostic_margin_grid or [])
    ]
    for epoch in range(1, epochs + 1):
        if selector_objective == SELECTOR_OBJECTIVE_BCE:
            if train_loader is None:
                raise RuntimeError("selector BCE objective requires a training loader")
            train_metrics = _train_selector_one_epoch(
                selector=selector,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
        else:
            train_metrics = _train_selector_group_rank_epoch(
                selector=selector,
                bundles=train_bundles,
                optimizer=optimizer,
                device=device,
                hard_shot_weight=selector_hard_shot_weight,
                identity_margin_loss_weight=selector_identity_margin_loss_weight,
                identity_margin=selector_identity_margin,
                harm_margin_loss_weight=selector_harm_margin_loss_weight,
                harm_margin=selector_harm_margin,
                negative_identity_margin_loss_weight=selector_negative_identity_margin_loss_weight,
                negative_identity_margin=selector_negative_identity_margin,
                benefit_harm_pairwise_loss_weight=selector_benefit_harm_pairwise_loss_weight,
                benefit_harm_pairwise_margin=selector_benefit_harm_pairwise_margin,
                positive_negative_hard_loss_weight=selector_positive_negative_hard_loss_weight,
                positive_negative_hard_margin=selector_positive_negative_hard_margin,
                cross_family_positive_negative_loss_weight=selector_cross_family_positive_negative_loss_weight,
                cross_family_positive_negative_margin=selector_cross_family_positive_negative_margin,
                risk_aware_benefit_loss_weight=effective_risk_aware_benefit_loss_weight,
                risk_aware_harm_loss_weight=effective_risk_aware_harm_loss_weight,
                risk_aware_benefit_pos_weight=effective_risk_aware_benefit_pos_weight,
                risk_aware_harm_pos_weight=effective_risk_aware_harm_pos_weight,
            )
        val_by_family: dict[str, Any] = {}
        val_logits_by_family: dict[str, np.ndarray] = {}
        for family, bundle in val_bundles_by_family.items():
            logits = _selector_logits_for_bundle(
                selector=selector,
                bundle=bundle,
                batch_size=batch_size,
                device=device,
            )
            val_logits_by_family[family] = logits
            selection_metric = _selector_selection_metric(bundle=bundle, selector_logits=logits)
            val_by_family[family] = {
                "selector_accuracy": selection_metric["selector_accuracy"],
                "mean_selected_target_score": selection_metric["mean_selected_target_score"],
                "candidate_oracle_accuracy": selection_metric["candidate_oracle_accuracy"],
                "mean_selected_edit_weight": selection_metric["mean_selected_edit_weight"],
                "fraction_with_any_selected_edit": selection_metric["fraction_with_any_selected_edit"],
            }
        proxy_val_metric = float(
            np.mean(
                [
                    float(result["selector_accuracy"] or 0.0) + 1e-3 * float(result["mean_selected_target_score"] or 0.0)
                    for result in val_by_family.values()
                ]
            )
        ) if val_by_family else 0.0
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val_by_family": val_by_family,
            "val_proxy_selection_metric": float(proxy_val_metric),
            "val_selection_metric": float(proxy_val_metric),
        }
        if diagnostic_margin_grid:
            diagnostics_by_family = {
                family: _selector_epoch_margin_diagnostics(
                    bundle=val_bundles_by_family[family],
                    selector_logits=logits,
                    emit_margin_grid=diagnostic_margin_grid,
                )
                for family, logits in val_logits_by_family.items()
            }
            epoch_record["val_margin_diagnostics_by_family"] = diagnostics_by_family
            diagnostic_selection = _selector_epoch_diagnostic_system_selection(diagnostics_by_family)
            if diagnostic_selection is not None:
                epoch_record["val_diagnostic_system_selection"] = diagnostic_selection
                if str(selector_epoch_selection_mode) == SELECTOR_EPOCH_SELECTION_DIAGNOSTIC_SYSTEM:
                    epoch_record["val_selection_metric"] = float(
                        diagnostic_selection["mean_selector_accuracy"]
                    )
        if str(selector_epoch_selection_mode) not in SELECTOR_EPOCH_SELECTION_CHOICES:
            raise ValueError(f"Unknown selector epoch selection mode: {selector_epoch_selection_mode!r}")
        val_metric = float(epoch_record["val_selection_metric"])
        epoch_history.append(epoch_record)
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(selector.state_dict()),
                "val_by_family": copy.deepcopy(val_by_family),
            }

    if best_state is None:
        return None
    selector.load_state_dict(best_state["model_state"])
    eval_by_family: dict[str, Any] = {}
    for family, bundle in val_bundles_by_family.items():
        eval_by_family[family] = _selector_system_metrics_for_subset(
            entry=entry_by_family[family],
            subset=subset_by_family[family],
            model=model,
            selector=selector,
            batch_size=batch_size,
            device=device,
            policy_specs=policy_specs,
            selector_score_edit_penalty=selector_score_edit_penalty,
            selector_target_mode=str(selector_target_mode),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
            selector_policy_candidate_mode=str(selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            motif_vocabulary=motif_vocabulary,
            local_motif_vocabulary=local_motif_vocabulary,
            local_motif_top_k=int(local_motif_top_k),
            selector_selection_mode=str(selector_selection_mode),
        )
    return {
        "selector": selector,
        "selector_kwargs": selector_kwargs,
        "epoch_history": epoch_history,
        "best_epoch": int(best_state["epoch"]),
        "best_val_selection_metric": float(best_val_metric),
        "best_val_by_family": best_state["val_by_family"],
        "eval_by_family": eval_by_family,
        "selector_model": str(selector_model_name),
        "selector_risk_aware_harm_logit_weight": float(selector_risk_aware_harm_logit_weight),
        "selector_risk_aware_benefit_loss_weight": float(effective_risk_aware_benefit_loss_weight),
        "selector_risk_aware_harm_loss_weight": float(effective_risk_aware_harm_loss_weight),
        "selector_risk_aware_benefit_pos_weight": float(effective_risk_aware_benefit_pos_weight),
        "selector_risk_aware_harm_pos_weight": float(effective_risk_aware_harm_pos_weight),
        "selector_objective": str(selector_objective),
        "selector_hard_shot_weight": float(selector_hard_shot_weight),
        "selector_identity_margin_loss_weight": float(selector_identity_margin_loss_weight),
        "selector_identity_margin": float(selector_identity_margin),
        "selector_harm_margin_loss_weight": float(selector_harm_margin_loss_weight),
        "selector_harm_margin": float(selector_harm_margin),
        "selector_negative_identity_margin_loss_weight": float(
            selector_negative_identity_margin_loss_weight
        ),
        "selector_negative_identity_margin": float(selector_negative_identity_margin),
        "selector_benefit_harm_pairwise_loss_weight": float(selector_benefit_harm_pairwise_loss_weight),
        "selector_benefit_harm_pairwise_margin": float(selector_benefit_harm_pairwise_margin),
        "selector_positive_negative_hard_loss_weight": float(
            selector_positive_negative_hard_loss_weight
        ),
        "selector_positive_negative_hard_margin": float(selector_positive_negative_hard_margin),
        "selector_cross_family_positive_negative_loss_weight": float(
            selector_cross_family_positive_negative_loss_weight
        ),
        "selector_cross_family_positive_negative_margin": float(
            selector_cross_family_positive_negative_margin
        ),
        "selector_patch_head": bool(selector_patch_head),
        "selector_patch_hidden_dim": int(selector_patch_hidden_dim),
        "policy_specs": [spec.to_dict() for spec in policy_specs],
        "selector_score_edit_penalty": float(selector_score_edit_penalty),
        "selector_target_mode": str(selector_target_mode),
        "selector_harm_weight": float(selector_harm_weight),
        "selector_miss_weight": float(selector_miss_weight),
        "selector_policy_candidate_mode": str(selector_policy_candidate_mode),
        "selector_candidate_geometry_features": bool(selector_candidate_geometry_features),
        "selector_candidate_pattern_features": bool(selector_candidate_pattern_features),
        "selector_candidate_local_evidence_features": bool(selector_candidate_local_evidence_features),
        "selector_candidate_local_patch_features": bool(selector_candidate_local_patch_features),
        "selector_selection_mode": str(selector_selection_mode),
        "selector_epoch_diagnostic_margin_grid": diagnostic_margin_grid,
        "selector_epoch_selection_mode": str(selector_epoch_selection_mode),
        "candidate_motif_vocab_num_classes": (
            int(motif_vocabulary.mask_table.shape[0]) if motif_vocabulary is not None else 0
        ),
        "candidate_local_motif_num_patterns": (
            int(len(local_motif_vocabulary.offset_patterns)) if local_motif_vocabulary is not None else 0
        ),
        "candidate_local_motif_top_k": int(local_motif_top_k),
    }


def _grid_search_selector_emit_margin_by_family(
    *,
    entry_by_family: dict[str, PreparedEditFamily],
    subset_by_family: dict[str, dict[str, np.ndarray]],
    model: nn.Module,
    selector: nn.Module,
    batch_size: int,
    device: torch.device,
    policy_specs: list[CandidatePolicySpec],
    selector_score_edit_penalty: float,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    emit_margin_grid: list[float],
    nonzero_bias_grid: list[float] | None = None,
    transition_prior_head: nn.Module | None = None,
    transition_prior_weight_grid: list[float] | None = None,
    transition_compat_top_k_grid: list[int] | None = None,
    candidate_compatibility_head: nn.Module | None = None,
    candidate_compat_threshold_grid: list[float] | None = None,
    candidate_compat_top_k_grid: list[int] | None = None,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_vocabulary: LocalMotifVocabulary | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    selector_selection_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
) -> dict[str, Any]:
    best_by_family: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    profile_records: list[dict[str, Any]] = []
    bias_grid = nonzero_bias_grid if nonzero_bias_grid is not None else list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID)
    prior_weight_grid = (
        transition_prior_weight_grid
        if transition_prior_weight_grid is not None
        else list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID)
    )
    compat_top_k_grid = (
        transition_compat_top_k_grid
        if transition_compat_top_k_grid is not None
        else list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID)
    )
    compat_threshold_grid = (
        candidate_compat_threshold_grid
        if candidate_compat_threshold_grid is not None
        else list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID)
    )
    compat_candidate_top_k_grid = (
        candidate_compat_top_k_grid
        if candidate_compat_top_k_grid is not None
        else list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID)
    )
    for candidate_compat_top_k in compat_candidate_top_k_grid:
        for candidate_compat_threshold in compat_threshold_grid:
            for transition_compat_top_k in compat_top_k_grid:
                for transition_prior_weight in prior_weight_grid:
                    for nonzero_bias in bias_grid:
                        for emit_margin in emit_margin_grid:
                            by_family = {
                                family: _selector_system_metrics_for_subset(
                                    entry=entry_by_family[family],
                                    subset=subset,
                                    model=model,
                                    selector=selector,
                                    batch_size=batch_size,
                                    device=device,
                                    policy_specs=policy_specs,
                                    selector_score_edit_penalty=selector_score_edit_penalty,
                                    selector_target_mode=str(selector_target_mode),
                                    selector_harm_weight=float(selector_harm_weight),
                                    selector_miss_weight=float(selector_miss_weight),
                                    selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                                    selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                                    selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                                    selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                                    selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                                    motif_vocabulary=motif_vocabulary,
                                    local_motif_vocabulary=local_motif_vocabulary,
                                    local_motif_top_k=int(local_motif_top_k),
                                    selector_selection_mode=str(selector_selection_mode),
                                    selector_emit_margin=float(emit_margin),
                                    selector_nonzero_bias=float(nonzero_bias),
                                    transition_prior_head=transition_prior_head,
                                    selector_transition_prior_weight=float(transition_prior_weight),
                                    selector_transition_compat_top_k=int(transition_compat_top_k),
                                    candidate_compatibility_head=candidate_compatibility_head,
                                    selector_candidate_compat_threshold=float(candidate_compat_threshold),
                                    selector_candidate_compat_top_k=int(candidate_compat_top_k),
                                )
                                for family, subset in subset_by_family.items()
                            }
                            profile_records.append(_selector_margin_profile_record(by_family))
                            mean_metric = _mean_system_metric(list(by_family.values()))
                            mean_harmed = float(
                                np.mean([
                                    float(result["change_summary"].get("num_harmed_vs_baseline") or 0.0)
                                    for result in by_family.values()
                                ])
                            )
                            mean_weight = float(
                                np.mean([
                                    float(result["change_summary"].get("mean_predicted_edit_weight") or 0.0)
                                    for result in by_family.values()
                                ])
                            )
                            key = (float(mean_metric), -mean_harmed, -mean_weight)
                            if best_key is None or key > best_key:
                                best_key = key
                                best_by_family = by_family
    if best_by_family is None:
        raise RuntimeError("Selector emit-margin grid search produced no family results")
    _attach_selector_margin_profile(best_by_family, profile_records)
    return best_by_family


def _grid_search_routed_selector_policy_by_family(
    *,
    entry_by_family: dict[str, PreparedEditFamily],
    subset_by_family: dict[str, dict[str, np.ndarray]],
    model: nn.Module,
    selector: nn.Module,
    router: nn.Module,
    batch_size: int,
    device: torch.device,
    policy_specs: list[CandidatePolicySpec],
    selector_score_edit_penalty: float,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    router_threshold_grid: list[float],
    selector_emit_margin_grid: list[float],
    selector_nonzero_bias_grid: list[float] | None = None,
    transition_prior_head: nn.Module | None = None,
    transition_prior_weight_grid: list[float] | None = None,
    transition_compat_top_k_grid: list[int] | None = None,
    candidate_compatibility_head: nn.Module | None = None,
    candidate_compat_threshold_grid: list[float] | None = None,
    motif_vocabulary: MotifVocabulary | None = None,
    local_motif_vocabulary: LocalMotifVocabulary | None = None,
    local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    selector_selection_mode: str = SELECTION_MODE_LOCAL_MOTIF_ROUTER,
) -> dict[str, Any]:
    best_by_family: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    profile_records: list[dict[str, Any]] = []
    bias_grid = selector_nonzero_bias_grid if selector_nonzero_bias_grid is not None else list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID)
    prior_weight_grid = (
        transition_prior_weight_grid
        if transition_prior_weight_grid is not None
        else list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID)
    )
    compat_top_k_grid = (
        transition_compat_top_k_grid
        if transition_compat_top_k_grid is not None
        else list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID)
    )
    compat_threshold_grid = (
        candidate_compat_threshold_grid
        if candidate_compat_threshold_grid is not None
        else list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID)
    )
    compat_candidate_top_k_grid = (
        candidate_compat_top_k_grid
        if candidate_compat_top_k_grid is not None
        else list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID)
    )
    for selector_candidate_compat_top_k in compat_candidate_top_k_grid:
        for selector_candidate_compat_threshold in compat_threshold_grid:
            for selector_transition_compat_top_k in compat_top_k_grid:
                for selector_transition_prior_weight in prior_weight_grid:
                    for router_threshold in router_threshold_grid:
                        for selector_nonzero_bias in bias_grid:
                            for selector_emit_margin in selector_emit_margin_grid:
                                by_family = {
                                    family: _selector_system_metrics_for_subset(
                                        entry=entry_by_family[family],
                                        subset=subset,
                                        model=model,
                                        selector=selector,
                                        batch_size=batch_size,
                                        device=device,
                                        policy_specs=policy_specs,
                                        selector_score_edit_penalty=selector_score_edit_penalty,
                                        selector_target_mode=str(selector_target_mode),
                                        selector_harm_weight=float(selector_harm_weight),
                                        selector_miss_weight=float(selector_miss_weight),
                                        selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                                        selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                                        selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                                        selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                                        selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                                        motif_vocabulary=motif_vocabulary,
                                        local_motif_vocabulary=local_motif_vocabulary,
                                        local_motif_top_k=int(local_motif_top_k),
                                        selector_selection_mode=str(selector_selection_mode),
                                        selector_emit_margin=float(selector_emit_margin),
                                        selector_nonzero_bias=float(selector_nonzero_bias),
                                        transition_prior_head=transition_prior_head,
                                        selector_transition_prior_weight=float(selector_transition_prior_weight),
                                        selector_transition_compat_top_k=int(selector_transition_compat_top_k),
                                        candidate_compatibility_head=candidate_compatibility_head,
                                        selector_candidate_compat_threshold=float(selector_candidate_compat_threshold),
                                        selector_candidate_compat_top_k=int(selector_candidate_compat_top_k),
                                        router=router,
                                        router_threshold=float(router_threshold),
                                        routed_nonzero_only=True,
                                    )
                                    for family, subset in subset_by_family.items()
                                }
                                profile_records.append(_selector_margin_profile_record(by_family))
                                mean_metric = _mean_system_metric(list(by_family.values()))
                                mean_harmed = float(
                                    np.mean([
                                        float(result["change_summary"].get("num_harmed_vs_baseline") or 0.0)
                                        for result in by_family.values()
                                    ])
                                )
                                mean_weight = float(
                                    np.mean([
                                        float(result["change_summary"].get("mean_predicted_edit_weight") or 0.0)
                                        for result in by_family.values()
                                    ])
                                )
                                key = (float(mean_metric), -mean_harmed, -mean_weight)
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best_by_family = by_family
    if best_by_family is None:
        raise RuntimeError("Routed selector grid search produced no family results")
    _attach_selector_margin_profile(best_by_family, profile_records)
    return best_by_family


def _system_metrics_for_subset(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    needs_edit_threshold: float,
    edit_threshold: float,
    max_predicted_edit_weight: int,
) -> dict[str, Any]:
    outputs = _collect_outputs(model=model, x=subset["x"], batch_size=batch_size, device=device)
    edit_logits = outputs["edit_logits"]
    needs_edit_logits = outputs["needs_edit_logits"]
    edit_probs = common._sigmoid_np(edit_logits)
    needs_edit_probs = common._sigmoid_np(needs_edit_logits)

    _edited_detector_events, chosen_edit_mask, _predicted_edit_weight = _apply_predicted_edits(
        detector_events=subset["detector_events"],
        edit_probs_volume=edit_probs,
        needs_edit_probs=needs_edit_probs,
        detector_time_index=entry.detector_time_index,
        row_index_by_detector=entry.row_index_by_detector,
        col_index_by_detector=entry.col_index_by_detector,
        needs_edit_threshold=needs_edit_threshold,
        edit_threshold=edit_threshold,
        max_predicted_edit_weight=max_predicted_edit_weight,
    )
    needs_metrics = _binary_metrics_from_probs(
        needs_edit_probs,
        subset["needs_edit"],
        threshold=needs_edit_threshold,
    )
    edit_metrics = _masked_edit_binary_metrics(
        edit_probs,
        subset["edit_target_volume"],
        valid_mask_volume=entry.valid_mask_volume,
        known_mask=subset["edit_target_known"],
        threshold=edit_threshold,
    )
    return _system_metrics_from_chosen_edit_mask(
        entry=entry,
        subset=subset,
        chosen_edit_mask=chosen_edit_mask,
        decision_payload={
            "selection_mode": SELECTION_MODE_GLOBAL_POLICY,
            "needs_edit_threshold": float(needs_edit_threshold),
            "edit_threshold": float(edit_threshold),
            "max_predicted_edit_weight": int(max_predicted_edit_weight),
        },
        needs_metrics=needs_metrics,
        edit_metrics=edit_metrics,
    )


def _grid_search_decision_policy(
    *,
    entry: PreparedEditFamily,
    subset: dict[str, np.ndarray],
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    needs_edit_threshold_grid: list[float],
    edit_threshold_grid: list[float],
    max_edit_weight_grid: list[int],
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for needs_thr in needs_edit_threshold_grid:
        for edit_thr in edit_threshold_grid:
            for max_weight in max_edit_weight_grid:
                result = _system_metrics_for_subset(
                    entry=entry,
                    subset=subset,
                    model=model,
                    batch_size=batch_size,
                    device=device,
                    needs_edit_threshold=needs_thr,
                    edit_threshold=edit_thr,
                    max_predicted_edit_weight=max_weight,
                )
                edited = result["edited_pymatching"]
                change = result["change_summary"]
                score = (
                    float(edited.get("accuracy") or 0.0),
                    float(edited.get("macro_f1") or 0.0),
                    float(change.get("num_improved_over_baseline") or 0.0)
                    - float(change.get("num_harmed_vs_baseline") or 0.0),
                    -float(change.get("num_harmed_vs_baseline") or 0.0),
                    -float(change.get("mean_predicted_edit_weight") or 0.0),
                    -float(change.get("fraction_with_any_predicted_edit") or 0.0),
                )
                if best_key is None or score > best_key:
                    best_key = score
                    best_result = result
    if best_result is None:
        raise RuntimeError("Decision grid search produced no candidates")
    return best_result


def _family_system_metric_value(result: dict[str, Any]) -> float:
    edited = result["edited_pymatching"]
    accuracy = edited.get("accuracy")
    macro_f1 = edited.get("macro_f1")
    accuracy_f = float(accuracy if accuracy is not None else 0.0)
    macro_f1_f = float(macro_f1 if macro_f1 is not None else 0.0)
    return accuracy_f + 1e-3 * macro_f1_f


def _mean_system_metric(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([_family_system_metric_value(result) for result in results]))


SELECTOR_MARGIN_PROFILE_DECISION_KEYS = (
    "selector_nonzero_bias",
    "selector_transition_prior_weight",
    "selector_transition_compat_top_k",
    "selector_candidate_compat_threshold",
    "selector_candidate_compat_top_k",
    "router_threshold",
)


def _selector_margin_profile_key_from_mapping(mapping: dict[str, Any]) -> tuple[Any, ...]:
    key: list[Any] = []
    for name in SELECTOR_MARGIN_PROFILE_DECISION_KEYS:
        value = mapping.get(name)
        if value is None:
            key.append(None)
        elif name.endswith("_top_k"):
            key.append(int(value))
        else:
            key.append(round(float(value), 12))
    return tuple(key)


def _selector_margin_profile_record(by_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first_result = next(iter(by_family.values()))
    decision = dict(first_result.get("decision") or {})
    total_examples = 0
    total_improved = 0
    total_harmed = 0
    total_nonzero = 0
    family_rows: dict[str, Any] = {}
    for family, result in by_family.items():
        change = result.get("change_summary") or {}
        num_examples = int(change.get("num_examples") or 0)
        improved = int(change.get("num_improved_over_baseline") or 0)
        harmed = int(change.get("num_harmed_vs_baseline") or 0)
        nonzero = _result_nonzero_count(result)
        total_examples += num_examples
        total_improved += improved
        total_harmed += harmed
        total_nonzero += nonzero
        family_rows[str(family)] = {
            "num_examples": int(num_examples),
            "validation_delta_over_no_edit": (
                float((improved - harmed) / num_examples) if num_examples > 0 else None
            ),
            "selected_nonzero": int(nonzero),
            "improved": int(improved),
            "harmed": int(harmed),
        }
    record = {
        "selector_emit_margin": float(decision.get("selector_emit_margin") or 0.0),
        "validation_delta_over_no_edit": (
            float((total_improved - total_harmed) / total_examples)
            if total_examples > 0
            else None
        ),
        "selected_nonzero": int(total_nonzero),
        "improved": int(total_improved),
        "harmed": int(total_harmed),
        "num_examples": int(total_examples),
        "by_family": family_rows,
    }
    for name in SELECTOR_MARGIN_PROFILE_DECISION_KEYS:
        if name in decision:
            value = decision.get(name)
            if value is None:
                record[name] = None
            elif name.endswith("_top_k"):
                record[name] = int(value)
            else:
                record[name] = float(value)
    return record


def _attach_selector_margin_profile(
    by_family: dict[str, dict[str, Any]],
    profile_records: list[dict[str, Any]],
) -> None:
    if not by_family or not profile_records:
        return
    selected_decision = dict(next(iter(by_family.values())).get("decision") or {})
    selected_key = _selector_margin_profile_key_from_mapping(selected_decision)
    selected_profile = [
        dict(record)
        for record in profile_records
        if _selector_margin_profile_key_from_mapping(record) == selected_key
    ]
    selected_profile.sort(key=lambda record: float(record.get("selector_emit_margin") or 0.0))
    for result in by_family.values():
        decision = result.setdefault("decision", {})
        decision["selector_margin_profile"] = copy.deepcopy(selected_profile)


def _selected_inference_mode(
    *,
    global_metric: float,
    selector_metric: float | None,
    requested_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    min_delta: float = DEFAULT_SELECTOR_ADOPTION_MIN_DELTA,
) -> str:
    if selector_metric is None:
        return SELECTION_MODE_GLOBAL_POLICY
    if float(selector_metric) + 1e-12 >= float(global_metric) + float(min_delta):
        return str(requested_mode)
    return SELECTION_MODE_GLOBAL_POLICY


def _result_nonzero_count(result: dict[str, Any] | None) -> int:
    if result is None:
        return 0
    change_summary = result.get("change_summary") or {}
    histogram = change_summary.get("predicted_edit_weight_histogram") or {}
    total = 0
    for key, value in histogram.items():
        try:
            weight = float(key)
        except (TypeError, ValueError):
            weight = 0.0
        if weight != 0.0:
            total += int(value)
    return int(total)


def _results_nonzero_count(results: list[dict[str, Any]] | None) -> int:
    if not results:
        return 0
    return int(sum(_result_nonzero_count(result) for result in results))


def _results_change_count(results: list[dict[str, Any]] | None, key: str) -> int:
    if not results:
        return 0
    total = 0
    for result in results:
        change_summary = result.get("change_summary") or {}
        total += int(change_summary.get(key) or 0)
    return int(total)


def _selector_emit_margin_from_results(results: list[dict[str, Any]] | None) -> float | None:
    if not results:
        return None
    for result in results:
        decision = result.get("decision") or {}
        if "selector_emit_margin" in decision and decision["selector_emit_margin"] is not None:
            return float(decision["selector_emit_margin"])
    return None


def _selector_margin_profile_from_results(results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not results:
        return []
    for result in results:
        decision = result.get("decision") or {}
        profile = decision.get("selector_margin_profile")
        if isinstance(profile, list):
            return [dict(row) for row in profile if isinstance(row, dict)]
    return []


def _selector_positive_plateau_summary(
    *,
    selector_results: list[dict[str, Any]] | None,
    selector_margin: float,
    positive_delta: float,
) -> dict[str, Any]:
    profile = _selector_margin_profile_from_results(selector_results)
    higher_positive = []
    for row in profile:
        margin = row.get("selector_emit_margin")
        delta = row.get("validation_delta_over_no_edit")
        if margin is None or delta is None:
            continue
        if float(margin) > float(selector_margin) + 1e-12 and float(delta) >= float(positive_delta):
            higher_positive.append(
                {
                    "selector_emit_margin": float(margin),
                    "validation_delta_over_no_edit": float(delta),
                    "selected_nonzero": int(row.get("selected_nonzero") or 0),
                    "improved": int(row.get("improved") or 0),
                    "harmed": int(row.get("harmed") or 0),
                }
            )
    return {
        "positive_delta": float(positive_delta),
        "selected_selector_emit_margin": float(selector_margin),
        "has_higher_positive_margin": bool(higher_positive),
        "higher_positive_margins": higher_positive,
        "profile": profile,
    }


def _candidate_first_safety_adoption_mode(
    *,
    no_edit_metric: float,
    global_metric: float | None,
    selector_metric: float | None,
    selector_results: list[dict[str, Any]] | None,
    requested_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    strong_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_STRONG_DELTA,
    positive_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_DELTA,
    positive_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MARGIN_FLOOR,
    positive_max_harmed: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED,
    positive_max_margin: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_MARGIN,
    positive_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MIN_NONZERO,
    positive_plateau_guard: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_PLATEAU_GUARD,
    tie_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_DELTA,
    tie_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MARGIN_FLOOR,
    tie_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_NONZERO,
    allow_global: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_ALLOW_GLOBAL,
    global_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_GLOBAL_MIN_DELTA,
) -> tuple[str, dict[str, Any]]:
    base = float(no_edit_metric)
    global_delta = None if global_metric is None else float(global_metric) - base
    selector_delta = None if selector_metric is None else float(selector_metric) - base
    selector_nonzero = _results_nonzero_count(selector_results)
    selector_margin_raw = _selector_emit_margin_from_results(selector_results)
    selector_margin = float(selector_margin_raw or 0.0)
    selector_improved = _results_change_count(
        selector_results,
        "num_improved_over_baseline",
    )
    selector_harmed = _results_change_count(
        selector_results,
        "num_harmed_vs_baseline",
    )
    selector_plateau_summary = _selector_positive_plateau_summary(
        selector_results=selector_results,
        selector_margin=selector_margin,
        positive_delta=float(positive_delta),
    )

    mode = SELECTION_MODE_RAW_NO_EDIT
    reason = "default_no_edit"
    if selector_delta is not None and selector_nonzero > 0:
        if selector_delta >= float(strong_delta):
            mode = str(requested_mode)
            reason = "candidate_strong_validation_delta"
        else:
            positive_harm_guard_failed = (
                int(positive_max_harmed) >= 0
                and selector_delta >= float(positive_delta)
                and selector_harmed > int(positive_max_harmed)
            )
            positive_margin_guard_failed = (
                float(positive_max_margin) >= 0.0
                and selector_delta >= float(positive_delta)
                and selector_margin > float(positive_max_margin)
            )
            positive_plateau_guard_failed = (
                bool(positive_plateau_guard)
                and selector_delta >= float(positive_delta)
                and bool(selector_plateau_summary.get("has_higher_positive_margin"))
            )
            positive_support_guard_failed = (
                int(positive_min_nonzero) > 0
                and selector_delta >= float(positive_delta)
                and selector_nonzero < int(positive_min_nonzero)
            )
            positive_guard_failed = (
                positive_harm_guard_failed
                or positive_margin_guard_failed
                or positive_plateau_guard_failed
                or positive_support_guard_failed
            )
            if (
                selector_delta >= float(positive_delta)
                and selector_margin >= float(positive_margin_floor)
                and not positive_guard_failed
            ):
                mode = str(requested_mode)
                reason = "candidate_positive_delta_with_margin"
            elif (
                selector_delta >= float(tie_min_delta)
                and selector_margin >= float(tie_margin_floor)
                and selector_nonzero >= int(tie_min_nonzero)
                and not positive_guard_failed
            ):
                mode = str(requested_mode)
                reason = "candidate_tie_with_high_margin_evidence"
            elif positive_harm_guard_failed:
                reason = "candidate_positive_delta_harm_guard"
            elif positive_margin_guard_failed:
                reason = "candidate_positive_delta_margin_guard"
            elif positive_plateau_guard_failed:
                reason = "candidate_positive_delta_plateau_guard"
            elif positive_support_guard_failed:
                reason = "candidate_positive_delta_support_guard"

    if (
        mode == SELECTION_MODE_RAW_NO_EDIT
        and bool(allow_global)
        and global_delta is not None
        and global_delta >= float(global_min_delta)
    ):
        mode = SELECTION_MODE_GLOBAL_POLICY
        reason = "global_delta_clears_guard"

    decision = {
        "policy": SELECTOR_ADOPTION_POLICY_CANDIDATE_FIRST_SAFETY,
        "requested_mode": str(requested_mode),
        "selected_mode": str(mode),
        "reason": str(reason),
        "no_edit_metric": float(no_edit_metric),
        "global_metric": (None if global_metric is None else float(global_metric)),
        "selector_metric": (None if selector_metric is None else float(selector_metric)),
        "global_delta_over_no_edit": global_delta,
        "selector_delta_over_no_edit": selector_delta,
        "selector_nonzero_count": int(selector_nonzero),
        "selector_improved_over_baseline": int(selector_improved),
        "selector_harmed_vs_baseline": int(selector_harmed),
        "selector_emit_margin": selector_margin_raw,
        "thresholds": {
            "strong_delta": float(strong_delta),
            "positive_delta": float(positive_delta),
            "positive_margin_floor": float(positive_margin_floor),
            "positive_max_harmed": int(positive_max_harmed),
            "positive_max_margin": float(positive_max_margin),
            "positive_min_nonzero": int(positive_min_nonzero),
            "positive_plateau_guard": bool(positive_plateau_guard),
            "tie_min_delta": float(tie_min_delta),
            "tie_margin_floor": float(tie_margin_floor),
            "tie_min_nonzero": int(tie_min_nonzero),
            "allow_global": bool(allow_global),
            "global_min_delta": float(global_min_delta),
        },
        "selector_positive_plateau": selector_plateau_summary,
    }
    return str(mode), decision


def _apply_no_edit_guardrail_to_mode(
    *,
    selected_mode: str,
    selected_metric: float,
    no_edit_metric: float,
    enabled: bool = DEFAULT_SELECTED_NO_EDIT_GUARDRAIL,
    min_delta: float = DEFAULT_SELECTED_NO_EDIT_MIN_DELTA,
) -> str:
    if not bool(enabled):
        return str(selected_mode)
    if str(selected_mode) == SELECTION_MODE_GLOBAL_POLICY:
        return SELECTION_MODE_RAW_NO_EDIT
    if float(no_edit_metric) + float(min_delta) + 1e-12 >= float(selected_metric):
        return SELECTION_MODE_RAW_NO_EDIT
    return str(selected_mode)


def _build_training_config(
    *,
    edit_target_volume: np.ndarray,
    needs_edit: np.ndarray,
    edit_target_known: np.ndarray,
    valid_mask_volume: np.ndarray,
    shot_sample_weights: np.ndarray | None = None,
    edit_supervision_mode: str = EDIT_SUPERVISION_MODE_ALL_KNOWN,
) -> dict[str, Any]:
    supervision_mask = _build_edit_supervision_mask(
        known_mask=edit_target_known,
        needs_edit=needs_edit,
        mode=edit_supervision_mode,
    )
    decision_aware_mask = (
        (np.asarray(edit_target_known, dtype=np.float32).reshape(-1) >= 0.5)
        & (np.asarray(needs_edit, dtype=np.float32).reshape(-1) >= 0.5)
    ).astype(np.float32, copy=False)
    known_weight = supervision_mask[:, None, None, None] * valid_mask_volume[None, :, :, :]
    edit_pos_weight = _compute_pos_weight_from_targets(edit_target_volume, known_weight)
    needs_pos_weight = _compute_pos_weight_from_targets(
        needs_edit[:, None],
        np.ones((needs_edit.shape[0], 1), dtype=np.float32),
    )
    out = {
        "edit_supervision_mode": str(edit_supervision_mode),
        "edit_pos_weight": float(edit_pos_weight),
        "needs_pos_weight": float(needs_pos_weight),
        "known_supervision_fraction": float(np.mean(edit_target_known)) if edit_target_known.size else None,
        "edit_supervision_fraction": float(np.mean(supervision_mask)) if supervision_mask.size else None,
        "needs_edit_fraction": float(np.mean(needs_edit)) if needs_edit.size else None,
        "decision_aware_hard_known_fraction": (
            float(np.mean(decision_aware_mask)) if decision_aware_mask.size else None
        ),
        "edit_positive_fraction_known": (
            float(np.sum(edit_target_volume * known_weight) / max(np.sum(known_weight), 1.0))
            if known_weight.size
            else None
        ),
    }
    if shot_sample_weights is not None:
        weights = np.asarray(shot_sample_weights, dtype=np.float32).reshape(-1)
        out["shot_sample_weight_stats"] = {
            "min": float(weights.min()) if weights.size else None,
            "max": float(weights.max()) if weights.size else None,
            "mean": float(weights.mean()) if weights.size else None,
        }
        out["hard_shot_sampling"] = {
            "fraction_weight_gt_1": float(np.mean(weights > 1.0)) if weights.size else None,
            "fraction_weight_eq_1": float(np.mean(np.isclose(weights, 1.0))) if weights.size else None,
        }
    else:
        out["shot_sample_weight_stats"] = None
        out["hard_shot_sampling"] = None
    return out


def _train_one_epoch(
    *,
    model: nn.Module,
    loader: Any,
    optimizer: Any,
    device: torch.device,
    valid_mask_volume: torch.Tensor,
    edit_pos_weight: torch.Tensor | None,
    needs_pos_weight: torch.Tensor | None,
    needs_edit_loss_weight: float,
    sparsity_loss_weight: float,
    decision_aware_loss_weight: float,
    decision_aware_margin: float,
    action_motif_loss_weight: float,
    action_motif_identity_margin: float,
    action_motif_mask_table: torch.Tensor | None,
    action_motif_detector_time_index: torch.Tensor | None,
    action_motif_row_index_by_detector: torch.Tensor | None,
    action_motif_col_index_by_detector: torch.Tensor | None,
    edit_supervision_mode: str,
) -> dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    edit_loss_sum = 0.0
    needs_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    decision_aware_loss_sum = 0.0
    decision_aware_advantage_sum = 0.0
    decision_aware_count_sum = 0.0
    action_motif_loss_sum = 0.0
    action_motif_ce_loss_sum = 0.0
    action_motif_identity_margin_loss_sum = 0.0
    action_motif_accuracy_sum = 0.0
    action_motif_active_count_sum = 0.0
    action_motif_nonzero_active_count_sum = 0.0
    action_motif_identity_gap_sum = 0.0
    action_motif_identity_gap_count_sum = 0.0
    count = 0
    for xb, y_edit, y_needs, y_known, y_action_motif_label, y_action_motif_active in loader:
        xb = xb.to(device)
        y_edit = y_edit.to(device)
        y_needs = y_needs.to(device)
        y_known = y_known.to(device)
        y_action_motif_label = y_action_motif_label.to(device)
        y_action_motif_active = y_action_motif_active.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        if edit_supervision_mode == EDIT_SUPERVISION_MODE_ALL_KNOWN:
            supervision_mask = y_known
        elif edit_supervision_mode == EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY:
            supervision_mask = y_known * (y_needs >= 0.5).to(dtype=y_known.dtype)
        else:
            raise ValueError(f"Unsupported edit_supervision_mode: {edit_supervision_mode!r}")
        edit_loss = _masked_edit_loss(
            outputs["edit_logits"],
            y_edit,
            valid_mask_volume=valid_mask_volume,
            known_mask=supervision_mask,
            pos_weight=edit_pos_weight,
        )
        needs_loss = F.binary_cross_entropy_with_logits(
            outputs["needs_edit_logits"],
            y_needs,
            pos_weight=needs_pos_weight,
        )
        sparsity_loss = (
            torch.sigmoid(outputs["edit_logits"]) * valid_mask_volume[None, :, :, :]
        ).mean()
        decision_aware_loss, decision_aware_advantage, decision_aware_mask = _decision_aware_ranking_loss(
            edit_logits=outputs["edit_logits"],
            needs_edit_logits=outputs["needs_edit_logits"],
            target_volume=y_edit,
            known_mask=y_known,
            needs_edit_target=y_needs,
            valid_mask_volume=valid_mask_volume,
            margin=decision_aware_margin,
        )
        action_motif_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        action_motif_metrics: dict[str, float | None] = {
            "action_motif_ce_loss": None,
            "action_motif_identity_margin_loss": None,
            "action_motif_accuracy": None,
            "action_motif_active_count": 0.0,
            "action_motif_nonzero_active_fraction": None,
            "action_motif_mean_identity_gap": None,
        }
        if (
            float(action_motif_loss_weight) > 0.0
            and action_motif_mask_table is not None
            and action_motif_detector_time_index is not None
            and action_motif_row_index_by_detector is not None
            and action_motif_col_index_by_detector is not None
        ):
            action_motif_loss, action_motif_metrics = _action_motif_competition_loss(
                edit_logits=outputs["edit_logits"],
                needs_edit_logits=outputs["needs_edit_logits"],
                action_motif_label=y_action_motif_label,
                action_motif_active=y_action_motif_active,
                motif_mask_table=action_motif_mask_table,
                detector_time_index=action_motif_detector_time_index,
                row_index_by_detector=action_motif_row_index_by_detector,
                col_index_by_detector=action_motif_col_index_by_detector,
                identity_margin=action_motif_identity_margin,
            )
        total_loss = (
            edit_loss
            + float(needs_edit_loss_weight) * needs_loss
            + float(sparsity_loss_weight) * sparsity_loss
            + float(decision_aware_loss_weight) * decision_aware_loss
            + float(action_motif_loss_weight) * action_motif_loss
        )
        total_loss.backward()
        optimizer.step()

        batch_count = int(xb.shape[0])
        total_loss_sum += float(total_loss.item()) * batch_count
        edit_loss_sum += float(edit_loss.item()) * batch_count
        needs_loss_sum += float(needs_loss.item()) * batch_count
        sparsity_loss_sum += float(sparsity_loss.item()) * batch_count
        decision_aware_loss_sum += float(decision_aware_loss.item()) * batch_count
        decision_aware_count_sum += float(decision_aware_mask.sum().item())
        if float(decision_aware_mask.sum().item()) > 0.0:
            masked_advantage = decision_aware_advantage * decision_aware_mask
            decision_aware_advantage_sum += float(masked_advantage.sum().item())
        active_count = float(action_motif_metrics["action_motif_active_count"] or 0.0)
        action_motif_loss_sum += float(action_motif_loss.item()) * batch_count
        if action_motif_metrics["action_motif_ce_loss"] is not None:
            action_motif_ce_loss_sum += float(action_motif_metrics["action_motif_ce_loss"]) * active_count
        if action_motif_metrics["action_motif_identity_margin_loss"] is not None:
            action_motif_identity_margin_loss_sum += float(action_motif_metrics["action_motif_identity_margin_loss"]) * active_count
        if action_motif_metrics["action_motif_accuracy"] is not None:
            action_motif_accuracy_sum += float(action_motif_metrics["action_motif_accuracy"]) * active_count
        action_motif_active_count_sum += active_count
        nonzero_fraction = action_motif_metrics["action_motif_nonzero_active_fraction"]
        if nonzero_fraction is not None:
            action_motif_nonzero_active_count_sum += float(nonzero_fraction) * active_count
        mean_identity_gap = action_motif_metrics["action_motif_mean_identity_gap"]
        if mean_identity_gap is not None:
            nonzero_count = float(nonzero_fraction or 0.0) * active_count
            action_motif_identity_gap_sum += float(mean_identity_gap) * nonzero_count
            action_motif_identity_gap_count_sum += nonzero_count
        count += batch_count

    return {
        "total_loss": float(total_loss_sum / count) if count else None,
        "edit_bce_loss": float(edit_loss_sum / count) if count else None,
        "needs_edit_bce_loss": float(needs_loss_sum / count) if count else None,
        "sparsity_loss": float(sparsity_loss_sum / count) if count else None,
        "decision_aware_loss": float(decision_aware_loss_sum / count) if count else None,
        "decision_aware_margin": float(decision_aware_margin),
        "decision_aware_hard_known_count": float(decision_aware_count_sum),
        "decision_aware_mean_advantage": (
            float(decision_aware_advantage_sum / decision_aware_count_sum)
            if decision_aware_count_sum > 0.0
            else None
        ),
        "action_motif_loss": float(action_motif_loss_sum / count) if count else None,
        "action_motif_loss_weight": float(action_motif_loss_weight),
        "action_motif_ce_loss": (
            float(action_motif_ce_loss_sum / action_motif_active_count_sum)
            if action_motif_active_count_sum > 0.0
            else None
        ),
        "action_motif_identity_margin_loss": (
            float(action_motif_identity_margin_loss_sum / action_motif_active_count_sum)
            if action_motif_active_count_sum > 0.0
            else None
        ),
        "action_motif_accuracy": (
            float(action_motif_accuracy_sum / action_motif_active_count_sum)
            if action_motif_active_count_sum > 0.0
            else None
        ),
        "action_motif_active_count": float(action_motif_active_count_sum),
        "action_motif_nonzero_active_fraction": (
            float(action_motif_nonzero_active_count_sum / action_motif_active_count_sum)
            if action_motif_active_count_sum > 0.0
            else None
        ),
        "action_motif_mean_identity_gap": (
            float(action_motif_identity_gap_sum / action_motif_identity_gap_count_sum)
            if action_motif_identity_gap_count_sum > 0.0
            else None
        ),
    }


def _make_loader(dataset: Any, *, batch_size: int, shuffle: bool) -> Any:
    return common._make_loader(dataset, batch_size=batch_size, shuffle=shuffle)


def _make_weighted_loader(dataset: Any, *, batch_size: int, sample_weights: np.ndarray) -> Any:
    weights = torch.as_tensor(np.asarray(sample_weights, dtype=np.float64))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=int(weights.shape[0]),
        replacement=True,
    )
    return common.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def train_family_dir(
    *,
    family_dir: Path,
    checkpoint_out: Path,
    out_json: Path,
    fill_value: float = -0.5,
    max_shots: int | None = None,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    hidden_channels: int = 24,
    num_blocks: int = 3,
    dense_hidden_dim: int = 64,
    dropout: float = 0.1,
    needs_edit_loss_weight: float = 0.5,
    sparsity_loss_weight: float = 0.01,
    decision_aware_loss_weight: float = DEFAULT_DECISION_AWARE_LOSS_WEIGHT,
    decision_aware_margin: float = DEFAULT_DECISION_AWARE_MARGIN,
    hard_shot_solved_weight: float = DEFAULT_HARD_SHOT_SOLVED_WEIGHT,
    hard_shot_unsolved_weight: float = DEFAULT_HARD_SHOT_UNSOLVED_WEIGHT,
    edit_supervision_mode: str = EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY,
    selection_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    selector_hidden_dim: int = DEFAULT_SELECTOR_HIDDEN_DIM,
    selector_epochs: int = DEFAULT_SELECTOR_EPOCHS,
    selector_lr: float = DEFAULT_SELECTOR_LR,
    selector_objective: str = SELECTOR_OBJECTIVE_GROUP_RANK,
    selector_hard_shot_weight: float = DEFAULT_SELECTOR_HARD_SHOT_WEIGHT,
    selector_identity_margin_loss_weight: float = DEFAULT_SELECTOR_IDENTITY_MARGIN_LOSS_WEIGHT,
    selector_identity_margin: float = DEFAULT_SELECTOR_IDENTITY_MARGIN,
    selector_harm_margin_loss_weight: float = DEFAULT_SELECTOR_HARM_MARGIN_LOSS_WEIGHT,
    selector_harm_margin: float = DEFAULT_SELECTOR_HARM_MARGIN,
    selector_negative_identity_margin_loss_weight: float = DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN_LOSS_WEIGHT,
    selector_negative_identity_margin: float = DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN,
    selector_benefit_harm_pairwise_loss_weight: float = DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_LOSS_WEIGHT,
    selector_benefit_harm_pairwise_margin: float = DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_MARGIN,
    selector_positive_negative_hard_loss_weight: float = DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_LOSS_WEIGHT,
    selector_positive_negative_hard_margin: float = DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_MARGIN,
    selector_cross_family_positive_negative_loss_weight: float = (
        DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_LOSS_WEIGHT
    ),
    selector_cross_family_positive_negative_margin: float = (
        DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_MARGIN
    ),
    selector_model: str = SELECTOR_MODEL_SCALAR,
    selector_risk_aware_harm_logit_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
    selector_risk_aware_benefit_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_LOSS_WEIGHT,
    selector_risk_aware_harm_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOSS_WEIGHT,
    selector_risk_aware_benefit_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_POS_WEIGHT,
    selector_risk_aware_harm_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_POS_WEIGHT,
    selector_patch_head: bool = DEFAULT_SELECTOR_PATCH_HEAD,
    selector_patch_hidden_dim: int = DEFAULT_SELECTOR_PATCH_HIDDEN_DIM,
    selector_emit_margin_grid: list[float] | None = None,
    selector_nonzero_bias_grid: list[float] | None = None,
    selector_transition_prior_weight_grid: list[float] | None = None,
    selector_transition_compat_top_k_grid: list[int] | None = None,
    selector_candidate_compat_threshold_grid: list[float] | None = None,
    selector_candidate_compat_top_k_grid: list[int] | None = None,
    candidate_compat_hidden_dim: int = DEFAULT_CANDIDATE_COMPAT_HIDDEN_DIM,
    candidate_compat_epochs: int = DEFAULT_CANDIDATE_COMPAT_EPOCHS,
    candidate_compat_lr: float = DEFAULT_CANDIDATE_COMPAT_LR,
    candidate_compat_objective: str = CANDIDATE_COMPAT_OBJECTIVE_BCE,
    candidate_compat_negative_ratio: float = DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO,
    candidate_compat_no_positive_negative_count: int = DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT,
    transition_prior_hidden_dim: int = DEFAULT_TRANSITION_PRIOR_HIDDEN_DIM,
    transition_prior_epochs: int = DEFAULT_TRANSITION_PRIOR_EPOCHS,
    transition_prior_lr: float = DEFAULT_TRANSITION_PRIOR_LR,
    selector_score_edit_penalty: float = DEFAULT_SELECTOR_SCORE_EDIT_PENALTY,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    selector_epoch_diagnostic_margin_grid: list[float] | None = None,
    selector_epoch_selection_mode: str = SELECTOR_EPOCH_SELECTION_PROXY,
    selector_adoption_min_delta: float = DEFAULT_SELECTOR_ADOPTION_MIN_DELTA,
    selector_adoption_policy: str = SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR,
    selector_candidate_first_strong_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_STRONG_DELTA,
    selector_candidate_first_positive_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_DELTA,
    selector_candidate_first_positive_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MARGIN_FLOOR,
    selector_candidate_first_positive_max_harmed: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED,
    selector_candidate_first_positive_max_margin: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_MARGIN,
    selector_candidate_first_positive_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MIN_NONZERO,
    selector_candidate_first_positive_plateau_guard: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_PLATEAU_GUARD,
    selector_candidate_first_tie_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_DELTA,
    selector_candidate_first_tie_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MARGIN_FLOOR,
    selector_candidate_first_tie_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_NONZERO,
    selector_candidate_first_allow_global: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_ALLOW_GLOBAL,
    selector_candidate_first_global_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_GLOBAL_MIN_DELTA,
    selected_no_edit_guardrail: bool = DEFAULT_SELECTED_NO_EDIT_GUARDRAIL,
    selected_no_edit_min_delta: float = DEFAULT_SELECTED_NO_EDIT_MIN_DELTA,
    selector_candidate_motif_max_classes: int = DEFAULT_SELECTOR_CANDIDATE_MOTIF_MAX_CLASSES,
    selector_local_motif_max_classes: int = DEFAULT_SELECTOR_LOCAL_MOTIF_MAX_CLASSES,
    selector_local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    router_hidden_dim: int = DEFAULT_ROUTER_HIDDEN_DIM,
    router_epochs: int = DEFAULT_ROUTER_EPOCHS,
    router_lr: float = DEFAULT_ROUTER_LR,
    router_pos_weight: float = DEFAULT_ROUTER_POS_WEIGHT,
    router_threshold_grid: list[float] | None = None,
    router_supervision_target: str = ROUTER_LABEL_IDENTITY_VS_NONZERO,
    router_pretrain_target: str = ROUTER_PRETRAIN_TARGET_NONE,
    router_pretrain_epochs: int = 0,
    router_pretrain_pos_weight: float | None = None,
    router_negative_ratio: float | None = None,
    action_motif_max_classes: int = DEFAULT_ACTION_MOTIF_MAX_CLASSES,
    action_motif_loss_weight: float = DEFAULT_ACTION_MOTIF_LOSS_WEIGHT,
    action_motif_identity_margin: float = DEFAULT_ACTION_MOTIF_IDENTITY_MARGIN,
    action_motif_emit_margin_grid: list[float] | None = None,
    local_motif_max_classes: int = DEFAULT_LOCAL_MOTIF_MAX_CLASSES,
    local_motif_emit_margin_grid: list[float] | None = None,
    local_motif_min_bit_logit_grid: list[float] | None = None,
    motif_max_classes: int = DEFAULT_MOTIF_MAX_CLASSES,
    motif_epochs: int = DEFAULT_MOTIF_EPOCHS,
    motif_lr: float = DEFAULT_MOTIF_LR,
    motif_hard_shot_weight: float = DEFAULT_MOTIF_HARD_SHOT_WEIGHT,
    needs_edit_threshold_grid: list[float] | None = None,
    edit_threshold_grid: list[float] | None = None,
    max_edit_weight_grid: list[int] | None = None,
) -> dict[str, Any]:
    _require_torch()
    common._set_random_seeds(int(seed))
    if str(selector_adoption_policy) not in SELECTOR_ADOPTION_POLICY_CHOICES:
        raise ValueError(f"Unknown selector adoption policy: {selector_adoption_policy!r}")
    entry = _prepare_edit_family(family_dir, fill_value=fill_value, max_shots=max_shots)
    split = _build_split_bundle(
        num_shots=int(entry.x.shape[0]),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_subset = _subset_family(entry, split.train)
    val_subset = _subset_family(entry, split.val)
    test_subset = _subset_family(entry, split.test)
    action_motif_vocabulary: MotifVocabulary | None = None
    local_motif_vocabulary: LocalMotifVocabulary | None = None
    train_action_motif_label = np.full((int(train_subset["x"].shape[0]),), -1, dtype=np.int64)
    train_action_motif_active = np.zeros((int(train_subset["x"].shape[0]),), dtype=np.float32)
    if int(action_motif_max_classes) >= 2 and float(action_motif_loss_weight) > 0.0:
        action_motif_vocabulary = _build_motif_vocabulary(
            [(entry, train_subset)],
            max_classes=int(action_motif_max_classes),
        )
        if action_motif_vocabulary.mask_table.shape[0] <= 1:
            action_motif_vocabulary = None
        else:
            train_action_motif_label, train_action_motif_active = _build_action_motif_supervision_arrays(
                entry=entry,
                subset=train_subset,
                vocabulary=action_motif_vocabulary,
            )
    if int(local_motif_max_classes) >= 2:
        local_motif_vocabulary = _build_local_motif_vocabulary(
            [(entry, train_subset)],
            max_classes=int(local_motif_max_classes),
        )
        if len(local_motif_vocabulary.offset_patterns) == 0:
            local_motif_vocabulary = None

    shot_sample_weights = _compute_shot_sample_weights(
        needs_edit=train_subset["needs_edit"],
        edit_target_known=train_subset["edit_target_known"],
        hard_shot_solved_weight=hard_shot_solved_weight,
        hard_shot_unsolved_weight=hard_shot_unsolved_weight,
    )
    training_config = _build_training_config(
        edit_target_volume=train_subset["edit_target_volume"],
        needs_edit=train_subset["needs_edit"],
        edit_target_known=train_subset["edit_target_known"],
        valid_mask_volume=entry.valid_mask_volume,
        shot_sample_weights=shot_sample_weights,
        edit_supervision_mode=edit_supervision_mode,
    )
    valid_mask_volume_t = torch.from_numpy(np.ascontiguousarray(entry.valid_mask_volume)).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = valid_mask_volume_t.device
    action_motif_mask_table_t = (
        torch.from_numpy(np.ascontiguousarray(action_motif_vocabulary.mask_table, dtype=np.float32)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_detector_time_index_t = (
        torch.from_numpy(np.ascontiguousarray(entry.detector_time_index, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_row_index_by_detector_t = (
        torch.from_numpy(np.ascontiguousarray(entry.row_index_by_detector, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_col_index_by_detector_t = (
        torch.from_numpy(np.ascontiguousarray(entry.col_index_by_detector, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )

    model_kwargs = {
        "in_channels": int(entry.x.shape[1]),
        "hidden_channels": int(hidden_channels),
        "num_blocks": int(num_blocks),
        "dense_hidden_dim": int(dense_hidden_dim),
        "dropout": float(dropout),
        "valid_mask_channel_index": int(_infer_valid_mask_channel_index(entry.bundle_info)),
    }
    model = SyndromeEditPreDecoder(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    edit_pos_weight_t = torch.tensor([training_config["edit_pos_weight"]], dtype=torch.float32, device=device)
    needs_pos_weight_t = torch.tensor([training_config["needs_pos_weight"]], dtype=torch.float32, device=device)

    train_dataset = _make_tensor_dataset(
        train_subset["x"],
        train_subset["edit_target_volume"],
        train_subset["needs_edit"],
        train_subset["edit_target_known"],
        train_action_motif_label,
        train_action_motif_active,
    )
    train_loader = _make_weighted_loader(
        train_dataset,
        batch_size=batch_size,
        sample_weights=shot_sample_weights,
    )

    needs_grid = needs_edit_threshold_grid or [0.3, 0.5, 0.7, 0.9]
    edit_grid = edit_threshold_grid or [0.3, 0.5, 0.7, 0.9]
    max_weight_grid = max_edit_weight_grid or [0, 1, 2]
    action_emit_grid = action_motif_emit_margin_grid or list(DEFAULT_ACTION_MOTIF_EMIT_MARGIN_GRID)
    local_emit_grid = local_motif_emit_margin_grid or list(DEFAULT_LOCAL_MOTIF_EMIT_MARGIN_GRID)
    local_min_bit_grid = local_motif_min_bit_logit_grid or list(DEFAULT_LOCAL_MOTIF_MIN_BIT_LOGIT_GRID)
    selector_emit_grid = selector_emit_margin_grid or list(DEFAULT_SELECTOR_EMIT_MARGIN_GRID)
    selector_nonzero_bias_values = selector_nonzero_bias_grid or list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID)
    selector_transition_prior_weight_values = (
        selector_transition_prior_weight_grid
        or list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID)
    )
    selector_transition_compat_top_k_values = (
        selector_transition_compat_top_k_grid
        or list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID)
    )
    selector_candidate_compat_threshold_values = (
        selector_candidate_compat_threshold_grid
        or list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID)
    )
    selector_candidate_compat_top_k_values = (
        selector_candidate_compat_top_k_grid
        or list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID)
    )
    router_threshold_grid_values = router_threshold_grid or list(DEFAULT_ROUTER_THRESHOLD_GRID)
    policy_specs = _build_candidate_policy_specs(
        needs_edit_threshold_grid=needs_grid,
        edit_threshold_grid=edit_grid,
        max_edit_weight_grid=max_weight_grid,
    )

    best_state: dict[str, Any] | None = None
    best_val_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            valid_mask_volume=valid_mask_volume_t,
            edit_pos_weight=edit_pos_weight_t,
            needs_pos_weight=needs_pos_weight_t,
            needs_edit_loss_weight=needs_edit_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            decision_aware_loss_weight=decision_aware_loss_weight,
            decision_aware_margin=decision_aware_margin,
            action_motif_loss_weight=action_motif_loss_weight,
            action_motif_identity_margin=action_motif_identity_margin,
            action_motif_mask_table=action_motif_mask_table_t,
            action_motif_detector_time_index=action_motif_detector_time_index_t,
            action_motif_row_index_by_detector=action_motif_row_index_by_detector_t,
            action_motif_col_index_by_detector=action_motif_col_index_by_detector_t,
            edit_supervision_mode=edit_supervision_mode,
        )
        val_result = _grid_search_decision_policy(
            entry=entry,
            subset=val_subset,
            model=model,
            batch_size=batch_size,
            device=device,
            needs_edit_threshold_grid=needs_grid,
            edit_threshold_grid=edit_grid,
            max_edit_weight_grid=max_weight_grid,
        )
        val_metric = _family_system_metric_value(val_result)
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_result,
            "val_selection_metric": float(val_metric),
        }
        epoch_history.append(epoch_record)
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "val_result": copy.deepcopy(val_result),
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    model.load_state_dict(best_state["model_state"])

    global_val_result = copy.deepcopy(best_state["val_result"])
    global_test_result = _system_metrics_for_subset(
        entry=entry,
        subset=test_subset,
        model=model,
        batch_size=batch_size,
        device=device,
        needs_edit_threshold=float(best_state["val_result"]["decision"]["needs_edit_threshold"]),
        edit_threshold=float(best_state["val_result"]["decision"]["edit_threshold"]),
        max_predicted_edit_weight=int(best_state["val_result"]["decision"]["max_predicted_edit_weight"]),
    )
    no_edit_val_result = _no_edit_system_metrics_for_subset(entry=entry, subset=val_subset)
    no_edit_test_result = _no_edit_system_metrics_for_subset(entry=entry, subset=test_subset)
    selector_summary: dict[str, Any] | None = None
    transition_prior_summary: dict[str, Any] | None = None
    candidate_compatibility_summary: dict[str, Any] | None = None
    router_summary: dict[str, Any] | None = None
    selector_val_result: dict[str, Any] | None = None
    selector_test_result: dict[str, Any] | None = None
    selector_candidate_motif_vocabulary: MotifVocabulary | None = None
    motif_summary: dict[str, Any] | None = None
    motif_val_result: dict[str, Any] | None = None
    motif_test_result: dict[str, Any] | None = None
    action_motif_val_result: dict[str, Any] | None = None
    action_motif_test_result: dict[str, Any] | None = None
    local_motif_val_result: dict[str, Any] | None = None
    local_motif_test_result: dict[str, Any] | None = None
    selected_inference_mode = SELECTION_MODE_GLOBAL_POLICY
    selected_val_result = global_val_result
    selected_test_result = global_test_result
    selector_adoption_decision: dict[str, Any] = {
        "policy": str(selector_adoption_policy),
        "requested_mode": str(selection_mode),
        "selected_mode": str(selected_inference_mode),
        "reason": "selector_not_evaluated",
    }
    selector_local_motif_vocabulary: LocalMotifVocabulary | None = None
    if selection_mode in (
        SELECTION_MODE_CANDIDATE_SELECTOR,
        SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
        SELECTION_MODE_LOCAL_MOTIF_ROUTER,
    ):
        if int(selector_candidate_motif_max_classes) >= 2:
            selector_candidate_motif_vocabulary = _build_motif_vocabulary(
                [(entry, train_subset)],
                max_classes=int(selector_candidate_motif_max_classes),
            )
            if selector_candidate_motif_vocabulary.mask_table.shape[0] <= 1:
                selector_candidate_motif_vocabulary = None
        if int(selector_local_motif_max_classes) >= 2:
            selector_local_motif_vocabulary = _build_local_motif_vocabulary(
                [(entry, train_subset)],
                max_classes=int(selector_local_motif_max_classes),
            )
            if len(selector_local_motif_vocabulary.offset_patterns) == 0:
                selector_local_motif_vocabulary = None
        train_selector_bundle = _build_selector_candidate_bundle(
            entry=entry,
            subset=train_subset,
            model=model,
            batch_size=batch_size,
            device=device,
            policy_specs=policy_specs,
            selector_score_edit_penalty=selector_score_edit_penalty,
            selector_target_mode=str(selector_target_mode),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
            selector_policy_candidate_mode=str(selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            motif_vocabulary=selector_candidate_motif_vocabulary,
            local_motif_vocabulary=selector_local_motif_vocabulary,
            local_motif_top_k=int(selector_local_motif_top_k),
        )
        val_selector_bundle = _build_selector_candidate_bundle(
            entry=entry,
            subset=val_subset,
            model=model,
            batch_size=batch_size,
            device=device,
            policy_specs=policy_specs,
            selector_score_edit_penalty=selector_score_edit_penalty,
            selector_target_mode=str(selector_target_mode),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
            selector_policy_candidate_mode=str(selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            motif_vocabulary=selector_candidate_motif_vocabulary,
            local_motif_vocabulary=selector_local_motif_vocabulary,
            local_motif_top_k=int(selector_local_motif_top_k),
        )
        selector_summary = _train_candidate_selector(
            model=model,
            train_bundles=[train_selector_bundle],
            val_bundles_by_family={entry.family: val_selector_bundle},
            entry_by_family={entry.family: entry},
            subset_by_family={entry.family: val_subset},
            batch_size=batch_size,
            device=device,
            hidden_dim=selector_hidden_dim,
            dropout=dropout,
            lr=selector_lr,
            epochs=selector_epochs,
            selector_objective=selector_objective,
            selector_hard_shot_weight=selector_hard_shot_weight,
            selector_identity_margin_loss_weight=selector_identity_margin_loss_weight,
            selector_identity_margin=selector_identity_margin,
            selector_harm_margin_loss_weight=selector_harm_margin_loss_weight,
            selector_harm_margin=selector_harm_margin,
            selector_negative_identity_margin_loss_weight=selector_negative_identity_margin_loss_weight,
            selector_negative_identity_margin=selector_negative_identity_margin,
            selector_benefit_harm_pairwise_loss_weight=selector_benefit_harm_pairwise_loss_weight,
            selector_benefit_harm_pairwise_margin=selector_benefit_harm_pairwise_margin,
            selector_positive_negative_hard_loss_weight=selector_positive_negative_hard_loss_weight,
            selector_positive_negative_hard_margin=selector_positive_negative_hard_margin,
            selector_cross_family_positive_negative_loss_weight=(
                selector_cross_family_positive_negative_loss_weight
            ),
            selector_cross_family_positive_negative_margin=selector_cross_family_positive_negative_margin,
            selector_model=str(selector_model),
            selector_risk_aware_harm_logit_weight=float(selector_risk_aware_harm_logit_weight),
            selector_risk_aware_benefit_loss_weight=float(selector_risk_aware_benefit_loss_weight),
            selector_risk_aware_harm_loss_weight=float(selector_risk_aware_harm_loss_weight),
            selector_risk_aware_benefit_pos_weight=float(selector_risk_aware_benefit_pos_weight),
            selector_risk_aware_harm_pos_weight=float(selector_risk_aware_harm_pos_weight),
            selector_patch_head=bool(selector_patch_head),
            selector_patch_hidden_dim=int(selector_patch_hidden_dim),
            policy_specs=policy_specs,
            selector_score_edit_penalty=selector_score_edit_penalty,
            selector_target_mode=str(selector_target_mode),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
            selector_policy_candidate_mode=str(selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            motif_vocabulary=selector_candidate_motif_vocabulary,
            local_motif_vocabulary=selector_local_motif_vocabulary,
            local_motif_top_k=int(selector_local_motif_top_k),
            selector_selection_mode=str(selection_mode),
            selector_epoch_diagnostic_margin_grid=selector_epoch_diagnostic_margin_grid,
            selector_epoch_selection_mode=str(selector_epoch_selection_mode),
        )
        if (
            selector_summary is not None
            and str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM
            and (
                max(float(x) for x in selector_transition_prior_weight_values) > 0.0
                or max(int(x) for x in selector_transition_compat_top_k_values) > 0
            )
        ):
            transition_prior_summary = _train_transition_prior_head(
                train_bundles=[train_selector_bundle],
                val_bundles_by_family={entry.family: val_selector_bundle},
                hidden_dim=int(transition_prior_hidden_dim),
                dropout=dropout,
                lr=float(transition_prior_lr),
                epochs=int(transition_prior_epochs),
                batch_size=batch_size,
                device=device,
            )
        if (
            selector_summary is not None
            and str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM
            and (
                max(float(x) for x in selector_candidate_compat_threshold_values) > 0.0
                or max(int(x) for x in selector_candidate_compat_top_k_values) > 0
            )
        ):
            candidate_compatibility_summary = _train_candidate_compatibility_head(
                train_bundles=[train_selector_bundle],
                val_bundles_by_family={entry.family: val_selector_bundle},
                hidden_dim=int(candidate_compat_hidden_dim),
                dropout=dropout,
                lr=float(candidate_compat_lr),
                epochs=int(candidate_compat_epochs),
                batch_size=batch_size,
                device=device,
                objective=str(candidate_compat_objective),
                negative_ratio=float(candidate_compat_negative_ratio),
                no_positive_negative_count=int(candidate_compat_no_positive_negative_count),
            )
        if selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER:
            router_summary = _train_hard_shot_router(
                train_bundles=[train_selector_bundle],
                val_bundles_by_family={entry.family: val_selector_bundle},
                hidden_dim=int(router_hidden_dim),
                dropout=dropout,
                lr=float(router_lr),
                epochs=int(router_epochs),
                batch_size=batch_size,
                device=device,
                positive_weight=float(router_pos_weight),
                supervision_target=str(router_supervision_target),
                pretrain_target=str(router_pretrain_target),
                pretrain_epochs=int(router_pretrain_epochs),
                pretrain_positive_weight=router_pretrain_pos_weight,
                negative_ratio=router_negative_ratio,
            )
        if selector_summary is not None:
            if selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER and router_summary is not None:
                selector_val_by_family = _grid_search_routed_selector_policy_by_family(
                    entry_by_family={entry.family: entry},
                    subset_by_family={entry.family: val_subset},
                    model=model,
                    selector=selector_summary["selector"],
                    router=router_summary["router"],
                    batch_size=batch_size,
                    device=device,
                    policy_specs=policy_specs,
                    selector_score_edit_penalty=selector_score_edit_penalty,
                    selector_target_mode=str(selector_target_mode),
                    selector_harm_weight=float(selector_harm_weight),
                    selector_miss_weight=float(selector_miss_weight),
                    selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                    selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                    selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                    selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                    selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                    router_threshold_grid=router_threshold_grid_values,
                    selector_emit_margin_grid=selector_emit_grid,
                    selector_nonzero_bias_grid=selector_nonzero_bias_values,
                    transition_prior_head=(
                        transition_prior_summary["head"]
                        if transition_prior_summary is not None
                        else None
                    ),
                    transition_prior_weight_grid=selector_transition_prior_weight_values,
                    transition_compat_top_k_grid=selector_transition_compat_top_k_values,
                    candidate_compatibility_head=(
                        candidate_compatibility_summary["head"]
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    candidate_compat_threshold_grid=selector_candidate_compat_threshold_values,
                    candidate_compat_top_k_grid=selector_candidate_compat_top_k_values,
                    motif_vocabulary=selector_candidate_motif_vocabulary,
                    local_motif_vocabulary=selector_local_motif_vocabulary,
                    local_motif_top_k=int(selector_local_motif_top_k),
                    selector_selection_mode=str(selection_mode),
                )
            else:
                selector_val_by_family = _grid_search_selector_emit_margin_by_family(
                    entry_by_family={entry.family: entry},
                    subset_by_family={entry.family: val_subset},
                    model=model,
                    selector=selector_summary["selector"],
                    batch_size=batch_size,
                    device=device,
                    policy_specs=policy_specs,
                    selector_score_edit_penalty=selector_score_edit_penalty,
                    selector_target_mode=str(selector_target_mode),
                    selector_harm_weight=float(selector_harm_weight),
                    selector_miss_weight=float(selector_miss_weight),
                    selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                    selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                    selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                    selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                    selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                    emit_margin_grid=selector_emit_grid,
                    nonzero_bias_grid=selector_nonzero_bias_values,
                    transition_prior_head=(
                        transition_prior_summary["head"]
                        if transition_prior_summary is not None
                        else None
                    ),
                    transition_prior_weight_grid=selector_transition_prior_weight_values,
                    transition_compat_top_k_grid=selector_transition_compat_top_k_values,
                    candidate_compatibility_head=(
                        candidate_compatibility_summary["head"]
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    candidate_compat_threshold_grid=selector_candidate_compat_threshold_values,
                    candidate_compat_top_k_grid=selector_candidate_compat_top_k_values,
                    motif_vocabulary=selector_candidate_motif_vocabulary,
                    local_motif_vocabulary=selector_local_motif_vocabulary,
                    local_motif_top_k=int(selector_local_motif_top_k),
                    selector_selection_mode=str(selection_mode),
                )
            selector_val_result = copy.deepcopy(selector_val_by_family[entry.family])
            selected_selector_emit_margin = float(selector_val_result["decision"]["selector_emit_margin"])
            selected_selector_nonzero_bias = float(selector_val_result["decision"].get("selector_nonzero_bias") or 0.0)
            selected_selector_transition_prior_weight = float(
                selector_val_result["decision"].get("selector_transition_prior_weight") or 0.0
            )
            selected_selector_transition_compat_top_k = int(
                selector_val_result["decision"].get("selector_transition_compat_top_k") or 0
            )
            selected_selector_candidate_compat_threshold = float(
                selector_val_result["decision"].get("selector_candidate_compat_threshold") or 0.0
            )
            selected_selector_candidate_compat_top_k = int(
                selector_val_result["decision"].get("selector_candidate_compat_top_k") or 0
            )
            selected_router_threshold = float(selector_val_result["decision"].get("router_threshold") or 0.5)
            selector_test_result = _selector_system_metrics_for_subset(
                entry=entry,
                subset=test_subset,
                model=model,
                selector=selector_summary["selector"],
                batch_size=batch_size,
                device=device,
                policy_specs=policy_specs,
                selector_score_edit_penalty=selector_score_edit_penalty,
                selector_target_mode=str(selector_target_mode),
                selector_harm_weight=float(selector_harm_weight),
                selector_miss_weight=float(selector_miss_weight),
                selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                motif_vocabulary=selector_candidate_motif_vocabulary,
                local_motif_vocabulary=selector_local_motif_vocabulary,
                local_motif_top_k=int(selector_local_motif_top_k),
                selector_selection_mode=str(selection_mode),
                selector_emit_margin=selected_selector_emit_margin,
                selector_nonzero_bias=selected_selector_nonzero_bias,
                transition_prior_head=(
                    transition_prior_summary["head"]
                    if transition_prior_summary is not None
                    else None
                ),
                selector_transition_prior_weight=selected_selector_transition_prior_weight,
                selector_transition_compat_top_k=selected_selector_transition_compat_top_k,
                candidate_compatibility_head=(
                    candidate_compatibility_summary["head"]
                    if candidate_compatibility_summary is not None
                    else None
                ),
                selector_candidate_compat_threshold=selected_selector_candidate_compat_threshold,
                selector_candidate_compat_top_k=selected_selector_candidate_compat_top_k,
                router=(router_summary["router"] if router_summary is not None else None),
                router_threshold=selected_router_threshold,
                routed_nonzero_only=(selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER),
            )
            global_val_metric = _family_system_metric_value(global_val_result)
            selector_val_metric = _family_system_metric_value(selector_val_result)
            no_edit_val_metric = _family_system_metric_value(no_edit_val_result)
            if str(selector_adoption_policy) == SELECTOR_ADOPTION_POLICY_CANDIDATE_FIRST_SAFETY:
                selected_inference_mode, selector_adoption_decision = _candidate_first_safety_adoption_mode(
                    no_edit_metric=float(no_edit_val_metric),
                    global_metric=float(global_val_metric),
                    selector_metric=float(selector_val_metric),
                    selector_results=[selector_val_result],
                    requested_mode=str(selection_mode),
                    strong_delta=float(selector_candidate_first_strong_delta),
                    positive_delta=float(selector_candidate_first_positive_delta),
                    positive_margin_floor=float(selector_candidate_first_positive_margin_floor),
                    positive_max_harmed=int(selector_candidate_first_positive_max_harmed),
                    positive_max_margin=float(selector_candidate_first_positive_max_margin),
                    positive_min_nonzero=int(selector_candidate_first_positive_min_nonzero),
                    positive_plateau_guard=bool(selector_candidate_first_positive_plateau_guard),
                    tie_min_delta=float(selector_candidate_first_tie_min_delta),
                    tie_margin_floor=float(selector_candidate_first_tie_margin_floor),
                    tie_min_nonzero=int(selector_candidate_first_tie_min_nonzero),
                    allow_global=bool(selector_candidate_first_allow_global),
                    global_min_delta=float(selector_candidate_first_global_min_delta),
                )
            else:
                selected_inference_mode = (
                    str(selection_mode)
                    if _selected_inference_mode(
                        global_metric=float(global_val_metric),
                        selector_metric=float(selector_val_metric),
                        requested_mode=str(selection_mode),
                        min_delta=float(selector_adoption_min_delta),
                    )
                    == str(selection_mode)
                    else SELECTION_MODE_GLOBAL_POLICY
                )
                selector_adoption_decision = {
                    "policy": SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR,
                    "requested_mode": str(selection_mode),
                    "selected_mode": str(selected_inference_mode),
                    "reason": (
                        "selector_noninferior"
                        if selected_inference_mode == str(selection_mode)
                        else "global_policy_better"
                    ),
                    "global_metric": float(global_val_metric),
                    "selector_metric": float(selector_val_metric),
                    "selector_delta_over_global": float(selector_val_metric) - float(global_val_metric),
                    "selector_adoption_min_delta": float(selector_adoption_min_delta),
                }
            if selected_inference_mode == selection_mode:
                selected_val_result = selector_val_result
                selected_test_result = selector_test_result
            elif selected_inference_mode == SELECTION_MODE_RAW_NO_EDIT:
                selected_val_result = no_edit_val_result
                selected_test_result = no_edit_test_result
            else:
                selected_val_result = global_val_result
                selected_test_result = global_test_result
    elif selection_mode == SELECTION_MODE_MOTIF_VOCAB:
        motif_summary = _train_motif_vocabulary_head(
            model=model,
            train_bundles=[(entry, train_subset)],
            val_bundles_by_family={entry.family: (entry, val_subset)},
            batch_size=batch_size,
            device=device,
            hidden_dim=dense_hidden_dim,
            dropout=dropout,
            max_classes=motif_max_classes,
            lr=motif_lr,
            epochs=motif_epochs,
            hard_shot_weight=motif_hard_shot_weight,
        )
        if motif_summary is not None:
            motif_val_result = copy.deepcopy(motif_summary["eval_by_family"][entry.family])
            motif_test_result = _motif_system_metrics_for_subset(
                entry=entry,
                subset=test_subset,
                model=model,
                motif_head=motif_summary["motif_head"],
                vocabulary=motif_summary["vocabulary"],
                batch_size=batch_size,
                device=device,
            )
            selected_inference_mode = (
                SELECTION_MODE_MOTIF_VOCAB
                if float(_family_system_metric_value(motif_val_result)) > float(_family_system_metric_value(global_val_result))
                else SELECTION_MODE_GLOBAL_POLICY
            )
            if selected_inference_mode == SELECTION_MODE_MOTIF_VOCAB:
                selected_val_result = motif_val_result
                selected_test_result = motif_test_result
    if action_motif_vocabulary is not None:
        action_motif_val_result = _grid_search_action_motif_policy(
            entry=entry,
            subset=val_subset,
            model=model,
            vocabulary=action_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin_grid=action_emit_grid,
        )
        action_motif_emit_margin = float(action_motif_val_result["decision"]["action_motif_emit_margin"])
        action_motif_test_result = _action_motif_system_metrics_for_subset(
            entry=entry,
            subset=test_subset,
            model=model,
            vocabulary=action_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin=action_motif_emit_margin,
        )
        if selection_mode == SELECTION_MODE_ACTION_MOTIF:
            selected_inference_mode = (
                SELECTION_MODE_ACTION_MOTIF
                if float(_family_system_metric_value(action_motif_val_result)) > float(_family_system_metric_value(global_val_result))
                else SELECTION_MODE_GLOBAL_POLICY
            )
            if selected_inference_mode == SELECTION_MODE_ACTION_MOTIF:
                selected_val_result = action_motif_val_result
                selected_test_result = action_motif_test_result
    if local_motif_vocabulary is not None:
        local_motif_val_result = _grid_search_local_motif_policy(
            entry=entry,
            subset=val_subset,
            model=model,
            vocabulary=local_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin_grid=local_emit_grid,
            min_bit_logit_grid=local_min_bit_grid,
        )
        local_motif_emit_margin = float(local_motif_val_result["decision"]["local_motif_emit_margin"])
        local_motif_min_bit_logit = float(local_motif_val_result["decision"]["local_motif_min_bit_logit"])
        local_motif_test_result = _local_motif_system_metrics_for_subset(
            entry=entry,
            subset=test_subset,
            model=model,
            vocabulary=local_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin=local_motif_emit_margin,
            min_bit_logit=local_motif_min_bit_logit,
        )
        if selection_mode == SELECTION_MODE_LOCAL_MOTIF:
            selected_inference_mode = (
                SELECTION_MODE_LOCAL_MOTIF
                if float(_family_system_metric_value(local_motif_val_result)) > float(_family_system_metric_value(global_val_result))
                else SELECTION_MODE_GLOBAL_POLICY
            )
            if selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF:
                selected_val_result = local_motif_val_result
                selected_test_result = local_motif_test_result

    mode_before_no_edit_guardrail = str(selected_inference_mode)
    guarded_mode = _apply_no_edit_guardrail_to_mode(
        selected_mode=str(selected_inference_mode),
        selected_metric=_family_system_metric_value(selected_val_result),
        no_edit_metric=_family_system_metric_value(no_edit_val_result),
        enabled=bool(selected_no_edit_guardrail),
        min_delta=float(selected_no_edit_min_delta),
    )
    if guarded_mode == SELECTION_MODE_RAW_NO_EDIT:
        selected_inference_mode = SELECTION_MODE_RAW_NO_EDIT
        selected_val_result = no_edit_val_result
        selected_test_result = no_edit_test_result
    selector_adoption_decision["mode_before_no_edit_guardrail"] = mode_before_no_edit_guardrail
    selector_adoption_decision["selected_mode"] = str(selected_inference_mode)
    selector_adoption_decision["no_edit_guardrail_applied"] = (
        mode_before_no_edit_guardrail != str(selected_inference_mode)
    )

    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "syndrome_edit_predecoder",
        "created_at_utc": _utc_now_iso(),
        "model_kwargs": model_kwargs,
        "model_state": model.state_dict(),
        "decision_policy": best_state["val_result"]["decision"],
        "inference": {
            "requested_selection_mode": str(selection_mode),
            "selected_inference_mode": str(selected_inference_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_adoption_decision": selector_adoption_decision,
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "candidate_policy_grid": [spec.to_dict() for spec in policy_specs],
        },
        "candidate_selector": (
            {
                "selector_model": selector_summary["selector_model"],
                "selector_kwargs": selector_summary["selector_kwargs"],
                "selector_state": selector_summary["selector"].state_dict(),
                "training_summary": {
                    "epoch_history": selector_summary["epoch_history"],
                    "best_epoch": selector_summary["best_epoch"],
                    "best_val_selection_metric": selector_summary["best_val_selection_metric"],
                    "best_val_by_family": selector_summary["best_val_by_family"],
                    "selector_model": selector_summary["selector_model"],
                    "selector_risk_aware_harm_logit_weight": selector_summary[
                        "selector_risk_aware_harm_logit_weight"
                    ],
                    "selector_risk_aware_benefit_loss_weight": selector_summary[
                        "selector_risk_aware_benefit_loss_weight"
                    ],
                    "selector_risk_aware_harm_loss_weight": selector_summary[
                        "selector_risk_aware_harm_loss_weight"
                    ],
                    "selector_risk_aware_benefit_pos_weight": selector_summary[
                        "selector_risk_aware_benefit_pos_weight"
                    ],
                    "selector_risk_aware_harm_pos_weight": selector_summary[
                        "selector_risk_aware_harm_pos_weight"
                    ],
                    "selector_objective": selector_summary["selector_objective"],
                    "selector_hard_shot_weight": selector_summary["selector_hard_shot_weight"],
                    "selector_identity_margin_loss_weight": selector_summary["selector_identity_margin_loss_weight"],
                    "selector_identity_margin": selector_summary["selector_identity_margin"],
                    "selector_harm_margin_loss_weight": selector_summary["selector_harm_margin_loss_weight"],
                    "selector_harm_margin": selector_summary["selector_harm_margin"],
                    "selector_negative_identity_margin_loss_weight": selector_summary[
                        "selector_negative_identity_margin_loss_weight"
                    ],
                    "selector_negative_identity_margin": selector_summary[
                        "selector_negative_identity_margin"
                    ],
                    "selector_benefit_harm_pairwise_loss_weight": selector_summary["selector_benefit_harm_pairwise_loss_weight"],
                    "selector_benefit_harm_pairwise_margin": selector_summary["selector_benefit_harm_pairwise_margin"],
                    "selector_positive_negative_hard_loss_weight": selector_summary[
                        "selector_positive_negative_hard_loss_weight"
                    ],
                    "selector_positive_negative_hard_margin": selector_summary[
                        "selector_positive_negative_hard_margin"
                    ],
                    "selector_cross_family_positive_negative_loss_weight": selector_summary[
                        "selector_cross_family_positive_negative_loss_weight"
                    ],
                    "selector_cross_family_positive_negative_margin": selector_summary[
                        "selector_cross_family_positive_negative_margin"
                    ],
                    "selector_patch_head": selector_summary["selector_patch_head"],
                    "selector_patch_hidden_dim": selector_summary["selector_patch_hidden_dim"],
                    "selector_score_edit_penalty": selector_summary["selector_score_edit_penalty"],
                    "selector_target_mode": selector_summary["selector_target_mode"],
                    "selector_harm_weight": selector_summary["selector_harm_weight"],
                    "selector_miss_weight": selector_summary["selector_miss_weight"],
                    "selector_policy_candidate_mode": selector_summary["selector_policy_candidate_mode"],
                    "selector_candidate_geometry_features": selector_summary["selector_candidate_geometry_features"],
                    "selector_candidate_pattern_features": selector_summary["selector_candidate_pattern_features"],
                    "selector_candidate_local_evidence_features": selector_summary["selector_candidate_local_evidence_features"],
                    "selector_candidate_local_patch_features": selector_summary["selector_candidate_local_patch_features"],
                    "selector_epoch_diagnostic_margin_grid": selector_summary["selector_epoch_diagnostic_margin_grid"],
                    "selector_epoch_selection_mode": selector_summary["selector_epoch_selection_mode"],
                    "selector_selection_mode": selector_summary["selector_selection_mode"],
                    "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
                    "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
                    "selector_transition_prior_weight_grid": [
                        float(x) for x in selector_transition_prior_weight_values
                    ],
                    "selector_transition_compat_top_k_grid": [
                        int(x) for x in selector_transition_compat_top_k_values
                    ],
                    "selector_candidate_compat_threshold_grid": [
                        float(x) for x in selector_candidate_compat_threshold_values
                    ],
                    "selector_candidate_compat_top_k_grid": [
                        int(x) for x in selector_candidate_compat_top_k_values
                    ],
                    "selector_transition_compat_top_k_grid": [
                        int(x) for x in selector_transition_compat_top_k_values
                    ],
                    "selector_candidate_compat_threshold_grid": [
                        float(x) for x in selector_candidate_compat_threshold_values
                    ],
                    "selector_candidate_compat_top_k_grid": [
                        int(x) for x in selector_candidate_compat_top_k_values
                    ],
                    "selected_selector_emit_margin": (
                        float(selector_val_result["decision"]["selector_emit_margin"])
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_nonzero_bias": (
                        float(selector_val_result["decision"].get("selector_nonzero_bias") or 0.0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_transition_prior_weight": (
                        float(selector_val_result["decision"].get("selector_transition_prior_weight") or 0.0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_transition_compat_top_k": (
                        int(selector_val_result["decision"].get("selector_transition_compat_top_k") or 0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_candidate_compat_threshold": (
                        float(selector_val_result["decision"].get("selector_candidate_compat_threshold") or 0.0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_candidate_compat_top_k": (
                        int(selector_val_result["decision"].get("selector_candidate_compat_top_k") or 0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selector_candidate_motif_max_classes": int(selector_candidate_motif_max_classes),
                    "selector_local_motif_max_classes": int(selector_local_motif_max_classes),
                    "selector_local_motif_top_k": int(selector_local_motif_top_k),
                },
                "transition_prior": (
                    {
                        "head_kwargs": transition_prior_summary["head_kwargs"],
                        "head_state": transition_prior_summary["head"].state_dict(),
                        "training_summary": {
                            "epoch_history": transition_prior_summary["epoch_history"],
                            "best_epoch": transition_prior_summary["best_epoch"],
                            "best_val_selection_metric": transition_prior_summary["best_val_selection_metric"],
                            "best_val_by_family": transition_prior_summary["best_val_by_family"],
                            "lr": transition_prior_summary["lr"],
                            "epochs": transition_prior_summary["epochs"],
                        },
                    }
                    if transition_prior_summary is not None
                    else None
                ),
                "candidate_compatibility": (
                    {
                        "head_kwargs": candidate_compatibility_summary["head_kwargs"],
                        "head_state": candidate_compatibility_summary["head"].state_dict(),
                        "training_summary": {
                            "epoch_history": candidate_compatibility_summary["epoch_history"],
                            "best_epoch": candidate_compatibility_summary["best_epoch"],
                            "best_val_selection_metric": candidate_compatibility_summary["best_val_selection_metric"],
                            "best_val_by_family": candidate_compatibility_summary["best_val_by_family"],
                            "lr": candidate_compatibility_summary["lr"],
                            "epochs": candidate_compatibility_summary["epochs"],
                            "objective": candidate_compatibility_summary["objective"],
                            "negative_ratio": candidate_compatibility_summary["negative_ratio"],
                            "no_positive_negative_count": candidate_compatibility_summary["no_positive_negative_count"],
                        },
                    }
                    if candidate_compatibility_summary is not None
                    else None
                ),
                "candidate_motif_vocabulary": (
                    {
                        "mask_table": selector_candidate_motif_vocabulary.mask_table,
                        "detector_index_lists": [list(x) for x in selector_candidate_motif_vocabulary.detector_index_lists],
                        "counts": list(selector_candidate_motif_vocabulary.counts),
                    }
                    if selector_candidate_motif_vocabulary is not None
                    else None
                ),
                "candidate_local_motif_vocabulary": (
                    {
                        "offset_patterns": [
                            [list(offset) for offset in pattern]
                            for pattern in selector_local_motif_vocabulary.offset_patterns
                        ],
                        "counts": list(selector_local_motif_vocabulary.counts),
                        "detector_count": int(selector_local_motif_vocabulary.detector_count),
                    }
                    if selector_local_motif_vocabulary is not None
                    else None
                ),
            }
            if selector_summary is not None
            else None
        ),
        "hard_shot_router": (
            {
                "router_kwargs": router_summary["router_kwargs"],
                "router_state": router_summary["router"].state_dict(),
                "training_summary": {
                    "epoch_history": router_summary["epoch_history"],
                    "best_epoch": router_summary["best_epoch"],
                    "best_val_selection_metric": router_summary["best_val_selection_metric"],
                    "best_val_by_family": router_summary["best_val_by_family"],
                    "positive_weight": router_summary["positive_weight"],
                    "supervision_target": router_summary["supervision_target"],
                    "pretrain_target": router_summary["pretrain_target"],
                    "pretrain_epochs": router_summary["pretrain_epochs"],
                    "pretrain_positive_weight": router_summary["pretrain_positive_weight"],
                    "pretrain_epoch_history": router_summary["pretrain_epoch_history"],
                    "negative_ratio": router_summary["negative_ratio"],
                    "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
                    "selected_router_threshold": (
                        float(selector_val_result["decision"].get("router_threshold"))
                        if selector_val_result is not None
                        and selector_val_result["decision"].get("router_threshold") is not None
                        else None
                    ),
                },
            }
            if router_summary is not None
            else None
        ),
        "motif_vocabulary_head": (
            {
                "motif_kwargs": motif_summary["motif_kwargs"],
                "motif_state": motif_summary["motif_head"].state_dict(),
                "vocabulary_mask_table": motif_summary["vocabulary"].mask_table,
                "vocabulary_detector_index_lists": [list(x) for x in motif_summary["vocabulary"].detector_index_lists],
                "vocabulary_counts": list(motif_summary["vocabulary"].counts),
                "training_summary": {
                    "epoch_history": motif_summary["epoch_history"],
                    "best_epoch": motif_summary["best_epoch"],
                    "best_val_selection_metric": motif_summary["best_val_selection_metric"],
                    "best_val_by_family": motif_summary["best_val_by_family"],
                    "hard_shot_weight": motif_summary["hard_shot_weight"],
                },
            }
            if motif_summary is not None
            else None
        ),
        "action_motif_vocabulary": (
            {
                "mask_table": action_motif_vocabulary.mask_table,
                "detector_index_lists": [list(x) for x in action_motif_vocabulary.detector_index_lists],
                "counts": list(action_motif_vocabulary.counts),
                "training_summary": {
                    "active_fraction": (
                        float(np.mean(train_action_motif_active >= 0.5))
                        if train_action_motif_active.size
                        else None
                    ),
                    "nonzero_active_fraction": (
                        float(
                            np.mean(
                                (train_action_motif_active >= 0.5)
                                & (train_action_motif_label > 0)
                            )
                        )
                        if train_action_motif_active.size
                        else None
                    ),
                    "loss_weight": float(action_motif_loss_weight),
                    "identity_margin": float(action_motif_identity_margin),
                    "emit_margin_grid": [float(x) for x in action_emit_grid],
                    "selected_emit_margin": (
                        float(action_motif_val_result["decision"]["action_motif_emit_margin"])
                        if action_motif_val_result is not None
                        else None
                    ),
                },
            }
            if action_motif_vocabulary is not None
            else None
        ),
        "local_motif_vocabulary": (
            {
                "offset_patterns": [
                    [list(offset) for offset in pattern]
                    for pattern in local_motif_vocabulary.offset_patterns
                ],
                "counts": list(local_motif_vocabulary.counts),
                "detector_count": int(local_motif_vocabulary.detector_count),
                "training_summary": {
                    "vocabulary_num_patterns": int(len(local_motif_vocabulary.offset_patterns)),
                    "emit_margin_grid": [float(x) for x in local_emit_grid],
                    "min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
                    "selected_emit_margin": (
                        float(local_motif_val_result["decision"]["local_motif_emit_margin"])
                        if local_motif_val_result is not None
                        else None
                    ),
                    "selected_min_bit_logit": (
                        float(local_motif_val_result["decision"]["local_motif_min_bit_logit"])
                        if local_motif_val_result is not None
                        else None
                    ),
                },
            }
            if local_motif_vocabulary is not None
            else None
        ),
        "bundle_info": entry.bundle_info,
        "training_config": {
            "fill_value": float(fill_value),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "needs_edit_loss_weight": float(needs_edit_loss_weight),
            "sparsity_loss_weight": float(sparsity_loss_weight),
            "decision_aware_loss_weight": float(decision_aware_loss_weight),
            "decision_aware_margin": float(decision_aware_margin),
            "action_motif_max_classes": int(action_motif_max_classes),
            "action_motif_loss_weight": float(action_motif_loss_weight),
            "action_motif_identity_margin": float(action_motif_identity_margin),
            "action_motif_emit_margin_grid": [float(x) for x in action_emit_grid],
            "local_motif_max_classes": int(local_motif_max_classes),
            "local_motif_emit_margin_grid": [float(x) for x in local_emit_grid],
            "local_motif_min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
            "hard_shot_solved_weight": float(hard_shot_solved_weight),
            "hard_shot_unsolved_weight": float(hard_shot_unsolved_weight),
            "edit_supervision_mode": str(edit_supervision_mode),
            "selection_mode": str(selection_mode),
            "selector_hidden_dim": int(selector_hidden_dim),
            "selector_epochs": int(selector_epochs),
            "selector_lr": float(selector_lr),
            "selector_model": str(selector_model),
            "selector_risk_aware_harm_logit_weight": float(selector_risk_aware_harm_logit_weight),
            "selector_risk_aware_benefit_loss_weight": float(selector_risk_aware_benefit_loss_weight),
            "selector_risk_aware_harm_loss_weight": float(selector_risk_aware_harm_loss_weight),
            "selector_risk_aware_benefit_pos_weight": float(selector_risk_aware_benefit_pos_weight),
            "selector_risk_aware_harm_pos_weight": float(selector_risk_aware_harm_pos_weight),
            "selector_objective": str(selector_objective),
            "selector_hard_shot_weight": float(selector_hard_shot_weight),
            "selector_identity_margin_loss_weight": float(selector_identity_margin_loss_weight),
            "selector_identity_margin": float(selector_identity_margin),
            "selector_harm_margin_loss_weight": float(selector_harm_margin_loss_weight),
            "selector_harm_margin": float(selector_harm_margin),
            "selector_negative_identity_margin_loss_weight": float(
                selector_negative_identity_margin_loss_weight
            ),
            "selector_negative_identity_margin": float(selector_negative_identity_margin),
            "selector_benefit_harm_pairwise_loss_weight": float(selector_benefit_harm_pairwise_loss_weight),
            "selector_benefit_harm_pairwise_margin": float(selector_benefit_harm_pairwise_margin),
            "selector_positive_negative_hard_loss_weight": float(
                selector_positive_negative_hard_loss_weight
            ),
            "selector_positive_negative_hard_margin": float(selector_positive_negative_hard_margin),
            "selector_cross_family_positive_negative_loss_weight": float(
                selector_cross_family_positive_negative_loss_weight
            ),
            "selector_cross_family_positive_negative_margin": float(
                selector_cross_family_positive_negative_margin
            ),
            "selector_patch_head": bool(selector_patch_head),
            "selector_patch_hidden_dim": int(selector_patch_hidden_dim),
            "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
            "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
            "selector_transition_prior_weight_grid": [
                float(x) for x in selector_transition_prior_weight_values
            ],
            "selector_transition_compat_top_k_grid": [
                int(x) for x in selector_transition_compat_top_k_values
            ],
            "transition_prior_hidden_dim": int(transition_prior_hidden_dim),
            "transition_prior_epochs": int(transition_prior_epochs),
            "transition_prior_lr": float(transition_prior_lr),
            "candidate_compat_hidden_dim": int(candidate_compat_hidden_dim),
            "candidate_compat_epochs": int(candidate_compat_epochs),
            "candidate_compat_lr": float(candidate_compat_lr),
            "candidate_compat_objective": str(candidate_compat_objective),
            "candidate_compat_negative_ratio": float(candidate_compat_negative_ratio),
            "candidate_compat_no_positive_negative_count": int(candidate_compat_no_positive_negative_count),
            "selector_candidate_compat_threshold_grid": [
                float(x) for x in selector_candidate_compat_threshold_values
            ],
            "selector_candidate_compat_top_k_grid": [
                int(x) for x in selector_candidate_compat_top_k_values
            ],
            "selector_score_edit_penalty": float(selector_score_edit_penalty),
            "selector_target_mode": str(selector_target_mode),
            "selector_harm_weight": float(selector_harm_weight),
            "selector_miss_weight": float(selector_miss_weight),
            "selector_policy_candidate_mode": str(selector_policy_candidate_mode),
            "selector_candidate_geometry_features": bool(selector_candidate_geometry_features),
            "selector_candidate_pattern_features": bool(selector_candidate_pattern_features),
            "selector_candidate_local_evidence_features": bool(selector_candidate_local_evidence_features),
            "selector_candidate_local_patch_features": bool(selector_candidate_local_patch_features),
            "selector_epoch_diagnostic_margin_grid": [
                float(x) for x in (selector_epoch_diagnostic_margin_grid or [])
            ],
            "selector_epoch_selection_mode": str(selector_epoch_selection_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_candidate_first_strong_delta": float(selector_candidate_first_strong_delta),
            "selector_candidate_first_positive_delta": float(selector_candidate_first_positive_delta),
            "selector_candidate_first_positive_margin_floor": float(selector_candidate_first_positive_margin_floor),
            "selector_candidate_first_positive_max_harmed": int(selector_candidate_first_positive_max_harmed),
            "selector_candidate_first_positive_max_margin": float(selector_candidate_first_positive_max_margin),
            "selector_candidate_first_positive_min_nonzero": int(selector_candidate_first_positive_min_nonzero),
            "selector_candidate_first_positive_plateau_guard": bool(selector_candidate_first_positive_plateau_guard),
            "selector_candidate_first_tie_min_delta": float(selector_candidate_first_tie_min_delta),
            "selector_candidate_first_tie_margin_floor": float(selector_candidate_first_tie_margin_floor),
            "selector_candidate_first_tie_min_nonzero": int(selector_candidate_first_tie_min_nonzero),
            "selector_candidate_first_allow_global": bool(selector_candidate_first_allow_global),
            "selector_candidate_first_global_min_delta": float(selector_candidate_first_global_min_delta),
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "selector_candidate_motif_max_classes": int(selector_candidate_motif_max_classes),
            "selector_local_motif_max_classes": int(selector_local_motif_max_classes),
            "selector_local_motif_top_k": int(selector_local_motif_top_k),
            "router_hidden_dim": int(router_hidden_dim),
            "router_epochs": int(router_epochs),
            "router_lr": float(router_lr),
            "router_pos_weight": float(router_pos_weight),
            "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
            "router_supervision_target": str(router_supervision_target),
            "router_pretrain_target": str(router_pretrain_target),
            "router_pretrain_epochs": int(router_pretrain_epochs),
            "router_pretrain_pos_weight": (
                float(router_pretrain_pos_weight) if router_pretrain_pos_weight is not None else None
            ),
            "router_negative_ratio": (
                float(router_negative_ratio) if router_negative_ratio is not None else None
            ),
            "motif_max_classes": int(motif_max_classes),
            "motif_epochs": int(motif_epochs),
            "motif_lr": float(motif_lr),
            "motif_hard_shot_weight": float(motif_hard_shot_weight),
            "seed": int(seed),
        },
        "source_family_dir": entry.source_family_dir.as_posix(),
        "edit_target_family_dir": entry.family_dir.as_posix(),
    }
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)

    split_summary = common.summarise_split_indices(
        {
            "train": split.train,
            "val": split.val,
            "test": split.test,
        },
        num_shots=int(entry.x.shape[0]),
        seed=seed,
    ).to_dict()

    result = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "syndrome_edit_predecoder",
        "created_at_utc": _utc_now_iso(),
        "input_mode": "family_dir",
        "family": entry.family,
        "stage": entry.stage,
        "family_dir": entry.family_dir.as_posix(),
        "source_family_dir": entry.source_family_dir.as_posix(),
        "model": {
            "model_class": "SyndromeEditPreDecoder",
            "model_kwargs": model_kwargs,
            "num_parameters": int(sum(param.numel() for param in model.parameters())),
        },
        "dataset": {
            "num_shots": int(entry.x.shape[0]),
            "input_shape": list(entry.x.shape),
            "channel_names": list(entry.bundle_info.get("channel_names", [])),
            "matching": entry.matching_info,
            "training_config": training_config,
        },
        "split": split_summary,
        "training": {
            "epoch_history": epoch_history,
            "best_epoch": int(best_state["epoch"]),
            "best_val_selection_metric": float(best_val_metric),
            "requested_selection_mode": str(selection_mode),
            "selected_inference_mode": str(selected_inference_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_adoption_decision": selector_adoption_decision,
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "selector_training": (
                {
                    "epoch_history": selector_summary["epoch_history"],
                    "best_epoch": selector_summary["best_epoch"],
                    "best_val_selection_metric": selector_summary["best_val_selection_metric"],
                    "best_val_by_family": selector_summary["best_val_by_family"],
                    "selector_model": selector_summary["selector_model"],
                    "selector_risk_aware_harm_logit_weight": selector_summary[
                        "selector_risk_aware_harm_logit_weight"
                    ],
                    "selector_risk_aware_benefit_loss_weight": selector_summary[
                        "selector_risk_aware_benefit_loss_weight"
                    ],
                    "selector_risk_aware_harm_loss_weight": selector_summary[
                        "selector_risk_aware_harm_loss_weight"
                    ],
                    "selector_risk_aware_benefit_pos_weight": selector_summary[
                        "selector_risk_aware_benefit_pos_weight"
                    ],
                    "selector_risk_aware_harm_pos_weight": selector_summary[
                        "selector_risk_aware_harm_pos_weight"
                    ],
                    "selector_objective": selector_summary["selector_objective"],
                    "selector_hard_shot_weight": selector_summary["selector_hard_shot_weight"],
                    "selector_identity_margin_loss_weight": selector_summary["selector_identity_margin_loss_weight"],
                    "selector_identity_margin": selector_summary["selector_identity_margin"],
                    "selector_harm_margin_loss_weight": selector_summary["selector_harm_margin_loss_weight"],
                    "selector_harm_margin": selector_summary["selector_harm_margin"],
                    "selector_negative_identity_margin_loss_weight": selector_summary[
                        "selector_negative_identity_margin_loss_weight"
                    ],
                    "selector_negative_identity_margin": selector_summary[
                        "selector_negative_identity_margin"
                    ],
                    "selector_benefit_harm_pairwise_loss_weight": selector_summary["selector_benefit_harm_pairwise_loss_weight"],
                    "selector_benefit_harm_pairwise_margin": selector_summary["selector_benefit_harm_pairwise_margin"],
                    "selector_positive_negative_hard_loss_weight": selector_summary[
                        "selector_positive_negative_hard_loss_weight"
                    ],
                    "selector_positive_negative_hard_margin": selector_summary[
                        "selector_positive_negative_hard_margin"
                    ],
                    "selector_cross_family_positive_negative_loss_weight": selector_summary[
                        "selector_cross_family_positive_negative_loss_weight"
                    ],
                    "selector_cross_family_positive_negative_margin": selector_summary[
                        "selector_cross_family_positive_negative_margin"
                    ],
                    "selector_patch_head": selector_summary["selector_patch_head"],
                    "selector_patch_hidden_dim": selector_summary["selector_patch_hidden_dim"],
                    "policy_specs": selector_summary["policy_specs"],
                    "selector_score_edit_penalty": selector_summary["selector_score_edit_penalty"],
                    "selector_target_mode": selector_summary["selector_target_mode"],
                    "selector_harm_weight": selector_summary["selector_harm_weight"],
                    "selector_miss_weight": selector_summary["selector_miss_weight"],
                    "selector_policy_candidate_mode": selector_summary["selector_policy_candidate_mode"],
                    "selector_candidate_geometry_features": selector_summary["selector_candidate_geometry_features"],
                    "selector_candidate_pattern_features": selector_summary["selector_candidate_pattern_features"],
                    "selector_candidate_local_evidence_features": selector_summary["selector_candidate_local_evidence_features"],
                    "selector_candidate_local_patch_features": selector_summary["selector_candidate_local_patch_features"],
                    "selector_epoch_diagnostic_margin_grid": selector_summary["selector_epoch_diagnostic_margin_grid"],
                    "selector_epoch_selection_mode": selector_summary["selector_epoch_selection_mode"],
                    "selector_selection_mode": selector_summary["selector_selection_mode"],
                    "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
                    "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
                    "selector_transition_prior_weight_grid": [
                        float(x) for x in selector_transition_prior_weight_values
                    ],
                    "selector_transition_compat_top_k_grid": [
                        int(x) for x in selector_transition_compat_top_k_values
                    ],
                    "selected_selector_emit_margin": (
                        float(selector_val_result["decision"]["selector_emit_margin"])
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_nonzero_bias": (
                        float(selector_val_result["decision"].get("selector_nonzero_bias") or 0.0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_transition_prior_weight": (
                        float(selector_val_result["decision"].get("selector_transition_prior_weight") or 0.0)
                        if selector_val_result is not None
                        else None
                    ),
                    "selected_selector_transition_compat_top_k": (
                        int(selector_val_result["decision"].get("selector_transition_compat_top_k") or 0)
                        if selector_val_result is not None
                        else None
                    ),
                    "transition_prior_training": (
                        {
                            "epoch_history": transition_prior_summary["epoch_history"],
                            "best_epoch": transition_prior_summary["best_epoch"],
                            "best_val_selection_metric": transition_prior_summary["best_val_selection_metric"],
                            "best_val_by_family": transition_prior_summary["best_val_by_family"],
                            "lr": transition_prior_summary["lr"],
                            "epochs": transition_prior_summary["epochs"],
                        }
                        if transition_prior_summary is not None
                        else None
                    ),
                    "candidate_compatibility_training": (
                        {
                            "epoch_history": candidate_compatibility_summary["epoch_history"],
                            "best_epoch": candidate_compatibility_summary["best_epoch"],
                            "best_val_selection_metric": candidate_compatibility_summary["best_val_selection_metric"],
                            "best_val_by_family": candidate_compatibility_summary["best_val_by_family"],
                            "lr": candidate_compatibility_summary["lr"],
                            "epochs": candidate_compatibility_summary["epochs"],
                        }
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    "candidate_motif_vocab_num_classes": (
                        int(selector_candidate_motif_vocabulary.mask_table.shape[0])
                        if selector_candidate_motif_vocabulary is not None
                        else 0
                    ),
                    "candidate_local_motif_num_patterns": (
                        int(len(selector_local_motif_vocabulary.offset_patterns))
                        if selector_local_motif_vocabulary is not None
                        else 0
                    ),
                    "candidate_local_motif_top_k": int(selector_local_motif_top_k),
                }
                if selector_summary is not None
                else None
            ),
            "router_training": (
                {
                    "epoch_history": router_summary["epoch_history"],
                    "best_epoch": router_summary["best_epoch"],
                    "best_val_selection_metric": router_summary["best_val_selection_metric"],
                    "best_val_by_family": router_summary["best_val_by_family"],
                    "positive_weight": router_summary["positive_weight"],
                    "supervision_target": router_summary["supervision_target"],
                    "pretrain_target": router_summary["pretrain_target"],
                    "pretrain_epochs": router_summary["pretrain_epochs"],
                    "pretrain_positive_weight": router_summary["pretrain_positive_weight"],
                    "pretrain_epoch_history": router_summary["pretrain_epoch_history"],
                    "negative_ratio": router_summary["negative_ratio"],
                    "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
                    "selected_router_threshold": (
                        float(selector_val_result["decision"].get("router_threshold"))
                        if selector_val_result is not None
                        and selector_val_result["decision"].get("router_threshold") is not None
                        else None
                    ),
                }
                if router_summary is not None
                else None
            ),
            "action_motif_training": (
                {
                    "vocabulary_num_classes": int(action_motif_vocabulary.mask_table.shape[0]),
                    "active_fraction": (
                        float(np.mean(train_action_motif_active >= 0.5))
                        if train_action_motif_active.size
                        else None
                    ),
                    "nonzero_active_fraction": (
                        float(
                            np.mean(
                                (train_action_motif_active >= 0.5)
                                & (train_action_motif_label > 0)
                            )
                        )
                        if train_action_motif_active.size
                        else None
                    ),
                    "loss_weight": float(action_motif_loss_weight),
                    "identity_margin": float(action_motif_identity_margin),
                    "emit_margin_grid": [float(x) for x in action_emit_grid],
                    "selected_emit_margin": (
                        float(action_motif_val_result["decision"]["action_motif_emit_margin"])
                        if action_motif_val_result is not None
                        else None
                    ),
                }
                if action_motif_vocabulary is not None
                else None
            ),
            "local_motif_training": (
                {
                    "vocabulary_num_patterns": int(len(local_motif_vocabulary.offset_patterns)),
                    "emit_margin_grid": [float(x) for x in local_emit_grid],
                    "min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
                    "selected_emit_margin": (
                        float(local_motif_val_result["decision"]["local_motif_emit_margin"])
                        if local_motif_val_result is not None
                        else None
                    ),
                    "selected_min_bit_logit": (
                        float(local_motif_val_result["decision"]["local_motif_min_bit_logit"])
                        if local_motif_val_result is not None
                        else None
                    ),
                }
                if local_motif_vocabulary is not None
                else None
            ),
            "motif_training": (
                {
                    "epoch_history": motif_summary["epoch_history"],
                    "best_epoch": motif_summary["best_epoch"],
                    "best_val_selection_metric": motif_summary["best_val_selection_metric"],
                    "best_val_by_family": motif_summary["best_val_by_family"],
                    "hard_shot_weight": motif_summary["hard_shot_weight"],
                    "vocabulary_num_classes": int(motif_summary["vocabulary"].mask_table.shape[0]),
                }
                if motif_summary is not None
                else None
            ),
        },
        "metrics": {
            "best_val": selected_val_result,
            "test": selected_test_result,
            "best_val_no_edit": no_edit_val_result,
            "test_no_edit": no_edit_test_result,
            "best_val_global_policy": global_val_result,
            "test_global_policy": global_test_result,
            "best_val_candidate_selector": selector_val_result,
            "test_candidate_selector": selector_test_result,
            "best_val_motif_vocab": motif_val_result,
            "test_motif_vocab": motif_test_result,
            "best_val_action_motif": action_motif_val_result,
            "test_action_motif": action_motif_test_result,
            "best_val_local_motif": local_motif_val_result,
            "test_local_motif": local_motif_test_result,
        },
        "artifacts": {
            "checkpoint": checkpoint_out.as_posix(),
        },
    }
    _write_json(out_json, result)
    return result


def evaluate_checkpoint_on_family(
    *,
    checkpoint_path: Path,
    family_dir: Path,
    out_json: Path,
    fill_value: float = -0.5,
    max_shots: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    entry = _prepare_edit_family(family_dir, fill_value=fill_value, max_shots=max_shots)
    model_kwargs = dict(checkpoint["model_kwargs"])
    model = SyndromeEditPreDecoder(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    decision = dict(checkpoint.get("decision_policy", {}))
    inference = dict(checkpoint.get("inference", {}))
    candidate_policy_grid = list(inference.get("candidate_policy_grid", []))
    policy_specs = [
        CandidatePolicySpec(
            needs_edit_threshold=float(item["needs_edit_threshold"]),
            edit_threshold=float(item["edit_threshold"]),
            max_predicted_edit_weight=int(item["max_predicted_edit_weight"]),
        )
        for item in candidate_policy_grid
    ]
    global_result = _system_metrics_for_subset(
        entry=entry,
        subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
        model=model,
        batch_size=batch_size,
        device=device,
        needs_edit_threshold=float(decision.get("needs_edit_threshold", 0.5)),
        edit_threshold=float(decision.get("edit_threshold", 0.5)),
        max_predicted_edit_weight=int(decision.get("max_predicted_edit_weight", 2)),
    )
    no_edit_result = _no_edit_system_metrics_for_subset(
        entry=entry,
        subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
    )
    selector_result: dict[str, Any] | None = None
    motif_result: dict[str, Any] | None = None
    action_motif_result: dict[str, Any] | None = None
    local_motif_result: dict[str, Any] | None = None
    candidate_selector_payload = checkpoint.get("candidate_selector")
    hard_shot_router_payload = checkpoint.get("hard_shot_router")
    motif_payload = checkpoint.get("motif_vocabulary_head")
    action_motif_payload = checkpoint.get("action_motif_vocabulary")
    local_motif_payload = checkpoint.get("local_motif_vocabulary")
    selected_inference_mode = str(inference.get("selected_inference_mode", SELECTION_MODE_GLOBAL_POLICY))
    if (
        isinstance(candidate_selector_payload, dict)
        and policy_specs
    ):
        selector_kwargs = dict(candidate_selector_payload.get("selector_kwargs", {}))
        selector_training_summary = dict(candidate_selector_payload.get("training_summary", {}))
        selector_model = str(
            candidate_selector_payload.get(
                "selector_model",
                selector_training_summary.get("selector_model", SELECTOR_MODEL_SCALAR),
            )
        )
        selector = _make_candidate_selector_module(
            selector_model=selector_model,
            selector_kwargs=selector_kwargs,
        )
        selector.load_state_dict(candidate_selector_payload["selector_state"])
        selector = selector.to(device)
        transition_prior_head: nn.Module | None = None
        transition_prior_payload = candidate_selector_payload.get("transition_prior")
        if isinstance(transition_prior_payload, dict):
            transition_prior_kwargs = dict(transition_prior_payload.get("head_kwargs", {}))
            transition_prior_head = MotifVocabularyHead(**transition_prior_kwargs)
            transition_prior_head.load_state_dict(transition_prior_payload["head_state"])
            transition_prior_head = transition_prior_head.to(device)
        candidate_compatibility_head: nn.Module | None = None
        candidate_compatibility_payload = candidate_selector_payload.get("candidate_compatibility")
        if isinstance(candidate_compatibility_payload, dict):
            candidate_compatibility_kwargs = dict(candidate_compatibility_payload.get("head_kwargs", {}))
            candidate_compatibility_head = CandidateEditSelector(**candidate_compatibility_kwargs)
            candidate_compatibility_head.load_state_dict(candidate_compatibility_payload["head_state"])
            candidate_compatibility_head = candidate_compatibility_head.to(device)
        router: nn.Module | None = None
        router_threshold = 0.5
        if isinstance(hard_shot_router_payload, dict):
            router_kwargs = dict(hard_shot_router_payload.get("router_kwargs", {}))
            router = HardShotRouter(**router_kwargs)
            router.load_state_dict(hard_shot_router_payload["router_state"])
            router = router.to(device)
            router_training_summary = dict(hard_shot_router_payload.get("training_summary", {}))
            raw_threshold = router_training_summary.get("selected_router_threshold", 0.5)
            router_threshold = 0.5 if raw_threshold is None else float(raw_threshold)
        candidate_motif_payload = candidate_selector_payload.get("candidate_motif_vocabulary")
        candidate_local_motif_payload = candidate_selector_payload.get("candidate_local_motif_vocabulary")
        candidate_motif_vocabulary: MotifVocabulary | None = None
        if isinstance(candidate_motif_payload, dict):
            candidate_motif_vocabulary = MotifVocabulary(
                mask_table=np.asarray(candidate_motif_payload["mask_table"], dtype=np.uint8),
                detector_index_lists=tuple(
                    tuple(int(x) for x in row) for row in candidate_motif_payload["detector_index_lists"]
                ),
                counts=tuple(int(x) for x in candidate_motif_payload["counts"]),
                detector_count=int(np.asarray(candidate_motif_payload["mask_table"]).shape[1]),
            )
        candidate_local_motif_vocabulary: LocalMotifVocabulary | None = None
        if isinstance(candidate_local_motif_payload, dict):
            candidate_local_motif_vocabulary = LocalMotifVocabulary(
                offset_patterns=tuple(
                    tuple(tuple(int(v) for v in offset) for offset in pattern)
                    for pattern in candidate_local_motif_payload["offset_patterns"]
                ),
                counts=tuple(int(x) for x in candidate_local_motif_payload["counts"]),
                detector_count=int(candidate_local_motif_payload.get("detector_count", 0) or 0),
            )
        selector_result = _selector_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
            model=model,
            selector=selector,
            batch_size=batch_size,
            device=device,
            policy_specs=policy_specs,
            selector_score_edit_penalty=float(selector_training_summary.get("selector_score_edit_penalty", DEFAULT_SELECTOR_SCORE_EDIT_PENALTY)),
            selector_target_mode=str(selector_training_summary.get("selector_target_mode", SELECTOR_TARGET_MODE_CORRECTNESS)),
            selector_harm_weight=float(selector_training_summary.get("selector_harm_weight", DEFAULT_SELECTOR_HARM_WEIGHT)),
            selector_miss_weight=float(selector_training_summary.get("selector_miss_weight", DEFAULT_SELECTOR_MISS_WEIGHT)),
            selector_policy_candidate_mode=str(selector_training_summary.get("selector_policy_candidate_mode", SELECTOR_POLICY_CANDIDATE_MODE_ALL)),
            selector_candidate_geometry_features=bool(
                selector_training_summary.get(
                    "selector_candidate_geometry_features",
                    DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
                )
            ),
            selector_candidate_pattern_features=bool(
                selector_training_summary.get(
                    "selector_candidate_pattern_features",
                    DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
                )
            ),
            selector_candidate_local_evidence_features=bool(
                selector_training_summary.get(
                    "selector_candidate_local_evidence_features",
                    DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
                )
            ),
            selector_candidate_local_patch_features=bool(
                selector_training_summary.get(
                    "selector_candidate_local_patch_features",
                    DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
                )
            ),
            motif_vocabulary=candidate_motif_vocabulary,
            local_motif_vocabulary=candidate_local_motif_vocabulary,
            local_motif_top_k=int(selector_training_summary.get("selector_local_motif_top_k", DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K)),
            selector_selection_mode=str(selector_training_summary.get("selector_selection_mode", selected_inference_mode)),
            selector_emit_margin=float(selector_training_summary.get("selected_selector_emit_margin", 0.0) or 0.0),
            selector_nonzero_bias=float(selector_training_summary.get("selected_selector_nonzero_bias", 0.0) or 0.0),
            transition_prior_head=transition_prior_head,
            selector_transition_prior_weight=float(
                selector_training_summary.get("selected_selector_transition_prior_weight", 0.0) or 0.0
            ),
            selector_transition_compat_top_k=int(
                selector_training_summary.get("selected_selector_transition_compat_top_k", 0) or 0
            ),
            candidate_compatibility_head=candidate_compatibility_head,
            selector_candidate_compat_threshold=float(
                selector_training_summary.get("selected_selector_candidate_compat_threshold", 0.0) or 0.0
            ),
            selector_candidate_compat_top_k=int(
                selector_training_summary.get("selected_selector_candidate_compat_top_k", 0) or 0
            ),
            router=router,
            router_threshold=float(router_threshold),
            routed_nonzero_only=(selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER),
        )
    if isinstance(motif_payload, dict):
        motif_kwargs = dict(motif_payload.get("motif_kwargs", {}))
        motif_head = MotifVocabularyHead(**motif_kwargs)
        motif_head.load_state_dict(motif_payload["motif_state"])
        motif_head = motif_head.to(device)
        vocabulary = MotifVocabulary(
            mask_table=np.asarray(motif_payload["vocabulary_mask_table"], dtype=np.uint8),
            detector_index_lists=tuple(tuple(int(x) for x in row) for row in motif_payload["vocabulary_detector_index_lists"]),
            counts=tuple(int(x) for x in motif_payload["vocabulary_counts"]),
            detector_count=int(np.asarray(motif_payload["vocabulary_mask_table"]).shape[1]),
        )
        motif_result = _motif_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
            model=model,
            motif_head=motif_head,
            vocabulary=vocabulary,
            batch_size=batch_size,
            device=device,
        )
    if isinstance(action_motif_payload, dict):
        action_motif_training_summary = dict(action_motif_payload.get("training_summary", {}))
        action_motif_emit_margin = float(action_motif_training_summary.get("selected_emit_margin", 0.0) or 0.0)
        action_motif_vocabulary = MotifVocabulary(
            mask_table=np.asarray(action_motif_payload["mask_table"], dtype=np.uint8),
            detector_index_lists=tuple(
                tuple(int(x) for x in row) for row in action_motif_payload["detector_index_lists"]
            ),
            counts=tuple(int(x) for x in action_motif_payload["counts"]),
            detector_count=int(np.asarray(action_motif_payload["mask_table"]).shape[1]),
        )
        action_motif_result = _action_motif_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
            model=model,
            vocabulary=action_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin=action_motif_emit_margin,
        )
    if isinstance(local_motif_payload, dict):
        local_motif_training_summary = dict(local_motif_payload.get("training_summary", {}))
        local_motif_emit_margin = float(local_motif_training_summary.get("selected_emit_margin", 0.0) or 0.0)
        local_motif_min_bit_logit_raw = local_motif_training_summary.get("selected_min_bit_logit", -1.0)
        local_motif_min_bit_logit = (
            -1.0 if local_motif_min_bit_logit_raw is None else float(local_motif_min_bit_logit_raw)
        )
        local_motif_vocabulary = LocalMotifVocabulary(
            offset_patterns=tuple(
                tuple(tuple(int(v) for v in offset) for offset in pattern)
                for pattern in local_motif_payload["offset_patterns"]
            ),
            counts=tuple(int(x) for x in local_motif_payload["counts"]),
            detector_count=int(local_motif_payload.get("detector_count", 0) or 0),
        )
        local_motif_result = _local_motif_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
            model=model,
            vocabulary=local_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin=local_motif_emit_margin,
            min_bit_logit=local_motif_min_bit_logit,
        )
    result_metrics = (
        selector_result
        if selector_result is not None and selected_inference_mode == SELECTION_MODE_CANDIDATE_SELECTOR
        else motif_result
        if motif_result is not None and selected_inference_mode == SELECTION_MODE_MOTIF_VOCAB
        else action_motif_result
        if action_motif_result is not None and selected_inference_mode == SELECTION_MODE_ACTION_MOTIF
        else local_motif_result
        if local_motif_result is not None and selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF
        else selector_result
        if selector_result is not None and selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_SELECTOR
        else selector_result
        if selector_result is not None and selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER
        else no_edit_result
        if selected_inference_mode == SELECTION_MODE_RAW_NO_EDIT
        else global_result
    )
    result = {
        "schema_version": SCHEMA_VERSION_EVAL,
        "decoder": "syndrome_edit_predecoder",
        "created_at_utc": _utc_now_iso(),
        "input_mode": "family_dir",
        "family": entry.family,
        "stage": entry.stage,
        "family_dir": entry.family_dir.as_posix(),
        "source_family_dir": entry.source_family_dir.as_posix(),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "decision_policy": decision,
            "inference": inference,
        },
        "model": {
            "model_class": "SyndromeEditPreDecoder",
            "model_kwargs": model_kwargs,
            "num_parameters": int(sum(param.numel() for param in model.parameters())),
        },
        "dataset": {
            "num_shots": int(entry.x.shape[0]),
            "input_shape": list(entry.x.shape),
            "channel_names": list(entry.bundle_info.get("channel_names", [])),
            "matching": entry.matching_info,
        },
        "metrics": {
            "selected": result_metrics,
            "no_edit": no_edit_result,
            "global_policy": global_result,
            "candidate_selector": selector_result,
            "motif_vocab": motif_result,
            "action_motif": action_motif_result,
            "local_motif": local_motif_result,
        },
    }
    _write_json(out_json, result)
    return result


def _resolve_manifest_family_entries(
    manifest_path: Path,
    requested_families: list[str] | None,
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    manifest_data = _read_json(manifest_path)
    family_dirs = manifest_data.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(f"manifest.json does not contain a non-empty family_dirs mapping: {manifest_path}")
    families = list(requested_families) if requested_families else list(family_dirs.keys())
    resolved: list[tuple[str, Path]] = []
    for family in families:
        if family not in family_dirs:
            raise KeyError(f"Requested family {family!r} missing from manifest. Available: {sorted(family_dirs)}")
        resolved.append((str(family), common._resolve_manifest_family_dir(manifest_path, family_dirs[family])))
    return manifest_data, resolved


def run_manifest_experiment(
    *,
    manifest_path: Path,
    train_families: list[str],
    eval_families: list[str] | None,
    out_dir: Path,
    fill_value: float = -0.5,
    max_shots: int | None = None,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    hidden_channels: int = 24,
    num_blocks: int = 3,
    dense_hidden_dim: int = 64,
    dropout: float = 0.1,
    needs_edit_loss_weight: float = 0.5,
    sparsity_loss_weight: float = 0.01,
    decision_aware_loss_weight: float = DEFAULT_DECISION_AWARE_LOSS_WEIGHT,
    decision_aware_margin: float = DEFAULT_DECISION_AWARE_MARGIN,
    hard_shot_solved_weight: float = DEFAULT_HARD_SHOT_SOLVED_WEIGHT,
    hard_shot_unsolved_weight: float = DEFAULT_HARD_SHOT_UNSOLVED_WEIGHT,
    edit_supervision_mode: str = EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY,
    selection_mode: str = SELECTION_MODE_CANDIDATE_SELECTOR,
    selector_hidden_dim: int = DEFAULT_SELECTOR_HIDDEN_DIM,
    selector_epochs: int = DEFAULT_SELECTOR_EPOCHS,
    selector_lr: float = DEFAULT_SELECTOR_LR,
    selector_objective: str = SELECTOR_OBJECTIVE_GROUP_RANK,
    selector_hard_shot_weight: float = DEFAULT_SELECTOR_HARD_SHOT_WEIGHT,
    selector_identity_margin_loss_weight: float = DEFAULT_SELECTOR_IDENTITY_MARGIN_LOSS_WEIGHT,
    selector_identity_margin: float = DEFAULT_SELECTOR_IDENTITY_MARGIN,
    selector_harm_margin_loss_weight: float = DEFAULT_SELECTOR_HARM_MARGIN_LOSS_WEIGHT,
    selector_harm_margin: float = DEFAULT_SELECTOR_HARM_MARGIN,
    selector_negative_identity_margin_loss_weight: float = DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN_LOSS_WEIGHT,
    selector_negative_identity_margin: float = DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN,
    selector_benefit_harm_pairwise_loss_weight: float = DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_LOSS_WEIGHT,
    selector_benefit_harm_pairwise_margin: float = DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_MARGIN,
    selector_positive_negative_hard_loss_weight: float = DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_LOSS_WEIGHT,
    selector_positive_negative_hard_margin: float = DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_MARGIN,
    selector_cross_family_positive_negative_loss_weight: float = (
        DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_LOSS_WEIGHT
    ),
    selector_cross_family_positive_negative_margin: float = (
        DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_MARGIN
    ),
    selector_model: str = SELECTOR_MODEL_SCALAR,
    selector_risk_aware_harm_logit_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
    selector_risk_aware_benefit_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_LOSS_WEIGHT,
    selector_risk_aware_harm_loss_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_LOSS_WEIGHT,
    selector_risk_aware_benefit_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_POS_WEIGHT,
    selector_risk_aware_harm_pos_weight: float = DEFAULT_SELECTOR_RISK_AWARE_HARM_POS_WEIGHT,
    selector_patch_head: bool = DEFAULT_SELECTOR_PATCH_HEAD,
    selector_patch_hidden_dim: int = DEFAULT_SELECTOR_PATCH_HIDDEN_DIM,
    selector_emit_margin_grid: list[float] | None = None,
    selector_nonzero_bias_grid: list[float] | None = None,
    selector_transition_prior_weight_grid: list[float] | None = None,
    selector_transition_compat_top_k_grid: list[int] | None = None,
    selector_candidate_compat_threshold_grid: list[float] | None = None,
    selector_candidate_compat_top_k_grid: list[int] | None = None,
    candidate_compat_hidden_dim: int = DEFAULT_CANDIDATE_COMPAT_HIDDEN_DIM,
    candidate_compat_epochs: int = DEFAULT_CANDIDATE_COMPAT_EPOCHS,
    candidate_compat_lr: float = DEFAULT_CANDIDATE_COMPAT_LR,
    candidate_compat_objective: str = CANDIDATE_COMPAT_OBJECTIVE_BCE,
    candidate_compat_negative_ratio: float = DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO,
    candidate_compat_no_positive_negative_count: int = DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT,
    transition_prior_hidden_dim: int = DEFAULT_TRANSITION_PRIOR_HIDDEN_DIM,
    transition_prior_epochs: int = DEFAULT_TRANSITION_PRIOR_EPOCHS,
    transition_prior_lr: float = DEFAULT_TRANSITION_PRIOR_LR,
    selector_score_edit_penalty: float = DEFAULT_SELECTOR_SCORE_EDIT_PENALTY,
    selector_target_mode: str = SELECTOR_TARGET_MODE_CORRECTNESS,
    selector_harm_weight: float = DEFAULT_SELECTOR_HARM_WEIGHT,
    selector_miss_weight: float = DEFAULT_SELECTOR_MISS_WEIGHT,
    selector_policy_candidate_mode: str = SELECTOR_POLICY_CANDIDATE_MODE_ALL,
    selector_candidate_geometry_features: bool = DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
    selector_candidate_pattern_features: bool = DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
    selector_candidate_local_evidence_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
    selector_candidate_local_patch_features: bool = DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
    selector_epoch_diagnostic_margin_grid: list[float] | None = None,
    selector_epoch_selection_mode: str = SELECTOR_EPOCH_SELECTION_PROXY,
    selector_adoption_min_delta: float = DEFAULT_SELECTOR_ADOPTION_MIN_DELTA,
    selector_adoption_policy: str = SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR,
    selector_candidate_first_strong_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_STRONG_DELTA,
    selector_candidate_first_positive_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_DELTA,
    selector_candidate_first_positive_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MARGIN_FLOOR,
    selector_candidate_first_positive_max_harmed: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED,
    selector_candidate_first_positive_max_margin: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_MARGIN,
    selector_candidate_first_positive_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MIN_NONZERO,
    selector_candidate_first_positive_plateau_guard: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_PLATEAU_GUARD,
    selector_candidate_first_tie_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_DELTA,
    selector_candidate_first_tie_margin_floor: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MARGIN_FLOOR,
    selector_candidate_first_tie_min_nonzero: int = DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_NONZERO,
    selector_candidate_first_allow_global: bool = DEFAULT_SELECTOR_CANDIDATE_FIRST_ALLOW_GLOBAL,
    selector_candidate_first_global_min_delta: float = DEFAULT_SELECTOR_CANDIDATE_FIRST_GLOBAL_MIN_DELTA,
    selected_no_edit_guardrail: bool = DEFAULT_SELECTED_NO_EDIT_GUARDRAIL,
    selected_no_edit_min_delta: float = DEFAULT_SELECTED_NO_EDIT_MIN_DELTA,
    selector_candidate_motif_max_classes: int = DEFAULT_SELECTOR_CANDIDATE_MOTIF_MAX_CLASSES,
    selector_local_motif_max_classes: int = DEFAULT_SELECTOR_LOCAL_MOTIF_MAX_CLASSES,
    selector_local_motif_top_k: int = DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K,
    router_hidden_dim: int = DEFAULT_ROUTER_HIDDEN_DIM,
    router_epochs: int = DEFAULT_ROUTER_EPOCHS,
    router_lr: float = DEFAULT_ROUTER_LR,
    router_pos_weight: float = DEFAULT_ROUTER_POS_WEIGHT,
    router_threshold_grid: list[float] | None = None,
    router_supervision_target: str = ROUTER_LABEL_IDENTITY_VS_NONZERO,
    router_pretrain_target: str = ROUTER_PRETRAIN_TARGET_NONE,
    router_pretrain_epochs: int = 0,
    router_pretrain_pos_weight: float | None = None,
    router_negative_ratio: float | None = None,
    action_motif_max_classes: int = DEFAULT_ACTION_MOTIF_MAX_CLASSES,
    action_motif_loss_weight: float = DEFAULT_ACTION_MOTIF_LOSS_WEIGHT,
    action_motif_identity_margin: float = DEFAULT_ACTION_MOTIF_IDENTITY_MARGIN,
    action_motif_emit_margin_grid: list[float] | None = None,
    local_motif_max_classes: int = DEFAULT_LOCAL_MOTIF_MAX_CLASSES,
    local_motif_emit_margin_grid: list[float] | None = None,
    local_motif_min_bit_logit_grid: list[float] | None = None,
    motif_max_classes: int = DEFAULT_MOTIF_MAX_CLASSES,
    motif_epochs: int = DEFAULT_MOTIF_EPOCHS,
    motif_lr: float = DEFAULT_MOTIF_LR,
    motif_hard_shot_weight: float = DEFAULT_MOTIF_HARD_SHOT_WEIGHT,
    needs_edit_threshold_grid: list[float] | None = None,
    edit_threshold_grid: list[float] | None = None,
    max_edit_weight_grid: list[int] | None = None,
) -> dict[str, Any]:
    _require_torch()
    common._set_random_seeds(int(seed))
    if str(selector_adoption_policy) not in SELECTOR_ADOPTION_POLICY_CHOICES:
        raise ValueError(f"Unknown selector adoption policy: {selector_adoption_policy!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_data, resolved_train = _resolve_manifest_family_entries(manifest_path, train_families)
    train_entries = [
        _prepare_edit_family(path, fill_value=fill_value, max_shots=max_shots)
        for _family, path in resolved_train
    ]
    compatibility = _validate_compatible_entries(train_entries)

    split_by_family: dict[str, SplitBundle] = {}
    train_subsets: list[dict[str, np.ndarray]] = []
    val_results_for_selection: list[dict[str, Any]] = []
    for offset, entry in enumerate(train_entries):
        split = _build_split_bundle(
            num_shots=int(entry.x.shape[0]),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + offset,
        )
        split_by_family[entry.family] = split
        train_subsets.append(_subset_family(entry, split.train))
    train_bundle = _concatenate_subsets(train_subsets)
    action_motif_vocabulary: MotifVocabulary | None = None
    local_motif_vocabulary: LocalMotifVocabulary | None = None
    train_action_motif_label = np.full((int(train_bundle["x"].shape[0]),), -1, dtype=np.int64)
    train_action_motif_active = np.zeros((int(train_bundle["x"].shape[0]),), dtype=np.float32)
    if int(action_motif_max_classes) >= 2 and float(action_motif_loss_weight) > 0.0:
        action_motif_vocabulary = _build_motif_vocabulary(
            [
                (entry, _subset_family(entry, split_by_family[entry.family].train))
                for entry in train_entries
            ],
            max_classes=int(action_motif_max_classes),
        )
        if action_motif_vocabulary.mask_table.shape[0] <= 1:
            action_motif_vocabulary = None
        else:
            label_parts: list[np.ndarray] = []
            active_parts: list[np.ndarray] = []
            for entry in train_entries:
                labels_part, active_part = _build_action_motif_supervision_arrays(
                    entry=entry,
                    subset=_subset_family(entry, split_by_family[entry.family].train),
                    vocabulary=action_motif_vocabulary,
                )
                label_parts.append(labels_part)
                active_parts.append(active_part)
            train_action_motif_label = np.ascontiguousarray(np.concatenate(label_parts, axis=0), dtype=np.int64)
            train_action_motif_active = np.ascontiguousarray(np.concatenate(active_parts, axis=0), dtype=np.float32)
    if int(local_motif_max_classes) >= 2:
        local_motif_vocabulary = _build_local_motif_vocabulary(
            [
                (entry, _subset_family(entry, split_by_family[entry.family].train))
                for entry in train_entries
            ],
            max_classes=int(local_motif_max_classes),
        )
        if len(local_motif_vocabulary.offset_patterns) == 0:
            local_motif_vocabulary = None

    shot_sample_weights = _compute_shot_sample_weights(
        needs_edit=train_bundle["needs_edit"],
        edit_target_known=train_bundle["edit_target_known"],
        hard_shot_solved_weight=hard_shot_solved_weight,
        hard_shot_unsolved_weight=hard_shot_unsolved_weight,
    )
    training_config = _build_training_config(
        edit_target_volume=train_bundle["edit_target_volume"],
        needs_edit=train_bundle["needs_edit"],
        edit_target_known=train_bundle["edit_target_known"],
        valid_mask_volume=train_entries[0].valid_mask_volume,
        shot_sample_weights=shot_sample_weights,
        edit_supervision_mode=edit_supervision_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_mask_volume_t = torch.from_numpy(np.ascontiguousarray(train_entries[0].valid_mask_volume)).to(device)
    action_motif_mask_table_t = (
        torch.from_numpy(np.ascontiguousarray(action_motif_vocabulary.mask_table, dtype=np.float32)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_detector_time_index_t = (
        torch.from_numpy(np.ascontiguousarray(train_entries[0].detector_time_index, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_row_index_by_detector_t = (
        torch.from_numpy(np.ascontiguousarray(train_entries[0].row_index_by_detector, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    action_motif_col_index_by_detector_t = (
        torch.from_numpy(np.ascontiguousarray(train_entries[0].col_index_by_detector, dtype=np.int64)).to(device)
        if action_motif_vocabulary is not None
        else None
    )
    model_kwargs = {
        "in_channels": int(train_entries[0].x.shape[1]),
        "hidden_channels": int(hidden_channels),
        "num_blocks": int(num_blocks),
        "dense_hidden_dim": int(dense_hidden_dim),
        "dropout": float(dropout),
        "valid_mask_channel_index": int(_infer_valid_mask_channel_index(train_entries[0].bundle_info)),
    }
    model = SyndromeEditPreDecoder(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    edit_pos_weight_t = torch.tensor([training_config["edit_pos_weight"]], dtype=torch.float32, device=device)
    needs_pos_weight_t = torch.tensor([training_config["needs_pos_weight"]], dtype=torch.float32, device=device)
    train_dataset = _make_tensor_dataset(
        train_bundle["x"],
        train_bundle["edit_target_volume"],
        train_bundle["needs_edit"],
        train_bundle["edit_target_known"],
        train_action_motif_label,
        train_action_motif_active,
    )
    train_loader = _make_weighted_loader(
        train_dataset,
        batch_size=batch_size,
        sample_weights=shot_sample_weights,
    )

    needs_grid = needs_edit_threshold_grid or [0.3, 0.5, 0.7, 0.9]
    edit_grid = edit_threshold_grid or [0.3, 0.5, 0.7, 0.9]
    max_weight_grid = max_edit_weight_grid or [0, 1, 2]
    action_emit_grid = action_motif_emit_margin_grid or list(DEFAULT_ACTION_MOTIF_EMIT_MARGIN_GRID)
    local_emit_grid = local_motif_emit_margin_grid or list(DEFAULT_LOCAL_MOTIF_EMIT_MARGIN_GRID)
    local_min_bit_grid = local_motif_min_bit_logit_grid or list(DEFAULT_LOCAL_MOTIF_MIN_BIT_LOGIT_GRID)
    selector_emit_grid = selector_emit_margin_grid or list(DEFAULT_SELECTOR_EMIT_MARGIN_GRID)
    selector_nonzero_bias_values = selector_nonzero_bias_grid or list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID)
    selector_transition_prior_weight_values = (
        selector_transition_prior_weight_grid
        or list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID)
    )
    selector_transition_compat_top_k_values = (
        selector_transition_compat_top_k_grid
        or list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID)
    )
    selector_candidate_compat_threshold_values = (
        selector_candidate_compat_threshold_grid
        or list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID)
    )
    selector_candidate_compat_top_k_values = (
        selector_candidate_compat_top_k_grid
        or list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID)
    )
    router_threshold_grid_values = router_threshold_grid or list(DEFAULT_ROUTER_THRESHOLD_GRID)
    policy_specs = _build_candidate_policy_specs(
        needs_edit_threshold_grid=needs_grid,
        edit_threshold_grid=edit_grid,
        max_edit_weight_grid=max_weight_grid,
    )
    best_state: dict[str, Any] | None = None
    best_val_metric = -math.inf
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            valid_mask_volume=valid_mask_volume_t,
            edit_pos_weight=edit_pos_weight_t,
            needs_pos_weight=needs_pos_weight_t,
            needs_edit_loss_weight=needs_edit_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            decision_aware_loss_weight=decision_aware_loss_weight,
            decision_aware_margin=decision_aware_margin,
            action_motif_loss_weight=action_motif_loss_weight,
            action_motif_identity_margin=action_motif_identity_margin,
            action_motif_mask_table=action_motif_mask_table_t,
            action_motif_detector_time_index=action_motif_detector_time_index_t,
            action_motif_row_index_by_detector=action_motif_row_index_by_detector_t,
            action_motif_col_index_by_detector=action_motif_col_index_by_detector_t,
            edit_supervision_mode=edit_supervision_mode,
        )
        val_family_results: dict[str, Any] = {}
        for entry in train_entries:
            subset = _subset_family(entry, split_by_family[entry.family].val)
            val_family_results[entry.family] = _grid_search_decision_policy(
                entry=entry,
                subset=subset,
                model=model,
                batch_size=batch_size,
                device=device,
                needs_edit_threshold_grid=needs_grid,
                edit_threshold_grid=edit_grid,
                max_edit_weight_grid=max_weight_grid,
            )
        val_metric = _mean_system_metric(list(val_family_results.values()))
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val_by_family": val_family_results,
            "val_selection_metric": float(val_metric),
        }
        epoch_history.append(epoch_record)
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "val_by_family": copy.deepcopy(val_family_results),
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    model.load_state_dict(best_state["model_state"])

    global_val_by_family = copy.deepcopy(best_state["val_by_family"])
    global_val_metric = _mean_system_metric(list(global_val_by_family.values()))
    no_edit_val_by_family = {
        entry.family: _no_edit_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, split_by_family[entry.family].val),
        )
        for entry in train_entries
    }
    no_edit_val_metric = _mean_system_metric(list(no_edit_val_by_family.values()))
    selected_decision = list(best_state["val_by_family"].values())[0]["decision"]
    checkpoint_path = out_dir / "sedp_checkpoint.pt"
    selector_summary: dict[str, Any] | None = None
    transition_prior_summary: dict[str, Any] | None = None
    candidate_compatibility_summary: dict[str, Any] | None = None
    router_summary: dict[str, Any] | None = None
    selector_val_by_family: dict[str, Any] | None = None
    selector_candidate_motif_vocabulary: MotifVocabulary | None = None
    selector_local_motif_vocabulary: LocalMotifVocabulary | None = None
    motif_summary: dict[str, Any] | None = None
    motif_val_by_family: dict[str, Any] | None = None
    action_motif_val_by_family: dict[str, Any] | None = None
    local_motif_val_by_family: dict[str, Any] | None = None
    selected_inference_mode = SELECTION_MODE_GLOBAL_POLICY
    selector_adoption_decision: dict[str, Any] = {
        "policy": str(selector_adoption_policy),
        "requested_mode": str(selection_mode),
        "selected_mode": str(selected_inference_mode),
        "reason": "selector_not_evaluated",
    }
    if selection_mode in (
        SELECTION_MODE_CANDIDATE_SELECTOR,
        SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
        SELECTION_MODE_LOCAL_MOTIF_ROUTER,
    ):
        if int(selector_candidate_motif_max_classes) >= 2:
            selector_candidate_motif_vocabulary = _build_motif_vocabulary(
                [
                    (entry, _subset_family(entry, split_by_family[entry.family].train))
                    for entry in train_entries
                ],
                max_classes=int(selector_candidate_motif_max_classes),
            )
            if selector_candidate_motif_vocabulary.mask_table.shape[0] <= 1:
                selector_candidate_motif_vocabulary = None
        if int(selector_local_motif_max_classes) >= 2:
            selector_local_motif_vocabulary = _build_local_motif_vocabulary(
                [
                    (entry, _subset_family(entry, split_by_family[entry.family].train))
                    for entry in train_entries
                ],
                max_classes=int(selector_local_motif_max_classes),
            )
            if len(selector_local_motif_vocabulary.offset_patterns) == 0:
                selector_local_motif_vocabulary = None
        train_selector_bundles = [
            _build_selector_candidate_bundle(
                entry=entry,
                subset=_subset_family(entry, split_by_family[entry.family].train),
                model=model,
                batch_size=batch_size,
                device=device,
                policy_specs=policy_specs,
                selector_score_edit_penalty=selector_score_edit_penalty,
                selector_target_mode=str(selector_target_mode),
                selector_harm_weight=float(selector_harm_weight),
                selector_miss_weight=float(selector_miss_weight),
                selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                motif_vocabulary=selector_candidate_motif_vocabulary,
                local_motif_vocabulary=selector_local_motif_vocabulary,
                local_motif_top_k=int(selector_local_motif_top_k),
            )
            for entry in train_entries
        ]
        val_selector_bundles_by_family = {
            entry.family: _build_selector_candidate_bundle(
                entry=entry,
                subset=_subset_family(entry, split_by_family[entry.family].val),
                model=model,
                batch_size=batch_size,
                device=device,
                policy_specs=policy_specs,
                selector_score_edit_penalty=selector_score_edit_penalty,
                selector_target_mode=str(selector_target_mode),
                selector_harm_weight=float(selector_harm_weight),
                selector_miss_weight=float(selector_miss_weight),
                selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                motif_vocabulary=selector_candidate_motif_vocabulary,
                local_motif_vocabulary=selector_local_motif_vocabulary,
                local_motif_top_k=int(selector_local_motif_top_k),
            )
            for entry in train_entries
        }
        selector_summary = _train_candidate_selector(
            model=model,
            train_bundles=train_selector_bundles,
            val_bundles_by_family=val_selector_bundles_by_family,
            entry_by_family={entry.family: entry for entry in train_entries},
            subset_by_family={
                entry.family: _subset_family(entry, split_by_family[entry.family].val)
                for entry in train_entries
            },
            batch_size=batch_size,
            device=device,
            hidden_dim=selector_hidden_dim,
            dropout=dropout,
            lr=selector_lr,
            epochs=selector_epochs,
            selector_objective=selector_objective,
            selector_hard_shot_weight=selector_hard_shot_weight,
            selector_identity_margin_loss_weight=selector_identity_margin_loss_weight,
            selector_identity_margin=selector_identity_margin,
            selector_harm_margin_loss_weight=selector_harm_margin_loss_weight,
            selector_harm_margin=selector_harm_margin,
            selector_negative_identity_margin_loss_weight=selector_negative_identity_margin_loss_weight,
            selector_negative_identity_margin=selector_negative_identity_margin,
            selector_benefit_harm_pairwise_loss_weight=selector_benefit_harm_pairwise_loss_weight,
            selector_benefit_harm_pairwise_margin=selector_benefit_harm_pairwise_margin,
            selector_positive_negative_hard_loss_weight=selector_positive_negative_hard_loss_weight,
            selector_positive_negative_hard_margin=selector_positive_negative_hard_margin,
            selector_cross_family_positive_negative_loss_weight=(
                selector_cross_family_positive_negative_loss_weight
            ),
            selector_cross_family_positive_negative_margin=selector_cross_family_positive_negative_margin,
            selector_model=str(selector_model),
            selector_risk_aware_harm_logit_weight=float(selector_risk_aware_harm_logit_weight),
            selector_risk_aware_benefit_loss_weight=float(selector_risk_aware_benefit_loss_weight),
            selector_risk_aware_harm_loss_weight=float(selector_risk_aware_harm_loss_weight),
            selector_risk_aware_benefit_pos_weight=float(selector_risk_aware_benefit_pos_weight),
            selector_risk_aware_harm_pos_weight=float(selector_risk_aware_harm_pos_weight),
            selector_patch_head=bool(selector_patch_head),
            selector_patch_hidden_dim=int(selector_patch_hidden_dim),
            policy_specs=policy_specs,
            selector_score_edit_penalty=selector_score_edit_penalty,
            selector_target_mode=str(selector_target_mode),
            selector_harm_weight=float(selector_harm_weight),
            selector_miss_weight=float(selector_miss_weight),
            selector_policy_candidate_mode=str(selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
            motif_vocabulary=selector_candidate_motif_vocabulary,
            local_motif_vocabulary=selector_local_motif_vocabulary,
            local_motif_top_k=int(selector_local_motif_top_k),
            selector_selection_mode=str(selection_mode),
            selector_epoch_diagnostic_margin_grid=selector_epoch_diagnostic_margin_grid,
            selector_epoch_selection_mode=str(selector_epoch_selection_mode),
        )
        if selector_summary is not None:
            if (
                str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM
                and (
                    max(float(x) for x in selector_transition_prior_weight_values) > 0.0
                    or max(int(x) for x in selector_transition_compat_top_k_values) > 0
                )
            ):
                transition_prior_summary = _train_transition_prior_head(
                    train_bundles=train_selector_bundles,
                    val_bundles_by_family=val_selector_bundles_by_family,
                    hidden_dim=int(transition_prior_hidden_dim),
                    dropout=dropout,
                    lr=float(transition_prior_lr),
                    epochs=int(transition_prior_epochs),
                    batch_size=batch_size,
                    device=device,
                )
            if (
                str(selector_target_mode) == SELECTOR_TARGET_MODE_BENEFIT_HARM
                and (
                    max(float(x) for x in selector_candidate_compat_threshold_values) > 0.0
                    or max(int(x) for x in selector_candidate_compat_top_k_values) > 0
                )
            ):
                candidate_compatibility_summary = _train_candidate_compatibility_head(
                    train_bundles=train_selector_bundles,
                    val_bundles_by_family=val_selector_bundles_by_family,
                    hidden_dim=int(candidate_compat_hidden_dim),
                    dropout=dropout,
                    lr=float(candidate_compat_lr),
                    epochs=int(candidate_compat_epochs),
                    batch_size=batch_size,
                    device=device,
                    objective=str(candidate_compat_objective),
                    negative_ratio=float(candidate_compat_negative_ratio),
                    no_positive_negative_count=int(candidate_compat_no_positive_negative_count),
                )
            if selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER:
                router_summary = _train_hard_shot_router(
                    train_bundles=train_selector_bundles,
                    val_bundles_by_family=val_selector_bundles_by_family,
                    hidden_dim=int(router_hidden_dim),
                    dropout=dropout,
                    lr=float(router_lr),
                    epochs=int(router_epochs),
                    batch_size=batch_size,
                    device=device,
                    positive_weight=float(router_pos_weight),
                    supervision_target=str(router_supervision_target),
                    pretrain_target=str(router_pretrain_target),
                    pretrain_epochs=int(router_pretrain_epochs),
                    pretrain_positive_weight=router_pretrain_pos_weight,
                    negative_ratio=router_negative_ratio,
                )
            if selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER and router_summary is not None:
                selector_val_by_family = _grid_search_routed_selector_policy_by_family(
                    entry_by_family={entry.family: entry for entry in train_entries},
                    subset_by_family={
                        entry.family: _subset_family(entry, split_by_family[entry.family].val)
                        for entry in train_entries
                    },
                    model=model,
                    selector=selector_summary["selector"],
                    router=router_summary["router"],
                    batch_size=batch_size,
                    device=device,
                    policy_specs=policy_specs,
                    selector_score_edit_penalty=selector_score_edit_penalty,
                    selector_target_mode=str(selector_target_mode),
                    selector_harm_weight=float(selector_harm_weight),
                    selector_miss_weight=float(selector_miss_weight),
                    selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                    selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                    selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                    selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                    selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                    router_threshold_grid=router_threshold_grid_values,
                    selector_emit_margin_grid=selector_emit_grid,
                    selector_nonzero_bias_grid=selector_nonzero_bias_values,
                    transition_prior_head=(
                        transition_prior_summary["head"]
                        if transition_prior_summary is not None
                        else None
                    ),
                    transition_prior_weight_grid=selector_transition_prior_weight_values,
                    transition_compat_top_k_grid=selector_transition_compat_top_k_values,
                    candidate_compatibility_head=(
                        candidate_compatibility_summary["head"]
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    candidate_compat_threshold_grid=selector_candidate_compat_threshold_values,
                    candidate_compat_top_k_grid=selector_candidate_compat_top_k_values,
                    motif_vocabulary=selector_candidate_motif_vocabulary,
                    local_motif_vocabulary=selector_local_motif_vocabulary,
                    local_motif_top_k=int(selector_local_motif_top_k),
                    selector_selection_mode=str(selection_mode),
                )
            else:
                selector_val_by_family = _grid_search_selector_emit_margin_by_family(
                    entry_by_family={entry.family: entry for entry in train_entries},
                    subset_by_family={
                        entry.family: _subset_family(entry, split_by_family[entry.family].val)
                        for entry in train_entries
                    },
                    model=model,
                    selector=selector_summary["selector"],
                    batch_size=batch_size,
                    device=device,
                    policy_specs=policy_specs,
                    selector_score_edit_penalty=selector_score_edit_penalty,
                    selector_target_mode=str(selector_target_mode),
                    selector_harm_weight=float(selector_harm_weight),
                    selector_miss_weight=float(selector_miss_weight),
                    selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                    selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                    selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                    selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                    selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                    emit_margin_grid=selector_emit_grid,
                    nonzero_bias_grid=selector_nonzero_bias_values,
                    transition_prior_head=(
                        transition_prior_summary["head"]
                        if transition_prior_summary is not None
                        else None
                    ),
                    transition_prior_weight_grid=selector_transition_prior_weight_values,
                    transition_compat_top_k_grid=selector_transition_compat_top_k_values,
                    candidate_compatibility_head=(
                        candidate_compatibility_summary["head"]
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    candidate_compat_threshold_grid=selector_candidate_compat_threshold_values,
                    candidate_compat_top_k_grid=selector_candidate_compat_top_k_values,
                    motif_vocabulary=selector_candidate_motif_vocabulary,
                    local_motif_vocabulary=selector_local_motif_vocabulary,
                    local_motif_top_k=int(selector_local_motif_top_k),
                    selector_selection_mode=str(selection_mode),
                )
            selector_val_metric = _mean_system_metric(list(selector_val_by_family.values()))
            if str(selector_adoption_policy) == SELECTOR_ADOPTION_POLICY_CANDIDATE_FIRST_SAFETY:
                selected_inference_mode, selector_adoption_decision = _candidate_first_safety_adoption_mode(
                    no_edit_metric=float(no_edit_val_metric),
                    global_metric=float(global_val_metric),
                    selector_metric=float(selector_val_metric),
                    selector_results=list(selector_val_by_family.values()),
                    requested_mode=str(selection_mode),
                    strong_delta=float(selector_candidate_first_strong_delta),
                    positive_delta=float(selector_candidate_first_positive_delta),
                    positive_margin_floor=float(selector_candidate_first_positive_margin_floor),
                    positive_max_harmed=int(selector_candidate_first_positive_max_harmed),
                    positive_max_margin=float(selector_candidate_first_positive_max_margin),
                    positive_min_nonzero=int(selector_candidate_first_positive_min_nonzero),
                    positive_plateau_guard=bool(selector_candidate_first_positive_plateau_guard),
                    tie_min_delta=float(selector_candidate_first_tie_min_delta),
                    tie_margin_floor=float(selector_candidate_first_tie_margin_floor),
                    tie_min_nonzero=int(selector_candidate_first_tie_min_nonzero),
                    allow_global=bool(selector_candidate_first_allow_global),
                    global_min_delta=float(selector_candidate_first_global_min_delta),
                )
            else:
                selected_inference_mode = (
                    str(selection_mode)
                    if _selected_inference_mode(
                        global_metric=float(global_val_metric),
                        selector_metric=float(selector_val_metric),
                        requested_mode=str(selection_mode),
                        min_delta=float(selector_adoption_min_delta),
                    )
                    == str(selection_mode)
                    else SELECTION_MODE_GLOBAL_POLICY
                )
                selector_adoption_decision = {
                    "policy": SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR,
                    "requested_mode": str(selection_mode),
                    "selected_mode": str(selected_inference_mode),
                    "reason": (
                        "selector_noninferior"
                        if selected_inference_mode == str(selection_mode)
                        else "global_policy_better"
                    ),
                    "global_metric": float(global_val_metric),
                    "selector_metric": float(selector_val_metric),
                    "selector_delta_over_global": float(selector_val_metric) - float(global_val_metric),
                    "selector_adoption_min_delta": float(selector_adoption_min_delta),
                }
    elif selection_mode == SELECTION_MODE_MOTIF_VOCAB:
        motif_summary = _train_motif_vocabulary_head(
            model=model,
            train_bundles=[
                (entry, _subset_family(entry, split_by_family[entry.family].train))
                for entry in train_entries
            ],
            val_bundles_by_family={
                entry.family: (entry, _subset_family(entry, split_by_family[entry.family].val))
                for entry in train_entries
            },
            batch_size=batch_size,
            device=device,
            hidden_dim=dense_hidden_dim,
            dropout=dropout,
            max_classes=motif_max_classes,
            lr=motif_lr,
            epochs=motif_epochs,
            hard_shot_weight=motif_hard_shot_weight,
        )
        if motif_summary is not None:
            motif_val_by_family = copy.deepcopy(motif_summary["eval_by_family"])
            motif_val_metric = _mean_system_metric(list(motif_val_by_family.values()))
            selected_inference_mode = (
                SELECTION_MODE_MOTIF_VOCAB if float(motif_val_metric) > float(global_val_metric) else SELECTION_MODE_GLOBAL_POLICY
            )
    if action_motif_vocabulary is not None:
        action_motif_val_by_family = _grid_search_action_motif_policy_by_family(
            entries=train_entries,
            split_by_family=split_by_family,
            model=model,
            vocabulary=action_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin_grid=action_emit_grid,
        )
        if selection_mode == SELECTION_MODE_ACTION_MOTIF:
            action_motif_val_metric = _mean_system_metric(list(action_motif_val_by_family.values()))
            selected_inference_mode = (
                SELECTION_MODE_ACTION_MOTIF
                if float(action_motif_val_metric) > float(global_val_metric)
                else SELECTION_MODE_GLOBAL_POLICY
            )
    if local_motif_vocabulary is not None:
        local_motif_val_by_family = _grid_search_local_motif_policy_by_family(
            entries=train_entries,
            split_by_family=split_by_family,
            model=model,
            vocabulary=local_motif_vocabulary,
            batch_size=batch_size,
            device=device,
            emit_margin_grid=local_emit_grid,
            min_bit_logit_grid=local_min_bit_grid,
        )
        if selection_mode == SELECTION_MODE_LOCAL_MOTIF:
            local_motif_val_metric = _mean_system_metric(list(local_motif_val_by_family.values()))
            selected_inference_mode = (
                SELECTION_MODE_LOCAL_MOTIF
                if float(local_motif_val_metric) > float(global_val_metric)
                else SELECTION_MODE_GLOBAL_POLICY
            )
    if selected_inference_mode == SELECTION_MODE_CANDIDATE_SELECTOR:
        selected_val_by_family_for_guardrail = selector_val_by_family
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_SELECTOR:
        selected_val_by_family_for_guardrail = selector_val_by_family
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER:
        selected_val_by_family_for_guardrail = selector_val_by_family
    elif selected_inference_mode == SELECTION_MODE_MOTIF_VOCAB:
        selected_val_by_family_for_guardrail = motif_val_by_family
    elif selected_inference_mode == SELECTION_MODE_ACTION_MOTIF:
        selected_val_by_family_for_guardrail = action_motif_val_by_family
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF:
        selected_val_by_family_for_guardrail = local_motif_val_by_family
    elif selected_inference_mode == SELECTION_MODE_RAW_NO_EDIT:
        selected_val_by_family_for_guardrail = no_edit_val_by_family
    else:
        selected_val_by_family_for_guardrail = global_val_by_family
    selected_val_metric_for_guardrail = _mean_system_metric(
        list((selected_val_by_family_for_guardrail or global_val_by_family).values())
    )
    mode_before_no_edit_guardrail = str(selected_inference_mode)
    selected_inference_mode = _apply_no_edit_guardrail_to_mode(
        selected_mode=str(selected_inference_mode),
        selected_metric=float(selected_val_metric_for_guardrail),
        no_edit_metric=float(no_edit_val_metric),
        enabled=bool(selected_no_edit_guardrail),
        min_delta=float(selected_no_edit_min_delta),
    )
    selector_adoption_decision["mode_before_no_edit_guardrail"] = mode_before_no_edit_guardrail
    selector_adoption_decision["selected_mode"] = str(selected_inference_mode)
    selector_adoption_decision["no_edit_guardrail_applied"] = (
        mode_before_no_edit_guardrail != str(selected_inference_mode)
    )
    checkpoint = {
        "schema_version": SCHEMA_VERSION_TRAIN,
        "decoder": "syndrome_edit_predecoder",
        "created_at_utc": _utc_now_iso(),
        "model_kwargs": model_kwargs,
        "model_state": model.state_dict(),
        "decision_policy": selected_decision,
        "inference": {
            "requested_selection_mode": str(selection_mode),
            "selected_inference_mode": str(selected_inference_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_adoption_decision": selector_adoption_decision,
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "candidate_policy_grid": [spec.to_dict() for spec in policy_specs],
        },
        "candidate_selector": (
            {
                "selector_model": selector_summary["selector_model"],
                "selector_kwargs": selector_summary["selector_kwargs"],
                "selector_state": selector_summary["selector"].state_dict(),
                "training_summary": {
                    "epoch_history": selector_summary["epoch_history"],
                    "best_epoch": selector_summary["best_epoch"],
                    "best_val_selection_metric": selector_summary["best_val_selection_metric"],
                    "best_val_by_family": selector_summary["best_val_by_family"],
                    "selector_model": selector_summary["selector_model"],
                    "selector_risk_aware_harm_logit_weight": selector_summary[
                        "selector_risk_aware_harm_logit_weight"
                    ],
                    "selector_risk_aware_benefit_loss_weight": selector_summary[
                        "selector_risk_aware_benefit_loss_weight"
                    ],
                    "selector_risk_aware_harm_loss_weight": selector_summary[
                        "selector_risk_aware_harm_loss_weight"
                    ],
                    "selector_risk_aware_benefit_pos_weight": selector_summary[
                        "selector_risk_aware_benefit_pos_weight"
                    ],
                    "selector_risk_aware_harm_pos_weight": selector_summary[
                        "selector_risk_aware_harm_pos_weight"
                    ],
                    "selector_objective": selector_summary["selector_objective"],
                    "selector_hard_shot_weight": selector_summary["selector_hard_shot_weight"],
                    "selector_identity_margin_loss_weight": selector_summary["selector_identity_margin_loss_weight"],
                    "selector_identity_margin": selector_summary["selector_identity_margin"],
                    "selector_harm_margin_loss_weight": selector_summary["selector_harm_margin_loss_weight"],
                    "selector_harm_margin": selector_summary["selector_harm_margin"],
                    "selector_negative_identity_margin_loss_weight": selector_summary[
                        "selector_negative_identity_margin_loss_weight"
                    ],
                    "selector_negative_identity_margin": selector_summary[
                        "selector_negative_identity_margin"
                    ],
                    "selector_benefit_harm_pairwise_loss_weight": selector_summary["selector_benefit_harm_pairwise_loss_weight"],
                    "selector_benefit_harm_pairwise_margin": selector_summary["selector_benefit_harm_pairwise_margin"],
                    "selector_positive_negative_hard_loss_weight": selector_summary[
                        "selector_positive_negative_hard_loss_weight"
                    ],
                    "selector_positive_negative_hard_margin": selector_summary[
                        "selector_positive_negative_hard_margin"
                    ],
                    "selector_cross_family_positive_negative_loss_weight": selector_summary[
                        "selector_cross_family_positive_negative_loss_weight"
                    ],
                    "selector_cross_family_positive_negative_margin": selector_summary[
                        "selector_cross_family_positive_negative_margin"
                    ],
                    "selector_patch_head": selector_summary["selector_patch_head"],
                    "selector_patch_hidden_dim": selector_summary["selector_patch_hidden_dim"],
                    "selector_score_edit_penalty": selector_summary["selector_score_edit_penalty"],
                    "selector_target_mode": selector_summary["selector_target_mode"],
                    "selector_harm_weight": selector_summary["selector_harm_weight"],
                    "selector_miss_weight": selector_summary["selector_miss_weight"],
                    "selector_policy_candidate_mode": selector_summary["selector_policy_candidate_mode"],
                    "selector_candidate_geometry_features": selector_summary["selector_candidate_geometry_features"],
                    "selector_candidate_pattern_features": selector_summary["selector_candidate_pattern_features"],
                    "selector_candidate_local_evidence_features": selector_summary["selector_candidate_local_evidence_features"],
                    "selector_candidate_local_patch_features": selector_summary["selector_candidate_local_patch_features"],
                    "selector_epoch_diagnostic_margin_grid": selector_summary["selector_epoch_diagnostic_margin_grid"],
                    "selector_epoch_selection_mode": selector_summary["selector_epoch_selection_mode"],
                    "selector_selection_mode": selector_summary["selector_selection_mode"],
                    "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
                    "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
                    "selector_transition_prior_weight_grid": [
                        float(x) for x in selector_transition_prior_weight_values
                    ],
                    "selected_selector_emit_margin": (
                        float(next(iter(selector_val_by_family.values()))["decision"]["selector_emit_margin"])
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_nonzero_bias": (
                        float(next(iter(selector_val_by_family.values()))["decision"].get("selector_nonzero_bias") or 0.0)
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_transition_prior_weight": (
                        float(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_transition_prior_weight"
                            )
                            or 0.0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_transition_compat_top_k": (
                        int(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_transition_compat_top_k"
                            )
                            or 0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_candidate_compat_threshold": (
                        float(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_candidate_compat_threshold"
                            )
                            or 0.0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_candidate_compat_top_k": (
                        int(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_candidate_compat_top_k"
                            )
                            or 0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "selector_candidate_motif_max_classes": int(selector_candidate_motif_max_classes),
                    "selector_local_motif_max_classes": int(selector_local_motif_max_classes),
                    "selector_local_motif_top_k": int(selector_local_motif_top_k),
                },
                "transition_prior": (
                    {
                        "head_kwargs": transition_prior_summary["head_kwargs"],
                        "head_state": transition_prior_summary["head"].state_dict(),
                        "training_summary": {
                            "epoch_history": transition_prior_summary["epoch_history"],
                            "best_epoch": transition_prior_summary["best_epoch"],
                            "best_val_selection_metric": transition_prior_summary["best_val_selection_metric"],
                            "best_val_by_family": transition_prior_summary["best_val_by_family"],
                            "lr": transition_prior_summary["lr"],
                            "epochs": transition_prior_summary["epochs"],
                        },
                    }
                    if transition_prior_summary is not None
                    else None
                ),
                "candidate_compatibility": (
                    {
                        "head_kwargs": candidate_compatibility_summary["head_kwargs"],
                        "head_state": candidate_compatibility_summary["head"].state_dict(),
                        "training_summary": {
                            "epoch_history": candidate_compatibility_summary["epoch_history"],
                            "best_epoch": candidate_compatibility_summary["best_epoch"],
                            "best_val_selection_metric": candidate_compatibility_summary["best_val_selection_metric"],
                            "best_val_by_family": candidate_compatibility_summary["best_val_by_family"],
                            "lr": candidate_compatibility_summary["lr"],
                            "epochs": candidate_compatibility_summary["epochs"],
                            "objective": candidate_compatibility_summary["objective"],
                            "negative_ratio": candidate_compatibility_summary["negative_ratio"],
                            "no_positive_negative_count": candidate_compatibility_summary["no_positive_negative_count"],
                        },
                    }
                    if candidate_compatibility_summary is not None
                    else None
                ),
                "candidate_motif_vocabulary": (
                    {
                        "mask_table": selector_candidate_motif_vocabulary.mask_table,
                        "detector_index_lists": [list(x) for x in selector_candidate_motif_vocabulary.detector_index_lists],
                        "counts": list(selector_candidate_motif_vocabulary.counts),
                    }
                    if selector_candidate_motif_vocabulary is not None
                    else None
                ),
                "candidate_local_motif_vocabulary": (
                    {
                        "offset_patterns": [
                            [list(offset) for offset in pattern]
                            for pattern in selector_local_motif_vocabulary.offset_patterns
                        ],
                        "counts": list(selector_local_motif_vocabulary.counts),
                        "detector_count": int(selector_local_motif_vocabulary.detector_count),
                    }
                    if selector_local_motif_vocabulary is not None
                    else None
                ),
            }
            if selector_summary is not None
            else None
        ),
        "hard_shot_router": (
            {
                "router_kwargs": router_summary["router_kwargs"],
                "router_state": router_summary["router"].state_dict(),
                "training_summary": {
                    "epoch_history": router_summary["epoch_history"],
                    "best_epoch": router_summary["best_epoch"],
                    "best_val_selection_metric": router_summary["best_val_selection_metric"],
                    "best_val_by_family": router_summary["best_val_by_family"],
                    "positive_weight": router_summary["positive_weight"],
                    "supervision_target": router_summary["supervision_target"],
                    "pretrain_target": router_summary["pretrain_target"],
                    "pretrain_epochs": router_summary["pretrain_epochs"],
                    "pretrain_positive_weight": router_summary["pretrain_positive_weight"],
                    "pretrain_epoch_history": router_summary["pretrain_epoch_history"],
                    "negative_ratio": router_summary["negative_ratio"],
                    "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
                    "selected_router_threshold": (
                        float(next(iter(selector_val_by_family.values()))["decision"].get("router_threshold"))
                        if selector_val_by_family
                        and next(iter(selector_val_by_family.values()))["decision"].get("router_threshold") is not None
                        else None
                    ),
                },
            }
            if router_summary is not None
            else None
        ),
        "motif_vocabulary_head": (
            {
                "motif_kwargs": motif_summary["motif_kwargs"],
                "motif_state": motif_summary["motif_head"].state_dict(),
                "vocabulary_mask_table": motif_summary["vocabulary"].mask_table,
                "vocabulary_detector_index_lists": [list(x) for x in motif_summary["vocabulary"].detector_index_lists],
                "vocabulary_counts": list(motif_summary["vocabulary"].counts),
                "training_summary": {
                    "epoch_history": motif_summary["epoch_history"],
                    "best_epoch": motif_summary["best_epoch"],
                    "best_val_selection_metric": motif_summary["best_val_selection_metric"],
                    "best_val_by_family": motif_summary["best_val_by_family"],
                    "hard_shot_weight": motif_summary["hard_shot_weight"],
                },
            }
            if motif_summary is not None
            else None
        ),
        "action_motif_vocabulary": (
            {
                "mask_table": action_motif_vocabulary.mask_table,
                "detector_index_lists": [list(x) for x in action_motif_vocabulary.detector_index_lists],
                "counts": list(action_motif_vocabulary.counts),
                "training_summary": {
                    "active_fraction": (
                        float(np.mean(train_action_motif_active >= 0.5))
                        if train_action_motif_active.size
                        else None
                    ),
                    "nonzero_active_fraction": (
                        float(
                            np.mean(
                                (train_action_motif_active >= 0.5)
                                & (train_action_motif_label > 0)
                            )
                        )
                        if train_action_motif_active.size
                        else None
                    ),
                    "loss_weight": float(action_motif_loss_weight),
                    "identity_margin": float(action_motif_identity_margin),
                    "emit_margin_grid": [float(x) for x in action_emit_grid],
                    "selected_emit_margin": (
                        float(next(iter(action_motif_val_by_family.values()))["decision"]["action_motif_emit_margin"])
                        if action_motif_val_by_family
                        else None
                    ),
                },
            }
            if action_motif_vocabulary is not None
            else None
        ),
        "local_motif_vocabulary": (
            {
                "offset_patterns": [
                    [list(offset) for offset in pattern]
                    for pattern in local_motif_vocabulary.offset_patterns
                ],
                "counts": list(local_motif_vocabulary.counts),
                "detector_count": int(local_motif_vocabulary.detector_count),
                "training_summary": {
                    "vocabulary_num_patterns": int(len(local_motif_vocabulary.offset_patterns)),
                    "emit_margin_grid": [float(x) for x in local_emit_grid],
                    "min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
                    "selected_emit_margin": (
                        float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_emit_margin"])
                        if local_motif_val_by_family
                        else None
                    ),
                    "selected_min_bit_logit": (
                        float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_min_bit_logit"])
                        if local_motif_val_by_family
                        else None
                    ),
                },
            }
            if local_motif_vocabulary is not None
            else None
        ),
        "bundle_info": train_entries[0].bundle_info,
        "training_config": {
            "fill_value": float(fill_value),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "needs_edit_loss_weight": float(needs_edit_loss_weight),
            "sparsity_loss_weight": float(sparsity_loss_weight),
            "decision_aware_loss_weight": float(decision_aware_loss_weight),
            "decision_aware_margin": float(decision_aware_margin),
            "action_motif_max_classes": int(action_motif_max_classes),
            "action_motif_loss_weight": float(action_motif_loss_weight),
            "action_motif_identity_margin": float(action_motif_identity_margin),
            "action_motif_emit_margin_grid": [float(x) for x in action_emit_grid],
            "local_motif_max_classes": int(local_motif_max_classes),
            "local_motif_emit_margin_grid": [float(x) for x in local_emit_grid],
            "local_motif_min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
            "hard_shot_solved_weight": float(hard_shot_solved_weight),
            "hard_shot_unsolved_weight": float(hard_shot_unsolved_weight),
            "edit_supervision_mode": str(edit_supervision_mode),
            "selection_mode": str(selection_mode),
            "selector_hidden_dim": int(selector_hidden_dim),
            "selector_epochs": int(selector_epochs),
            "selector_lr": float(selector_lr),
            "selector_model": str(selector_model),
            "selector_risk_aware_harm_logit_weight": float(selector_risk_aware_harm_logit_weight),
            "selector_risk_aware_benefit_loss_weight": float(selector_risk_aware_benefit_loss_weight),
            "selector_risk_aware_harm_loss_weight": float(selector_risk_aware_harm_loss_weight),
            "selector_risk_aware_benefit_pos_weight": float(selector_risk_aware_benefit_pos_weight),
            "selector_risk_aware_harm_pos_weight": float(selector_risk_aware_harm_pos_weight),
            "selector_objective": str(selector_objective),
            "selector_hard_shot_weight": float(selector_hard_shot_weight),
            "selector_identity_margin_loss_weight": float(selector_identity_margin_loss_weight),
            "selector_identity_margin": float(selector_identity_margin),
            "selector_harm_margin_loss_weight": float(selector_harm_margin_loss_weight),
            "selector_harm_margin": float(selector_harm_margin),
            "selector_negative_identity_margin_loss_weight": float(
                selector_negative_identity_margin_loss_weight
            ),
            "selector_negative_identity_margin": float(selector_negative_identity_margin),
            "selector_benefit_harm_pairwise_loss_weight": float(selector_benefit_harm_pairwise_loss_weight),
            "selector_benefit_harm_pairwise_margin": float(selector_benefit_harm_pairwise_margin),
            "selector_positive_negative_hard_loss_weight": float(
                selector_positive_negative_hard_loss_weight
            ),
            "selector_positive_negative_hard_margin": float(selector_positive_negative_hard_margin),
            "selector_cross_family_positive_negative_loss_weight": float(
                selector_cross_family_positive_negative_loss_weight
            ),
            "selector_cross_family_positive_negative_margin": float(
                selector_cross_family_positive_negative_margin
            ),
            "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
            "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
            "selector_transition_prior_weight_grid": [
                float(x) for x in selector_transition_prior_weight_values
            ],
            "selector_transition_compat_top_k_grid": [
                int(x) for x in selector_transition_compat_top_k_values
            ],
            "transition_prior_hidden_dim": int(transition_prior_hidden_dim),
            "transition_prior_epochs": int(transition_prior_epochs),
            "transition_prior_lr": float(transition_prior_lr),
            "candidate_compat_hidden_dim": int(candidate_compat_hidden_dim),
            "candidate_compat_epochs": int(candidate_compat_epochs),
            "candidate_compat_lr": float(candidate_compat_lr),
            "candidate_compat_objective": str(candidate_compat_objective),
            "candidate_compat_negative_ratio": float(candidate_compat_negative_ratio),
            "candidate_compat_no_positive_negative_count": int(candidate_compat_no_positive_negative_count),
            "selector_candidate_compat_threshold_grid": [
                float(x) for x in selector_candidate_compat_threshold_values
            ],
            "selector_candidate_compat_top_k_grid": [
                int(x) for x in selector_candidate_compat_top_k_values
            ],
            "selector_score_edit_penalty": float(selector_score_edit_penalty),
            "selector_target_mode": str(selector_target_mode),
            "selector_harm_weight": float(selector_harm_weight),
            "selector_miss_weight": float(selector_miss_weight),
            "selector_policy_candidate_mode": str(selector_policy_candidate_mode),
            "selector_candidate_geometry_features": bool(selector_candidate_geometry_features),
            "selector_candidate_pattern_features": bool(selector_candidate_pattern_features),
            "selector_candidate_local_evidence_features": bool(selector_candidate_local_evidence_features),
            "selector_candidate_local_patch_features": bool(selector_candidate_local_patch_features),
            "selector_epoch_diagnostic_margin_grid": [
                float(x) for x in (selector_epoch_diagnostic_margin_grid or [])
            ],
            "selector_epoch_selection_mode": str(selector_epoch_selection_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_candidate_first_strong_delta": float(selector_candidate_first_strong_delta),
            "selector_candidate_first_positive_delta": float(selector_candidate_first_positive_delta),
            "selector_candidate_first_positive_margin_floor": float(selector_candidate_first_positive_margin_floor),
            "selector_candidate_first_positive_max_harmed": int(selector_candidate_first_positive_max_harmed),
            "selector_candidate_first_positive_max_margin": float(selector_candidate_first_positive_max_margin),
            "selector_candidate_first_positive_min_nonzero": int(selector_candidate_first_positive_min_nonzero),
            "selector_candidate_first_positive_plateau_guard": bool(selector_candidate_first_positive_plateau_guard),
            "selector_candidate_first_tie_min_delta": float(selector_candidate_first_tie_min_delta),
            "selector_candidate_first_tie_margin_floor": float(selector_candidate_first_tie_margin_floor),
            "selector_candidate_first_tie_min_nonzero": int(selector_candidate_first_tie_min_nonzero),
            "selector_candidate_first_allow_global": bool(selector_candidate_first_allow_global),
            "selector_candidate_first_global_min_delta": float(selector_candidate_first_global_min_delta),
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "selector_candidate_motif_max_classes": int(selector_candidate_motif_max_classes),
            "selector_local_motif_max_classes": int(selector_local_motif_max_classes),
            "selector_local_motif_top_k": int(selector_local_motif_top_k),
            "router_hidden_dim": int(router_hidden_dim),
            "router_epochs": int(router_epochs),
            "router_lr": float(router_lr),
            "router_pos_weight": float(router_pos_weight),
            "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
            "router_supervision_target": str(router_supervision_target),
            "router_pretrain_target": str(router_pretrain_target),
            "router_pretrain_epochs": int(router_pretrain_epochs),
            "router_pretrain_pos_weight": (
                float(router_pretrain_pos_weight) if router_pretrain_pos_weight is not None else None
            ),
            "router_negative_ratio": (
                float(router_negative_ratio) if router_negative_ratio is not None else None
            ),
            "motif_max_classes": int(motif_max_classes),
            "motif_epochs": int(motif_epochs),
            "motif_lr": float(motif_lr),
            "motif_hard_shot_weight": float(motif_hard_shot_weight),
            "seed": int(seed),
            "train_families": [entry.family for entry in train_entries],
        },
    }
    torch.save(checkpoint, checkpoint_path)

    requested_eval = eval_families or list(dict.fromkeys([entry.family for entry in train_entries]))
    _manifest_data, resolved_eval = _resolve_manifest_family_entries(manifest_path, requested_eval)
    eval_results_global: dict[str, Any] = {}
    eval_results_no_edit: dict[str, Any] = {}
    eval_results_selector: dict[str, Any] = {}
    eval_results_motif: dict[str, Any] = {}
    eval_results_action_motif: dict[str, Any] = {}
    eval_results_local_motif: dict[str, Any] = {}
    for family, path in resolved_eval:
        entry = _prepare_edit_family(path, fill_value=fill_value, max_shots=max_shots)
        eval_results_no_edit[family] = _no_edit_system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
        )
        eval_results_global[family] = _system_metrics_for_subset(
            entry=entry,
            subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
            model=model,
            batch_size=batch_size,
            device=device,
            needs_edit_threshold=float(selected_decision["needs_edit_threshold"]),
            edit_threshold=float(selected_decision["edit_threshold"]),
            max_predicted_edit_weight=int(selected_decision["max_predicted_edit_weight"]),
        )
        if selector_summary is not None:
            selected_selector_emit_margin = (
                float(next(iter(selector_val_by_family.values()))["decision"]["selector_emit_margin"])
                if selector_val_by_family
                else 0.0
            )
            selected_selector_nonzero_bias = (
                float(next(iter(selector_val_by_family.values()))["decision"].get("selector_nonzero_bias") or 0.0)
                if selector_val_by_family
                else 0.0
            )
            selected_selector_transition_prior_weight = (
                float(
                    next(iter(selector_val_by_family.values()))["decision"].get(
                        "selector_transition_prior_weight"
                    )
                    or 0.0
                )
                if selector_val_by_family
                else 0.0
            )
            selected_selector_transition_compat_top_k = (
                int(
                    next(iter(selector_val_by_family.values()))["decision"].get(
                        "selector_transition_compat_top_k"
                    )
                    or 0
                )
                if selector_val_by_family
                else 0
            )
            selected_selector_candidate_compat_threshold = (
                float(
                    next(iter(selector_val_by_family.values()))["decision"].get(
                        "selector_candidate_compat_threshold"
                    )
                    or 0.0
                )
                if selector_val_by_family
                else 0.0
            )
            selected_selector_candidate_compat_top_k = (
                int(
                    next(iter(selector_val_by_family.values()))["decision"].get(
                        "selector_candidate_compat_top_k"
                    )
                    or 0
                )
                if selector_val_by_family
                else 0
            )
            selected_router_threshold = (
                float(next(iter(selector_val_by_family.values()))["decision"].get("router_threshold"))
                if selector_val_by_family
                and next(iter(selector_val_by_family.values()))["decision"].get("router_threshold") is not None
                else 0.5
            )
            eval_results_selector[family] = _selector_system_metrics_for_subset(
                entry=entry,
                subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
                model=model,
                selector=selector_summary["selector"],
                batch_size=batch_size,
                device=device,
                policy_specs=policy_specs,
                selector_score_edit_penalty=selector_score_edit_penalty,
                selector_target_mode=str(selector_target_mode),
                selector_harm_weight=float(selector_harm_weight),
                selector_miss_weight=float(selector_miss_weight),
                selector_policy_candidate_mode=str(selector_policy_candidate_mode),
                selector_candidate_geometry_features=bool(selector_candidate_geometry_features),
                selector_candidate_pattern_features=bool(selector_candidate_pattern_features),
                selector_candidate_local_evidence_features=bool(selector_candidate_local_evidence_features),
                selector_candidate_local_patch_features=bool(selector_candidate_local_patch_features),
                motif_vocabulary=selector_candidate_motif_vocabulary,
                local_motif_vocabulary=selector_local_motif_vocabulary,
                local_motif_top_k=int(selector_local_motif_top_k),
                selector_selection_mode=str(selection_mode),
                selector_emit_margin=selected_selector_emit_margin,
                selector_nonzero_bias=selected_selector_nonzero_bias,
                transition_prior_head=(
                    transition_prior_summary["head"]
                    if transition_prior_summary is not None
                    else None
                ),
                selector_transition_prior_weight=selected_selector_transition_prior_weight,
                selector_transition_compat_top_k=selected_selector_transition_compat_top_k,
                candidate_compatibility_head=(
                    candidate_compatibility_summary["head"]
                    if candidate_compatibility_summary is not None
                    else None
                ),
                selector_candidate_compat_threshold=selected_selector_candidate_compat_threshold,
                selector_candidate_compat_top_k=selected_selector_candidate_compat_top_k,
                router=(router_summary["router"] if router_summary is not None else None),
                router_threshold=selected_router_threshold,
                routed_nonzero_only=(selection_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER),
            )
        if motif_summary is not None:
            eval_results_motif[family] = _motif_system_metrics_for_subset(
                entry=entry,
                subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
                model=model,
                motif_head=motif_summary["motif_head"],
                vocabulary=motif_summary["vocabulary"],
                batch_size=batch_size,
                device=device,
            )
        if action_motif_vocabulary is not None:
            action_motif_emit_margin = (
                float(next(iter(action_motif_val_by_family.values()))["decision"]["action_motif_emit_margin"])
                if action_motif_val_by_family
                else 0.0
            )
            eval_results_action_motif[family] = _action_motif_system_metrics_for_subset(
                entry=entry,
                subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
                model=model,
                vocabulary=action_motif_vocabulary,
                batch_size=batch_size,
                device=device,
                emit_margin=action_motif_emit_margin,
            )
        if local_motif_vocabulary is not None:
            local_motif_emit_margin = (
                float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_emit_margin"])
                if local_motif_val_by_family
                else 0.0
            )
            local_motif_min_bit_logit = (
                float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_min_bit_logit"])
                if local_motif_val_by_family
                else -1.0
            )
            eval_results_local_motif[family] = _local_motif_system_metrics_for_subset(
                entry=entry,
                subset=_subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64)),
                model=model,
                vocabulary=local_motif_vocabulary,
                batch_size=batch_size,
                device=device,
                emit_margin=local_motif_emit_margin,
                min_bit_logit=local_motif_min_bit_logit,
            )

    if selected_inference_mode == SELECTION_MODE_CANDIDATE_SELECTOR:
        selected_eval_results = eval_results_selector
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_SELECTOR:
        selected_eval_results = eval_results_selector
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF_ROUTER:
        selected_eval_results = eval_results_selector
    elif selected_inference_mode == SELECTION_MODE_MOTIF_VOCAB:
        selected_eval_results = eval_results_motif
    elif selected_inference_mode == SELECTION_MODE_ACTION_MOTIF:
        selected_eval_results = eval_results_action_motif
    elif selected_inference_mode == SELECTION_MODE_LOCAL_MOTIF:
        selected_eval_results = eval_results_local_motif
    elif selected_inference_mode == SELECTION_MODE_RAW_NO_EDIT:
        selected_eval_results = eval_results_no_edit
    else:
        selected_eval_results = eval_results_global

    result = {
        "schema_version": SCHEMA_VERSION_EXPERIMENT,
        "decoder": "syndrome_edit_predecoder",
        "created_at_utc": _utc_now_iso(),
        "input_mode": "manifest",
        "manifest_path": manifest_path.as_posix(),
        "model": {
            "model_class": "SyndromeEditPreDecoder",
            "model_kwargs": model_kwargs,
            "num_parameters": int(sum(param.numel() for param in model.parameters())),
        },
        "dataset": {
            "compatibility": compatibility,
            "training_config": training_config,
            "source_manifest_schema": manifest_data.get("schema_version"),
        },
        "training": {
            "train_families": [entry.family for entry in train_entries],
            "epoch_history": epoch_history,
            "best_epoch": int(best_state["epoch"]),
            "best_val_selection_metric": float(best_val_metric),
            "selected_decision_policy": selected_decision,
            "requested_selection_mode": str(selection_mode),
            "selected_inference_mode": str(selected_inference_mode),
            "selector_adoption_min_delta": float(selector_adoption_min_delta),
            "selector_adoption_policy": str(selector_adoption_policy),
            "selector_adoption_decision": selector_adoption_decision,
            "selected_no_edit_guardrail": bool(selected_no_edit_guardrail),
            "selected_no_edit_min_delta": float(selected_no_edit_min_delta),
            "best_val_no_edit_by_family": no_edit_val_by_family,
            "best_val_global_policy_by_family": global_val_by_family,
            "best_val_candidate_selector_by_family": selector_val_by_family,
            "best_val_motif_vocab_by_family": motif_val_by_family,
            "best_val_action_motif_by_family": action_motif_val_by_family,
            "best_val_local_motif_by_family": local_motif_val_by_family,
            "selector_training": (
                {
                    "epoch_history": selector_summary["epoch_history"],
                    "best_epoch": selector_summary["best_epoch"],
                    "best_val_selection_metric": selector_summary["best_val_selection_metric"],
                    "best_val_by_family": selector_summary["best_val_by_family"],
                    "selector_model": selector_summary["selector_model"],
                    "selector_risk_aware_harm_logit_weight": selector_summary[
                        "selector_risk_aware_harm_logit_weight"
                    ],
                    "selector_risk_aware_benefit_loss_weight": selector_summary[
                        "selector_risk_aware_benefit_loss_weight"
                    ],
                    "selector_risk_aware_harm_loss_weight": selector_summary[
                        "selector_risk_aware_harm_loss_weight"
                    ],
                    "selector_risk_aware_benefit_pos_weight": selector_summary[
                        "selector_risk_aware_benefit_pos_weight"
                    ],
                    "selector_risk_aware_harm_pos_weight": selector_summary[
                        "selector_risk_aware_harm_pos_weight"
                    ],
                    "selector_objective": selector_summary["selector_objective"],
                    "selector_hard_shot_weight": selector_summary["selector_hard_shot_weight"],
                    "selector_identity_margin_loss_weight": selector_summary["selector_identity_margin_loss_weight"],
                    "selector_identity_margin": selector_summary["selector_identity_margin"],
                    "selector_harm_margin_loss_weight": selector_summary["selector_harm_margin_loss_weight"],
                    "selector_harm_margin": selector_summary["selector_harm_margin"],
                    "selector_negative_identity_margin_loss_weight": selector_summary[
                        "selector_negative_identity_margin_loss_weight"
                    ],
                    "selector_negative_identity_margin": selector_summary[
                        "selector_negative_identity_margin"
                    ],
                    "selector_benefit_harm_pairwise_loss_weight": selector_summary["selector_benefit_harm_pairwise_loss_weight"],
                    "selector_benefit_harm_pairwise_margin": selector_summary["selector_benefit_harm_pairwise_margin"],
                    "selector_positive_negative_hard_loss_weight": selector_summary[
                        "selector_positive_negative_hard_loss_weight"
                    ],
                    "selector_positive_negative_hard_margin": selector_summary[
                        "selector_positive_negative_hard_margin"
                    ],
                    "selector_cross_family_positive_negative_loss_weight": selector_summary[
                        "selector_cross_family_positive_negative_loss_weight"
                    ],
                    "selector_cross_family_positive_negative_margin": selector_summary[
                        "selector_cross_family_positive_negative_margin"
                    ],
                    "selector_patch_head": selector_summary["selector_patch_head"],
                    "selector_patch_hidden_dim": selector_summary["selector_patch_hidden_dim"],
                    "policy_specs": selector_summary["policy_specs"],
                    "selector_score_edit_penalty": selector_summary["selector_score_edit_penalty"],
                    "selector_target_mode": selector_summary["selector_target_mode"],
                    "selector_harm_weight": selector_summary["selector_harm_weight"],
                    "selector_miss_weight": selector_summary["selector_miss_weight"],
                    "selector_policy_candidate_mode": selector_summary["selector_policy_candidate_mode"],
                    "selector_candidate_geometry_features": selector_summary["selector_candidate_geometry_features"],
                    "selector_candidate_pattern_features": selector_summary["selector_candidate_pattern_features"],
                    "selector_candidate_local_evidence_features": selector_summary["selector_candidate_local_evidence_features"],
                    "selector_candidate_local_patch_features": selector_summary["selector_candidate_local_patch_features"],
                    "selector_epoch_diagnostic_margin_grid": selector_summary["selector_epoch_diagnostic_margin_grid"],
                    "selector_epoch_selection_mode": selector_summary["selector_epoch_selection_mode"],
                    "selector_selection_mode": selector_summary["selector_selection_mode"],
                    "selector_emit_margin_grid": [float(x) for x in selector_emit_grid],
                    "selector_nonzero_bias_grid": [float(x) for x in selector_nonzero_bias_values],
                    "selector_transition_prior_weight_grid": [
                        float(x) for x in selector_transition_prior_weight_values
                    ],
                    "selector_transition_compat_top_k_grid": [
                        int(x) for x in selector_transition_compat_top_k_values
                    ],
                    "selected_selector_emit_margin": (
                        float(next(iter(selector_val_by_family.values()))["decision"]["selector_emit_margin"])
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_nonzero_bias": (
                        float(next(iter(selector_val_by_family.values()))["decision"].get("selector_nonzero_bias") or 0.0)
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_transition_prior_weight": (
                        float(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_transition_prior_weight"
                            )
                            or 0.0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "selected_selector_transition_compat_top_k": (
                        int(
                            next(iter(selector_val_by_family.values()))["decision"].get(
                                "selector_transition_compat_top_k"
                            )
                            or 0
                        )
                        if selector_val_by_family
                        else None
                    ),
                    "transition_prior_training": (
                        {
                            "epoch_history": transition_prior_summary["epoch_history"],
                            "best_epoch": transition_prior_summary["best_epoch"],
                            "best_val_selection_metric": transition_prior_summary["best_val_selection_metric"],
                            "best_val_by_family": transition_prior_summary["best_val_by_family"],
                            "lr": transition_prior_summary["lr"],
                            "epochs": transition_prior_summary["epochs"],
                        }
                        if transition_prior_summary is not None
                        else None
                    ),
                    "candidate_compatibility_training": (
                        {
                            "epoch_history": candidate_compatibility_summary["epoch_history"],
                            "best_epoch": candidate_compatibility_summary["best_epoch"],
                            "best_val_selection_metric": candidate_compatibility_summary["best_val_selection_metric"],
                            "best_val_by_family": candidate_compatibility_summary["best_val_by_family"],
                            "lr": candidate_compatibility_summary["lr"],
                            "epochs": candidate_compatibility_summary["epochs"],
                        }
                        if candidate_compatibility_summary is not None
                        else None
                    ),
                    "candidate_motif_vocab_num_classes": (
                        int(selector_candidate_motif_vocabulary.mask_table.shape[0])
                        if selector_candidate_motif_vocabulary is not None
                        else 0
                    ),
                    "candidate_local_motif_num_patterns": (
                        int(len(selector_local_motif_vocabulary.offset_patterns))
                        if selector_local_motif_vocabulary is not None
                        else 0
                    ),
                    "candidate_local_motif_top_k": int(selector_local_motif_top_k),
                }
                if selector_summary is not None
                else None
            ),
            "router_training": (
                {
                    "epoch_history": router_summary["epoch_history"],
                    "best_epoch": router_summary["best_epoch"],
                    "best_val_selection_metric": router_summary["best_val_selection_metric"],
                    "best_val_by_family": router_summary["best_val_by_family"],
                    "positive_weight": router_summary["positive_weight"],
                    "supervision_target": router_summary["supervision_target"],
                    "pretrain_target": router_summary["pretrain_target"],
                    "pretrain_epochs": router_summary["pretrain_epochs"],
                    "pretrain_positive_weight": router_summary["pretrain_positive_weight"],
                    "pretrain_epoch_history": router_summary["pretrain_epoch_history"],
                    "negative_ratio": router_summary["negative_ratio"],
                    "router_threshold_grid": [float(x) for x in router_threshold_grid_values],
                    "selected_router_threshold": (
                        float(next(iter(selector_val_by_family.values()))["decision"].get("router_threshold"))
                        if selector_val_by_family
                        and next(iter(selector_val_by_family.values()))["decision"].get("router_threshold") is not None
                        else None
                    ),
                }
                if router_summary is not None
                else None
            ),
            "motif_training": (
                {
                    "epoch_history": motif_summary["epoch_history"],
                    "best_epoch": motif_summary["best_epoch"],
                    "best_val_selection_metric": motif_summary["best_val_selection_metric"],
                    "best_val_by_family": motif_summary["best_val_by_family"],
                    "hard_shot_weight": motif_summary["hard_shot_weight"],
                    "vocabulary_num_classes": int(motif_summary["vocabulary"].mask_table.shape[0]),
                }
                if motif_summary is not None
                else None
            ),
            "action_motif_training": (
                {
                    "vocabulary_num_classes": int(action_motif_vocabulary.mask_table.shape[0]),
                    "active_fraction": (
                        float(np.mean(train_action_motif_active >= 0.5))
                        if train_action_motif_active.size
                        else None
                    ),
                    "nonzero_active_fraction": (
                        float(
                            np.mean(
                                (train_action_motif_active >= 0.5)
                                & (train_action_motif_label > 0)
                            )
                        )
                        if train_action_motif_active.size
                        else None
                    ),
                    "loss_weight": float(action_motif_loss_weight),
                    "identity_margin": float(action_motif_identity_margin),
                    "emit_margin_grid": [float(x) for x in action_emit_grid],
                    "selected_emit_margin": (
                        float(next(iter(action_motif_val_by_family.values()))["decision"]["action_motif_emit_margin"])
                        if action_motif_val_by_family
                        else None
                    ),
                }
                if action_motif_vocabulary is not None
                else None
            ),
            "local_motif_training": (
                {
                    "vocabulary_num_patterns": int(len(local_motif_vocabulary.offset_patterns)),
                    "emit_margin_grid": [float(x) for x in local_emit_grid],
                    "min_bit_logit_grid": [float(x) for x in local_min_bit_grid],
                    "selected_emit_margin": (
                        float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_emit_margin"])
                        if local_motif_val_by_family
                        else None
                    ),
                    "selected_min_bit_logit": (
                        float(next(iter(local_motif_val_by_family.values()))["decision"]["local_motif_min_bit_logit"])
                        if local_motif_val_by_family
                        else None
                    ),
                }
                if local_motif_vocabulary is not None
                else None
            ),
        },
        "eval_families": selected_eval_results,
        "eval_families_no_edit": eval_results_no_edit,
        "eval_families_global_policy": eval_results_global,
        "eval_families_candidate_selector": (eval_results_selector or None),
        "eval_families_motif_vocab": (eval_results_motif or None),
        "eval_families_action_motif": (eval_results_action_motif or None),
        "eval_families_local_motif": (eval_results_local_motif or None),
        "artifacts": {
            "checkpoint": checkpoint_path.as_posix(),
        },
    }
    _write_json(out_dir / "experiment_summary.json", result)
    return result


def _parse_float_list(values: list[str] | None, *, default: list[float]) -> list[float]:
    if not values:
        return list(default)
    return [float(value) for value in values]


def _parse_int_list(values: list[str] | None, *, default: list[int]) -> list[int]:
    if not values:
        return list(default)
    return [int(value) for value in values]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the first syndrome-edit pre-decoder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train on one predecoder target family directory")
    train.add_argument("--family-dir", type=Path, required=True)
    train.add_argument("--checkpoint-out", type=Path, required=True)
    train.add_argument("--out-json", type=Path, required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained checkpoint on one predecoder target family")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--family-dir", type=Path, required=True)
    eval_parser.add_argument("--out-json", type=Path, required=True)

    experiment = subparsers.add_parser("experiment", help="Run a manifest-based predecoder experiment")
    experiment.add_argument("--manifest", type=Path, required=True)
    experiment.add_argument("--train-families", nargs="+", required=True)
    experiment.add_argument("--eval-families", nargs="*", default=None)
    experiment.add_argument("--out-dir", type=Path, required=True)

    for sub in (train, eval_parser, experiment):
        sub.add_argument("--fill-value", type=float, default=-0.5)
        sub.add_argument("--max-shots", type=int, default=None)
        sub.add_argument("--batch-size", type=int, default=64)

    for sub in (train, experiment):
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--train-ratio", type=float, default=0.7)
        sub.add_argument("--val-ratio", type=float, default=0.15)
        sub.add_argument("--test-ratio", type=float, default=0.15)
        sub.add_argument("--epochs", type=int, default=8)
        sub.add_argument("--lr", type=float, default=1e-3)
        sub.add_argument("--hidden-channels", type=int, default=24)
        sub.add_argument("--num-blocks", type=int, default=3)
        sub.add_argument("--dense-hidden-dim", type=int, default=64)
        sub.add_argument("--dropout", type=float, default=0.1)
        sub.add_argument("--needs-edit-loss-weight", type=float, default=0.5)
        sub.add_argument("--sparsity-loss-weight", type=float, default=0.01)
        sub.add_argument("--decision-aware-loss-weight", type=float, default=DEFAULT_DECISION_AWARE_LOSS_WEIGHT)
        sub.add_argument("--decision-aware-margin", type=float, default=DEFAULT_DECISION_AWARE_MARGIN)
        sub.add_argument("--hard-shot-solved-weight", type=float, default=DEFAULT_HARD_SHOT_SOLVED_WEIGHT)
        sub.add_argument("--hard-shot-unsolved-weight", type=float, default=DEFAULT_HARD_SHOT_UNSOLVED_WEIGHT)
        sub.add_argument("--edit-supervision-mode", choices=EDIT_SUPERVISION_MODE_CHOICES, default=EDIT_SUPERVISION_MODE_HARD_SHOTS_ONLY)
        sub.add_argument("--selection-mode", choices=SELECTION_MODE_CHOICES, default=SELECTION_MODE_CANDIDATE_SELECTOR)
        sub.add_argument("--selector-hidden-dim", type=int, default=DEFAULT_SELECTOR_HIDDEN_DIM)
        sub.add_argument("--selector-epochs", type=int, default=DEFAULT_SELECTOR_EPOCHS)
        sub.add_argument("--selector-lr", type=float, default=DEFAULT_SELECTOR_LR)
        sub.add_argument("--selector-model", choices=SELECTOR_MODEL_CHOICES, default=SELECTOR_MODEL_SCALAR)
        sub.add_argument(
            "--selector-risk-aware-harm-logit-weight",
            type=float,
            default=DEFAULT_SELECTOR_RISK_AWARE_HARM_LOGIT_WEIGHT,
        )
        sub.add_argument(
            "--selector-risk-aware-benefit-loss-weight",
            type=float,
            default=DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_LOSS_WEIGHT,
        )
        sub.add_argument(
            "--selector-risk-aware-harm-loss-weight",
            type=float,
            default=DEFAULT_SELECTOR_RISK_AWARE_HARM_LOSS_WEIGHT,
        )
        sub.add_argument(
            "--selector-risk-aware-benefit-pos-weight",
            type=float,
            default=DEFAULT_SELECTOR_RISK_AWARE_BENEFIT_POS_WEIGHT,
        )
        sub.add_argument(
            "--selector-risk-aware-harm-pos-weight",
            type=float,
            default=DEFAULT_SELECTOR_RISK_AWARE_HARM_POS_WEIGHT,
        )
        sub.add_argument("--selector-objective", choices=SELECTOR_OBJECTIVE_CHOICES, default=SELECTOR_OBJECTIVE_GROUP_RANK)
        sub.add_argument("--selector-hard-shot-weight", type=float, default=DEFAULT_SELECTOR_HARD_SHOT_WEIGHT)
        sub.add_argument("--selector-identity-margin-loss-weight", type=float, default=DEFAULT_SELECTOR_IDENTITY_MARGIN_LOSS_WEIGHT)
        sub.add_argument("--selector-identity-margin", type=float, default=DEFAULT_SELECTOR_IDENTITY_MARGIN)
        sub.add_argument("--selector-harm-margin-loss-weight", type=float, default=DEFAULT_SELECTOR_HARM_MARGIN_LOSS_WEIGHT)
        sub.add_argument("--selector-harm-margin", type=float, default=DEFAULT_SELECTOR_HARM_MARGIN)
        sub.add_argument("--selector-negative-identity-margin-loss-weight", type=float, default=DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN_LOSS_WEIGHT)
        sub.add_argument("--selector-negative-identity-margin", type=float, default=DEFAULT_SELECTOR_NEGATIVE_IDENTITY_MARGIN)
        sub.add_argument("--selector-benefit-harm-pairwise-loss-weight", type=float, default=DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_LOSS_WEIGHT)
        sub.add_argument("--selector-benefit-harm-pairwise-margin", type=float, default=DEFAULT_SELECTOR_BENEFIT_HARM_PAIRWISE_MARGIN)
        sub.add_argument("--selector-positive-negative-hard-loss-weight", type=float, default=DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_LOSS_WEIGHT)
        sub.add_argument("--selector-positive-negative-hard-margin", type=float, default=DEFAULT_SELECTOR_POSITIVE_NEGATIVE_HARD_MARGIN)
        sub.add_argument(
            "--selector-cross-family-positive-negative-loss-weight",
            type=float,
            default=DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_LOSS_WEIGHT,
        )
        sub.add_argument(
            "--selector-cross-family-positive-negative-margin",
            type=float,
            default=DEFAULT_SELECTOR_CROSS_FAMILY_POSITIVE_NEGATIVE_MARGIN,
        )
        sub.add_argument("--selector-patch-head", action="store_true", default=DEFAULT_SELECTOR_PATCH_HEAD)
        sub.add_argument("--selector-patch-hidden-dim", type=int, default=DEFAULT_SELECTOR_PATCH_HIDDEN_DIM)
        sub.add_argument("--selector-emit-margin-grid", nargs="*", default=None)
        sub.add_argument("--selector-nonzero-bias-grid", nargs="*", default=None)
        sub.add_argument("--selector-transition-prior-weight-grid", nargs="*", default=None)
        sub.add_argument("--selector-transition-compat-top-k-grid", nargs="*", default=None)
        sub.add_argument("--selector-candidate-compat-threshold-grid", nargs="*", default=None)
        sub.add_argument("--selector-candidate-compat-top-k-grid", nargs="*", default=None)
        sub.add_argument("--selector-score-edit-penalty", type=float, default=DEFAULT_SELECTOR_SCORE_EDIT_PENALTY)
        sub.add_argument("--selector-target-mode", choices=SELECTOR_TARGET_MODE_CHOICES, default=SELECTOR_TARGET_MODE_CORRECTNESS)
        sub.add_argument("--selector-harm-weight", type=float, default=DEFAULT_SELECTOR_HARM_WEIGHT)
        sub.add_argument("--selector-miss-weight", type=float, default=DEFAULT_SELECTOR_MISS_WEIGHT)
        sub.add_argument("--selector-policy-candidate-mode", choices=SELECTOR_POLICY_CANDIDATE_MODE_CHOICES, default=SELECTOR_POLICY_CANDIDATE_MODE_ALL)
        sub.add_argument("--selector-candidate-geometry-features", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES)
        sub.add_argument("--selector-candidate-pattern-features", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES)
        sub.add_argument("--selector-candidate-local-evidence-features", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES)
        sub.add_argument("--selector-candidate-local-patch-features", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES)
        sub.add_argument("--selector-epoch-diagnostic-margin-grid", nargs="*", default=None)
        sub.add_argument("--selector-epoch-selection-mode", choices=SELECTOR_EPOCH_SELECTION_CHOICES, default=SELECTOR_EPOCH_SELECTION_PROXY)
        sub.add_argument("--selector-adoption-min-delta", type=float, default=DEFAULT_SELECTOR_ADOPTION_MIN_DELTA)
        sub.add_argument("--selector-adoption-policy", choices=SELECTOR_ADOPTION_POLICY_CHOICES, default=SELECTOR_ADOPTION_POLICY_GLOBAL_NONINFERIOR)
        sub.add_argument("--selector-candidate-first-strong-delta", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_STRONG_DELTA)
        sub.add_argument("--selector-candidate-first-positive-delta", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_DELTA)
        sub.add_argument("--selector-candidate-first-positive-margin-floor", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MARGIN_FLOOR)
        sub.add_argument("--selector-candidate-first-positive-max-harmed", type=int, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED)
        sub.add_argument("--selector-candidate-first-positive-max-margin", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_MARGIN)
        sub.add_argument("--selector-candidate-first-positive-min-nonzero", type=int, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MIN_NONZERO)
        sub.add_argument("--selector-candidate-first-positive-plateau-guard", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_PLATEAU_GUARD)
        sub.add_argument("--selector-candidate-first-tie-min-delta", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_DELTA)
        sub.add_argument("--selector-candidate-first-tie-margin-floor", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MARGIN_FLOOR)
        sub.add_argument("--selector-candidate-first-tie-min-nonzero", type=int, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_TIE_MIN_NONZERO)
        sub.add_argument("--selector-candidate-first-allow-global", action="store_true", default=DEFAULT_SELECTOR_CANDIDATE_FIRST_ALLOW_GLOBAL)
        sub.add_argument("--selector-candidate-first-global-min-delta", type=float, default=DEFAULT_SELECTOR_CANDIDATE_FIRST_GLOBAL_MIN_DELTA)
        sub.add_argument("--selected-no-edit-guardrail", action="store_true", default=DEFAULT_SELECTED_NO_EDIT_GUARDRAIL)
        sub.add_argument("--selected-no-edit-min-delta", type=float, default=DEFAULT_SELECTED_NO_EDIT_MIN_DELTA)
        sub.add_argument("--selector-candidate-motif-max-classes", type=int, default=DEFAULT_SELECTOR_CANDIDATE_MOTIF_MAX_CLASSES)
        sub.add_argument("--selector-local-motif-max-classes", type=int, default=DEFAULT_SELECTOR_LOCAL_MOTIF_MAX_CLASSES)
        sub.add_argument("--selector-local-motif-top-k", type=int, default=DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K)
        sub.add_argument("--transition-prior-hidden-dim", type=int, default=DEFAULT_TRANSITION_PRIOR_HIDDEN_DIM)
        sub.add_argument("--transition-prior-epochs", type=int, default=DEFAULT_TRANSITION_PRIOR_EPOCHS)
        sub.add_argument("--transition-prior-lr", type=float, default=DEFAULT_TRANSITION_PRIOR_LR)
        sub.add_argument("--candidate-compat-hidden-dim", type=int, default=DEFAULT_CANDIDATE_COMPAT_HIDDEN_DIM)
        sub.add_argument("--candidate-compat-epochs", type=int, default=DEFAULT_CANDIDATE_COMPAT_EPOCHS)
        sub.add_argument("--candidate-compat-lr", type=float, default=DEFAULT_CANDIDATE_COMPAT_LR)
        sub.add_argument("--candidate-compat-objective", choices=CANDIDATE_COMPAT_OBJECTIVE_CHOICES, default=CANDIDATE_COMPAT_OBJECTIVE_BCE)
        sub.add_argument("--candidate-compat-negative-ratio", type=float, default=DEFAULT_CANDIDATE_COMPAT_NEGATIVE_RATIO)
        sub.add_argument("--candidate-compat-no-positive-negative-count", type=int, default=DEFAULT_CANDIDATE_COMPAT_NO_POSITIVE_NEGATIVE_COUNT)
        sub.add_argument("--router-hidden-dim", type=int, default=DEFAULT_ROUTER_HIDDEN_DIM)
        sub.add_argument("--router-epochs", type=int, default=DEFAULT_ROUTER_EPOCHS)
        sub.add_argument("--router-lr", type=float, default=DEFAULT_ROUTER_LR)
        sub.add_argument("--router-pos-weight", type=float, default=DEFAULT_ROUTER_POS_WEIGHT)
        sub.add_argument("--router-threshold-grid", nargs="*", default=None)
        sub.add_argument("--router-supervision-target", choices=ROUTER_LABEL_CHOICES, default=ROUTER_LABEL_IDENTITY_VS_NONZERO)
        sub.add_argument("--router-pretrain-target", choices=ROUTER_PRETRAIN_TARGET_CHOICES, default=ROUTER_PRETRAIN_TARGET_NONE)
        sub.add_argument("--router-pretrain-epochs", type=int, default=0)
        sub.add_argument("--router-pretrain-pos-weight", type=float, default=None)
        sub.add_argument("--router-negative-ratio", type=float, default=None)
        sub.add_argument("--action-motif-max-classes", type=int, default=DEFAULT_ACTION_MOTIF_MAX_CLASSES)
        sub.add_argument("--action-motif-loss-weight", type=float, default=DEFAULT_ACTION_MOTIF_LOSS_WEIGHT)
        sub.add_argument("--action-motif-identity-margin", type=float, default=DEFAULT_ACTION_MOTIF_IDENTITY_MARGIN)
        sub.add_argument("--action-motif-emit-margin-grid", nargs="*", default=None)
        sub.add_argument("--local-motif-max-classes", type=int, default=DEFAULT_LOCAL_MOTIF_MAX_CLASSES)
        sub.add_argument("--local-motif-emit-margin-grid", nargs="*", default=None)
        sub.add_argument("--local-motif-min-bit-logit-grid", nargs="*", default=None)
        sub.add_argument("--motif-max-classes", type=int, default=DEFAULT_MOTIF_MAX_CLASSES)
        sub.add_argument("--motif-epochs", type=int, default=DEFAULT_MOTIF_EPOCHS)
        sub.add_argument("--motif-lr", type=float, default=DEFAULT_MOTIF_LR)
        sub.add_argument("--motif-hard-shot-weight", type=float, default=DEFAULT_MOTIF_HARD_SHOT_WEIGHT)
        sub.add_argument("--needs-edit-threshold-grid", nargs="*", default=None)
        sub.add_argument("--edit-threshold-grid", nargs="*", default=None)
        sub.add_argument("--max-edit-weight-grid", nargs="*", default=None)
    return parser


def main() -> None:
    _require_torch()
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.command == "train":
        result = train_family_dir(
            family_dir=args.family_dir,
            checkpoint_out=args.checkpoint_out,
            out_json=args.out_json,
            fill_value=float(args.fill_value),
            max_shots=args.max_shots,
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            hidden_channels=int(args.hidden_channels),
            num_blocks=int(args.num_blocks),
            dense_hidden_dim=int(args.dense_hidden_dim),
            dropout=float(args.dropout),
            needs_edit_loss_weight=float(args.needs_edit_loss_weight),
            sparsity_loss_weight=float(args.sparsity_loss_weight),
            decision_aware_loss_weight=float(args.decision_aware_loss_weight),
            decision_aware_margin=float(args.decision_aware_margin),
            hard_shot_solved_weight=float(args.hard_shot_solved_weight),
            hard_shot_unsolved_weight=float(args.hard_shot_unsolved_weight),
            edit_supervision_mode=str(args.edit_supervision_mode),
            selection_mode=str(args.selection_mode),
            selector_hidden_dim=int(args.selector_hidden_dim),
            selector_epochs=int(args.selector_epochs),
            selector_lr=float(args.selector_lr),
            selector_model=str(args.selector_model),
            selector_risk_aware_harm_logit_weight=float(args.selector_risk_aware_harm_logit_weight),
            selector_risk_aware_benefit_loss_weight=float(args.selector_risk_aware_benefit_loss_weight),
            selector_risk_aware_harm_loss_weight=float(args.selector_risk_aware_harm_loss_weight),
            selector_risk_aware_benefit_pos_weight=float(args.selector_risk_aware_benefit_pos_weight),
            selector_risk_aware_harm_pos_weight=float(args.selector_risk_aware_harm_pos_weight),
            selector_objective=str(args.selector_objective),
            selector_hard_shot_weight=float(args.selector_hard_shot_weight),
            selector_identity_margin_loss_weight=float(args.selector_identity_margin_loss_weight),
            selector_identity_margin=float(args.selector_identity_margin),
            selector_harm_margin_loss_weight=float(args.selector_harm_margin_loss_weight),
            selector_harm_margin=float(args.selector_harm_margin),
            selector_negative_identity_margin_loss_weight=float(
                args.selector_negative_identity_margin_loss_weight
            ),
            selector_negative_identity_margin=float(args.selector_negative_identity_margin),
            selector_benefit_harm_pairwise_loss_weight=float(args.selector_benefit_harm_pairwise_loss_weight),
            selector_benefit_harm_pairwise_margin=float(args.selector_benefit_harm_pairwise_margin),
            selector_positive_negative_hard_loss_weight=float(
                args.selector_positive_negative_hard_loss_weight
            ),
            selector_positive_negative_hard_margin=float(args.selector_positive_negative_hard_margin),
            selector_cross_family_positive_negative_loss_weight=float(
                args.selector_cross_family_positive_negative_loss_weight
            ),
            selector_cross_family_positive_negative_margin=float(
                args.selector_cross_family_positive_negative_margin
            ),
            selector_patch_head=bool(args.selector_patch_head),
            selector_patch_hidden_dim=int(args.selector_patch_hidden_dim),
            selector_emit_margin_grid=_parse_float_list(
                args.selector_emit_margin_grid,
                default=list(DEFAULT_SELECTOR_EMIT_MARGIN_GRID),
            ),
            selector_nonzero_bias_grid=_parse_float_list(
                args.selector_nonzero_bias_grid,
                default=list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID),
            ),
            selector_transition_prior_weight_grid=_parse_float_list(
                args.selector_transition_prior_weight_grid,
                default=list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID),
            ),
            selector_transition_compat_top_k_grid=_parse_int_list(
                args.selector_transition_compat_top_k_grid,
                default=list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID),
            ),
            selector_candidate_compat_threshold_grid=_parse_float_list(
                args.selector_candidate_compat_threshold_grid,
                default=list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID),
            ),
            selector_candidate_compat_top_k_grid=_parse_int_list(
                args.selector_candidate_compat_top_k_grid,
                default=list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID),
            ),
            candidate_compat_hidden_dim=int(args.candidate_compat_hidden_dim),
            candidate_compat_epochs=int(args.candidate_compat_epochs),
            candidate_compat_lr=float(args.candidate_compat_lr),
            candidate_compat_objective=str(args.candidate_compat_objective),
            candidate_compat_negative_ratio=float(args.candidate_compat_negative_ratio),
            candidate_compat_no_positive_negative_count=int(args.candidate_compat_no_positive_negative_count),
            selector_score_edit_penalty=float(args.selector_score_edit_penalty),
            selector_target_mode=str(args.selector_target_mode),
            selector_harm_weight=float(args.selector_harm_weight),
            selector_miss_weight=float(args.selector_miss_weight),
            selector_policy_candidate_mode=str(args.selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(args.selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(args.selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(args.selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(args.selector_candidate_local_patch_features),
            selector_epoch_diagnostic_margin_grid=_parse_float_list(
                args.selector_epoch_diagnostic_margin_grid,
                default=[],
            ),
            selector_epoch_selection_mode=str(args.selector_epoch_selection_mode),
            selector_adoption_min_delta=float(args.selector_adoption_min_delta),
            selector_adoption_policy=str(args.selector_adoption_policy),
            selector_candidate_first_strong_delta=float(args.selector_candidate_first_strong_delta),
            selector_candidate_first_positive_delta=float(args.selector_candidate_first_positive_delta),
            selector_candidate_first_positive_margin_floor=float(args.selector_candidate_first_positive_margin_floor),
            selector_candidate_first_positive_max_harmed=int(args.selector_candidate_first_positive_max_harmed),
            selector_candidate_first_positive_max_margin=float(args.selector_candidate_first_positive_max_margin),
            selector_candidate_first_positive_min_nonzero=int(args.selector_candidate_first_positive_min_nonzero),
            selector_candidate_first_positive_plateau_guard=bool(args.selector_candidate_first_positive_plateau_guard),
            selector_candidate_first_tie_min_delta=float(args.selector_candidate_first_tie_min_delta),
            selector_candidate_first_tie_margin_floor=float(args.selector_candidate_first_tie_margin_floor),
            selector_candidate_first_tie_min_nonzero=int(args.selector_candidate_first_tie_min_nonzero),
            selector_candidate_first_allow_global=bool(args.selector_candidate_first_allow_global),
            selector_candidate_first_global_min_delta=float(args.selector_candidate_first_global_min_delta),
            selected_no_edit_guardrail=bool(args.selected_no_edit_guardrail),
            selected_no_edit_min_delta=float(args.selected_no_edit_min_delta),
            selector_candidate_motif_max_classes=int(args.selector_candidate_motif_max_classes),
            selector_local_motif_max_classes=int(args.selector_local_motif_max_classes),
            selector_local_motif_top_k=int(args.selector_local_motif_top_k),
            transition_prior_hidden_dim=int(args.transition_prior_hidden_dim),
            transition_prior_epochs=int(args.transition_prior_epochs),
            transition_prior_lr=float(args.transition_prior_lr),
            router_hidden_dim=int(args.router_hidden_dim),
            router_epochs=int(args.router_epochs),
            router_lr=float(args.router_lr),
            router_pos_weight=float(args.router_pos_weight),
            router_threshold_grid=_parse_float_list(
                args.router_threshold_grid,
                default=list(DEFAULT_ROUTER_THRESHOLD_GRID),
            ),
            router_supervision_target=str(args.router_supervision_target),
            router_pretrain_target=str(args.router_pretrain_target),
            router_pretrain_epochs=int(args.router_pretrain_epochs),
            router_pretrain_pos_weight=(
                float(args.router_pretrain_pos_weight)
                if args.router_pretrain_pos_weight is not None
                else None
            ),
            router_negative_ratio=(
                float(args.router_negative_ratio) if args.router_negative_ratio is not None else None
            ),
            action_motif_max_classes=int(args.action_motif_max_classes),
            action_motif_loss_weight=float(args.action_motif_loss_weight),
            action_motif_identity_margin=float(args.action_motif_identity_margin),
            action_motif_emit_margin_grid=_parse_float_list(
                args.action_motif_emit_margin_grid,
                default=list(DEFAULT_ACTION_MOTIF_EMIT_MARGIN_GRID),
            ),
            local_motif_max_classes=int(args.local_motif_max_classes),
            local_motif_emit_margin_grid=_parse_float_list(
                args.local_motif_emit_margin_grid,
                default=list(DEFAULT_LOCAL_MOTIF_EMIT_MARGIN_GRID),
            ),
            local_motif_min_bit_logit_grid=_parse_float_list(
                args.local_motif_min_bit_logit_grid,
                default=list(DEFAULT_LOCAL_MOTIF_MIN_BIT_LOGIT_GRID),
            ),
            motif_max_classes=int(args.motif_max_classes),
            motif_epochs=int(args.motif_epochs),
            motif_lr=float(args.motif_lr),
            motif_hard_shot_weight=float(args.motif_hard_shot_weight),
            needs_edit_threshold_grid=_parse_float_list(args.needs_edit_threshold_grid, default=[0.3, 0.5, 0.7, 0.9]),
            edit_threshold_grid=_parse_float_list(args.edit_threshold_grid, default=[0.3, 0.5, 0.7, 0.9]),
            max_edit_weight_grid=_parse_int_list(args.max_edit_weight_grid, default=[0, 1, 2]),
        )
    elif args.command == "eval":
        result = evaluate_checkpoint_on_family(
            checkpoint_path=args.checkpoint,
            family_dir=args.family_dir,
            out_json=args.out_json,
            fill_value=float(args.fill_value),
            max_shots=args.max_shots,
            batch_size=int(args.batch_size),
        )
    else:
        result = run_manifest_experiment(
            manifest_path=args.manifest,
            train_families=list(args.train_families),
            eval_families=(list(args.eval_families) if args.eval_families else None),
            out_dir=args.out_dir,
            fill_value=float(args.fill_value),
            max_shots=args.max_shots,
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            hidden_channels=int(args.hidden_channels),
            num_blocks=int(args.num_blocks),
            dense_hidden_dim=int(args.dense_hidden_dim),
            dropout=float(args.dropout),
            needs_edit_loss_weight=float(args.needs_edit_loss_weight),
            sparsity_loss_weight=float(args.sparsity_loss_weight),
            decision_aware_loss_weight=float(args.decision_aware_loss_weight),
            decision_aware_margin=float(args.decision_aware_margin),
            hard_shot_solved_weight=float(args.hard_shot_solved_weight),
            hard_shot_unsolved_weight=float(args.hard_shot_unsolved_weight),
            edit_supervision_mode=str(args.edit_supervision_mode),
            selection_mode=str(args.selection_mode),
            selector_hidden_dim=int(args.selector_hidden_dim),
            selector_epochs=int(args.selector_epochs),
            selector_lr=float(args.selector_lr),
            selector_model=str(args.selector_model),
            selector_risk_aware_harm_logit_weight=float(args.selector_risk_aware_harm_logit_weight),
            selector_risk_aware_benefit_loss_weight=float(args.selector_risk_aware_benefit_loss_weight),
            selector_risk_aware_harm_loss_weight=float(args.selector_risk_aware_harm_loss_weight),
            selector_risk_aware_benefit_pos_weight=float(args.selector_risk_aware_benefit_pos_weight),
            selector_risk_aware_harm_pos_weight=float(args.selector_risk_aware_harm_pos_weight),
            selector_objective=str(args.selector_objective),
            selector_hard_shot_weight=float(args.selector_hard_shot_weight),
            selector_identity_margin_loss_weight=float(args.selector_identity_margin_loss_weight),
            selector_identity_margin=float(args.selector_identity_margin),
            selector_harm_margin_loss_weight=float(args.selector_harm_margin_loss_weight),
            selector_harm_margin=float(args.selector_harm_margin),
            selector_negative_identity_margin_loss_weight=float(
                args.selector_negative_identity_margin_loss_weight
            ),
            selector_negative_identity_margin=float(args.selector_negative_identity_margin),
            selector_benefit_harm_pairwise_loss_weight=float(args.selector_benefit_harm_pairwise_loss_weight),
            selector_benefit_harm_pairwise_margin=float(args.selector_benefit_harm_pairwise_margin),
            selector_positive_negative_hard_loss_weight=float(
                args.selector_positive_negative_hard_loss_weight
            ),
            selector_positive_negative_hard_margin=float(args.selector_positive_negative_hard_margin),
            selector_cross_family_positive_negative_loss_weight=float(
                args.selector_cross_family_positive_negative_loss_weight
            ),
            selector_cross_family_positive_negative_margin=float(
                args.selector_cross_family_positive_negative_margin
            ),
            selector_patch_head=bool(args.selector_patch_head),
            selector_patch_hidden_dim=int(args.selector_patch_hidden_dim),
            selector_emit_margin_grid=_parse_float_list(
                args.selector_emit_margin_grid,
                default=list(DEFAULT_SELECTOR_EMIT_MARGIN_GRID),
            ),
            selector_nonzero_bias_grid=_parse_float_list(
                args.selector_nonzero_bias_grid,
                default=list(DEFAULT_SELECTOR_NONZERO_BIAS_GRID),
            ),
            selector_transition_prior_weight_grid=_parse_float_list(
                args.selector_transition_prior_weight_grid,
                default=list(DEFAULT_SELECTOR_TRANSITION_PRIOR_WEIGHT_GRID),
            ),
            selector_transition_compat_top_k_grid=_parse_int_list(
                args.selector_transition_compat_top_k_grid,
                default=list(DEFAULT_SELECTOR_TRANSITION_COMPAT_TOP_K_GRID),
            ),
            selector_candidate_compat_threshold_grid=_parse_float_list(
                args.selector_candidate_compat_threshold_grid,
                default=list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_THRESHOLD_GRID),
            ),
            selector_candidate_compat_top_k_grid=_parse_int_list(
                args.selector_candidate_compat_top_k_grid,
                default=list(DEFAULT_SELECTOR_CANDIDATE_COMPAT_TOP_K_GRID),
            ),
            candidate_compat_hidden_dim=int(args.candidate_compat_hidden_dim),
            candidate_compat_epochs=int(args.candidate_compat_epochs),
            candidate_compat_lr=float(args.candidate_compat_lr),
            candidate_compat_objective=str(args.candidate_compat_objective),
            candidate_compat_negative_ratio=float(args.candidate_compat_negative_ratio),
            candidate_compat_no_positive_negative_count=int(args.candidate_compat_no_positive_negative_count),
            selector_score_edit_penalty=float(args.selector_score_edit_penalty),
            selector_target_mode=str(args.selector_target_mode),
            selector_harm_weight=float(args.selector_harm_weight),
            selector_miss_weight=float(args.selector_miss_weight),
            selector_policy_candidate_mode=str(args.selector_policy_candidate_mode),
            selector_candidate_geometry_features=bool(args.selector_candidate_geometry_features),
            selector_candidate_pattern_features=bool(args.selector_candidate_pattern_features),
            selector_candidate_local_evidence_features=bool(args.selector_candidate_local_evidence_features),
            selector_candidate_local_patch_features=bool(args.selector_candidate_local_patch_features),
            selector_epoch_diagnostic_margin_grid=_parse_float_list(
                args.selector_epoch_diagnostic_margin_grid,
                default=[],
            ),
            selector_epoch_selection_mode=str(args.selector_epoch_selection_mode),
            selector_adoption_min_delta=float(args.selector_adoption_min_delta),
            selector_adoption_policy=str(args.selector_adoption_policy),
            selector_candidate_first_strong_delta=float(args.selector_candidate_first_strong_delta),
            selector_candidate_first_positive_delta=float(args.selector_candidate_first_positive_delta),
            selector_candidate_first_positive_margin_floor=float(args.selector_candidate_first_positive_margin_floor),
            selector_candidate_first_positive_max_harmed=int(args.selector_candidate_first_positive_max_harmed),
            selector_candidate_first_positive_max_margin=float(args.selector_candidate_first_positive_max_margin),
            selector_candidate_first_positive_min_nonzero=int(args.selector_candidate_first_positive_min_nonzero),
            selector_candidate_first_positive_plateau_guard=bool(args.selector_candidate_first_positive_plateau_guard),
            selector_candidate_first_tie_min_delta=float(args.selector_candidate_first_tie_min_delta),
            selector_candidate_first_tie_margin_floor=float(args.selector_candidate_first_tie_margin_floor),
            selector_candidate_first_tie_min_nonzero=int(args.selector_candidate_first_tie_min_nonzero),
            selector_candidate_first_allow_global=bool(args.selector_candidate_first_allow_global),
            selector_candidate_first_global_min_delta=float(args.selector_candidate_first_global_min_delta),
            selected_no_edit_guardrail=bool(args.selected_no_edit_guardrail),
            selected_no_edit_min_delta=float(args.selected_no_edit_min_delta),
            selector_candidate_motif_max_classes=int(args.selector_candidate_motif_max_classes),
            selector_local_motif_max_classes=int(args.selector_local_motif_max_classes),
            selector_local_motif_top_k=int(args.selector_local_motif_top_k),
            transition_prior_hidden_dim=int(args.transition_prior_hidden_dim),
            transition_prior_epochs=int(args.transition_prior_epochs),
            transition_prior_lr=float(args.transition_prior_lr),
            router_hidden_dim=int(args.router_hidden_dim),
            router_epochs=int(args.router_epochs),
            router_lr=float(args.router_lr),
            router_pos_weight=float(args.router_pos_weight),
            router_threshold_grid=_parse_float_list(
                args.router_threshold_grid,
                default=list(DEFAULT_ROUTER_THRESHOLD_GRID),
            ),
            router_supervision_target=str(args.router_supervision_target),
            router_pretrain_target=str(args.router_pretrain_target),
            router_pretrain_epochs=int(args.router_pretrain_epochs),
            router_pretrain_pos_weight=(
                float(args.router_pretrain_pos_weight)
                if args.router_pretrain_pos_weight is not None
                else None
            ),
            router_negative_ratio=(
                float(args.router_negative_ratio) if args.router_negative_ratio is not None else None
            ),
            action_motif_max_classes=int(args.action_motif_max_classes),
            action_motif_loss_weight=float(args.action_motif_loss_weight),
            action_motif_identity_margin=float(args.action_motif_identity_margin),
            action_motif_emit_margin_grid=_parse_float_list(
                args.action_motif_emit_margin_grid,
                default=list(DEFAULT_ACTION_MOTIF_EMIT_MARGIN_GRID),
            ),
            local_motif_max_classes=int(args.local_motif_max_classes),
            local_motif_emit_margin_grid=_parse_float_list(
                args.local_motif_emit_margin_grid,
                default=list(DEFAULT_LOCAL_MOTIF_EMIT_MARGIN_GRID),
            ),
            local_motif_min_bit_logit_grid=_parse_float_list(
                args.local_motif_min_bit_logit_grid,
                default=list(DEFAULT_LOCAL_MOTIF_MIN_BIT_LOGIT_GRID),
            ),
            motif_max_classes=int(args.motif_max_classes),
            motif_epochs=int(args.motif_epochs),
            motif_lr=float(args.motif_lr),
            motif_hard_shot_weight=float(args.motif_hard_shot_weight),
            needs_edit_threshold_grid=_parse_float_list(args.needs_edit_threshold_grid, default=[0.3, 0.5, 0.7, 0.9]),
            edit_threshold_grid=_parse_float_list(args.edit_threshold_grid, default=[0.3, 0.5, 0.7, 0.9]),
            max_edit_weight_grid=_parse_int_list(args.max_edit_weight_grid, default=[0, 1, 2]),
        )
    print(json.dumps({"decoder": result["decoder"], "created_at_utc": result["created_at_utc"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
