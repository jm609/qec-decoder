from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackLayout:
    num_tracks: int
    max_time_steps: int
    unique_xy: np.ndarray
    inverse_track: np.ndarray
    detector_time_index: np.ndarray
    canonical_detector_index: np.ndarray
    track_mask: np.ndarray
    track_static: np.ndarray
    track_static_names: list[str]
    track_lengths: np.ndarray
    metadata_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrackTensorBundle:
    x: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    layout: TrackLayout
    feature_names: list[str]
    dataset_summary: dict[str, Any]


def _as_uint8_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={out.shape}")
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


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _binary_one_hot(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((v.shape[0], 2), dtype=np.float32)
    clipped = np.clip(v, 0, 1)
    out[np.arange(v.shape[0]), clipped] = 1.0
    return out


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if math.isclose(xmin, xmax):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / float(xmax - xmin)).astype(np.float32, copy=False)


def build_track_layout(arrays: dict[str, np.ndarray]) -> TrackLayout:
    detector_coordinates = _as_float32_2d(arrays["detector_coordinates"], name="detector_coordinates")
    detector_time_index = _as_int16_1d(arrays["detector_time_index"], name="detector_time_index")
    detector_final_round_flag = _as_uint8_1d(arrays["detector_final_round_flag"], name="detector_final_round_flag")
    detector_boundary_flag = _as_uint8_1d(arrays["detector_boundary_flag"], name="detector_boundary_flag")
    detector_checkerboard_class = _as_uint8_1d(arrays["detector_checkerboard_class"], name="detector_checkerboard_class")
    detector_type = _as_uint8_1d(arrays["detector_type"], name="detector_type")

    num_detectors = int(detector_coordinates.shape[0])
    for key, arr in {
        "detector_time_index": detector_time_index,
        "detector_final_round_flag": detector_final_round_flag,
        "detector_boundary_flag": detector_boundary_flag,
        "detector_checkerboard_class": detector_checkerboard_class,
        "detector_type": detector_type,
    }.items():
        if int(arr.shape[0]) != num_detectors:
            raise ValueError(
                f"{key} length does not match detector dimension: {arr.shape[0]} vs {num_detectors}"
            )

    if detector_coordinates.shape[1] < 2:
        raise ValueError(
            f"detector_coordinates must have at least 2 columns for (x, y), got {detector_coordinates.shape}"
        )
    if np.any(detector_time_index < 0):
        raise ValueError("detector_time_index must be nonnegative for track layout building")

    xy = np.ascontiguousarray(detector_coordinates[:, :2], dtype=np.float32)
    unique_xy, inverse_track, track_lengths = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    inverse_track = np.asarray(inverse_track, dtype=np.int64)
    track_lengths = np.asarray(track_lengths, dtype=np.int64)
    num_tracks = int(unique_xy.shape[0])
    max_time_steps = int(detector_time_index.max()) + 1 if detector_time_index.size else 0

    canonical_detector_index = np.full((num_tracks, max_time_steps), -1, dtype=np.int64)
    track_mask = np.zeros((num_tracks, max_time_steps), dtype=np.uint8)

    for det_idx in range(num_detectors):
        track_id = int(inverse_track[det_idx])
        t = int(detector_time_index[det_idx])
        if canonical_detector_index[track_id, t] != -1:
            prev = int(canonical_detector_index[track_id, t])
            raise ValueError(
                "Multiple detectors map to the same (track_id, time_index) slot: "
                f"track_id={track_id}, time_index={t}, prev_detector={prev}, detector={det_idx}"
            )
        canonical_detector_index[track_id, t] = det_idx
        track_mask[track_id, t] = 1

    track_lengths_from_mask = track_mask.sum(axis=1).astype(np.int64, copy=False)
    if not np.array_equal(track_lengths_from_mask, track_lengths):
        raise ValueError(
            "Track length mismatch between np.unique counts and canonical track mask counts"
        )

    x_norm = _normalize_vector(unique_xy[:, 0])
    y_norm = _normalize_vector(unique_xy[:, 1])
    track_boundary = np.zeros(num_tracks, dtype=np.float32)
    track_checkerboard = np.zeros(num_tracks, dtype=np.uint8)
    track_detector_type = np.zeros(num_tracks, dtype=np.uint8)

    type_constant_tracks = 0
    checker_constant_tracks = 0
    boundary_constant_tracks = 0
    final_round_singleton_tracks = 0
    contiguous_time_tracks = 0

    for track_id in range(num_tracks):
        idx = np.flatnonzero(inverse_track == track_id)
        if idx.size == 0:
            raise ValueError(f"Empty track found at track_id={track_id}")

        if np.unique(detector_type[idx]).size != 1:
            raise ValueError(f"detector_type is not constant within track_id={track_id}")
        type_constant_tracks += 1
        track_detector_type[track_id] = detector_type[idx[0]]

        if np.unique(detector_checkerboard_class[idx]).size != 1:
            raise ValueError(f"detector_checkerboard_class is not constant within track_id={track_id}")
        checker_constant_tracks += 1
        track_checkerboard[track_id] = detector_checkerboard_class[idx[0]]

        if np.unique(detector_boundary_flag[idx]).size != 1:
            raise ValueError(f"detector_boundary_flag is not constant within track_id={track_id}")
        boundary_constant_tracks += 1
        track_boundary[track_id] = float(detector_boundary_flag[idx[0]])

        times = np.sort(detector_time_index[idx].astype(np.int64, copy=False))
        if times.size <= 1 or np.all(np.diff(times) == 1):
            contiguous_time_tracks += 1

        if int(detector_final_round_flag[idx].sum()) > 1:
            raise ValueError(f"detector_final_round_flag appears more than once in track_id={track_id}")
        final_round_singleton_tracks += 1

    checker_one_hot = _binary_one_hot(track_checkerboard)
    is_x_check = (track_detector_type == 1).astype(np.float32)
    is_z_check = (track_detector_type == 2).astype(np.float32)

    track_static = np.stack(
        [
            x_norm,
            y_norm,
            track_boundary.astype(np.float32, copy=False),
            checker_one_hot[:, 0],
            checker_one_hot[:, 1],
            is_x_check,
            is_z_check,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    track_static_names = [
        "x_norm",
        "y_norm",
        "boundary_flag",
        "checkerboard_class_0",
        "checkerboard_class_1",
        "is_x_check",
        "is_z_check",
    ]

    metadata_summary = {
        "num_detectors": num_detectors,
        "num_tracks": num_tracks,
        "max_time_steps": max_time_steps,
        "track_length_histogram": _counter_dict(track_lengths),
        "track_length_min": int(track_lengths.min()) if track_lengths.size else None,
        "track_length_max": int(track_lengths.max()) if track_lengths.size else None,
        "type_constant_tracks": int(type_constant_tracks),
        "checker_constant_tracks": int(checker_constant_tracks),
        "boundary_constant_tracks": int(boundary_constant_tracks),
        "contiguous_time_tracks": int(contiguous_time_tracks),
        "final_round_singleton_tracks": int(final_round_singleton_tracks),
    }

    return TrackLayout(
        num_tracks=num_tracks,
        max_time_steps=max_time_steps,
        unique_xy=unique_xy.astype(np.float32, copy=False),
        inverse_track=inverse_track,
        detector_time_index=detector_time_index,
        canonical_detector_index=canonical_detector_index,
        track_mask=track_mask,
        track_static=track_static,
        track_static_names=track_static_names,
        track_lengths=track_lengths.astype(np.int64, copy=False),
        metadata_summary=metadata_summary,
    )


def validate_layout_against_known_geometry(layout: TrackLayout, metadata: dict[str, Any]) -> None:
    circuit = metadata.get("circuit", {}) if isinstance(metadata, dict) else {}
    distance = circuit.get("distance")
    expected_by_distance = {
        3: {"num_tracks": 8, "max_time_steps": 4, "track_hist": {"2": 4, "4": 4}},
        5: {"num_tracks": 24, "max_time_steps": 11, "track_hist": {"9": 12, "11": 12}},
        7: {"num_tracks": 48, "max_time_steps": 15, "track_hist": {"13": 24, "15": 24}},
    }
    if distance not in expected_by_distance:
        return

    expected = expected_by_distance[int(distance)]
    observed_hist = layout.metadata_summary.get("track_length_histogram", {})
    if int(layout.num_tracks) != int(expected["num_tracks"]):
        raise ValueError(
            f"Geometry sanity failed for d={distance}: num_tracks={layout.num_tracks}, "
            f"expected {expected['num_tracks']}"
        )
    if int(layout.max_time_steps) != int(expected["max_time_steps"]):
        raise ValueError(
            f"Geometry sanity failed for d={distance}: max_time_steps={layout.max_time_steps}, "
            f"expected {expected['max_time_steps']}"
        )
    if observed_hist != expected["track_hist"]:
        raise ValueError(
            f"Mask sanity failed for d={distance}: track_length_histogram={observed_hist}, "
            f"expected {expected['track_hist']}"
        )


def build_track_tensor_bundle(
    arrays: dict[str, np.ndarray],
    *,
    layout: TrackLayout,
    include_xy: bool = True,
    include_boundary: bool = True,
    include_checkerboard: bool = True,
    include_detector_type: bool = True,
    include_final_round: bool = True,
    include_valid_mask_feature: bool = True,
) -> TrackTensorBundle:
    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")
    detector_final_round_flag = _as_uint8_1d(arrays["detector_final_round_flag"], name="detector_final_round_flag")

    num_shots, num_detectors = detector_events.shape
    if int(layout.inverse_track.shape[0]) != num_detectors:
        raise ValueError(
            "Track layout detector dimension does not match detector_events: "
            f"layout={layout.inverse_track.shape[0]}, detector_events={num_detectors}"
        )
    if logical_label.shape[0] != num_shots:
        raise ValueError(
            f"logical_label length does not match detector_events shots: {logical_label.shape[0]} vs {num_shots}"
        )

    num_tracks = int(layout.num_tracks)
    tmax = int(layout.max_time_steps)
    track_mask_2d = layout.track_mask.astype(np.float32, copy=False)
    feature_planes: list[np.ndarray] = []
    feature_names: list[str] = []

    event_plane = np.zeros((num_shots, num_tracks, tmax), dtype=np.float32)
    final_round_plane = np.zeros((num_tracks, tmax), dtype=np.float32)
    for track_id in range(num_tracks):
        for t in range(tmax):
            det_idx = int(layout.canonical_detector_index[track_id, t])
            if det_idx < 0:
                continue
            event_plane[:, track_id, t] = detector_events[:, det_idx].astype(np.float32, copy=False)
            final_round_plane[track_id, t] = float(detector_final_round_flag[det_idx])

    feature_planes.append(event_plane)
    feature_names.append("event")

    if include_valid_mask_feature:
        feature_planes.append(
            np.broadcast_to(track_mask_2d[None, :, :], (num_shots, num_tracks, tmax)).astype(np.float32, copy=False)
        )
        feature_names.append("valid_mask")

    if include_final_round:
        feature_planes.append(
            np.broadcast_to(final_round_plane[None, :, :], (num_shots, num_tracks, tmax)).astype(np.float32, copy=False)
        )
        feature_names.append("final_round_flag")

    static_name_to_col = {name: idx for idx, name in enumerate(layout.track_static_names)}

    def add_static_feature(name: str) -> None:
        col = static_name_to_col[name]
        static_track = layout.track_static[:, col][:, None] * track_mask_2d
        broadcast = np.broadcast_to(static_track[None, :, :], (num_shots, num_tracks, tmax)).astype(
            np.float32,
            copy=False,
        )
        feature_planes.append(broadcast)
        feature_names.append(name)

    if include_xy:
        add_static_feature("x_norm")
        add_static_feature("y_norm")
    if include_boundary:
        add_static_feature("boundary_flag")
    if include_checkerboard:
        add_static_feature("checkerboard_class_0")
        add_static_feature("checkerboard_class_1")
    if include_detector_type:
        add_static_feature("is_x_check")
        add_static_feature("is_z_check")

    x = np.stack(feature_planes, axis=-1).astype(np.float32, copy=False)
    mask = np.broadcast_to(layout.track_mask[None, :, :], (num_shots, num_tracks, tmax)).astype(
        np.uint8,
        copy=False,
    )
    y = logical_label.astype(np.float32, copy=False)

    dataset_summary = {
        "num_shots": num_shots,
        "num_detectors": num_detectors,
        "num_tracks": num_tracks,
        "max_time_steps": tmax,
        "num_features": int(x.shape[-1]),
        "feature_names": feature_names,
        "logical_positive_rate": float(y.mean()) if y.size else 0.0,
        "layout_summary": layout.metadata_summary,
    }

    return TrackTensorBundle(
        x=np.ascontiguousarray(x),
        mask=np.ascontiguousarray(mask),
        y=np.ascontiguousarray(y),
        layout=layout,
        feature_names=feature_names,
        dataset_summary=dataset_summary,
    )


def describe_track_bundle(bundle: TrackTensorBundle) -> dict[str, Any]:
    return {
        "representation": "track_tensor",
        "shape": {
            "batch": int(bundle.x.shape[0]),
            "num_tracks": int(bundle.x.shape[1]),
            "max_time_steps": int(bundle.x.shape[2]),
            "num_features": int(bundle.x.shape[3]),
        },
        "feature_names": list(bundle.feature_names),
        "layout_summary": bundle.layout.metadata_summary,
    }
