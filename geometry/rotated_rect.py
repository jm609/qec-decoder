from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class RectangularSyndromeLayout:
    time_steps: int
    height: int
    width: int
    detector_time_index: np.ndarray
    row_index_by_detector: np.ndarray
    col_index_by_detector: np.ndarray
    detector_index_volume: np.ndarray
    valid_mask: np.ndarray
    checkerboard_volume: np.ndarray
    detector_type_volume: np.ndarray
    boundary_volume: np.ndarray
    final_round_volume: np.ndarray
    metadata_summary: dict[str, Any]


def _as_float32_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_int16_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int16).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_uint8_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _round_to_rect_index(values: np.ndarray, *, axis_name: str) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32) / np.float32(2.0)
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-4, rtol=0.0):
        bad = np.flatnonzero(~np.isclose(scaled, rounded, atol=1e-4, rtol=0.0))
        preview = bad[:8].tolist()
        raise ValueError(
            f"{axis_name} coordinates are not aligned to the expected even lattice; "
            f"bad_indices={preview}"
        )
    out = rounded.astype(np.int16, copy=False)
    if np.any(out < 0):
        raise ValueError(f"{axis_name} rectangular indices must be nonnegative")
    return np.ascontiguousarray(out)


def build_rectangular_syndrome_layout(arrays: dict[str, np.ndarray]) -> RectangularSyndromeLayout:
    detector_coordinates = _as_float32_2d(arrays["detector_coordinates"], name="detector_coordinates")
    detector_time_index = _as_int16_1d(arrays["detector_time_index"], name="detector_time_index")
    detector_final_round_flag = _as_uint8_1d(
        arrays["detector_final_round_flag"],
        name="detector_final_round_flag",
    )
    detector_boundary_flag = _as_uint8_1d(
        arrays["detector_boundary_flag"],
        name="detector_boundary_flag",
    )
    detector_checkerboard_class = _as_uint8_1d(
        arrays["detector_checkerboard_class"],
        name="detector_checkerboard_class",
    )
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
        raise ValueError("detector_time_index must be nonnegative for rectangular layout building")

    row_index_by_detector = _round_to_rect_index(detector_coordinates[:, 1], axis_name="y")
    col_index_by_detector = _round_to_rect_index(detector_coordinates[:, 0], axis_name="x")

    time_steps = int(detector_time_index.max()) + 1 if detector_time_index.size else 0
    height = int(row_index_by_detector.max()) + 1 if row_index_by_detector.size else 0
    width = int(col_index_by_detector.max()) + 1 if col_index_by_detector.size else 0

    detector_index_volume = np.full((time_steps, height, width), -1, dtype=np.int32)
    valid_mask = np.zeros((time_steps, height, width), dtype=np.uint8)
    checkerboard_volume = np.full((time_steps, height, width), 255, dtype=np.uint8)
    detector_type_volume = np.zeros((time_steps, height, width), dtype=np.uint8)
    boundary_volume = np.zeros((time_steps, height, width), dtype=np.uint8)
    final_round_volume = np.zeros((time_steps, height, width), dtype=np.uint8)

    for det_idx in range(num_detectors):
        t = int(detector_time_index[det_idx])
        r = int(row_index_by_detector[det_idx])
        c = int(col_index_by_detector[det_idx])
        if detector_index_volume[t, r, c] != -1:
            prev = int(detector_index_volume[t, r, c])
            raise ValueError(
                "Multiple detectors map to the same (time, row, col) slot: "
                f"time={t}, row={r}, col={c}, prev_detector={prev}, detector={det_idx}"
            )
        detector_index_volume[t, r, c] = int(det_idx)
        valid_mask[t, r, c] = 1
        checkerboard_volume[t, r, c] = np.uint8(detector_checkerboard_class[det_idx])
        detector_type_volume[t, r, c] = np.uint8(detector_type[det_idx])
        boundary_volume[t, r, c] = np.uint8(detector_boundary_flag[det_idx])
        final_round_volume[t, r, c] = np.uint8(detector_final_round_flag[det_idx])

    per_time_active = valid_mask.reshape(time_steps, -1).sum(axis=1).astype(np.int64, copy=False)
    metadata_summary = {
        "num_detectors": num_detectors,
        "time_steps": time_steps,
        "grid_height": height,
        "grid_width": width,
        "grid_area": int(height * width),
        "active_slots_total": int(valid_mask.sum()),
        "active_slots_per_time_histogram": _counter_dict(per_time_active),
        "checkerboard_class_histogram": _counter_dict(
            detector_checkerboard_class.astype(np.int64, copy=False)
        ),
        "detector_type_histogram": _counter_dict(detector_type.astype(np.int64, copy=False)),
        "boundary_fraction": float(detector_boundary_flag.mean()) if detector_boundary_flag.size else 0.0,
        "final_round_fraction": float(detector_final_round_flag.mean()) if detector_final_round_flag.size else 0.0,
    }

    return RectangularSyndromeLayout(
        time_steps=time_steps,
        height=height,
        width=width,
        detector_time_index=detector_time_index,
        row_index_by_detector=row_index_by_detector,
        col_index_by_detector=col_index_by_detector,
        detector_index_volume=detector_index_volume,
        valid_mask=valid_mask,
        checkerboard_volume=checkerboard_volume,
        detector_type_volume=detector_type_volume,
        boundary_volume=boundary_volume,
        final_round_volume=final_round_volume,
        metadata_summary=metadata_summary,
    )


def build_rectangular_syndrome_volume(
    arrays: dict[str, np.ndarray],
    *,
    layout: RectangularSyndromeLayout,
    fill_value: float = -0.5,
) -> np.ndarray:
    detector_events = np.asarray(arrays["detector_events"], dtype=np.uint8)
    if detector_events.ndim != 2:
        raise ValueError(f"detector_events must be rank-2, got shape={detector_events.shape}")
    if int(detector_events.shape[1]) != int(layout.row_index_by_detector.shape[0]):
        raise ValueError(
            "Rectangular layout detector dimension does not match detector_events: "
            f"layout={layout.row_index_by_detector.shape[0]}, detector_events={detector_events.shape[1]}"
        )

    num_shots = int(detector_events.shape[0])
    volume = np.full(
        (num_shots, layout.time_steps, layout.height, layout.width),
        np.float32(fill_value),
        dtype=np.float32,
    )
    t = layout.detector_time_index.astype(np.intp, copy=False)
    r = layout.row_index_by_detector.astype(np.intp, copy=False)
    c = layout.col_index_by_detector.astype(np.intp, copy=False)
    volume[:, t, r, c] = detector_events.astype(np.float32, copy=False)
    return np.ascontiguousarray(volume)


def describe_rectangular_syndrome_layout(layout: RectangularSyndromeLayout) -> dict[str, Any]:
    occupancy = 0.0
    if layout.time_steps > 0 and layout.height > 0 and layout.width > 0:
        occupancy = float(layout.valid_mask.mean())
    return {
        "representation": "rectangular_syndrome_layout",
        "shape": {
            "time_steps": int(layout.time_steps),
            "grid_height": int(layout.height),
            "grid_width": int(layout.width),
        },
        "occupancy_fraction": occupancy,
        "layout_summary": dict(layout.metadata_summary),
    }
