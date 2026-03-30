from __future__ import annotations

"""
inspect_samples.py

Dataset-level inspection utility for samples.npz files produced by sample_dataset.py.

Main goals
----------
- Inspect one or more family directories and/or manifest.json files.
- Summarize detector-event geometry and static detector metadata.
- Expose same-(x,y) track repetition across time for geometry-aware model design.
- Produce both readable console output and optional structured JSON.

Supported inputs
----------------
1. --family-dir <dir>
   Directory containing samples.npz, optionally metadata.json.
2. --manifest <manifest.json>
   Manifest produced by sample_dataset.py. By default all families in the manifest are inspected.
   Use repeated --family <name> to restrict to selected family names.
3. --samples-npz <path> [--metadata-json <path>] [--label <name>]
   Direct file mode for standalone uploaded artifacts.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json

import numpy as np


SCHEMA_VERSION = "inspect_samples.report.v1"


@dataclass(frozen=True, slots=True)
class InspectTarget:
    kind: str
    name: str
    samples_npz: Path
    metadata_json: Path | None
    family_dir: Path | None
    manifest_path: Path | None


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


def _counter_dict(values: np.ndarray | Sequence[int] | None) -> dict[str, int] | None:
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {}
    uniq, counts = np.unique(arr, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=True)}


def _histogram_of_counts(counts: np.ndarray) -> dict[str, int]:
    if counts.size == 0:
        return {}
    uniq, freq = np.unique(counts.astype(np.int64), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq.tolist(), freq.tolist(), strict=True)}


def _as_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _maybe_get_1d(arrays: dict[str, np.ndarray], key: str, *, dtype: Any | None = None) -> np.ndarray | None:
    if key not in arrays:
        return None
    out = _as_1d(arrays[key], name=key)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def _infer_time_index(detector_coordinates: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if detector_coordinates.ndim != 2 or detector_coordinates.shape[1] < 3:
        return None, None
    candidate = detector_coordinates[:, 2]
    rounded = np.rint(candidate)
    if np.allclose(candidate, rounded, atol=1e-6):
        return rounded.astype(np.int16, copy=False), "detector_coordinates[:, 2]"
    return None, None


def _safe_float(x: Any) -> float:
    return float(np.asarray(x).item())


def _resolve_manifest_family_dir(manifest_path: Path, raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    return manifest_path.parent / raw


def _resolve_targets(args: argparse.Namespace) -> list[InspectTarget]:
    targets: list[InspectTarget] = []

    for family_dir_raw in args.family_dir:
        family_dir = Path(family_dir_raw)
        samples_npz = family_dir / "samples.npz"
        metadata_json = family_dir / "metadata.json"
        if not samples_npz.exists():
            raise FileNotFoundError(f"Missing samples.npz in family_dir={family_dir}")
        targets.append(
            InspectTarget(
                kind="family_dir",
                name=family_dir.name,
                samples_npz=samples_npz,
                metadata_json=metadata_json if metadata_json.exists() else None,
                family_dir=family_dir,
                manifest_path=None,
            )
        )

    for manifest_raw in args.manifest:
        manifest_path = Path(manifest_raw)
        manifest = _read_json(manifest_path)
        family_dirs = manifest.get("family_dirs", {})
        if not isinstance(family_dirs, dict) or not family_dirs:
            raise ValueError(
                f"manifest does not contain a non-empty family_dirs mapping: {manifest_path}"
            )
        selected_families = args.family or sorted(family_dirs)
        for family in selected_families:
            if family not in family_dirs:
                raise KeyError(
                    f"Requested family {family!r} missing from manifest={manifest_path}. "
                    f"Available={sorted(family_dirs)}"
                )
            family_dir = _resolve_manifest_family_dir(manifest_path, family_dirs[family])
            samples_npz = family_dir / "samples.npz"
            metadata_json = family_dir / "metadata.json"
            if not samples_npz.exists():
                raise FileNotFoundError(
                    f"Missing samples.npz for family={family!r} resolved from manifest={manifest_path}: "
                    f"{samples_npz}"
                )
            targets.append(
                InspectTarget(
                    kind="manifest",
                    name=f"{manifest_path.stem}:{family}",
                    samples_npz=samples_npz,
                    metadata_json=metadata_json if metadata_json.exists() else None,
                    family_dir=family_dir,
                    manifest_path=manifest_path,
                )
            )

    if args.samples_npz is not None:
        samples_npz = Path(args.samples_npz)
        metadata_json = Path(args.metadata_json) if args.metadata_json else None
        label = args.label or samples_npz.stem
        targets.append(
            InspectTarget(
                kind="direct",
                name=label,
                samples_npz=samples_npz,
                metadata_json=metadata_json if metadata_json and metadata_json.exists() else None,
                family_dir=samples_npz.parent,
                manifest_path=None,
            )
        )

    if not targets:
        raise ValueError(
            "No input target provided. Use at least one of: --family-dir, --manifest, --samples-npz"
        )

    return targets


def _format_xy(values: np.ndarray) -> list[float]:
    out = []
    for v in values.tolist():
        fv = float(v)
        out.append(int(fv) if fv.is_integer() else fv)
    return out


def _collect_track_preview(
    *,
    unique_xy: np.ndarray,
    inverse_track: np.ndarray,
    detector_time_index: np.ndarray | None,
    detector_type: np.ndarray | None,
    checkerboard: np.ndarray | None,
    boundary: np.ndarray | None,
    final_round: np.ndarray | None,
    max_tracks: int,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    num_tracks = int(unique_xy.shape[0])
    for track_id in range(min(num_tracks, max_tracks)):
        idx = np.flatnonzero(inverse_track == track_id)
        entry: dict[str, Any] = {
            "track_id": track_id,
            "xy": _format_xy(unique_xy[track_id]),
            "detector_indices": idx.astype(int, copy=False).tolist(),
            "num_detectors": int(idx.size),
        }
        if detector_time_index is not None:
            entry["times"] = detector_time_index[idx].astype(int, copy=False).tolist()
        if detector_type is not None:
            entry["detector_type_values"] = sorted({int(v) for v in detector_type[idx].tolist()})
        if checkerboard is not None:
            entry["checkerboard_values"] = sorted({int(v) for v in checkerboard[idx].tolist()})
        if boundary is not None:
            entry["boundary_values"] = sorted({int(v) for v in boundary[idx].tolist()})
        if final_round is not None:
            entry["final_round_values"] = sorted({int(v) for v in final_round[idx].tolist()})
        previews.append(entry)
    return previews


def _make_model_hints(
    *,
    num_detectors: int,
    num_unique_xy: int | None,
    xy_repeat_histogram: dict[str, int] | None,
    logical_positive_rate: float,
    avg_detector_weight_per_shot: float,
    track_length_min: int | None,
    track_length_max: int | None,
) -> list[str]:
    hints: list[str] = []

    if num_unique_xy is not None and num_unique_xy < num_detectors:
        hints.append(
            "same-(x,y) detector coordinates repeat across time, so the data is better viewed as stabilizer tracks than as an independent flat feature vector"
        )
    if xy_repeat_histogram and len(xy_repeat_histogram) >= 2:
        hints.append(
            "track lengths are not uniform across all (x,y) locations, so any track tensor builder should support masking instead of assuming one fixed valid-length pattern"
        )
    if track_length_max is not None and track_length_min is not None and track_length_max > track_length_min:
        hints.append(
            "boundary and interior tracks likely have different temporal support; this is a strong reason to keep boundary/final-round metadata in the model input"
        )
    if num_detectors >= 128:
        hints.append(
            "the detector dimension is already large enough that a flat MLP baseline is likely to scale poorly compared with a temporal track encoder or other geometry-aware architecture"
        )
    if avg_detector_weight_per_shot > 4.0:
        hints.append(
            "shots activate multiple detectors on average, so per-shot structure is not purely ultra-sparse; temporal and spatial aggregation should help more than isolated per-detector scoring"
        )
    if 0.05 <= logical_positive_rate <= 0.35:
        hints.append(
            "the label distribution is imbalanced but not extreme, so binary classification is still workable without resorting immediately to anomaly-detection style objectives"
        )
    if not hints:
        hints.append(
            "the current dataset looks simple enough for a flat baseline, but keep the inspector results around before deciding against a geometry-aware model"
        )
    return hints


def inspect_target(target: InspectTarget, *, max_track_preview: int) -> dict[str, Any]:
    arrays_raw = np.load(target.samples_npz)
    arrays = {name: arrays_raw[name] for name in arrays_raw.files}

    metadata = _read_json(target.metadata_json) if target.metadata_json is not None else None

    detector_events = _as_2d(arrays["detector_events"], name="detector_events").astype(np.uint8, copy=False)
    observable_flips = _as_2d(arrays["observable_flips"], name="observable_flips").astype(np.uint8, copy=False)
    logical_label = _as_1d(arrays["logical_label"], name="logical_label").astype(np.uint8, copy=False)
    detector_coordinates = _as_2d(arrays["detector_coordinates"], name="detector_coordinates").astype(
        np.float32,
        copy=False,
    )

    num_shots, num_detectors = detector_events.shape
    if logical_label.shape[0] != num_shots:
        raise ValueError(
            f"logical_label length does not match detector_events shots: {logical_label.shape[0]} vs {num_shots}"
        )
    if detector_coordinates.shape[0] != num_detectors:
        raise ValueError(
            f"detector_coordinates detector dimension does not match detector_events: "
            f"{detector_coordinates.shape[0]} vs {num_detectors}"
        )

    detector_time_index = _maybe_get_1d(arrays, "detector_time_index", dtype=np.int16)
    time_index_source = "samples.npz:detector_time_index" if detector_time_index is not None else None
    if detector_time_index is None:
        detector_time_index, inferred_source = _infer_time_index(detector_coordinates)
        time_index_source = inferred_source

    detector_final_round_flag = _maybe_get_1d(arrays, "detector_final_round_flag", dtype=np.uint8)
    detector_boundary_flag = _maybe_get_1d(arrays, "detector_boundary_flag", dtype=np.uint8)
    detector_checkerboard_class = _maybe_get_1d(arrays, "detector_checkerboard_class", dtype=np.uint8)
    detector_type = _maybe_get_1d(arrays, "detector_type", dtype=np.uint8)

    xy = detector_coordinates[:, :2]
    unique_xy, inverse_track, track_lengths = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    logical_positive_rate = float(logical_label.mean()) if logical_label.size else 0.0
    avg_detector_weight_per_shot = float(detector_events.sum(axis=1).mean()) if num_shots else 0.0
    detector_event_fraction = float(detector_events.mean()) if detector_events.size else 0.0
    avg_track_activation_per_shot = None
    active_track_hist = None
    if unique_xy.shape[0] > 0:
        active_track = np.zeros((num_shots, unique_xy.shape[0]), dtype=np.uint8)
        for track_id in range(unique_xy.shape[0]):
            idx = inverse_track == track_id
            active_track[:, track_id] = detector_events[:, idx].any(axis=1).astype(np.uint8)
        avg_track_activation_per_shot = float(active_track.sum(axis=1).mean())
        active_track_hist = _counter_dict(active_track.sum(axis=1))

    time_hist = _counter_dict(detector_time_index) if detector_time_index is not None else None
    event_rate_per_time_index: dict[str, float] | None = None
    if detector_time_index is not None:
        event_rate_per_time_index = {}
        for t in sorted({int(v) for v in detector_time_index.tolist()}):
            idx = detector_time_index == t
            if idx.any():
                event_rate_per_time_index[str(t)] = float(detector_events[:, idx].mean())

    type_constant_tracks = None
    checker_constant_tracks = None
    boundary_constant_tracks = None
    contiguous_time_tracks = None
    final_round_singleton_tracks = None
    if unique_xy.shape[0] > 0:
        type_const = 0
        checker_const = 0
        boundary_const = 0
        contiguous_time = 0
        final_singleton = 0
        for track_id in range(unique_xy.shape[0]):
            idx = np.flatnonzero(inverse_track == track_id)
            if detector_type is not None and np.unique(detector_type[idx]).size == 1:
                type_const += 1
            if detector_checkerboard_class is not None and np.unique(detector_checkerboard_class[idx]).size == 1:
                checker_const += 1
            if detector_boundary_flag is not None and np.unique(detector_boundary_flag[idx]).size == 1:
                boundary_const += 1
            if detector_time_index is not None:
                times = np.sort(detector_time_index[idx].astype(np.int64))
                if times.size <= 1 or np.all(np.diff(times) == 1):
                    contiguous_time += 1
            if detector_final_round_flag is not None:
                if int(detector_final_round_flag[idx].sum()) <= 1:
                    final_singleton += 1
        type_constant_tracks = type_const if detector_type is not None else None
        checker_constant_tracks = checker_const if detector_checkerboard_class is not None else None
        boundary_constant_tracks = boundary_const if detector_boundary_flag is not None else None
        contiguous_time_tracks = contiguous_time if detector_time_index is not None else None
        final_round_singleton_tracks = final_singleton if detector_final_round_flag is not None else None

    family = metadata.get("family") if metadata else None
    stage = metadata.get("stage") if metadata else None
    circuit_meta = metadata.get("circuit", {}) if metadata else {}

    report = {
        "target": {
            "kind": target.kind,
            "name": target.name,
            "samples_npz": target.samples_npz,
            "metadata_json": target.metadata_json,
            "family_dir": target.family_dir,
            "manifest_path": target.manifest_path,
        },
        "dataset": {
            "family": family,
            "stage": stage,
            "distance": circuit_meta.get("distance"),
            "rounds": circuit_meta.get("rounds"),
            "basis": circuit_meta.get("basis"),
            "variant": circuit_meta.get("variant"),
            "num_shots": int(num_shots),
            "num_detectors": int(num_detectors),
            "num_observables": int(observable_flips.shape[1]),
            "array_keys": sorted(arrays.keys()),
        },
        "shapes": {
            "detector_events": list(detector_events.shape),
            "observable_flips": list(observable_flips.shape),
            "logical_label": list(logical_label.shape),
            "detector_coordinates": list(detector_coordinates.shape),
            "detector_time_index": list(detector_time_index.shape) if detector_time_index is not None else None,
            "detector_final_round_flag": list(detector_final_round_flag.shape) if detector_final_round_flag is not None else None,
            "detector_boundary_flag": list(detector_boundary_flag.shape) if detector_boundary_flag is not None else None,
            "detector_checkerboard_class": list(detector_checkerboard_class.shape) if detector_checkerboard_class is not None else None,
            "detector_type": list(detector_type.shape) if detector_type is not None else None,
        },
        "rates": {
            "logical_positive_rate": logical_positive_rate,
            "avg_detector_weight_per_shot": avg_detector_weight_per_shot,
            "detector_event_fraction": detector_event_fraction,
            "avg_track_activation_per_shot": avg_track_activation_per_shot,
        },
        "track_geometry": {
            "num_unique_xy": int(unique_xy.shape[0]),
            "xy_repeat_histogram": _histogram_of_counts(track_lengths),
            "track_length_min": int(track_lengths.min()) if track_lengths.size else None,
            "track_length_max": int(track_lengths.max()) if track_lengths.size else None,
            "track_length_mean": float(track_lengths.mean()) if track_lengths.size else None,
            "time_index_source": time_index_source,
            "time_index_range": [int(detector_time_index.min()), int(detector_time_index.max())] if detector_time_index is not None and detector_time_index.size else None,
            "time_index_histogram": time_hist,
            "event_rate_per_time_index": event_rate_per_time_index,
            "active_track_count_per_shot_histogram": active_track_hist,
        },
        "semantics": {
            "detector_type_counts": _counter_dict(detector_type),
            "checkerboard_class_counts": _counter_dict(detector_checkerboard_class),
            "boundary_flag_counts": _counter_dict(detector_boundary_flag),
            "final_round_flag_counts": _counter_dict(detector_final_round_flag),
        },
        "consistency_checks": {
            "tracks_with_constant_detector_type": type_constant_tracks,
            "tracks_with_constant_checkerboard_class": checker_constant_tracks,
            "tracks_with_constant_boundary_flag": boundary_constant_tracks,
            "tracks_with_contiguous_time_index": contiguous_time_tracks,
            "tracks_with_at_most_one_final_round_flag": final_round_singleton_tracks,
            "num_tracks": int(unique_xy.shape[0]),
        },
        "track_preview": _collect_track_preview(
            unique_xy=unique_xy,
            inverse_track=inverse_track,
            detector_time_index=detector_time_index,
            detector_type=detector_type,
            checkerboard=detector_checkerboard_class,
            boundary=detector_boundary_flag,
            final_round=detector_final_round_flag,
            max_tracks=max_track_preview,
        ),
        "model_hints": _make_model_hints(
            num_detectors=int(num_detectors),
            num_unique_xy=int(unique_xy.shape[0]),
            xy_repeat_histogram=_histogram_of_counts(track_lengths),
            logical_positive_rate=logical_positive_rate,
            avg_detector_weight_per_shot=avg_detector_weight_per_shot,
            track_length_min=int(track_lengths.min()) if track_lengths.size else None,
            track_length_max=int(track_lengths.max()) if track_lengths.size else None,
        ),
    }
    return report


def _fmt_dict_compact(d: dict[str, Any] | None) -> str:
    if d is None:
        return "n/a"
    if not d:
        return "{}"
    return "{" + ", ".join(f"{k}:{v}" for k, v in d.items()) + "}"


def print_case_report(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    rates = report["rates"]
    geom = report["track_geometry"]
    sem = report["semantics"]
    checks = report["consistency_checks"]

    header = report["target"]["name"]
    print("=" * 100)
    print(header)
    print("-" * 100)
    print(
        f"family={dataset.get('family')} stage={dataset.get('stage')} "
        f"d={dataset.get('distance')} r={dataset.get('rounds')} "
        f"basis={dataset.get('basis')} variant={dataset.get('variant')}"
    )
    print(
        f"shots={dataset['num_shots']} detectors={dataset['num_detectors']} "
        f"observables={dataset['num_observables']}"
    )
    print(
        f"logical_positive_rate={rates['logical_positive_rate']:.6f} "
        f"avg_detector_weight_per_shot={rates['avg_detector_weight_per_shot']:.4f} "
        f"detector_event_fraction={rates['detector_event_fraction']:.6f}"
    )
    if rates.get("avg_track_activation_per_shot") is not None:
        print(f"avg_active_tracks_per_shot={rates['avg_track_activation_per_shot']:.4f}")
    print(
        f"num_unique_xy={geom['num_unique_xy']} "
        f"xy_repeat_histogram={_fmt_dict_compact(geom['xy_repeat_histogram'])} "
        f"time_index_range={geom['time_index_range']}"
    )
    print(f"time_index_histogram={_fmt_dict_compact(geom['time_index_histogram'])}")
    print(f"event_rate_per_time_index={_fmt_dict_compact(geom['event_rate_per_time_index'])}")
    print(f"active_track_count_per_shot_histogram={_fmt_dict_compact(geom['active_track_count_per_shot_histogram'])}")
    print(f"detector_type_counts={_fmt_dict_compact(sem['detector_type_counts'])}")
    print(f"checkerboard_class_counts={_fmt_dict_compact(sem['checkerboard_class_counts'])}")
    print(f"boundary_flag_counts={_fmt_dict_compact(sem['boundary_flag_counts'])}")
    print(f"final_round_flag_counts={_fmt_dict_compact(sem['final_round_flag_counts'])}")
    print(
        "consistency_checks="
        f"constant_type={checks['tracks_with_constant_detector_type']}/{checks['num_tracks']} "
        f"constant_checker={checks['tracks_with_constant_checkerboard_class']}/{checks['num_tracks']} "
        f"constant_boundary={checks['tracks_with_constant_boundary_flag']}/{checks['num_tracks']} "
        f"contiguous_time={checks['tracks_with_contiguous_time_index']}/{checks['num_tracks']}"
    )
    preview = report.get("track_preview", [])
    if preview:
        print("track_preview:")
        for entry in preview:
            extra_bits = []
            if "times" in entry:
                extra_bits.append(f"times={entry['times']}")
            if "detector_type_values" in entry:
                extra_bits.append(f"type={entry['detector_type_values']}")
            if "boundary_values" in entry:
                extra_bits.append(f"boundary={entry['boundary_values']}")
            print(
                f"  track_id={entry['track_id']:>2} xy={entry['xy']} "
                f"n={entry['num_detectors']} " + " ".join(extra_bits)
            )
    print("model_hints:")
    for hint in report.get("model_hints", []):
        print(f"  - {hint}")
    print()


def print_summary_table(reports: list[dict[str, Any]]) -> None:
    if not reports:
        return
    print("=" * 100)
    print("summary")
    print("-" * 100)
    header = (
        f"{'name':<24} {'family':<16} {'d':>3} {'r':>3} {'shots':>8} {'det':>6} "
        f"{'xy':>5} {'pos_rate':>10} {'avg_wt':>10} {'repeat_hist':<18}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        dataset = report["dataset"]
        rates = report["rates"]
        geom = report["track_geometry"]
        name = str(report["target"]["name"])[:24]
        family = str(dataset.get("family") or "-")[:16]
        repeat_hist = _fmt_dict_compact(geom.get("xy_repeat_histogram"))[:18]
        d = dataset.get("distance")
        r = dataset.get("rounds")
        print(
            f"{name:<24} {family:<16} {str(d):>3} {str(r):>3} "
            f"{dataset['num_shots']:>8} {dataset['num_detectors']:>6} {geom['num_unique_xy']:>5} "
            f"{rates['logical_positive_rate']:>10.4f} {rates['avg_detector_weight_per_shot']:>10.4f} {repeat_hist:<18}"
        )
    print()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect samples.npz geometry and detector metadata for QEC decoder design."
    )
    parser.add_argument(
        "--family-dir",
        action="append",
        default=[],
        help="Family directory containing samples.npz and optionally metadata.json. Repeatable.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest.json produced by sample_dataset.py. Repeatable.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Restrict manifest inspection to specific family names. Repeatable.",
    )
    parser.add_argument(
        "--samples-npz",
        help="Direct path to samples.npz for standalone inspection.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional metadata.json paired with --samples-npz.",
    )
    parser.add_argument(
        "--label",
        help="Optional label used with --samples-npz mode.",
    )
    parser.add_argument(
        "--max-track-preview",
        type=int,
        default=12,
        help="Maximum number of track previews to print/store per target.",
    )
    parser.add_argument(
        "--out-json",
        help="Optional path to save the structured report JSON.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    reports = [inspect_target(target, max_track_preview=args.max_track_preview) for target in _resolve_targets(args)]

    for report in reports:
        print_case_report(report)
    if len(reports) > 1:
        print_summary_table(reports)

    if args.out_json:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "num_reports": len(reports),
            "reports": reports,
        }
        _write_json(Path(args.out_json), payload)


if __name__ == "__main__":
    main()
