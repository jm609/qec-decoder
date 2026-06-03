from __future__ import annotations

"""
build_pymatching_edit_targets.py

Build derived supervision targets for an Ising-style syndrome-edit pre-decoder.

The tool consumes existing class4 datasets produced by sample_dataset.py and
derives, per shot:

- whether baseline PyMatching is already correct
- whether the shot appears to need an edit
- a bounded local detector-bit edit mask that makes PyMatching correct, when
  such a mask is found within the configured search budget

The output is a new artifact layer that keeps the original family directories as
the source of truth and stores only the derived target arrays and summary
metadata needed for later pre-decoder training.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import datetime as dt
import itertools
import json
import math
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoders.baseline_pymatching import (
    _build_matching,
    _decode_batch,
    _load_family_payload,
)


SCHEMA_VERSION = "pymatching_edit_targets.v1"
MANIFEST_SCHEMA_VERSION = "pymatching_edit_targets.manifest.v1"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_edit_weight: int
    time_radius: int
    space_radius: int
    max_candidates: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShotSearchResult:
    solved: bool
    edit_weight: int
    candidate_count: int
    variants_tested: int
    edit_indices: tuple[int, ...]


def _utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def _as_uint8_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape={out.shape}")
    return np.ascontiguousarray(out)


def _as_int16_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int16).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1 after reshape, got shape={out.shape}")
    return np.ascontiguousarray(out)


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


def _path_for_manifest(path: Path, out_root: Path) -> str:
    return path.relative_to(out_root).as_posix()


def _family_output_dir(out_root: Path, family_dir: Path) -> Path:
    return out_root / family_dir.name


def _ordered_candidate_indices(
    shot_events: np.ndarray,
    *,
    detector_time_index: np.ndarray,
    rect_row_index_by_detector: np.ndarray,
    rect_col_index_by_detector: np.ndarray,
    rect_detector_index_volume: np.ndarray,
    search_config: SearchConfig,
) -> list[int]:
    active = np.flatnonzero(np.asarray(shot_events, dtype=np.uint8))
    if active.size == 0:
        return []

    candidate_priority: dict[int, tuple[int, int, int, int, int]] = {}

    for det_idx in active.tolist():
        t0 = int(detector_time_index[det_idx])
        r0 = int(rect_row_index_by_detector[det_idx])
        c0 = int(rect_col_index_by_detector[det_idx])

        current = candidate_priority.get(det_idx)
        active_priority = (0, 0, 0, 0, det_idx)
        if current is None or active_priority < current:
            candidate_priority[det_idx] = active_priority

        t_lo = max(0, t0 - search_config.time_radius)
        t_hi = min(rect_detector_index_volume.shape[0] - 1, t0 + search_config.time_radius)
        r_lo = max(0, r0 - search_config.space_radius)
        r_hi = min(rect_detector_index_volume.shape[1] - 1, r0 + search_config.space_radius)
        c_lo = max(0, c0 - search_config.space_radius)
        c_hi = min(rect_detector_index_volume.shape[2] - 1, c0 + search_config.space_radius)

        for t in range(t_lo, t_hi + 1):
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    neighbor_idx = int(rect_detector_index_volume[t, r, c])
                    if neighbor_idx < 0:
                        continue
                    dt = abs(t - t0)
                    dr = abs(r - r0)
                    dc = abs(c - c0)
                    dist = dt + dr + dc
                    priority = (1, dist, dt, dr + dc, neighbor_idx)
                    current = candidate_priority.get(neighbor_idx)
                    if current is None or priority < current:
                        candidate_priority[neighbor_idx] = priority

    ordered = sorted(candidate_priority.items(), key=lambda kv: kv[1])
    if search_config.max_candidates > 0:
        ordered = ordered[: search_config.max_candidates]
    return [idx for idx, _priority in ordered]


def _search_local_edit_mask(
    matching: Any,
    shot_events: np.ndarray,
    target_observables: np.ndarray,
    *,
    detector_time_index: np.ndarray,
    rect_row_index_by_detector: np.ndarray,
    rect_col_index_by_detector: np.ndarray,
    rect_detector_index_volume: np.ndarray,
    search_config: SearchConfig,
) -> ShotSearchResult:
    candidates = _ordered_candidate_indices(
        shot_events,
        detector_time_index=detector_time_index,
        rect_row_index_by_detector=rect_row_index_by_detector,
        rect_col_index_by_detector=rect_col_index_by_detector,
        rect_detector_index_volume=rect_detector_index_volume,
        search_config=search_config,
    )
    variants_tested = 0
    if not candidates:
        return ShotSearchResult(
            solved=False,
            edit_weight=-1,
            candidate_count=0,
            variants_tested=0,
            edit_indices=(),
        )

    target = _as_uint8_2d(target_observables, name="target_observables")[0]
    base_shot = np.asarray(shot_events, dtype=np.uint8).reshape(1, -1)

    max_weight = min(search_config.max_edit_weight, len(candidates))
    for edit_weight in range(1, max_weight + 1):
        combinations = list(itertools.combinations(candidates, edit_weight))
        if not combinations:
            continue
        variants_tested += len(combinations)
        edited_shots = np.repeat(base_shot, len(combinations), axis=0)
        for row_idx, edit_indices in enumerate(combinations):
            edited_shots[row_idx, list(edit_indices)] ^= np.uint8(1)
        predicted = _decode_batch(matching, edited_shots)
        solved_rows = np.flatnonzero(np.all(predicted == target[None, :], axis=1))
        if solved_rows.size:
            best_indices = tuple(int(x) for x in combinations[int(solved_rows[0])])
            return ShotSearchResult(
                solved=True,
                edit_weight=edit_weight,
                candidate_count=len(candidates),
                variants_tested=variants_tested,
                edit_indices=best_indices,
            )

    return ShotSearchResult(
        solved=False,
        edit_weight=-1,
        candidate_count=len(candidates),
        variants_tested=variants_tested,
        edit_indices=(),
    )


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


def _build_family_targets(
    family_dir: Path,
    *,
    out_root: Path,
    search_config: SearchConfig,
    max_shots: int | None,
    allow_circuit_fallback: bool,
) -> dict[str, Any]:
    artifacts, metadata, arrays = _load_family_payload(family_dir)
    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")
    logical_class4 = (
        _as_uint8_2d(arrays["logical_class4"], name="logical_class4").reshape(-1)
        if "logical_class4" in arrays
        else None
    )
    detector_time_index = _as_int16_1d(arrays["detector_time_index"], name="detector_time_index")
    rect_row_index_by_detector = _as_int16_1d(
        arrays["rect_row_index_by_detector"],
        name="rect_row_index_by_detector",
    )
    rect_col_index_by_detector = _as_int16_1d(
        arrays["rect_col_index_by_detector"],
        name="rect_col_index_by_detector",
    )
    rect_detector_index_volume = np.asarray(arrays["rect_detector_index_volume"], dtype=np.int32)
    if rect_detector_index_volume.ndim != 3:
        raise ValueError(
            f"rect_detector_index_volume must be rank-3, got shape={rect_detector_index_volume.shape}"
        )

    if max_shots is not None:
        detector_events = detector_events[:max_shots]
        observable_flips = observable_flips[:max_shots]
        if logical_class4 is not None:
            logical_class4 = logical_class4[:max_shots]

    matching, matching_info = _build_matching(
        dem_path=artifacts.dem_path,
        circuit_path=artifacts.circuit_path,
        allow_circuit_fallback=allow_circuit_fallback,
    )
    baseline_predicted_observables = _decode_batch(matching, detector_events)
    baseline_correct = np.all(
        baseline_predicted_observables == observable_flips,
        axis=1,
    ).astype(np.uint8, copy=False)

    num_shots = int(detector_events.shape[0])
    num_detectors = int(detector_events.shape[1])
    detector_edit_target_mask = np.zeros((num_shots, num_detectors), dtype=np.uint8)
    needs_edit = (1 - baseline_correct).astype(np.uint8, copy=False)
    edit_target_known = baseline_correct.copy()
    solved_by_local_edit = np.zeros(num_shots, dtype=np.uint8)
    found_edit_weight = np.full(num_shots, -1, dtype=np.int16)
    found_edit_weight[baseline_correct.astype(bool)] = np.int16(0)
    search_candidate_count = np.zeros(num_shots, dtype=np.int16)
    search_variants_tested = np.zeros(num_shots, dtype=np.int32)

    t0 = time.perf_counter()
    incorrect_indices = np.flatnonzero(baseline_correct == 0)
    for shot_idx in incorrect_indices.tolist():
        result = _search_local_edit_mask(
            matching,
            detector_events[shot_idx],
            observable_flips[shot_idx : shot_idx + 1],
            detector_time_index=detector_time_index,
            rect_row_index_by_detector=rect_row_index_by_detector,
            rect_col_index_by_detector=rect_col_index_by_detector,
            rect_detector_index_volume=rect_detector_index_volume,
            search_config=search_config,
        )
        search_candidate_count[shot_idx] = np.int16(result.candidate_count)
        search_variants_tested[shot_idx] = np.int32(result.variants_tested)
        if not result.solved:
            continue
        solved_by_local_edit[shot_idx] = np.uint8(1)
        edit_target_known[shot_idx] = np.uint8(1)
        found_edit_weight[shot_idx] = np.int16(result.edit_weight)
        detector_edit_target_mask[shot_idx, list(result.edit_indices)] = np.uint8(1)
    search_wall_seconds = time.perf_counter() - t0

    oracle_correct = np.logical_or(
        baseline_correct.astype(bool),
        solved_by_local_edit.astype(bool),
    ).astype(np.uint8, copy=False)
    unsolved_by_local_edit = np.logical_and(
        baseline_correct == 0,
        solved_by_local_edit == 0,
    ).astype(np.uint8, copy=False)

    family_out_dir = _family_output_dir(out_root, artifacts.family_dir)
    family_out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = family_out_dir / "edit_targets.npz"
    np.savez_compressed(
        npz_path,
        detector_edit_target_mask=detector_edit_target_mask,
        baseline_pymatching_correct=baseline_correct,
        needs_edit=needs_edit,
        edit_target_known=edit_target_known,
        solved_by_local_edit=solved_by_local_edit,
        unsolved_by_local_edit=unsolved_by_local_edit,
        found_edit_weight=found_edit_weight,
        search_candidate_count=search_candidate_count,
        search_variants_tested=search_variants_tested,
        baseline_predicted_observables=baseline_predicted_observables.astype(np.uint8, copy=False),
        oracle_pymatching_correct=oracle_correct,
    )

    oracle_accuracy = float(oracle_correct.mean())
    baseline_accuracy = float(baseline_correct.mean())
    num_incorrect = int((baseline_correct == 0).sum())
    num_solved = int(solved_by_local_edit.sum())
    solved_edit_weights = found_edit_weight[solved_by_local_edit.astype(bool)]
    logical_class4_accuracy: float | None = None
    baseline_logical_class4_accuracy: float | None = None
    if logical_class4 is not None and observable_flips.shape[1] == 2:
        baseline_predicted_class4 = _logical_class4_from_observable_flips(baseline_predicted_observables)
        baseline_logical_class4_accuracy = float(np.mean(baseline_predicted_class4 == logical_class4))
        oracle_logical_class4 = np.logical_or(
            baseline_predicted_class4 == logical_class4,
            solved_by_local_edit.astype(bool),
        )
        logical_class4_accuracy = float(np.mean(oracle_logical_class4))

    metadata_out = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "family": metadata.get("family", artifacts.family_dir.name),
        "stage": metadata.get("stage"),
        "source_family_dir": artifacts.family_dir.as_posix(),
        "source_artifacts": {
            "samples_npz": artifacts.samples_npz.as_posix(),
            "metadata_json": artifacts.metadata_json.as_posix(),
            "detector_error_model_dem": artifacts.dem_path.as_posix(),
            "circuit_stim": artifacts.circuit_path.as_posix() if artifacts.circuit_path is not None else None,
        },
        "matching": matching_info.to_dict(),
        "search_config": search_config.to_dict(),
        "dataset": {
            "shots_processed": num_shots,
            "num_detectors": num_detectors,
            "num_observables": int(observable_flips.shape[1]),
            "distance": metadata.get("circuit", {}).get("distance"),
            "rounds": metadata.get("circuit", {}).get("rounds"),
            "basis": metadata.get("circuit", {}).get("basis"),
            "variant": metadata.get("circuit", {}).get("variant"),
            "target_mode": metadata.get("targets", {}).get("target_mode"),
        },
        "oracle_stats": {
            "baseline_pymatching_accuracy": baseline_accuracy,
            "baseline_pymatching_logical_class4_accuracy": baseline_logical_class4_accuracy,
            "oracle_pymatching_accuracy_after_edit_targets": oracle_accuracy,
            "oracle_pymatching_logical_class4_accuracy_after_edit_targets": logical_class4_accuracy,
            "num_incorrect_baseline_shots": num_incorrect,
            "num_solved_by_local_edit": num_solved,
            "num_unsolved_by_local_edit": int(unsolved_by_local_edit.sum()),
            "fraction_of_incorrect_solved": float(num_solved / num_incorrect) if num_incorrect else None,
            "mean_edit_weight_over_solved_hard_shots": (
                float(np.mean(solved_edit_weights)) if solved_edit_weights.size else None
            ),
            "max_edit_weight_over_solved_hard_shots": (
                int(np.max(solved_edit_weights)) if solved_edit_weights.size else None
            ),
            "mean_search_candidates_over_incorrect_shots": (
                float(search_candidate_count[incorrect_indices].mean())
                if incorrect_indices.size
                else None
            ),
            "mean_variants_tested_over_incorrect_shots": (
                float(search_variants_tested[incorrect_indices].mean())
                if incorrect_indices.size
                else None
            ),
            "search_wall_seconds": search_wall_seconds,
        },
        "edit_weight_histogram": {
            str(int(weight)): int(count)
            for weight, count in zip(
                *np.unique(found_edit_weight[found_edit_weight >= 0], return_counts=True),
                strict=False,
            )
        },
        "stored_arrays": {
            "edit_targets_npz": npz_path.name,
            "detector_edit_target_mask_shape": list(detector_edit_target_mask.shape),
            "baseline_pymatching_correct_shape": list(baseline_correct.shape),
            "needs_edit_shape": list(needs_edit.shape),
            "edit_target_known_shape": list(edit_target_known.shape),
            "solved_by_local_edit_shape": list(solved_by_local_edit.shape),
            "unsolved_by_local_edit_shape": list(unsolved_by_local_edit.shape),
            "found_edit_weight_shape": list(found_edit_weight.shape),
            "search_candidate_count_shape": list(search_candidate_count.shape),
            "search_variants_tested_shape": list(search_variants_tested.shape),
        },
    }
    metadata_path = family_out_dir / "metadata.json"
    _write_json(metadata_path, metadata_out)

    return {
        "family": metadata_out["family"],
        "family_dir": family_out_dir,
        "metadata_path": metadata_path,
        "summary": metadata_out["oracle_stats"],
    }


def build_from_manifest(
    manifest_path: Path,
    *,
    out_root: Path,
    families: Iterable[str] | None,
    search_config: SearchConfig,
    max_shots: int | None,
    allow_circuit_fallback: bool,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    family_dirs = manifest.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(f"Manifest has no family_dirs: {manifest_path}")

    selected_families = list(families) if families is not None else list(family_dirs.keys())
    results: dict[str, dict[str, Any]] = {}
    built_family_dirs: dict[str, str] = {}
    for family in selected_families:
        if family not in family_dirs:
            raise KeyError(f"Family '{family}' not found in manifest {manifest_path}")
        family_dir = _resolve_manifest_family_dir(manifest_path, family_dirs[family])
        built = _build_family_targets(
            family_dir,
            out_root=out_root,
            search_config=search_config,
            max_shots=max_shots,
            allow_circuit_fallback=allow_circuit_fallback,
        )
        results[family] = built["summary"]
        built_family_dirs[family] = _path_for_manifest(built["family_dir"], out_root)

    output_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "source_manifest": manifest_path.as_posix(),
        "search_config": search_config.to_dict(),
        "max_shots": max_shots,
        "family_dirs": built_family_dirs,
        "oracle_stats_by_family": results,
    }
    _write_json(out_root / "manifest.json", output_manifest)
    return output_manifest


def build_from_family_dir(
    family_dir: Path,
    *,
    out_root: Path,
    search_config: SearchConfig,
    max_shots: int | None,
    allow_circuit_fallback: bool,
) -> dict[str, Any]:
    built = _build_family_targets(
        family_dir,
        out_root=out_root,
        search_config=search_config,
        max_shots=max_shots,
        allow_circuit_fallback=allow_circuit_fallback,
    )
    output_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "source_family_dir": family_dir.as_posix(),
        "search_config": search_config.to_dict(),
        "max_shots": max_shots,
        "family_dirs": {
            str(built["family"]): _path_for_manifest(built["family_dir"], out_root),
        },
        "oracle_stats_by_family": {
            str(built["family"]): built["summary"],
        },
    }
    _write_json(out_root / "manifest.json", output_manifest)
    return output_manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build derived PyMatching edit-target artifacts for the first syndrome-edit "
            "pre-decoder branch."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="sample_dataset manifest.json to process")
    source.add_argument("--family-dir", type=Path, help="single dataset family directory to process")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for derived target artifacts")
    parser.add_argument("--families", nargs="*", default=None, help="Optional subset of manifest families")
    parser.add_argument("--max-shots", type=int, default=None, help="Optional shot limit per family")
    parser.add_argument("--max-edit-weight", type=int, default=2, help="Maximum detector toggle weight to search")
    parser.add_argument("--time-radius", type=int, default=1, help="Temporal neighborhood radius around active detectors")
    parser.add_argument("--space-radius", type=int, default=1, help="Spatial neighborhood radius around active detectors")
    parser.add_argument("--max-candidates", type=int, default=24, help="Maximum detector candidates per hard shot")
    parser.add_argument(
        "--allow-circuit-fallback",
        action="store_true",
        help="Allow PyMatching graph construction from circuit.stim when DEM-based construction is unavailable",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.max_edit_weight < 1:
        raise ValueError("--max-edit-weight must be >= 1")
    if args.time_radius < 0 or args.space_radius < 0:
        raise ValueError("--time-radius and --space-radius must be >= 0")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")

    search_config = SearchConfig(
        max_edit_weight=int(args.max_edit_weight),
        time_radius=int(args.time_radius),
        space_radius=int(args.space_radius),
        max_candidates=int(args.max_candidates),
    )
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        result = build_from_manifest(
            args.manifest,
            out_root=args.out_root,
            families=args.families,
            search_config=search_config,
            max_shots=args.max_shots,
            allow_circuit_fallback=bool(args.allow_circuit_fallback),
        )
    else:
        result = build_from_family_dir(
            args.family_dir,
            out_root=args.out_root,
            search_config=search_config,
            max_shots=args.max_shots,
            allow_circuit_fallback=bool(args.allow_circuit_fallback),
        )

    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "manifest_path": str((args.out_root / "manifest.json").resolve()),
                "families": list(result["family_dirs"].keys()),
                "search_config": result["search_config"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
