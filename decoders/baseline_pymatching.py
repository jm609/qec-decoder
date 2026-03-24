from __future__ import annotations

"""
baseline_pymatching.py

PyMatching baseline decoder + minimal evaluation pipeline for datasets produced by
sample_dataset.py.

Supported inputs
----------------
1. A single family directory containing:
   - samples.npz
   - detector_error_model.dem
   - metadata.json
   - optionally circuit.stim
2. A manifest.json produced by sample_dataset.py, which maps family names to
   family directories.

Design choices
--------------
- Use detector_error_model.dem first.
- Use samples.npz detector_events and logical_label / observable_flips.
- Support both single-family and manifest-wide evaluation.
- Save structured JSON results.
- Keep imports of optional runtime dependencies (pymatching, stim) lazy enough
  that this file can still be syntax-checked in environments where those
  packages are absent.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import datetime as dt
import json
import math
import time

import numpy as np

try:
    import pymatching  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    pymatching = None

try:
    import stim  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    stim = None


SCHEMA_VERSION = "baseline_pymatching.eval.v1"


class MissingPyMatchingError(ImportError):
    """Raised when PyMatching-dependent functionality is used without PyMatching."""


@dataclass(frozen=True, slots=True)
class FamilyArtifacts:
    family_dir: Path
    samples_npz: Path
    dem_path: Path
    metadata_json: Path
    circuit_path: Path | None


@dataclass(frozen=True, slots=True)
class MatchingBuildInfo:
    source: str
    builder: str
    allow_circuit_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FamilyEvaluationResult:
    schema_version: str
    decoder: str
    created_at_utc: str
    input_mode: str
    family: str
    stage: str
    family_dir: str
    matching: dict[str, Any]
    dataset: dict[str, Any]
    metrics: dict[str, Any]
    timing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


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


def _require_pymatching() -> Any:
    if pymatching is None:
        raise MissingPyMatchingError(
            "PyMatching is required for baseline decoding but is not installed in this "
            "Python environment. Install it with `pip install pymatching` in the same "
            "environment where Stim is available."
        )
    return pymatching


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _estimate_ler_per_cycle(frame_error_rate: float, rounds: int | None) -> float | None:
    """
    Estimate logical error per cycle from the mismatch rate measured on fixed-length
    memory experiments.

    Using the standard relation for a shot length n:
        LER = 0.5 * (1 - (1 - 2E(n)) ** (1/n))

    This is only meaningful when rounds is known and 0 <= E(n) < 0.5.
    """
    if rounds is None or rounds < 1:
        return None
    if not (0.0 <= frame_error_rate < 0.5):
        return None
    fidelity = 1.0 - 2.0 * frame_error_rate
    if fidelity < 0.0:
        return None
    return 0.5 * (1.0 - fidelity ** (1.0 / rounds))


def _normalise_predictions(pred: np.ndarray, *, num_shots: int) -> np.ndarray:
    out = np.asarray(pred, dtype=np.uint8)
    if out.ndim == 1:
        if out.shape[0] == num_shots:
            out = out.reshape(num_shots, 1)
        else:
            out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(f"Predictions must be rank-1 or rank-2, got shape={out.shape}")
    if out.shape[0] != num_shots:
        raise ValueError(
            f"Prediction shot dimension mismatch: got {out.shape[0]}, expected {num_shots}"
        )
    return np.ascontiguousarray(out)


def _resolve_family_artifacts(family_dir: str | Path) -> FamilyArtifacts:
    family_dir = Path(family_dir)
    samples_npz = family_dir / "samples.npz"
    dem_path = family_dir / "detector_error_model.dem"
    metadata_json = family_dir / "metadata.json"
    circuit_path = family_dir / "circuit.stim"

    missing = [
        p.name
        for p in (samples_npz, dem_path, metadata_json)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Family directory is missing required artifacts: {missing}. family_dir={family_dir}"
        )

    return FamilyArtifacts(
        family_dir=family_dir,
        samples_npz=samples_npz,
        dem_path=dem_path,
        metadata_json=metadata_json,
        circuit_path=circuit_path if circuit_path.exists() else None,
    )


def _load_family_payload(family_dir: str | Path) -> tuple[FamilyArtifacts, dict[str, Any], dict[str, np.ndarray]]:
    artifacts = _resolve_family_artifacts(family_dir)
    metadata = _read_json(artifacts.metadata_json)
    with np.load(artifacts.samples_npz) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return artifacts, metadata, arrays


# ---------------------------------------------------------------------------
# Matching graph construction and decoding
# ---------------------------------------------------------------------------


def _build_matching(
    *,
    dem_path: Path,
    circuit_path: Path | None,
    allow_circuit_fallback: bool,
) -> tuple[Any, MatchingBuildInfo]:
    pm = _require_pymatching()
    Matching = pm.Matching

    # DEM-first path (preferred and default).
    if dem_path.exists():
        from_dem_file = getattr(Matching, "from_detector_error_model_file", None)
        if callable(from_dem_file):
            return (
                from_dem_file(str(dem_path)),
                MatchingBuildInfo(
                    source="detector_error_model.dem",
                    builder="Matching.from_detector_error_model_file",
                    allow_circuit_fallback=allow_circuit_fallback,
                ),
            )

        from_dem = getattr(Matching, "from_detector_error_model", None)
        if callable(from_dem):
            if stim is None:
                raise RuntimeError(
                    "PyMatching is available but does not expose "
                    "Matching.from_detector_error_model_file, and Stim is not installed for "
                    "the fallback path via stim.DetectorErrorModel.from_file."
                )
            dem = stim.DetectorErrorModel.from_file(str(dem_path))
            return (
                from_dem(dem),
                MatchingBuildInfo(
                    source="detector_error_model.dem",
                    builder="Matching.from_detector_error_model",
                    allow_circuit_fallback=allow_circuit_fallback,
                ),
            )

    # Optional circuit fallback if DEM path is missing or unsupported.
    if allow_circuit_fallback and circuit_path is not None and circuit_path.exists():
        from_circuit_file = getattr(Matching, "from_stim_circuit_file", None)
        if callable(from_circuit_file):
            return (
                from_circuit_file(str(circuit_path)),
                MatchingBuildInfo(
                    source="circuit.stim",
                    builder="Matching.from_stim_circuit_file",
                    allow_circuit_fallback=allow_circuit_fallback,
                ),
            )

        from_circuit = getattr(Matching, "from_stim_circuit", None)
        if callable(from_circuit):
            if stim is None:
                raise RuntimeError(
                    "PyMatching is available but does not expose Matching.from_stim_circuit_file, "
                    "and Stim is not installed for the fallback path via stim.Circuit.from_file."
                )
            circuit = stim.Circuit.from_file(str(circuit_path))
            return (
                from_circuit(circuit),
                MatchingBuildInfo(
                    source="circuit.stim",
                    builder="Matching.from_stim_circuit",
                    allow_circuit_fallback=allow_circuit_fallback,
                ),
            )

    raise RuntimeError(
        "Failed to build a PyMatching graph. Expected detector_error_model.dem-first construction, "
        "or explicit --allow-circuit-fallback with a valid circuit.stim artifact."
    )


def _decode_batch(matching: Any, detector_events: np.ndarray) -> np.ndarray:
    shots = _as_uint8_2d(detector_events, name="detector_events")

    decode_batch = getattr(matching, "decode_batch", None)
    if callable(decode_batch):
        pred = decode_batch(shots)
        return _normalise_predictions(pred, num_shots=shots.shape[0])

    decode = getattr(matching, "decode", None)
    if not callable(decode):
        raise RuntimeError(
            "The PyMatching object exposes neither decode_batch nor decode."
        )

    pred_rows = [np.asarray(decode(shots[k]), dtype=np.uint8) for k in range(shots.shape[0])]
    pred = np.asarray(pred_rows, dtype=np.uint8)
    return _normalise_predictions(pred, num_shots=shots.shape[0])


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


def evaluate_family_dir(
    family_dir: str | Path,
    *,
    max_shots: int | None = None,
    allow_circuit_fallback: bool = False,
) -> dict[str, Any]:
    artifacts, metadata, arrays = _load_family_payload(family_dir)

    detector_events = _as_uint8_2d(arrays["detector_events"], name="detector_events")
    observable_flips = _as_uint8_2d(arrays["observable_flips"], name="observable_flips")
    logical_label = _as_uint8_1d(arrays["logical_label"], name="logical_label")

    if detector_events.shape[0] != observable_flips.shape[0]:
        raise ValueError(
            "detector_events and observable_flips have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {observable_flips.shape[0]}"
        )
    if detector_events.shape[0] != logical_label.shape[0]:
        raise ValueError(
            "detector_events and logical_label have inconsistent shot counts: "
            f"{detector_events.shape[0]} vs {logical_label.shape[0]}"
        )
    if observable_flips.shape[1] == 1:
        if not np.array_equal(observable_flips[:, 0], logical_label):
            raise ValueError(
                "logical_label does not match observable_flips[:, 0] for a single-observable dataset."
            )

    if max_shots is not None:
        if max_shots < 1:
            raise ValueError("max_shots must be >= 1 when provided")
        detector_events = detector_events[:max_shots]
        observable_flips = observable_flips[:max_shots]
        logical_label = logical_label[:max_shots]

    matching, matching_info = _build_matching(
        dem_path=artifacts.dem_path,
        circuit_path=artifacts.circuit_path,
        allow_circuit_fallback=allow_circuit_fallback,
    )

    t0 = time.perf_counter()
    predicted_observables = _decode_batch(matching, detector_events)
    decode_wall_seconds = time.perf_counter() - t0

    if predicted_observables.shape != observable_flips.shape:
        raise ValueError(
            "Predicted observable shape does not match target observable shape: "
            f"predicted={predicted_observables.shape}, target={observable_flips.shape}"
        )

    shot_error_mask = np.any(predicted_observables != observable_flips, axis=1)
    num_shots = int(detector_events.shape[0])
    num_shot_errors = int(shot_error_mask.sum())
    frame_error_rate = float(num_shot_errors / num_shots)
    accuracy = float(1.0 - frame_error_rate)

    pred_logical_label: np.ndarray | None = None
    label_error_rate: float | None = None
    confusion: dict[str, int] | None = None
    if predicted_observables.shape[1] == 1:
        pred_logical_label = predicted_observables[:, 0].astype(np.uint8, copy=False)
        label_error_rate = float(np.mean(pred_logical_label != logical_label))
        tn = int(np.sum((pred_logical_label == 0) & (logical_label == 0)))
        fp = int(np.sum((pred_logical_label == 1) & (logical_label == 0)))
        fn = int(np.sum((pred_logical_label == 0) & (logical_label == 1)))
        tp = int(np.sum((pred_logical_label == 1) & (logical_label == 1)))
        confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    rounds = metadata.get("circuit", {}).get("rounds")
    estimated_ler_per_cycle = _estimate_ler_per_cycle(frame_error_rate, int(rounds) if rounds is not None else None)

    dataset_info = {
        "family": metadata.get("family"),
        "stage": metadata.get("stage"),
        "shots_evaluated": num_shots,
        "num_detectors": int(detector_events.shape[1]),
        "num_observables": int(observable_flips.shape[1]),
        "rounds": rounds,
        "distance": metadata.get("circuit", {}).get("distance"),
        "basis": metadata.get("circuit", {}).get("basis"),
        "variant": metadata.get("circuit", {}).get("variant"),
        "metadata_json": artifacts.metadata_json.as_posix(),
        "samples_npz": artifacts.samples_npz.as_posix(),
        "detector_error_model_dem": artifacts.dem_path.as_posix(),
        "circuit_stim": artifacts.circuit_path.as_posix() if artifacts.circuit_path is not None else None,
        "qc_stats_from_dataset": metadata.get("qc_stats", {}),
    }

    metrics = {
        "num_shot_errors": num_shot_errors,
        "frame_error_rate": frame_error_rate,
        "accuracy": accuracy,
        "label_error_rate": label_error_rate,
        "estimated_ler_per_cycle": estimated_ler_per_cycle,
        "mean_detector_weight_per_shot": float(detector_events.sum(axis=1).mean()),
        "mean_target_observable_weight_per_shot": float(observable_flips.sum(axis=1).mean()),
        "mean_predicted_observable_weight_per_shot": float(predicted_observables.sum(axis=1).mean()),
        "confusion_matrix_logical_label": confusion,
    }

    timing = {
        "decode_wall_seconds": decode_wall_seconds,
        "shots_per_second": float(num_shots / decode_wall_seconds) if decode_wall_seconds > 0 else None,
    }

    result = FamilyEvaluationResult(
        schema_version=SCHEMA_VERSION,
        decoder="pymatching",
        created_at_utc=_utc_now_iso(),
        input_mode="family_dir",
        family=str(metadata.get("family", artifacts.family_dir.name)),
        stage=str(metadata.get("stage", "unknown")),
        family_dir=artifacts.family_dir.as_posix(),
        matching=matching_info.to_dict(),
        dataset=dataset_info,
        metrics=metrics,
        timing=timing,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Manifest evaluation
# ---------------------------------------------------------------------------


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


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    families: Iterable[str] | None = None,
    max_shots: int | None = None,
    allow_circuit_fallback: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    family_dirs = manifest.get("family_dirs", {})
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(
            f"manifest.json does not contain a non-empty family_dirs mapping: {manifest_path}"
        )

    requested_families = list(families) if families is not None else list(family_dirs)
    missing_families = [family for family in requested_families if family not in family_dirs]
    if missing_families:
        raise KeyError(
            f"Requested families are missing from manifest: {missing_families}. "
            f"Available families: {sorted(family_dirs)}"
        )

    by_family: dict[str, dict[str, Any]] = {}
    for family in requested_families:
        family_dir = _resolve_manifest_family_dir(manifest_path, family_dirs[family])
        by_family[family] = evaluate_family_dir(
            family_dir,
            max_shots=max_shots,
            allow_circuit_fallback=allow_circuit_fallback,
        )

    ranking = sorted(
        (
            {
                "family": family,
                "frame_error_rate": float(result["metrics"]["frame_error_rate"]),
                "estimated_ler_per_cycle": result["metrics"].get("estimated_ler_per_cycle"),
            }
            for family, result in by_family.items()
        ),
        key=lambda item: (item["frame_error_rate"], item["family"]),
    )

    ideal_fer = None
    if "ideal" in by_family:
        ideal_fer = float(by_family["ideal"]["metrics"]["frame_error_rate"])
        for family, result in by_family.items():
            result["metrics"]["delta_frame_error_rate_vs_ideal"] = (
                float(result["metrics"]["frame_error_rate"]) - ideal_fer
            )

    summary = {
        "families_evaluated": requested_families,
        "best_family_by_frame_error_rate": ranking[0]["family"] if ranking else None,
        "ideal_frame_error_rate": ideal_fer,
        "ranking_by_frame_error_rate": ranking,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "decoder": "pymatching",
        "created_at_utc": _utc_now_iso(),
        "input_mode": "manifest",
        "manifest_path": manifest_path.as_posix(),
        "manifest_summary": {
            "distance": manifest.get("distance"),
            "rounds": manifest.get("rounds"),
            "basis": manifest.get("basis"),
            "variant": manifest.get("variant"),
            "shots": manifest.get("shots"),
            "requested_families": manifest.get("requested_families"),
        },
        "summary": summary,
        "by_family": by_family,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sample_dataset.py outputs with a PyMatching baseline decoder. "
            "Supports a single family directory or a manifest.json covering multiple families."
        )
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--family-dir", type=Path, help="Path to one family directory")
    src.add_argument("--manifest", type=Path, help="Path to manifest.json")

    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset of families to evaluate when --manifest is used",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Optionally evaluate only the first N shots for quick tests",
    )
    parser.add_argument(
        "--allow-circuit-fallback",
        action="store_true",
        help=(
            "Permit circuit.stim fallback only when detector_error_model.dem cannot be used. "
            "Default behavior is DEM-first without fallback."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation result as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.family_dir is not None:
        result = evaluate_family_dir(
            args.family_dir,
            max_shots=args.max_shots,
            allow_circuit_fallback=args.allow_circuit_fallback,
        )
    else:
        result = evaluate_manifest(
            args.manifest,
            families=args.families,
            max_shots=args.max_shots,
            allow_circuit_fallback=args.allow_circuit_fallback,
        )

    if args.out_json is not None:
        _write_json(args.out_json, result)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
