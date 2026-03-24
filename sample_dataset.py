from __future__ import annotations

"""
sample_dataset.py

Dataset generator aligned with the current qec_decoder codebase.

Supported families
------------------
- ideal            : ideal circuit from circuits.py
- stage_a_si1000   : Stage A from noise_si1000.py
- stage_b_local    : Stage B from noise_willowcore.py
- stage_c_corr     : Stage C from noise_willowcore.py

Design rules
------------
1. Use the current project APIs directly. No builder-name guessing.
2. Treat the current practical scaffold as `stim_rotated` by default.
3. Allow `variant="xzzx"`, but record explicitly that it still reuses the
   current Stim rotated scaffold and is not yet a real Willow schedule.
4. Save the minimum fields needed by generate/train/eval:
   detector events, observable flips, logical label, detector coordinates, metadata.
5. Also save circuit and DEM artifacts for reproducibility/debugging.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import datetime as dt
import hashlib
import json
import shutil

import numpy as np

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from config import (
    CircuitConfig,
    ExperimentConfig,
    make_stage_a_config,
    make_stage_b_config,
    make_stage_c_config,
)
from circuits import (
    MissingStimError as CircuitsMissingStimError,
    build_memory_circuit,
    export_dataset_metadata,
    summarize_circuit,
)
from noise_si1000 import (
    build_si1000_memory_circuit,
    export_noisy_metadata as export_stage_a_metadata,
    summarize_si1000_circuit,
)
from noise_willowcore import (
    build_willowcore_memory_circuit,
    export_noisy_metadata as export_willow_metadata,
    summarize_willowcore_circuit,
)


SCHEMA_VERSION = "sample_dataset.v5"
SUPPORTED_FAMILIES = (
    "ideal",
    "stage_a_si1000",
    "stage_b_local",
    "stage_c_corr",
)


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    stage: str
    description: str


FAMILY_SPECS: dict[str, FamilySpec] = {
    "ideal": FamilySpec(
        family="ideal",
        stage="ideal",
        description="Ideal surface-code memory circuit on the current scaffold",
    ),
    "stage_a_si1000": FamilySpec(
        family="stage_a_si1000",
        stage="A",
        description="Uniform SI1000-Base noisy circuit",
    ),
    "stage_b_local": FamilySpec(
        family="stage_b_local",
        stage="B",
        description="Stage B Willow-inspired local heterogeneity",
    ),
    "stage_c_corr": FamilySpec(
        family="stage_c_corr",
        stage="C",
        description="Stage C Willow-inspired correlated stray surrogate",
    ),
}


@dataclass(frozen=True, slots=True)
class BuiltFamily:
    family: str
    stage: str
    spec: FamilySpec
    circuit: Any
    cfg: CircuitConfig | ExperimentConfig
    upstream_metadata: dict[str, Any]
    upstream_summary: dict[str, Any] | None
    dem_kwargs: dict[str, Any]


def _require_stim() -> Any:
    if stim is None:
        raise CircuitsMissingStimError(
            "Stim is required but not installed in this Python environment."
        )
    return stim


def _utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=_json_default),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    """
    Shape-aware, dtype-aware array hash for reproducibility.

    This avoids accidental collisions between arrays that share raw bytes but
    differ in semantic structure, such as shape=(N, 1) versus shape=(N,).
    """
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(str(tuple(int(x) for x in arr.shape)).encode("utf-8"))
    h.update(b"|")
    h.update(arr.tobytes())
    return h.hexdigest()


def _instruction_histogram(circuit: Any) -> dict[str, int]:
    counter: dict[str, int] = {}

    def walk(block: Any, repeat_mult: int = 1) -> None:
        for op in block:
            if op.__class__.__name__ == "CircuitRepeatBlock":
                walk(op.body_copy(), repeat_mult * int(op.repeat_count))
                continue
            name = str(op.name)
            counter[name] = counter.get(name, 0) + repeat_mult

    walk(circuit)
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def _is_real_willow_schedule(variant: str) -> bool:
    # Current project status: even variant="xzzx" still reuses the Stim rotated scaffold.
    return False


def _schedule_source(variant: str) -> str:
    if variant == "xzzx":
        return "xzzx_api_reuses_stim_rotated_scaffold"
    return "stim_rotated"


def _family_output_dir(
    out_root: Path,
    *,
    family: str,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
) -> Path:
    tag = f"{family}__d{distance}_r{rounds}_{basis}_{variant}"
    return out_root / tag


def _path_for_manifest(path: Path, out_root: Path) -> str:
    return path.relative_to(out_root).as_posix()


def _build_ideal(
    *,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
) -> BuiltFamily:
    spec = FAMILY_SPECS["ideal"]
    cfg = CircuitConfig(distance=distance, rounds=rounds, basis=basis, variant=variant)
    circuit = build_memory_circuit(cfg)
    upstream_metadata = export_dataset_metadata(cfg, circuit)
    upstream_summary = summarize_circuit(circuit, cfg).to_dict()
    return BuiltFamily(
        family="ideal",
        stage=spec.stage,
        spec=spec,
        circuit=circuit,
        cfg=cfg,
        upstream_metadata=upstream_metadata,
        upstream_summary=upstream_summary,
        dem_kwargs={"decompose_errors": True},
    )


def _build_stage_a(
    *,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
    physical_error_rate: float,
    shots: int,
) -> BuiltFamily:
    spec = FAMILY_SPECS["stage_a_si1000"]
    cfg = make_stage_a_config(
        distance=distance,
        rounds=rounds,
        basis=basis,
        variant=variant,
        physical_error_rate=physical_error_rate,
        shots=shots,
    )
    circuit = build_si1000_memory_circuit(cfg)
    upstream_metadata = export_stage_a_metadata(cfg, circuit)
    upstream_summary = summarize_si1000_circuit(cfg, circuit).to_dict()
    return BuiltFamily(
        family="stage_a_si1000",
        stage=spec.stage,
        spec=spec,
        circuit=circuit,
        cfg=cfg,
        upstream_metadata=upstream_metadata,
        upstream_summary=upstream_summary,
        dem_kwargs={"decompose_errors": True},
    )


def _build_stage_b_or_c(
    *,
    family: str,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
    physical_error_rate: float,
    shots: int,
) -> BuiltFamily:
    spec = FAMILY_SPECS[family]
    if family == "stage_b_local":
        cfg = make_stage_b_config(
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
            physical_error_rate=physical_error_rate,
            shots=shots,
        )
    elif family == "stage_c_corr":
        cfg = make_stage_c_config(
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
            physical_error_rate=physical_error_rate,
            shots=shots,
        )
    else:
        raise ValueError(f"Unsupported family for Willow builder: {family!r}")

    circuit = build_willowcore_memory_circuit(cfg)
    upstream_metadata = export_willow_metadata(cfg, circuit)
    upstream_summary = summarize_willowcore_circuit(cfg, circuit).to_dict()

    dem_kwargs: dict[str, Any] = {"decompose_errors": True}
    if cfg.willow.corr.enabled and cfg.willow.corr.enable_swap_component:
        dem_kwargs["approximate_disjoint_errors"] = True

    return BuiltFamily(
        family=family,
        stage=spec.stage,
        spec=spec,
        circuit=circuit,
        cfg=cfg,
        upstream_metadata=upstream_metadata,
        upstream_summary=upstream_summary,
        dem_kwargs=dem_kwargs,
    )


def build_family(
    *,
    family: str,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
    physical_error_rate: float,
    shots: int,
) -> BuiltFamily:
    if family == "ideal":
        return _build_ideal(
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
        )
    if family == "stage_a_si1000":
        return _build_stage_a(
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
            physical_error_rate=physical_error_rate,
            shots=shots,
        )
    if family in {"stage_b_local", "stage_c_corr"}:
        return _build_stage_b_or_c(
            family=family,
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
            physical_error_rate=physical_error_rate,
            shots=shots,
        )
    raise ValueError(f"Unknown family: {family!r}")


def _sample_arrays(
    circuit: Any,
    *,
    shots: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    return det_events, obs_flips


def _derive_logical_label(obs_flips: np.ndarray) -> np.ndarray:
    if obs_flips.ndim != 2:
        raise ValueError(f"observable_flips must be rank-2, got shape={obs_flips.shape}")
    if obs_flips.shape[1] != 1:
        raise ValueError(
            "logical_label derivation currently requires exactly one observable. "
            f"Got observable_flips.shape={obs_flips.shape}"
        )
    return obs_flips[:, 0].astype(np.uint8, copy=False)


def save_family_dataset(
    *,
    out_root: Path,
    family: str,
    distance: int,
    rounds: int,
    basis: str,
    variant: str,
    physical_error_rate: float,
    shots: int,
    seed: int | None,
    overwrite: bool,
) -> Path:
    built = build_family(
        family=family,
        distance=distance,
        rounds=rounds,
        basis=basis,
        variant=variant,
        physical_error_rate=physical_error_rate,
        shots=shots,
    )

    family_dir = _family_output_dir(
        out_root,
        family=family,
        distance=distance,
        rounds=rounds,
        basis=basis,
        variant=variant,
    )
    if family_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {family_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True, exist_ok=False)

    circuit = built.circuit
    det_events, obs_flips = _sample_arrays(circuit, shots=shots, seed=seed)
    logical_label = _derive_logical_label(obs_flips)
    detector_coords = np.asarray(
        built.upstream_metadata["detector_coordinates"],
        dtype=np.float32,
    )

    dem = circuit.detector_error_model(**built.dem_kwargs)
    circuit_text = str(circuit)
    dem_text = str(dem)

    det_events_u8 = det_events.astype(np.uint8, copy=False)
    obs_flips_u8 = obs_flips.astype(np.uint8, copy=False)
    logical_label_u8 = logical_label.astype(np.uint8, copy=False)
    detector_coords_f32 = detector_coords.astype(np.float32, copy=False)

    npz_path = family_dir / "samples.npz"
    circuit_path = family_dir / "circuit.stim"
    dem_path = family_dir / "detector_error_model.dem"
    metadata_path = family_dir / "metadata.json"
    upstream_metadata_path = family_dir / "upstream_metadata.json"
    config_path = family_dir / "config.json"

    np.savez_compressed(
        npz_path,
        detector_events=det_events_u8,
        observable_flips=obs_flips_u8,
        logical_label=logical_label_u8,
        detector_coordinates=detector_coords_f32,
    )
    circuit_path.write_text(circuit_text, encoding="utf-8")
    dem_path.write_text(dem_text, encoding="utf-8")
    _write_json(upstream_metadata_path, built.upstream_metadata)

    cfg_obj = built.cfg
    if isinstance(cfg_obj, ExperimentConfig):
        config_obj = cfg_obj.to_dict()
    else:
        config_obj = asdict(cfg_obj)
    _write_json(config_path, {"config": config_obj})

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "family": built.family,
        "stage": built.stage,
        "description": built.spec.description,
        "scaffold": {
            "variant": variant,
            "schedule_source": _schedule_source(variant),
            "is_real_willow_schedule": _is_real_willow_schedule(variant),
        },
        "circuit": {
            "distance": distance,
            "rounds": rounds,
            "basis": basis,
            "variant": variant,
            "num_qubits": int(circuit.num_qubits),
            "num_measurements": int(circuit.num_measurements),
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
        },
        "sampling": {
            "shots": int(shots),
            "seed": seed,
            "stored_detector_events_dtype": "uint8",
            "stored_observable_flips_dtype": "uint8",
            "stored_logical_label_dtype": "uint8",
            "stored_detector_coordinates_dtype": "float32",
            "stored_detector_events_shape": list(det_events_u8.shape),
            "stored_observable_flips_shape": list(obs_flips_u8.shape),
            "stored_logical_label_shape": list(logical_label_u8.shape),
            "stored_detector_coordinates_shape": list(detector_coords_f32.shape),
            "logical_label_definition": "observable_flips[:, 0] when num_observables == 1",
        },
        "qc_stats": {
            "detector_event_fraction": float(det_events_u8.mean()),
            "logical_flip_fraction": float(logical_label_u8.mean()),
            "observable_flip_fraction_per_observable": [
                float(obs_flips_u8[:, k].mean()) for k in range(obs_flips_u8.shape[1])
            ],
            "avg_detector_weight_per_shot": float(det_events_u8.sum(axis=1).mean()),
            "avg_observable_weight_per_shot": float(obs_flips_u8.sum(axis=1).mean()),
        },
        "dem": {
            "kwargs": built.dem_kwargs,
            "num_detectors": int(dem.num_detectors),
            "num_observables": int(dem.num_observables),
        },
        "instruction_histogram": _instruction_histogram(circuit),
        "artifacts": {
            "samples_npz": npz_path.name,
            "circuit_stim": circuit_path.name,
            "detector_error_model_dem": dem_path.name,
            "metadata_json": metadata_path.name,
            "upstream_metadata_json": upstream_metadata_path.name,
            "config_json": config_path.name,
        },
        "hashes": {
            "circuit_sha256": _sha256_text(circuit_text),
            "dem_sha256": _sha256_text(dem_text),
            "detector_events_sha256": _sha256_array(det_events_u8),
            "observable_flips_sha256": _sha256_array(obs_flips_u8),
            "logical_label_sha256": _sha256_array(logical_label_u8),
            "detector_coordinates_sha256": _sha256_array(detector_coords_f32),
        },
        "upstream_summary": built.upstream_summary,
        "generator": {
            "script": "sample_dataset.py",
            "created_at_utc": _utc_now_iso(),
            "stim_version": getattr(stim, "__version__", "unknown"),
            "numpy_version": np.__version__,
        },
    }
    _write_json(metadata_path, metadata)
    return family_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detector-event datasets for ideal / Stage A / Stage B / Stage C."
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--basis", choices=["x", "z"], required=True)
    parser.add_argument("--variant", choices=["stim_rotated", "xzzx"], default="stim_rotated")
    parser.add_argument("--p", type=float, default=0.0015, help="Base physical error rate for SI1000-derived families")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(SUPPORTED_FAMILIES),
        choices=list(SUPPORTED_FAMILIES),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    _require_stim()
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "manifest_format": {
        "family_dirs_base": "manifest_parent",
        "path_style": "posix_relative",
    },
    "created_at_utc": _utc_now_iso(),
    "distance": args.distance,
    "rounds": args.rounds,
    "basis": args.basis,
    "variant": args.variant,
    "shots": args.shots,
    "requested_families": list(args.families),
    "family_dirs": {},
    }

    for family in args.families:
        family_dir = save_family_dataset(
            out_root=args.out_root,
            family=family,
            distance=args.distance,
            rounds=args.rounds,
            basis=args.basis,
            variant=args.variant,
            physical_error_rate=args.p,
            shots=args.shots,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        manifest["family_dirs"][family] = _path_for_manifest(family_dir, args.out_root)

    _write_json(args.out_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
