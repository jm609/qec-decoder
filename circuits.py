from __future__ import annotations

"""
circuits.py

Ideal surface-code memory circuit builder.

Responsibilities
----------------
1. Build an ideal memory circuit from CircuitConfig / ExperimentConfig.
2. Provide stable detector / observable sampling helpers.
3. Export detector coordinates and dataset metadata.
4. Run a small smoke test before any noise module is added.

Non-responsibilities
--------------------
- No SI1000 noise here.
- No WillowCore effects here.
- No dataset file saving logic here.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import warnings

import numpy as np

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from config import CircuitConfig, ExperimentConfig


class MissingStimError(ImportError):
    """Raised when Stim-dependent functionality is used without Stim installed."""


DETECTOR_TYPE_VOCAB = {
    0: "unknown",
    1: "x_check",
    2: "z_check",
}

VALIDATED_STIM_ROTATED_CHECKERBOARD_TYPE_MAP = {
    0: 2,  # z_check
    1: 1,  # x_check
}

def get_validated_checkerboard_type_map(
    cfg: CircuitConfig | ExperimentConfig,
) -> dict[int, int]:
    """
    Return the locally validated checkerboard->detector_type mapping
    for the current practical scaffold.

    Current validation status:
      - stim_rotated: validated
      - xzzx: currently reuses the same stim_rotated scaffold, so the same
        structural mapping is used for now

    IMPORTANT:
      If the future xzzx path is replaced with a real Willow-native schedule,
      this mapping must be revalidated.
    """
    circuit_cfg, _ = _resolve_experiment_config(cfg)

    if circuit_cfg.variant in {"stim_rotated", "xzzx"}:
        return dict(VALIDATED_STIM_ROTATED_CHECKERBOARD_TYPE_MAP)

    raise ValueError(
        f"No validated checkerboard_type_map for variant={circuit_cfg.variant!r}"
    )

def _require_stim() -> Any:
    if stim is None:
        raise MissingStimError(
            "Stim is required but not installed in this Python environment."
        )
    return stim


def _resolve_experiment_config(
    cfg: CircuitConfig | ExperimentConfig,
) -> tuple[CircuitConfig, ExperimentConfig | None]:
    if isinstance(cfg, ExperimentConfig):
        cfg.validate()
        return cfg.circuit, cfg
    if isinstance(cfg, CircuitConfig):
        cfg.validate()
        return cfg, None
    raise TypeError(
        "Expected CircuitConfig or ExperimentConfig, "
        f"got {type(cfg).__name__}"
    )


def _stim_task_for_basis(basis: str) -> str:
    if basis == "x":
        return "surface_code:rotated_memory_x"
    if basis == "z":
        return "surface_code:rotated_memory_z"
    raise ValueError(f"Unsupported basis: {basis!r}")


def _build_stim_rotated_memory(circuit_cfg: CircuitConfig) -> Any:
    stim_mod = _require_stim()
    return stim_mod.Circuit.generated(
        _stim_task_for_basis(circuit_cfg.basis),
        distance=circuit_cfg.distance,
        rounds=circuit_cfg.rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )


def _build_xzzx_seed_memory(circuit_cfg: CircuitConfig) -> Any:
    """
    Stage-0 future-facing XZZX entry point.

    For now, this intentionally reuses Stim's rotated memory scaffold.
    The API is already separated so that later we can replace only this
    function with an explicit Willow-style CZ-native XZZX schedule.
    """
    warnings.warn(
        "variant='xzzx' currently reuses the ideal Stim rotated memory seed circuit. "
        "Explicit Willow-style XZZX scheduling is not implemented yet.",
        stacklevel=2,
    )
    return _build_stim_rotated_memory(circuit_cfg)


def build_memory_circuit(cfg: CircuitConfig | ExperimentConfig) -> Any:
    """
    Public entry point for ideal memory-circuit generation.
    """
    circuit_cfg, _ = _resolve_experiment_config(cfg)

    if circuit_cfg.variant == "stim_rotated":
        return _build_stim_rotated_memory(circuit_cfg)
    if circuit_cfg.variant == "xzzx":
        return _build_xzzx_seed_memory(circuit_cfg)

    raise AssertionError(f"Unreachable variant: {circuit_cfg.variant!r}")


@dataclass(frozen=True, slots=True)
class CircuitSummary:
    distance: int
    rounds: int
    basis: str
    variant: str
    num_qubits: int
    num_measurements: int
    num_detectors: int
    num_observables: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class CircuitSmokeTestReport:
    summary: CircuitSummary
    dem_num_detectors: int
    dem_num_observables: int
    detector_sample_shape: tuple[int, int]
    detector_sample_dtype: str
    observable_sample_shape: tuple[int, int]
    observable_sample_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def summarize_circuit(
    circuit: Any,
    cfg: CircuitConfig | ExperimentConfig,
) -> CircuitSummary:
    circuit_cfg, _ = _resolve_experiment_config(cfg)
    return CircuitSummary(
        distance=circuit_cfg.distance,
        rounds=circuit_cfg.rounds,
        basis=circuit_cfg.basis,
        variant=circuit_cfg.variant,
        num_qubits=int(circuit.num_qubits),
        num_measurements=int(circuit.num_measurements),
        num_detectors=int(circuit.num_detectors),
        num_observables=int(circuit.num_observables),
    )


def get_detector_coordinates(circuit: Any) -> list[list[float]]:
    """
    Return detector coordinates ordered by detector index.

    Stim typically uses coordinates like (x, y, t).
    """
    coord_map = circuit.get_detector_coordinates()
    coords: list[list[float]] = []
    for det_idx in range(int(circuit.num_detectors)):
        coord = coord_map.get(det_idx, [])
        coords.append([float(v) for v in coord])
    return coords


def get_detector_coordinate_array(
    circuit: Any,
    *,
    coord_dim: int = 3,
    pad_value: float = np.nan,
) -> np.ndarray:
    """
    Return detector coordinates as a fixed-shape float32 array.

    Default shape is (num_detectors, 3), corresponding to the common
    Stim coordinate convention (x, y, t). Missing coordinates are padded
    with NaN.
    """
    raw = get_detector_coordinates(circuit)
    arr = np.full((len(raw), coord_dim), pad_value, dtype=np.float32)

    for i, coord in enumerate(raw):
        n = min(len(coord), coord_dim)
        if n > 0:
            arr[i, :n] = np.asarray(coord[:n], dtype=np.float32)

    return arr


def infer_stim_rotated_detector_semantics(
    detector_coords: np.ndarray,
    *,
    checkerboard_type_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """
    Infer detector-level semantic metadata for the current stim_rotated scaffold.

    Safe fields:
      - detector_time_index
      - detector_final_round_flag
      - detector_boundary_flag
      - detector_checkerboard_class

    detector_type is left as 'unknown' unless a validated
    checkerboard_type_map is explicitly provided.
    """
    coords = np.asarray(detector_coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(
            f"detector_coords must have shape (N, >=3), got {coords.shape}"
        )

    x = coords[:, 0]
    y = coords[:, 1]
    t = coords[:, 2]

    if np.isnan(t).any():
        raise ValueError(
            "stim_rotated semantic inference requires valid detector time coordinates"
        )

    detector_time_index = np.rint(t).astype(np.int16)

    max_t = int(np.max(detector_time_index))
    detector_final_round_flag = (detector_time_index == max_t).astype(np.uint8)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))

    detector_boundary_flag = (
        np.isclose(x, x_min)
        | np.isclose(x, x_max)
        | np.isclose(y, y_min)
        | np.isclose(y, y_max)
    ).astype(np.uint8)

    # Safe structural class for the current scaffold.
    #
    # Stim rotated-memory detector coordinates on this scaffold commonly live on
    # an even lattice such as x,y in {0, 2, 4, 6, ...}. In that case, using
    # round(2*x), round(2*y) collapses the parity and can incorrectly assign all
    # detectors to the same checkerboard class.
    #
    # We therefore infer the checkerboard class from the integer lattice itself.
    # For the current scaffold, x and y are expected to be integer-valued
    # coordinates (often even integers). Dividing (x+y) by 2 before taking parity
    # preserves the alternating plaquette structure on the observed lattice.
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError(
            "stim_rotated semantic inference requires valid detector x/y coordinates"
        )

    xi = np.rint(x).astype(np.int16)
    yi = np.rint(y).astype(np.int16)

    if not np.allclose(x, xi) or not np.allclose(y, yi):
        raise ValueError(
            "stim_rotated semantic inference currently expects integer-valued "
            "detector x/y coordinates"
        )

    detector_checkerboard_class = (((xi + yi) // 2) & 1).astype(np.uint8)

    detector_type = np.zeros_like(detector_checkerboard_class, dtype=np.uint8)
    semantic_source = "stim_rotated_checkerboard_only"

    if checkerboard_type_map is not None:
        for cls, type_id in checkerboard_type_map.items():
            detector_type[detector_checkerboard_class == int(cls)] = np.uint8(type_id)
        semantic_source = "stim_rotated_type_map_validated"

    return {
        "detector_time_index": detector_time_index,
        "detector_final_round_flag": detector_final_round_flag,
        "detector_boundary_flag": detector_boundary_flag,
        "detector_checkerboard_class": detector_checkerboard_class,
        "detector_type": detector_type,
        "detector_type_vocab": DETECTOR_TYPE_VOCAB,
        "semantic_source": semantic_source,
    }


def export_detector_semantics(
    cfg: CircuitConfig | ExperimentConfig,
    circuit: Any,
    *,
    checkerboard_type_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """
    Export detector semantic metadata for the current scaffold.

    For now:
      - stim_rotated: structural semantics inferred directly
      - xzzx: still reuses the current stim_rotated scaffold, so we keep the
        same structural inference but mark the source explicitly
    """
    circuit_cfg, _ = _resolve_experiment_config(cfg)
    coords = get_detector_coordinate_array(circuit, coord_dim=3)

    out = infer_stim_rotated_detector_semantics(
        coords,
        checkerboard_type_map=checkerboard_type_map,
    )

    if circuit_cfg.variant == "xzzx":
        out["semantic_source"] = "xzzx_api_reuses_stim_rotated_scaffold"

    return out


def sample_detection_events_and_observables(
    circuit: Any,
    *,
    shots: int,
    separate_observables: bool = True,
) -> Any:
    """
    Sample detector events and logical observable flips.

    Returns:
        - (det_events, obs_flips) if separate_observables=True
        - combined array otherwise
    """
    if shots < 1:
        raise ValueError("shots must be >= 1")

    sampler = circuit.compile_detector_sampler()

    if separate_observables:
        det_events, obs_flips = sampler.sample(
            shots=shots,
            separate_observables=True,
        )
        return det_events, obs_flips

    return sampler.sample(
        shots=shots,
        separate_observables=False,
    )


def export_dataset_metadata(
    cfg: CircuitConfig | ExperimentConfig,
    circuit: Any,
) -> dict[str, Any]:
    circuit_cfg, experiment_cfg = _resolve_experiment_config(cfg)

    metadata: dict[str, Any] = {
        "distance": circuit_cfg.distance,
        "rounds": circuit_cfg.rounds,
        "basis": circuit_cfg.basis,
        "variant": circuit_cfg.variant,
        "num_qubits": int(circuit.num_qubits),
        "num_measurements": int(circuit.num_measurements),
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
        "detector_coordinates": get_detector_coordinates(circuit),
    }

    if experiment_cfg is not None:
        metadata["experiment_name"] = experiment_cfg.name
        metadata["experiment_tag"] = experiment_cfg.experiment_tag
        metadata["noise_stage"] = experiment_cfg.noise_stage
        metadata["noise_version"] = experiment_cfg.noise_version

    return metadata


def smoke_test_circuit(
    cfg: CircuitConfig | ExperimentConfig,
    *,
    sample_shots: int = 4,
) -> CircuitSmokeTestReport:
    """
    Minimum runtime checks before any noise module is added.

    Checks:
        1. circuit generation succeeds
        2. detector sampler compiles
        3. detector samples can actually be drawn
        4. logical observables can be sampled
        5. DEM generation succeeds
        6. detector / observable counts are consistent
    """
    if sample_shots < 1:
        raise ValueError("sample_shots must be >= 1")

    circuit = build_memory_circuit(cfg)
    summary = summarize_circuit(circuit, cfg)

    det_events, obs_flips = sample_detection_events_and_observables(
        circuit,
        shots=sample_shots,
        separate_observables=True,
    )

    dem = circuit.detector_error_model(decompose_errors=True)

    if summary.num_detectors < 1:
        raise RuntimeError("expected at least one detector")
    if summary.num_observables < 1:
        raise RuntimeError("expected at least one logical observable")

    det_shape = tuple(int(x) for x in det_events.shape)
    obs_shape = tuple(int(x) for x in obs_flips.shape)

    if det_shape != (sample_shots, summary.num_detectors):
        raise RuntimeError(
            f"unexpected detector sample shape: got {det_shape}, "
            f"expected {(sample_shots, summary.num_detectors)}"
        )
    if obs_shape != (sample_shots, summary.num_observables):
        raise RuntimeError(
            f"unexpected observable sample shape: got {obs_shape}, "
            f"expected {(sample_shots, summary.num_observables)}"
        )

    if int(dem.num_detectors) != summary.num_detectors:
        raise RuntimeError(
            f"DEM detector mismatch: dem={int(dem.num_detectors)} "
            f"summary={summary.num_detectors}"
        )
    if int(dem.num_observables) != summary.num_observables:
        raise RuntimeError(
            f"DEM observable mismatch: dem={int(dem.num_observables)} "
            f"summary={summary.num_observables}"
        )

    return CircuitSmokeTestReport(
        summary=summary,
        dem_num_detectors=int(dem.num_detectors),
        dem_num_observables=int(dem.num_observables),
        detector_sample_shape=det_shape,
        detector_sample_dtype=str(det_events.dtype),
        observable_sample_shape=obs_shape,
        observable_sample_dtype=str(obs_flips.dtype),
    )


def build_and_summarize(
    cfg: CircuitConfig | ExperimentConfig,
) -> tuple[Any, CircuitSummary]:
    circuit = build_memory_circuit(cfg)
    summary = summarize_circuit(circuit, cfg)
    return circuit, summary


def write_circuit_text(circuit: Any, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(str(circuit), encoding="utf-8")
    return out


def write_json_text(text: str, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test an ideal surface-code memory circuit."
    )
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=["x", "z"], default="z")
    parser.add_argument("--variant", choices=["stim_rotated", "xzzx"], default="stim_rotated")
    parser.add_argument("--sample-shots", type=int, default=4)
    parser.add_argument("--out-circuit", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-metadata", type=str, default=None)
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def _build_cfg_from_args(args: argparse.Namespace) -> CircuitConfig | ExperimentConfig:
    if args.config_json is not None:
        return ExperimentConfig.load_json(args.config_json)

    return CircuitConfig(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        variant=args.variant,
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg_from_args(args)

    circuit = build_memory_circuit(cfg)

    if args.out_circuit is not None:
        write_circuit_text(circuit, args.out_circuit)

    if args.out_metadata is not None:
        metadata_text = json.dumps(
            export_dataset_metadata(cfg, circuit),
            indent=2,
            sort_keys=True,
        )
        write_json_text(metadata_text, args.out_metadata)

    if args.skip_smoke_test:
        summary = summarize_circuit(circuit, cfg)
        text = summary.to_json(indent=2)
        if args.out_report is not None:
            write_json_text(text, args.out_report)
        print(text)
        return

    report = smoke_test_circuit(cfg, sample_shots=args.sample_shots)
    text = report.to_json(indent=2)

    if args.out_report is not None:
        write_json_text(text, args.out_report)

    print(text)


if __name__ == "__main__":
    main()