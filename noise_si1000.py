from __future__ import annotations

"""
noise_si1000.py

Uniform SI1000-Base noise injection for ideal surface-code memory circuits.

Responsibilities
----------------
1. Read ExperimentConfig / SI1000Config.
2. Build the ideal seed circuit from circuits.py.
3. Rewrite the seed into a noisy circuit using uniform SI1000-Base channels.
4. Provide smoke tests and metadata export.

Non-responsibilities
--------------------
- No local heterogeneity here.
- No correlated stray interaction here.
- No leakage surrogate here.
- No DQLR here.

Module boundary
---------------
- Stage A is implemented here.
- Stage B/C belong to noise_willowcore.py.
- Stage D/E are reserved for a future shot-level postprocess module.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from config import ExperimentConfig, SI1000Config, make_stage_a_config
from circuits import build_memory_circuit, export_dataset_metadata


class MissingStimError(ImportError):
    """Raised when Stim-dependent functionality is used without Stim installed."""


def _require_stim() -> Any:
    if stim is None:
        raise MissingStimError(
            "Stim is required but not installed in this Python environment."
        )
    return stim


def _ensure_stage_a_only(cfg: ExperimentConfig) -> None:
    cfg.validate()
    if cfg.noise_stage != "A":
        raise ValueError(
            "noise_si1000.py only supports Stage A (uniform SI1000-Base). "
            f"Current config is Stage {cfg.noise_stage} / {cfg.noise_version}. "
            "Use noise_willowcore.py for Stage B/C. "
            "Stage D/E are reserved for a future shot-level postprocess module."
        )


@dataclass(frozen=True, slots=True)
class SI1000Summary:
    experiment_name: str
    experiment_tag: str
    noise_stage: str
    noise_version: str
    distance: int
    rounds: int
    basis: str
    variant: str
    p: float
    p_cz: float
    p_1q: float
    p_reset: float
    p_meas: float
    p_idle: float
    p_ridle: float
    num_qubits: int
    num_measurements: int
    num_detectors: int
    num_observables: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class SI1000SmokeTestReport:
    summary: SI1000Summary
    dem_num_detectors: int
    dem_num_observables: int
    detector_sample_shape: tuple[int, int]
    detector_sample_dtype: str
    observable_sample_shape: tuple[int, int]
    observable_sample_dtype: str
    detector_event_fraction: float
    logical_flip_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# ----------------------------
# Stim instruction families
# ----------------------------

ANNOTATION_GATES = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS",
    "QUBIT_COORDS",
    "TICK",
    "MPAD",
}

PASSTHROUGH_NOISE_GATES = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
}

ONE_QUBIT_CLIFFORD_GATES = {
    "H",
    "S",
    "S_DAG",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "H_XY",
    "H_YZ",
    "H_XZ",
    "C_XYZ",
    "C_ZYX",
    "X",
    "Y",
    "Z",
}

TWO_QUBIT_CLIFFORD_GATES = {
    "CX",
    "CY",
    "CZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "SWAP",
    "ISWAP",
    "ISWAP_DAG",
}

RESET_GATES = {"R", "RX", "RY", "RZ"}
MEASURE_GATES = {"M", "MX", "MY", "MZ"}
MEASURE_RESET_GATES = {"MR", "MRX", "MRY", "MRZ"}

UNSUPPORTED_FOR_STAGE_A = {
    "MPP",
}


# ----------------------------
# Low-level Stim helpers
# ----------------------------

def _call_or_value(x: Any) -> Any:
    return x() if callable(x) else x


def _targets_of(inst: Any) -> list[Any]:
    if hasattr(inst, "targets_copy"):
        return list(inst.targets_copy())
    if hasattr(inst, "targets"):
        return list(inst.targets)
    raise TypeError(f"Cannot extract targets from instruction type {type(inst)}")


def _gate_args_of(inst: Any) -> list[float]:
    if hasattr(inst, "gate_args_copy"):
        return list(inst.gate_args_copy())
    if hasattr(inst, "args_copy"):
        return list(inst.args_copy())
    if hasattr(inst, "gate_args"):
        return list(inst.gate_args)
    return []


def _extract_qubit_targets(inst: Any) -> list[int]:
    qubits: list[int] = []
    for t in _targets_of(inst):
        is_qubit_target = getattr(t, "is_qubit_target", None)
        if is_qubit_target is not None and _call_or_value(is_qubit_target):
            qubits.append(int(getattr(t, "value")))
            continue

        qubit_value = getattr(t, "qubit_value", None)
        if qubit_value is not None:
            qubits.append(int(_call_or_value(qubit_value)))
            continue

        if isinstance(t, int):
            qubits.append(int(t))

    return qubits


def _append_op(
    out: Any,
    name: str,
    targets: list[Any],
    gate_args: list[float] | None = None,
) -> None:
    gate_args = [] if gate_args is None else list(gate_args)
    if len(gate_args) == 0:
        out.append(name, targets)
    elif len(gate_args) == 1:
        out.append(name, targets, gate_args[0])
    else:
        out.append(name, targets, gate_args)


def _append_original_instruction(out: Any, inst: Any) -> None:
    _append_op(out, inst.name, _targets_of(inst), _gate_args_of(inst))


def _append_depolarize1(out: Any, p: float, qubits: list[int]) -> None:
    if p <= 0.0 or len(qubits) == 0:
        return
    _append_op(out, "DEPOLARIZE1", [int(q) for q in qubits], [float(p)])


def _append_depolarize2(out: Any, p: float, qubits: list[int]) -> None:
    if p <= 0.0 or len(qubits) == 0:
        return
    if len(qubits) % 2 != 0:
        raise ValueError(
            f"DEPOLARIZE2 requires an even number of targets, got {qubits}"
        )
    _append_op(out, "DEPOLARIZE2", [int(q) for q in qubits], [float(p)])


def _append_single_qubit_error(out: Any, gate_name: str, p: float, qubits: list[int]) -> None:
    if p <= 0.0 or len(qubits) == 0:
        return
    _append_op(out, gate_name, [int(q) for q in qubits], [float(p)])


def _measure_flip_gate(name: str) -> str:
    if name in {"M", "MZ", "MR", "MRZ"}:
        return "X_ERROR"
    if name in {"MX", "MRX"}:
        return "Z_ERROR"
    if name in {"MY", "MRY"}:
        return "X_ERROR"
    raise ValueError(f"Unsupported measurement gate for flip mapping: {name}")


def _reset_flip_gate(name: str) -> str:
    if name in {"R", "RZ", "MR", "MRZ"}:
        return "X_ERROR"
    if name in {"RX", "MRX"}:
        return "Z_ERROR"
    if name in {"RY", "MRY"}:
        return "X_ERROR"
    raise ValueError(f"Unsupported reset gate for flip mapping: {name}")


def _instruction_is_repeat_block(op: Any) -> bool:
    return op.__class__.__name__ == "CircuitRepeatBlock"


# ----------------------------
# Layer/block rewriting
# ----------------------------

def _rewrite_layer(
    layer_ops: list[Any],
    *,
    num_qubits: int,
    si_cfg: SI1000Config,
    stim_mod: Any,
) -> Any:
    """
    Rewrite one TICK-delimited layer with SI1000-Base noise.
    """
    out = stim_mod.Circuit()

    active_qubits: set[int] = set()
    meas_reset_qubits: set[int] = set()
    layer_has_meas_or_reset = False

    for inst in layer_ops:
        name = inst.name

        if name in UNSUPPORTED_FOR_STAGE_A:
            raise NotImplementedError(
                f"Instruction {name!r} is not supported by Stage-A SI1000 rewriting."
            )

        qubits = _extract_qubit_targets(inst)

        if name in ANNOTATION_GATES or name in PASSTHROUGH_NOISE_GATES:
            _append_original_instruction(out, inst)
            continue

        if name in MEASURE_GATES:
            _append_single_qubit_error(
                out,
                _measure_flip_gate(name),
                si_cfg.p_meas,
                qubits,
            )
            _append_original_instruction(out, inst)
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in RESET_GATES:
            _append_original_instruction(out, inst)
            _append_single_qubit_error(
                out,
                _reset_flip_gate(name),
                si_cfg.p_reset,
                qubits,
            )
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in MEASURE_RESET_GATES:
            _append_single_qubit_error(
                out,
                _measure_flip_gate(name),
                si_cfg.p_meas,
                qubits,
            )
            _append_original_instruction(out, inst)
            _append_single_qubit_error(
                out,
                _reset_flip_gate(name),
                si_cfg.p_reset,
                qubits,
            )
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in ONE_QUBIT_CLIFFORD_GATES:
            _append_original_instruction(out, inst)
            _append_depolarize1(out, si_cfg.p_1q, qubits)
            active_qubits.update(qubits)
            continue

        if name in TWO_QUBIT_CLIFFORD_GATES:
            _append_original_instruction(out, inst)
            _append_depolarize2(out, si_cfg.p_cz, qubits)
            active_qubits.update(qubits)
            continue

        raise NotImplementedError(
            f"Unsupported instruction {name!r} encountered during SI1000 rewrite."
        )

    all_qubits = list(range(int(num_qubits)))
    inactive_qubits = sorted(set(all_qubits) - active_qubits)
    _append_depolarize1(out, si_cfg.p_idle, inactive_qubits)

    if layer_has_meas_or_reset:
        ridle_qubits = sorted(set(all_qubits) - meas_reset_qubits)
        _append_depolarize1(out, si_cfg.p_ridle, ridle_qubits)

    return out


def _rewrite_block(
    block: Any,
    *,
    num_qubits: int,
    si_cfg: SI1000Config,
    stim_mod: Any,
) -> Any:
    """
    Recursively rewrite a Stim circuit block.

    REPEAT blocks are intentionally unrolled to reduce version-specific
    dependence on Stim repeat-block constructors.
    """
    out = stim_mod.Circuit()
    pending_layer: list[Any] = []

    def flush_pending_layer() -> None:
        nonlocal pending_layer, out
        if not pending_layer:
            return
        out += _rewrite_layer(
            pending_layer,
            num_qubits=num_qubits,
            si_cfg=si_cfg,
            stim_mod=stim_mod,
        )
        pending_layer = []

    for op in block:
        if _instruction_is_repeat_block(op):
            flush_pending_layer()
            body = op.body_copy()
            repeat_count = int(op.repeat_count)
            rewritten_body = _rewrite_block(
                body,
                num_qubits=num_qubits,
                si_cfg=si_cfg,
                stim_mod=stim_mod,
            )
            for _ in range(repeat_count):
                out += rewritten_body
            continue

        if op.name == "TICK":
            flush_pending_layer()
            out.append("TICK")
            continue

        pending_layer.append(op)

    flush_pending_layer()
    return out


# ----------------------------
# Public API
# ----------------------------

def apply_si1000_noise(seed_circuit: Any, si_cfg: SI1000Config) -> Any:
    """
    Rewrite an ideal seed circuit into a uniform SI1000-Base noisy circuit.
    """
    stim_mod = _require_stim()
    si_cfg.validate()
    return _rewrite_block(
        seed_circuit,
        num_qubits=int(seed_circuit.num_qubits),
        si_cfg=si_cfg,
        stim_mod=stim_mod,
    )


def build_si1000_memory_circuit(cfg: ExperimentConfig) -> Any:
    """
    Build ideal seed from circuits.py, then inject uniform SI1000-Base noise.
    """
    _ensure_stage_a_only(cfg)
    seed = build_memory_circuit(cfg)
    noisy = apply_si1000_noise(seed, cfg.si1000)
    return noisy


def summarize_si1000_circuit(cfg: ExperimentConfig, noisy_circuit: Any) -> SI1000Summary:
    _ensure_stage_a_only(cfg)
    rates = cfg.si1000.resolved_rates()
    return SI1000Summary(
        experiment_name=cfg.name,
        experiment_tag=cfg.experiment_tag,
        noise_stage=cfg.noise_stage,
        noise_version=cfg.noise_version,
        distance=cfg.circuit.distance,
        rounds=cfg.circuit.rounds,
        basis=cfg.circuit.basis,
        variant=cfg.circuit.variant,
        p=rates["p"],
        p_cz=rates["p_cz"],
        p_1q=rates["p_1q"],
        p_reset=rates["p_reset"],
        p_meas=rates["p_meas"],
        p_idle=rates["p_idle"],
        p_ridle=rates["p_ridle"],
        num_qubits=int(noisy_circuit.num_qubits),
        num_measurements=int(noisy_circuit.num_measurements),
        num_detectors=int(noisy_circuit.num_detectors),
        num_observables=int(noisy_circuit.num_observables),
    )


def export_noisy_metadata(cfg: ExperimentConfig, noisy_circuit: Any) -> dict[str, Any]:
    """
    Metadata block for dataset generation on the noisy Stage-A circuit.
    """
    _ensure_stage_a_only(cfg)
    metadata = export_dataset_metadata(cfg, noisy_circuit)
    metadata["si1000_rates"] = cfg.si1000.resolved_rates()
    metadata["notes"] = cfg.notes
    metadata["supported_by_module"] = {
        "this_module": ["A"],
        "noise_willowcore.py": ["B", "C"],
        "postprocess.py": ["D", "E"],
    }
    return metadata


def smoke_test_si1000(
    cfg: ExperimentConfig,
    *,
    sample_shots: int = 64,
) -> SI1000SmokeTestReport:
    """
    Runtime validation for the noisy Stage-A circuit.
    """
    if sample_shots < 1:
        raise ValueError("sample_shots must be >= 1")

    noisy = build_si1000_memory_circuit(cfg)
    summary = summarize_si1000_circuit(cfg, noisy)

    sampler = noisy.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(
        shots=sample_shots,
        separate_observables=True,
    )

    dem = noisy.detector_error_model(decompose_errors=True)

    det_shape = tuple(int(x) for x in det_events.shape)
    obs_shape = tuple(int(x) for x in obs_flips.shape)

    if summary.num_detectors < 1:
        raise RuntimeError("expected at least one detector")
    if summary.num_observables < 1:
        raise RuntimeError("expected at least one logical observable")

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

    return SI1000SmokeTestReport(
        summary=summary,
        dem_num_detectors=int(dem.num_detectors),
        dem_num_observables=int(dem.num_observables),
        detector_sample_shape=det_shape,
        detector_sample_dtype=str(det_events.dtype),
        observable_sample_shape=obs_shape,
        observable_sample_dtype=str(obs_flips.dtype),
        detector_event_fraction=float(det_events.mean()),
        logical_flip_fraction=float(obs_flips.mean()),
    )


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


# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test a Stage-A SI1000-Base noisy surface-code circuit."
    )
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=["x", "z"], default="z")
    parser.add_argument("--variant", choices=["stim_rotated", "xzzx"], default="stim_rotated")
    parser.add_argument("--p", type=float, default=0.0015)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--sample-shots", type=int, default=64)
    parser.add_argument("--out-circuit", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-metadata", type=str, default=None)
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def _build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config_json is not None:
        cfg = ExperimentConfig.load_json(args.config_json)
        _ensure_stage_a_only(cfg)
        return cfg

    cfg = make_stage_a_config(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        variant=args.variant,
        physical_error_rate=args.p,
        shots=args.shots,
    )
    _ensure_stage_a_only(cfg)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg_from_args(args)

    noisy = build_si1000_memory_circuit(cfg)

    if args.out_circuit is not None:
        write_circuit_text(noisy, args.out_circuit)

    if args.out_metadata is not None:
        metadata_text = json.dumps(
            export_noisy_metadata(cfg, noisy),
            indent=2,
            sort_keys=True,
        )
        write_json_text(metadata_text, args.out_metadata)

    if args.skip_smoke_test:
        summary = summarize_si1000_circuit(cfg, noisy)
        text = summary.to_json(indent=2)
        if args.out_report is not None:
            write_json_text(text, args.out_report)
        print(text)
        return

    report = smoke_test_si1000(cfg, sample_shots=args.sample_shots)
    text = report.to_json(indent=2)

    if args.out_report is not None:
        write_json_text(text, args.out_report)

    print(text)


if __name__ == "__main__":
    main()
