from __future__ import annotations

"""
noise_willowcore.py

Willow-inspired in-circuit noise injection for ideal surface-code memory circuits.

Responsibilities
----------------
1. Read ExperimentConfig for Stage B/C.
2. Build the ideal seed circuit from circuits.py.
3. Reuse the Stage-A rewrite style from noise_si1000.py.
4. Add Stage B local heterogeneity.
5. Add Stage C correlated stray-interaction surrogates.
6. Export smoke-test reports and metadata for dataset generation.

Explicit non-responsibilities
-----------------------------
- Stage A is handled by noise_si1000.py.
- Stage D/E are intentionally NOT implemented here.
- Leakage surrogate and DQLR belong in a future shot-level postprocess module.

Important note
--------------
Stim may serialize CORRELATED_ERROR as the alias "E". This file counts both
"CORRELATED_ERROR" and "E" together.
"""

from dataclasses import asdict, dataclass
from typing import Any
import argparse
import json
import random

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from config import (
    ExperimentConfig,
    make_stage_b_config,
    make_stage_c_config,
)
from circuits import build_memory_circuit, export_dataset_metadata
from noise_si1000 import (
    MissingStimError,
    _append_depolarize1,
    _append_depolarize2,
    _append_op,
    _append_original_instruction,
    _append_single_qubit_error,
    _extract_qubit_targets,
    _instruction_is_repeat_block,
    _measure_flip_gate,
    _require_stim,
    _reset_flip_gate,
    ANNOTATION_GATES,
    MEASURE_GATES,
    MEASURE_RESET_GATES,
    ONE_QUBIT_CLIFFORD_GATES,
    PASSTHROUGH_NOISE_GATES,
    RESET_GATES,
    TWO_QUBIT_CLIFFORD_GATES,
    UNSUPPORTED_FOR_STAGE_A,
    write_circuit_text,
    write_json_text,
)


SUPPORTED_WILLOW_STAGES = frozenset({"B", "C"})


class UnsupportedWillowStageError(NotImplementedError):
    """Raised when a non-B/C stage is requested from this module."""


@dataclass(frozen=True, slots=True)
class LocalRateMaps:
    qubit_mult: dict[int, float]
    edge_mult: dict[tuple[int, int], float]


@dataclass(frozen=True, slots=True)
class CorrelatedEdgeMaps:
    p_corr_zz: dict[tuple[int, int], float]
    p_corr_swap: dict[tuple[int, int], float]


@dataclass(frozen=True, slots=True)
class WillowCoreSummary:
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
    correlated_instruction_counts: dict[str, int]
    supported_stages: tuple[str, ...]
    stage_d_postprocess_pending: bool
    stage_e_postprocess_pending: bool
    dem_uses_approximate_disjoint_errors: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class WillowCoreSmokeTestReport:
    summary: WillowCoreSummary
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


@dataclass(frozen=True, slots=True)
class WillowCoreArtifacts:
    noisy_circuit: Any
    local_maps: LocalRateMaps | None
    corr_maps: CorrelatedEdgeMaps | None


def _ensure_stage_bc_only(cfg: ExperimentConfig) -> None:
    cfg.validate()
    stage = cfg.noise_stage

    if stage == "A":
        raise ValueError(
            "noise_willowcore.py only supports Stage B/C. "
            "Use noise_si1000.py for Stage A."
        )

    if stage in {"D", "E"}:
        raise UnsupportedWillowStageError(
            f"Stage {stage} is intentionally disabled in noise_willowcore.py. "
            "Stage D/E leakage and DQLR are shot-level dynamics and should be implemented "
            "later in postprocess.py, not by silently reusing the Stage-C circuit."
        )

    if stage not in SUPPORTED_WILLOW_STAGES:
        raise ValueError(f"Unsupported Willow stage: {stage!r}")


def _extract_qubit_pairs_from_inst(inst: Any) -> list[tuple[int, int]]:
    qubits = _extract_qubit_targets(inst)
    if len(qubits) % 2 != 0:
        raise ValueError(f"Expected an even number of qubit targets, got {qubits}")
    return [(int(qubits[k]), int(qubits[k + 1])) for k in range(0, len(qubits), 2)]


def _extract_entangling_edges(seed_circuit: Any) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()

    def walk(block: Any) -> None:
        for op in block:
            if _instruction_is_repeat_block(op):
                walk(op.body_copy())
                continue
            if op.name not in TWO_QUBIT_CLIFFORD_GATES:
                continue
            for q0, q1 in _extract_qubit_pairs_from_inst(op):
                edges.add(tuple(sorted((q0, q1))))

    walk(seed_circuit)
    return sorted(edges)


def _qubit_scaled_rate(base_rate: float, q: int, local_maps: LocalRateMaps | None) -> float:
    if local_maps is None:
        return float(base_rate)
    return float(base_rate) * float(local_maps.qubit_mult.get(int(q), 1.0))


def _edge_scaled_rate(
    base_rate: float,
    edge: tuple[int, int],
    local_maps: LocalRateMaps | None,
) -> float:
    if local_maps is None:
        return float(base_rate)
    return float(base_rate) * float(local_maps.edge_mult.get(tuple(sorted(edge)), 1.0))


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _dem_kwargs(cfg: ExperimentConfig) -> dict[str, bool]:
    kwargs: dict[str, bool] = {"decompose_errors": True}
    if cfg.willow.corr.enabled and cfg.willow.corr.enable_swap_component:
        kwargs["approximate_disjoint_errors"] = True
    return kwargs


def build_local_rate_maps(seed_circuit: Any, cfg: ExperimentConfig) -> LocalRateMaps:
    local_cfg = cfg.willow.local
    rng = random.Random(local_cfg.seed)

    qubits = list(range(int(seed_circuit.num_qubits)))
    edges = _extract_entangling_edges(seed_circuit)

    qubit_mult: dict[int, float] = {}
    for q in qubits:
        x = rng.lognormvariate(0.0, local_cfg.qubit_lognormal_sigma)
        qubit_mult[int(q)] = min(max(x, local_cfg.qubit_clip_min), local_cfg.qubit_clip_max)

    edge_mult: dict[tuple[int, int], float] = {}
    for edge in edges:
        x = rng.lognormvariate(0.0, local_cfg.edge_lognormal_sigma)
        edge_mult[edge] = min(max(x, local_cfg.edge_clip_min), local_cfg.edge_clip_max)

    return LocalRateMaps(qubit_mult=qubit_mult, edge_mult=edge_mult)


def build_correlated_edge_maps(
    cfg: ExperimentConfig,
    local_maps: LocalRateMaps,
) -> CorrelatedEdgeMaps:
    corr_cfg = cfg.willow.corr

    p_corr_zz: dict[tuple[int, int], float] = {}
    p_corr_swap: dict[tuple[int, int], float] = {}

    sorted_edges = sorted(local_maps.edge_mult)
    for edge_index, edge in enumerate(sorted_edges):
        hotspot_scale = (
            corr_cfg.hotspot_scale
            if edge_index in corr_cfg.hotspot_edge_indices
            else 1.0
        )
        effective_p_cz = cfg.si1000.p_cz * local_maps.edge_mult[edge] * hotspot_scale
        p_corr_zz[edge] = corr_cfg.zz_factor * effective_p_cz
        p_corr_swap[edge] = (
            corr_cfg.swap_factor * effective_p_cz
            if corr_cfg.enable_swap_component
            else 0.0
        )

    return CorrelatedEdgeMaps(
        p_corr_zz=p_corr_zz,
        p_corr_swap=p_corr_swap,
    )


def _append_correlated_surrogates(
    out: Any,
    q0: int,
    q1: int,
    edge: tuple[int, int],
    corr_maps: CorrelatedEdgeMaps | None,
    stim_mod: Any,
) -> None:
    if corr_maps is None:
        return

    edge = tuple(sorted(edge))
    pzz = float(corr_maps.p_corr_zz.get(edge, 0.0))
    pswap = float(corr_maps.p_corr_swap.get(edge, 0.0))

    if pzz > 0.0:
        _append_op(
            out,
            "CORRELATED_ERROR",
            [stim_mod.target_z(int(q0)), stim_mod.target_z(int(q1))],
            [pzz],
        )

    if pswap > 0.0:
        # Conservative surrogate for "swap-like" faults:
        # small XX/YY mixture instead of claiming exact coherent SWAP dynamics.
        _append_op(
            out,
            "PAULI_CHANNEL_2",
            [int(q0), int(q1)],
            [
                0.0,          # IX
                0.0,          # IY
                0.0,          # IZ
                0.0,          # XI
                pswap * 0.5,  # XX
                0.0,          # XY
                0.0,          # XZ
                0.0,          # YI
                0.0,          # YX
                pswap * 0.5,  # YY
                0.0,          # YZ
                0.0,          # ZI
                0.0,          # ZX
                0.0,          # ZY
                0.0,          # ZZ
            ],
        )


def _rewrite_layer_willow(
    layer_ops: list[Any],
    *,
    num_qubits: int,
    cfg: ExperimentConfig,
    local_maps: LocalRateMaps | None,
    corr_maps: CorrelatedEdgeMaps | None,
    stim_mod: Any,
) -> Any:
    out = stim_mod.Circuit()

    active_qubits: set[int] = set()
    meas_reset_qubits: set[int] = set()
    layer_has_meas_or_reset = False

    for inst in layer_ops:
        name = inst.name

        if name in UNSUPPORTED_FOR_STAGE_A:
            raise NotImplementedError(
                f"Instruction {name!r} is not supported by WillowCore rewriting."
            )

        qubits = _extract_qubit_targets(inst)

        if name in ANNOTATION_GATES or name in PASSTHROUGH_NOISE_GATES:
            _append_original_instruction(out, inst)
            continue

        if name in MEASURE_GATES:
            for q in qubits:
                _append_single_qubit_error(
                    out,
                    _measure_flip_gate(name),
                    _qubit_scaled_rate(cfg.si1000.p_meas, q, local_maps),
                    [q],
                )
            _append_original_instruction(out, inst)
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in RESET_GATES:
            _append_original_instruction(out, inst)
            for q in qubits:
                _append_single_qubit_error(
                    out,
                    _reset_flip_gate(name),
                    _qubit_scaled_rate(cfg.si1000.p_reset, q, local_maps),
                    [q],
                )
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in MEASURE_RESET_GATES:
            for q in qubits:
                _append_single_qubit_error(
                    out,
                    _measure_flip_gate(name),
                    _qubit_scaled_rate(cfg.si1000.p_meas, q, local_maps),
                    [q],
                )
            _append_original_instruction(out, inst)
            for q in qubits:
                _append_single_qubit_error(
                    out,
                    _reset_flip_gate(name),
                    _qubit_scaled_rate(cfg.si1000.p_reset, q, local_maps),
                    [q],
                )
            active_qubits.update(qubits)
            meas_reset_qubits.update(qubits)
            layer_has_meas_or_reset = True
            continue

        if name in ONE_QUBIT_CLIFFORD_GATES:
            _append_original_instruction(out, inst)
            for q in qubits:
                _append_depolarize1(
                    out,
                    _qubit_scaled_rate(cfg.si1000.p_1q, q, local_maps),
                    [q],
                )
            active_qubits.update(qubits)
            continue

        if name in TWO_QUBIT_CLIFFORD_GATES:
            for q0, q1 in _extract_qubit_pairs_from_inst(inst):
                _append_op(out, name, [q0, q1])
                _append_depolarize2(
                    out,
                    _edge_scaled_rate(cfg.si1000.p_cz, (q0, q1), local_maps),
                    [q0, q1],
                )
                _append_correlated_surrogates(
                    out,
                    q0,
                    q1,
                    (q0, q1),
                    corr_maps,
                    stim_mod,
                )
                active_qubits.add(q0)
                active_qubits.add(q1)
            continue

        raise NotImplementedError(
            f"Unsupported instruction {name!r} encountered during WillowCore rewrite."
        )

    all_qubits = list(range(int(num_qubits)))
    inactive_qubits = sorted(set(all_qubits) - active_qubits)
    for q in inactive_qubits:
        _append_depolarize1(out, _qubit_scaled_rate(cfg.si1000.p_idle, q, local_maps), [q])

    if layer_has_meas_or_reset:
        ridle_qubits = sorted(set(all_qubits) - meas_reset_qubits)
        for q in ridle_qubits:
            _append_depolarize1(out, _qubit_scaled_rate(cfg.si1000.p_ridle, q, local_maps), [q])

    return out


def _rewrite_block_willow(
    block: Any,
    *,
    num_qubits: int,
    cfg: ExperimentConfig,
    local_maps: LocalRateMaps | None,
    corr_maps: CorrelatedEdgeMaps | None,
    stim_mod: Any,
) -> Any:
    out = stim_mod.Circuit()
    pending_layer: list[Any] = []

    def flush_pending_layer() -> None:
        nonlocal pending_layer, out
        if not pending_layer:
            return
        out += _rewrite_layer_willow(
            pending_layer,
            num_qubits=num_qubits,
            cfg=cfg,
            local_maps=local_maps,
            corr_maps=corr_maps,
            stim_mod=stim_mod,
        )
        pending_layer = []

    for op in block:
        if _instruction_is_repeat_block(op):
            flush_pending_layer()
            body = op.body_copy()
            repeat_count = int(op.repeat_count)
            rewritten_body = _rewrite_block_willow(
                body,
                num_qubits=num_qubits,
                cfg=cfg,
                local_maps=local_maps,
                corr_maps=corr_maps,
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


def apply_willowcore_noise(seed_circuit: Any, cfg: ExperimentConfig) -> WillowCoreArtifacts:
    stim_mod = _require_stim()
    _ensure_stage_bc_only(cfg)

    local_maps = build_local_rate_maps(seed_circuit, cfg) if cfg.willow.local.enabled else None
    corr_maps = (
        build_correlated_edge_maps(cfg, local_maps)
        if (cfg.willow.corr.enabled and local_maps is not None)
        else None
    )

    noisy = _rewrite_block_willow(
        seed_circuit,
        num_qubits=int(seed_circuit.num_qubits),
        cfg=cfg,
        local_maps=local_maps,
        corr_maps=corr_maps,
        stim_mod=stim_mod,
    )

    return WillowCoreArtifacts(
        noisy_circuit=noisy,
        local_maps=local_maps,
        corr_maps=corr_maps,
    )


def build_willowcore_memory_circuit(cfg: ExperimentConfig) -> Any:
    _ensure_stage_bc_only(cfg)
    seed = build_memory_circuit(cfg)
    return apply_willowcore_noise(seed, cfg).noisy_circuit


def count_correlated_instructions(circuit: Any) -> dict[str, int]:
    counts = {
        "CORRELATED_ERROR": 0,
        "ELSE_CORRELATED_ERROR": 0,
        "PAULI_CHANNEL_2": 0,
    }

    def walk(block: Any) -> None:
        for op in block:
            if _instruction_is_repeat_block(op):
                walk(op.body_copy())
                continue

            name = op.name
            if name in {"CORRELATED_ERROR", "E"}:
                counts["CORRELATED_ERROR"] += 1
            elif name == "ELSE_CORRELATED_ERROR":
                counts["ELSE_CORRELATED_ERROR"] += 1
            elif name == "PAULI_CHANNEL_2":
                counts["PAULI_CHANNEL_2"] += 1

    walk(circuit)
    return counts


def summarize_willowcore_circuit(
    cfg: ExperimentConfig,
    noisy_circuit: Any,
) -> WillowCoreSummary:
    _ensure_stage_bc_only(cfg)
    rates = cfg.si1000.resolved_rates()
    return WillowCoreSummary(
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
        correlated_instruction_counts=count_correlated_instructions(noisy_circuit),
        supported_stages=tuple(sorted(SUPPORTED_WILLOW_STAGES)),
        stage_d_postprocess_pending=True,
        stage_e_postprocess_pending=True,
        dem_uses_approximate_disjoint_errors=_dem_kwargs(cfg).get(
            "approximate_disjoint_errors",
            False,
        ),
    )


def export_noisy_metadata(
    cfg: ExperimentConfig,
    noisy_circuit: Any,
    *,
    local_maps: LocalRateMaps | None = None,
    corr_maps: CorrelatedEdgeMaps | None = None,
) -> dict[str, Any]:
    _ensure_stage_bc_only(cfg)

    metadata = export_dataset_metadata(cfg, noisy_circuit)
    metadata["si1000_rates"] = cfg.si1000.resolved_rates()
    metadata["notes"] = cfg.notes
    metadata["correlated_instruction_counts"] = count_correlated_instructions(noisy_circuit)
    metadata["supported_stages"] = sorted(SUPPORTED_WILLOW_STAGES)
    metadata["stage_d_postprocess_pending"] = True
    metadata["stage_e_postprocess_pending"] = True
    metadata["dem_kwargs"] = _dem_kwargs(cfg)

    if local_maps is not None:
        metadata["local_qubit_multiplier_stats"] = _stats(
            list(local_maps.qubit_mult.values())
        )
        metadata["local_edge_multiplier_stats"] = _stats(
            list(local_maps.edge_mult.values())
        )

    if corr_maps is not None:
        metadata["corr_zz_prob_stats"] = _stats(
            list(corr_maps.p_corr_zz.values())
        )
        metadata["corr_swap_prob_stats"] = _stats(
            list(corr_maps.p_corr_swap.values())
        )

    return metadata


def smoke_test_willowcore(
    cfg: ExperimentConfig,
    *,
    sample_shots: int = 64,
) -> WillowCoreSmokeTestReport:
    if sample_shots < 1:
        raise ValueError("sample_shots must be >= 1")

    seed = build_memory_circuit(cfg)
    artifacts = apply_willowcore_noise(seed, cfg)
    noisy = artifacts.noisy_circuit
    summary = summarize_willowcore_circuit(cfg, noisy)

    sampler = noisy.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(
        shots=sample_shots,
        separate_observables=True,
    )

    dem = noisy.detector_error_model(**_dem_kwargs(cfg))

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

    return WillowCoreSmokeTestReport(
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test a Willow-inspired noisy surface-code circuit (Stage B/C only)."
    )
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--stage", choices=["B", "C"], default="B")
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
        _ensure_stage_bc_only(cfg)
        return cfg

    common = dict(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        variant=args.variant,
        physical_error_rate=args.p,
        shots=args.shots,
    )

    if args.stage == "B":
        return make_stage_b_config(**common)
    if args.stage == "C":
        return make_stage_c_config(**common)
    raise AssertionError(f"Unreachable stage: {args.stage!r}")


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg_from_args(args)

    seed = build_memory_circuit(cfg)
    artifacts = apply_willowcore_noise(seed, cfg)
    noisy = artifacts.noisy_circuit

    if args.out_circuit is not None:
        write_circuit_text(noisy, args.out_circuit)

    if args.out_metadata is not None:
        metadata_text = json.dumps(
            export_noisy_metadata(
                cfg,
                noisy,
                local_maps=artifacts.local_maps,
                corr_maps=artifacts.corr_maps,
            ),
            indent=2,
            sort_keys=True,
        )
        write_json_text(metadata_text, args.out_metadata)

    if args.skip_smoke_test:
        summary = summarize_willowcore_circuit(cfg, noisy)
        text = summary.to_json(indent=2)
        if args.out_report is not None:
            write_json_text(text, args.out_report)
        print(text)
        return

    report = smoke_test_willowcore(cfg, sample_shots=args.sample_shots)
    text = report.to_json(indent=2)

    if args.out_report is not None:
        write_json_text(text, args.out_report)

    print(text)


if __name__ == "__main__":
    main()
