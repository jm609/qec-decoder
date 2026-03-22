from __future__ import annotations

"""
config.py

Single source of truth for experiment configuration.

Project scope
-------------
This project starts from SI1000-Base and incrementally adds WillowCore:
    Stage A: SI1000-Base
    Stage B: + local heterogeneity
    Stage C: + correlated stray interaction
    Stage D: + leakage surrogate
    Stage E: + DQLR

Design principles
-----------------
1. Keep all experiment knobs in one place.
2. Separate ideal circuit config from noise config.
3. Make stage progression explicit and reproducible.
4. Use standard-library-only dataclasses for portability.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
import math


Basis = Literal["x", "z"]
CircuitVariant = Literal["stim_rotated", "xzzx"]
SaveFormat = Literal["npz", "pt", "jsonl"]
NoiseVersion = Literal[
    "sc_si1000_base_v1",
    "sc_si1000_willowcore_local_v1",
    "sc_si1000_willowcore_corr_v1",
    "sc_si1000_willowcore_leak_v1",
    "sc_si1000_willowcore_dqlr_v1",
]


def _ensure_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _ensure_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _ensure_split_sum_to_one(train: float, val: float, test: float, tol: float = 1e-9) -> None:
    total = train + val + test
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"dataset splits must sum to 1.0, got train+val+test={total}"
        )


@dataclass(frozen=True, slots=True)
class CircuitConfig:
    """
    Ideal surface-code memory circuit configuration.

    Notes
    -----
    - `variant="stim_rotated"` is the current practical scaffold.
    - `variant="xzzx"` is the future-facing API path for the Willow-style circuit.
    """

    distance: int = 3
    rounds: int = 3
    basis: Basis = "z"
    variant: CircuitVariant = "stim_rotated"

    def validate(self) -> None:
        if self.distance < 3:
            raise ValueError("distance must be >= 3")
        if self.distance % 2 == 0:
            raise ValueError("distance must be odd for rotated planar surface code")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.basis not in {"x", "z"}:
            raise ValueError(f"unsupported basis: {self.basis!r}")
        if self.variant not in {"stim_rotated", "xzzx"}:
            raise ValueError(f"unsupported variant: {self.variant!r}")

    @property
    def tag(self) -> str:
        return f"d{self.distance}_r{self.rounds}_{self.basis}_{self.variant}"


@dataclass(frozen=True, slots=True)
class SI1000Config:
    """
    Uniform SI1000-Base configuration.

    Base physical scale:
        p = physical_error_rate

    Derived channels:
        p_cz    = p
        p_1q    = p / 10
        p_reset = 2p
        p_meas  = 5p
        p_idle  = p / 10
        p_ridle = 2p

    This follows the superconducting-inspired SI1000 family, and matches
    the modern AQ2 interpretation where measurement and reset are treated
    separately and both idle-like terms are accounted for around them.
    """

    physical_error_rate: float = 0.0015

    override_p_cz: float | None = None
    override_p_1q: float | None = None
    override_p_reset: float | None = None
    override_p_meas: float | None = None
    override_p_idle: float | None = None
    override_p_ridle: float | None = None

    noise_version: str = "sc_si1000_base_v1"

    def validate(self) -> None:
        _ensure_probability("physical_error_rate", self.physical_error_rate)
        _ensure_probability("p_cz", self.p_cz)
        _ensure_probability("p_1q", self.p_1q)
        _ensure_probability("p_reset", self.p_reset)
        _ensure_probability("p_meas", self.p_meas)
        _ensure_probability("p_idle", self.p_idle)
        _ensure_probability("p_ridle", self.p_ridle)

    @property
    def p(self) -> float:
        return self.physical_error_rate

    @property
    def p_cz(self) -> float:
        return self.p if self.override_p_cz is None else self.override_p_cz

    @property
    def p_1q(self) -> float:
        return self.p / 10 if self.override_p_1q is None else self.override_p_1q

    @property
    def p_reset(self) -> float:
        return 2 * self.p if self.override_p_reset is None else self.override_p_reset

    @property
    def p_meas(self) -> float:
        return 5 * self.p if self.override_p_meas is None else self.override_p_meas

    @property
    def p_idle(self) -> float:
        return self.p / 10 if self.override_p_idle is None else self.override_p_idle

    @property
    def p_ridle(self) -> float:
        return 2 * self.p if self.override_p_ridle is None else self.override_p_ridle

    def resolved_rates(self) -> dict[str, float]:
        return {
            "p": self.p,
            "p_cz": self.p_cz,
            "p_1q": self.p_1q,
            "p_reset": self.p_reset,
            "p_meas": self.p_meas,
            "p_idle": self.p_idle,
            "p_ridle": self.p_ridle,
        }


@dataclass(frozen=True, slots=True)
class WillowLocalHeterogeneityConfig:
    """
    Local multiplicative heterogeneity.

    Suggested starting point:
    - qubit multiplier ~ LogNormal(0, 0.20), clipped to [0.5, 2.0]
    - edge  multiplier ~ LogNormal(0, 0.25), clipped to [0.5, 2.5]
    """

    enabled: bool = False

    qubit_lognormal_sigma: float = 0.20
    qubit_clip_min: float = 0.5
    qubit_clip_max: float = 2.0

    edge_lognormal_sigma: float = 0.25
    edge_clip_min: float = 0.5
    edge_clip_max: float = 2.5

    seed: int = 12345

    def validate(self) -> None:
        _ensure_nonnegative("qubit_lognormal_sigma", self.qubit_lognormal_sigma)
        _ensure_nonnegative("edge_lognormal_sigma", self.edge_lognormal_sigma)
        _ensure_positive("qubit_clip_min", self.qubit_clip_min)
        _ensure_positive("edge_clip_min", self.edge_clip_min)
        if self.qubit_clip_min > self.qubit_clip_max:
            raise ValueError("qubit_clip_min must be <= qubit_clip_max")
        if self.edge_clip_min > self.edge_clip_max:
            raise ValueError("edge_clip_min must be <= edge_clip_max")

    @property
    def noise_version(self) -> str:
        return "sc_si1000_willowcore_local_v1"


@dataclass(frozen=True, slots=True)
class WillowCorrelatedStrayConfig:
    """
    Correlated stray interaction around CZ gates.

    Baseline surrogate:
    - p_corr_zz = zz_factor * p_cz(edge)
    - p_swap    = swap_factor * p_cz(edge)
    """

    enabled: bool = False

    zz_factor: float = 0.05
    swap_factor: float = 0.02
    enable_swap_component: bool = True

    hotspot_scale: float = 1.0
    hotspot_edge_indices: tuple[int, ...] = field(default_factory=tuple)

    seed: int = 12346

    def validate(self) -> None:
        _ensure_nonnegative("zz_factor", self.zz_factor)
        _ensure_nonnegative("swap_factor", self.swap_factor)
        _ensure_positive("hotspot_scale", self.hotspot_scale)
        for idx in self.hotspot_edge_indices:
            if idx < 0:
                raise ValueError("hotspot_edge_indices must be nonnegative")

    @property
    def noise_version(self) -> str:
        return "sc_si1000_willowcore_corr_v1"


@dataclass(frozen=True, slots=True)
class WillowLeakageConfig:
    """
    Leakage surrogate config.

    Baseline surrogate:
    - leakage can occur after CZ with probability proportional to p_cz(edge)
    - leaked state persists for a few cycles
    - leakage boosts local readout / detector / neighboring fault rates
    """

    enabled: bool = False

    p_leak_after_cz_factor: float = 0.005
    mean_lifetime_cycles: float = 2.5

    readout_boost_factor: float = 2.0
    neighboring_cz_boost_factor: float = 1.5
    local_detector_boost_factor: float = 1.5

    seed: int = 12347

    def validate(self) -> None:
        _ensure_nonnegative("p_leak_after_cz_factor", self.p_leak_after_cz_factor)
        _ensure_positive("mean_lifetime_cycles", self.mean_lifetime_cycles)
        _ensure_positive("readout_boost_factor", self.readout_boost_factor)
        _ensure_positive("neighboring_cz_boost_factor", self.neighboring_cz_boost_factor)
        _ensure_positive("local_detector_boost_factor", self.local_detector_boost_factor)

    @property
    def noise_version(self) -> str:
        return "sc_si1000_willowcore_leak_v1"


@dataclass(frozen=True, slots=True)
class WillowDQLRConfig:
    """
    DQLR surrogate config.

    DQLR primarily acts by making leakage shorter-lived.
    """

    enabled: bool = False

    lifetime_reduction_factor: float = 0.5
    apply_every_n_cycles: int = 1

    def validate(self) -> None:
        _ensure_positive("lifetime_reduction_factor", self.lifetime_reduction_factor)
        if self.apply_every_n_cycles < 1:
            raise ValueError("apply_every_n_cycles must be >= 1")

    @property
    def noise_version(self) -> str:
        return "sc_si1000_willowcore_dqlr_v1"


@dataclass(frozen=True, slots=True)
class WillowCoreConfig:
    """
    Bundle of Willow-inspired non-uniform / correlated effects.

    Staging:
        local -> corr -> leak -> dqlr
    """

    local: WillowLocalHeterogeneityConfig = field(default_factory=WillowLocalHeterogeneityConfig)
    corr: WillowCorrelatedStrayConfig = field(default_factory=WillowCorrelatedStrayConfig)
    leak: WillowLeakageConfig = field(default_factory=WillowLeakageConfig)
    dqlr: WillowDQLRConfig = field(default_factory=WillowDQLRConfig)

    def validate(self) -> None:
        self.local.validate()
        self.corr.validate()
        self.leak.validate()
        self.dqlr.validate()

        if self.dqlr.enabled and not self.leak.enabled:
            raise ValueError("DQLR requires leakage surrogate to be enabled first")

        if self.corr.enabled and not self.local.enabled:
            raise ValueError(
                "For this project scaffold, correlated stray interaction must be added on top "
                "of the local heterogeneity stage"
            )

        if self.leak.enabled and not self.corr.enabled:
            raise ValueError(
                "For this project scaffold, leakage surrogate must be added after the correlated stage"
            )

    @property
    def stage(self) -> str:
        if self.dqlr.enabled:
            return "E"
        if self.leak.enabled:
            return "D"
        if self.corr.enabled:
            return "C"
        if self.local.enabled:
            return "B"
        return "A"

    @property
    def noise_version(self) -> str:
        if self.dqlr.enabled:
            return self.dqlr.noise_version
        if self.leak.enabled:
            return self.leak.noise_version
        if self.corr.enabled:
            return self.corr.noise_version
        if self.local.enabled:
            return self.local.noise_version
        return "sc_si1000_base_v1"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """
    Dataset generation config.

    The intended stored fields are:
    - detector_event_tensor
    - final_measurement / terminal information
    - logical_label
    - meta
    """

    shots: int = 10000
    sample_batch_size: int = 1024
    seed: int = 20260319

    output_dir: str = "artifacts"
    output_prefix: str = "dataset"
    save_format: SaveFormat = "npz"

    train_split: float = 0.90
    val_split: float = 0.05
    test_split: float = 0.05

    include_detector_coordinates: bool = True
    include_terminal_measurements: bool = True
    include_observable_flips: bool = True
    include_metadata: bool = True

    def validate(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be >= 1")
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be >= 1")
        _ensure_split_sum_to_one(self.train_split, self.val_split, self.test_split)
        if self.save_format not in {"npz", "pt", "jsonl"}:
            raise ValueError(f"unsupported save_format: {self.save_format!r}")

    @property
    def output_path_prefix(self) -> Path:
        return Path(self.output_dir) / self.output_prefix


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Top-level configuration object for the whole experiment.
    """

    name: str = "qec_decoder_experiment"

    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    si1000: SI1000Config = field(default_factory=SI1000Config)
    willow: WillowCoreConfig = field(default_factory=WillowCoreConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    notes: str = ""

    def validate(self) -> None:
        self.circuit.validate()
        self.si1000.validate()
        self.willow.validate()
        self.dataset.validate()

    @property
    def noise_stage(self) -> str:
        return self.willow.stage

    @property
    def noise_version(self) -> str:
        return self.willow.noise_version

    @property
    def experiment_tag(self) -> str:
        return f"{self.name}__{self.circuit.tag}__{self.noise_version}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resolved"] = {
            "noise_stage": self.noise_stage,
            "noise_version": self.noise_version,
            "experiment_tag": self.experiment_tag,
            "si1000_rates": self.si1000.resolved_rates(),
        }
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "qec_decoder_experiment"),
            circuit=CircuitConfig(**d.get("circuit", {})),
            si1000=SI1000Config(**d.get("si1000", {})),
            willow=WillowCoreConfig(
                local=WillowLocalHeterogeneityConfig(**d.get("willow", {}).get("local", {})),
                corr=WillowCorrelatedStrayConfig(**d.get("willow", {}).get("corr", {})),
                leak=WillowLeakageConfig(**d.get("willow", {}).get("leak", {})),
                dqlr=WillowDQLRConfig(**d.get("willow", {}).get("dqlr", {})),
            ),
            dataset=DatasetConfig(**d.get("dataset", {})),
            notes=d.get("notes", ""),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ------------------------------------------------------------------
# Stage presets
# ------------------------------------------------------------------

def make_stage_a_config(
    *,
    name: str = "stage_a_si1000_base",
    distance: int = 3,
    rounds: int | None = None,
    basis: Basis = "z",
    variant: CircuitVariant = "stim_rotated",
    physical_error_rate: float = 0.0015,
    shots: int = 10000,
) -> ExperimentConfig:
    rounds = distance if rounds is None else rounds
    cfg = ExperimentConfig(
        name=name,
        circuit=CircuitConfig(
            distance=distance,
            rounds=rounds,
            basis=basis,
            variant=variant,
        ),
        si1000=SI1000Config(
            physical_error_rate=physical_error_rate,
            noise_version="sc_si1000_base_v1",
        ),
        willow=WillowCoreConfig(),
        dataset=DatasetConfig(shots=shots),
        notes="Stage A: uniform SI1000-Base",
    )
    cfg.validate()
    return cfg


def make_stage_b_config(
    *,
    name: str = "stage_b_willow_local",
    distance: int = 3,
    rounds: int | None = None,
    basis: Basis = "z",
    variant: CircuitVariant = "stim_rotated",
    physical_error_rate: float = 0.0015,
    shots: int = 10000,
    local_seed: int = 12345,
) -> ExperimentConfig:
    rounds = distance if rounds is None else rounds
    cfg = ExperimentConfig(
        name=name,
        circuit=CircuitConfig(distance=distance, rounds=rounds, basis=basis, variant=variant),
        si1000=SI1000Config(physical_error_rate=physical_error_rate),
        willow=WillowCoreConfig(
            local=WillowLocalHeterogeneityConfig(enabled=True, seed=local_seed),
        ),
        dataset=DatasetConfig(shots=shots),
        notes="Stage B: SI1000-Base + local heterogeneity",
    )
    cfg.validate()
    return cfg


def make_stage_c_config(
    *,
    name: str = "stage_c_willow_corr",
    distance: int = 3,
    rounds: int | None = None,
    basis: Basis = "z",
    variant: CircuitVariant = "stim_rotated",
    physical_error_rate: float = 0.0015,
    shots: int = 10000,
    local_seed: int = 12345,
    corr_seed: int = 12346,
) -> ExperimentConfig:
    rounds = distance if rounds is None else rounds
    cfg = ExperimentConfig(
        name=name,
        circuit=CircuitConfig(distance=distance, rounds=rounds, basis=basis, variant=variant),
        si1000=SI1000Config(physical_error_rate=physical_error_rate),
        willow=WillowCoreConfig(
            local=WillowLocalHeterogeneityConfig(enabled=True, seed=local_seed),
            corr=WillowCorrelatedStrayConfig(enabled=True, seed=corr_seed),
        ),
        dataset=DatasetConfig(shots=shots),
        notes="Stage C: Stage B + correlated stray interaction",
    )
    cfg.validate()
    return cfg


def make_stage_d_config(
    *,
    name: str = "stage_d_willow_leak",
    distance: int = 3,
    rounds: int | None = None,
    basis: Basis = "z",
    variant: CircuitVariant = "stim_rotated",
    physical_error_rate: float = 0.0015,
    shots: int = 10000,
    local_seed: int = 12345,
    corr_seed: int = 12346,
    leak_seed: int = 12347,
) -> ExperimentConfig:
    rounds = distance if rounds is None else rounds
    cfg = ExperimentConfig(
        name=name,
        circuit=CircuitConfig(distance=distance, rounds=rounds, basis=basis, variant=variant),
        si1000=SI1000Config(physical_error_rate=physical_error_rate),
        willow=WillowCoreConfig(
            local=WillowLocalHeterogeneityConfig(enabled=True, seed=local_seed),
            corr=WillowCorrelatedStrayConfig(enabled=True, seed=corr_seed),
            leak=WillowLeakageConfig(enabled=True, seed=leak_seed),
        ),
        dataset=DatasetConfig(shots=shots),
        notes="Stage D: Stage C + leakage surrogate",
    )
    cfg.validate()
    return cfg


def make_stage_e_config(
    *,
    name: str = "stage_e_willow_dqlr",
    distance: int = 3,
    rounds: int | None = None,
    basis: Basis = "z",
    variant: CircuitVariant = "stim_rotated",
    physical_error_rate: float = 0.0015,
    shots: int = 10000,
    local_seed: int = 12345,
    corr_seed: int = 12346,
    leak_seed: int = 12347,
    dqlr_every_n_cycles: int = 1,
    dqlr_lifetime_reduction_factor: float = 0.5,
) -> ExperimentConfig:
    rounds = distance if rounds is None else rounds
    cfg = ExperimentConfig(
        name=name,
        circuit=CircuitConfig(distance=distance, rounds=rounds, basis=basis, variant=variant),
        si1000=SI1000Config(physical_error_rate=physical_error_rate),
        willow=WillowCoreConfig(
            local=WillowLocalHeterogeneityConfig(enabled=True, seed=local_seed),
            corr=WillowCorrelatedStrayConfig(enabled=True, seed=corr_seed),
            leak=WillowLeakageConfig(enabled=True, seed=leak_seed),
            dqlr=WillowDQLRConfig(
                enabled=True,
                apply_every_n_cycles=dqlr_every_n_cycles,
                lifetime_reduction_factor=dqlr_lifetime_reduction_factor,
            ),
        ),
        dataset=DatasetConfig(shots=shots),
        notes="Stage E: Stage D + DQLR",
    )
    cfg.validate()
    return cfg


# ------------------------------------------------------------------
# Minimal self-check
# ------------------------------------------------------------------

def _demo() -> None:
    cfg = make_stage_a_config(distance=3, rounds=3, basis="z", shots=128)
    print(cfg.to_json(indent=2))


if __name__ == "__main__":
    _demo()