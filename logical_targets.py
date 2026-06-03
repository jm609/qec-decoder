from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BELL_PAIR_Z_READOUT_MODE = "bell_pair_z_readout"


@dataclass(frozen=True, slots=True)
class LogicalTargetCapability:
    mode: str
    source_basis: str
    variant: str
    num_observables: int
    supports_per_shot_logical_class4: bool
    directly_measured_logical_observable: str
    supervised_logical_error_axis: str | None
    logical_axis_target_name: str | None
    available_targets: list[str]
    planned_future_targets: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def supervised_logical_error_axis_for_basis(basis: str) -> str:
    if basis == "z":
        return "x"
    if basis == "x":
        return "z"
    raise ValueError(f"Unsupported basis for logical-axis mapping: {basis!r}")


def logical_axis_target_name_for_basis(basis: str) -> str:
    axis = supervised_logical_error_axis_for_basis(basis)
    return f"logical_{axis}_flip"


def directly_measured_logical_observable_for_basis(basis: str) -> str:
    if basis == "z":
        return "z"
    if basis == "x":
        return "x"
    raise ValueError(f"Unsupported basis for direct logical observable mapping: {basis!r}")


def describe_logical_target_capability(
    *,
    basis: str,
    variant: str,
    num_observables: int,
    mode: str = "single_basis_memory",
) -> LogicalTargetCapability:
    if mode == BELL_PAIR_Z_READOUT_MODE:
        if basis != "z":
            raise ValueError(
                f"{BELL_PAIR_Z_READOUT_MODE!r} currently requires basis='z', got {basis!r}."
            )
        notes = [
            "This supervision path uses an added reference qubit and Bell-stabilizer readout.",
            "observable_flips[:, 0] is logical_z_flip from the XX Bell stabilizer.",
            "observable_flips[:, 1] is logical_x_flip from the ZZ Bell stabilizer.",
            "logical_label and logical_axis_flip remain aliases of logical_x_flip for backward compatibility on the rebuilt mainline.",
            "This path supports true per-shot logical_class4 labels.",
        ]
        if num_observables != 2:
            notes.append(
                f"Expected num_observables=2 for {BELL_PAIR_Z_READOUT_MODE}, got {num_observables}."
            )
        return LogicalTargetCapability(
            mode=mode,
            source_basis=basis,
            variant=variant,
            num_observables=num_observables,
            supports_per_shot_logical_class4=(num_observables == 2),
            directly_measured_logical_observable="bell_pair_xx_zz",
            supervised_logical_error_axis="x",
            logical_axis_target_name="logical_x_flip",
            available_targets=[
                "logical_label",
                "logical_axis_flip",
                "logical_x_flip",
                "logical_z_flip",
                "logical_class4",
            ],
            planned_future_targets=[],
            notes=notes,
        )

    if mode != "single_basis_memory":
        raise ValueError(f"Unsupported logical target capability mode: {mode!r}")

    if num_observables != 1:
        notes = [
            "Current logical target capability helper assumes one observable per basis-specific memory experiment.",
            f"Received num_observables={num_observables}; this path requires revisiting the supervision design.",
        ]
        return LogicalTargetCapability(
            mode=mode,
            source_basis=basis,
            variant=variant,
            num_observables=num_observables,
            supports_per_shot_logical_class4=False,
            directly_measured_logical_observable=directly_measured_logical_observable_for_basis(basis),
            supervised_logical_error_axis=None,
            logical_axis_target_name=None,
            available_targets=["logical_label"],
            planned_future_targets=["logical_x_flip", "logical_z_flip", "logical_class4"],
            notes=notes,
        )

    supervised_axis = supervised_logical_error_axis_for_basis(basis)
    target_name = logical_axis_target_name_for_basis(basis)
    return LogicalTargetCapability(
        mode=mode,
        source_basis=basis,
        variant=variant,
        num_observables=num_observables,
        supports_per_shot_logical_class4=False,
        directly_measured_logical_observable=directly_measured_logical_observable_for_basis(basis),
        supervised_logical_error_axis=supervised_axis,
        logical_axis_target_name=target_name,
        available_targets=["logical_label", "logical_axis_flip"],
        planned_future_targets=["logical_x_flip", "logical_z_flip", "logical_class4"],
        notes=[
            "A basis-specific surface-code memory experiment supervises only one logical Pauli error axis.",
            f"basis={basis} directly measures the logical {basis.upper()} observable.",
            f"basis={basis} supervises {target_name}.",
            "This is sufficient for axis-wise experiments but not for true per-shot logical_class4 supervision.",
        ],
    )
