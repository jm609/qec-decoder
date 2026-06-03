from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import re

import numpy as np

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from logical_frame import infer_rotated_logical_string_qubits


BELL_PAIR_Z_READOUT_MODE = "bell_pair_z_readout"
LOGICAL_CLASS4_LABELS = {
    0: "I",
    1: "X",
    2: "Z",
    3: "Y",
}
OBSERVABLE_INDEX_TO_TARGET = {
    0: "logical_z_flip",
    1: "logical_x_flip",
}
_REC_TARGET_RE = re.compile(r"rec\[-(\d+)\]")


class MissingStimError(ImportError):
    """Raised when Stim-dependent logical Bell helpers are used without Stim."""


def _require_stim() -> Any:
    if stim is None:
        raise MissingStimError(
            "Stim is required but not installed in this Python environment."
        )
    return stim


@dataclass(frozen=True, slots=True)
class LogicalBellPairReadout:
    mode: str
    source_basis: str
    variant: str
    reference_qubit: int
    logical_x_qubits: list[int]
    logical_z_qubits: list[int]
    observable_index_to_target: dict[int, str]
    logical_class4_mapping: dict[int, str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_qcoords_block_end(lines: list[str]) -> int:
    index = 0
    while index < len(lines) and lines[index].startswith("QUBIT_COORDS"):
        index += 1
    return index


def _find_first_tick_after_qcoords(lines: list[str], qcoords_end: int) -> int:
    for index in range(qcoords_end, len(lines)):
        if lines[index] == "TICK":
            return index
    raise ValueError("Could not find first TICK after QUBIT_COORDS block.")


def _find_terminal_z_measurement_line(lines: list[str]) -> tuple[int, list[int]]:
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if line.startswith("M "):
            return index, [int(token) for token in line.split()[1:]]
    raise ValueError("Could not find terminal Z-basis data measurement line.")


def _shift_tail_measurement_refs(line: str, terminal_data_count: int) -> str:
    def repl(match: re.Match[str]) -> str:
        offset = int(match.group(1))
        if offset <= terminal_data_count:
            return f"rec[-{offset + 1}]"
        return f"rec[-{offset + 2}]"

    return _REC_TARGET_RE.sub(repl, line)


def describe_bell_pair_z_readout(
    *,
    circuit: Any,
    basis: str,
    variant: str,
) -> LogicalBellPairReadout:
    if basis != "z":
        raise ValueError(
            f"{BELL_PAIR_Z_READOUT_MODE!r} currently requires basis='z', got {basis!r}."
        )

    logical_strings = infer_rotated_logical_string_qubits(circuit)
    reference_qubit = int(circuit.num_qubits)
    notes = [
        "This supervision path adds an ideal noiseless reference qubit.",
        "The reference qubit is Bell-paired with the logical data qubit using the logical X string.",
        "Observable 0 is the flip of the XX Bell stabilizer and equals logical_z_flip.",
        "Observable 1 is the flip of the ZZ Bell stabilizer and equals logical_x_flip.",
        "logical_class4 is encoded as logical_x_flip + 2 * logical_z_flip with 0=I, 1=X, 2=Z, 3=Y.",
    ]
    if variant == "xzzx":
        notes.append(
            "variant='xzzx' still reuses the rotated Stim scaffold, so this Bell readout path is scaffold-level and not Willow-native."
        )

    return LogicalBellPairReadout(
        mode=BELL_PAIR_Z_READOUT_MODE,
        source_basis=basis,
        variant=variant,
        reference_qubit=reference_qubit,
        logical_x_qubits=list(logical_strings["x"]),
        logical_z_qubits=list(logical_strings["z"]),
        observable_index_to_target=dict(OBSERVABLE_INDEX_TO_TARGET),
        logical_class4_mapping=dict(LOGICAL_CLASS4_LABELS),
        notes=notes,
    )


def build_bell_pair_z_readout_circuit(
    *,
    circuit: Any,
    basis: str,
    variant: str,
) -> tuple[Any, LogicalBellPairReadout]:
    stim_mod = _require_stim()
    readout = describe_bell_pair_z_readout(
        circuit=circuit,
        basis=basis,
        variant=variant,
    )
    lines = str(circuit).splitlines()
    qcoords_end = _find_qcoords_block_end(lines)
    first_tick_index = _find_first_tick_after_qcoords(lines, qcoords_end)
    terminal_measure_index, terminal_qubits = _find_terminal_z_measurement_line(lines)
    terminal_measurement_line = lines[terminal_measure_index]
    terminal_data_count = len(terminal_qubits)

    out: list[str] = []
    out.extend(lines[:qcoords_end])
    out.append(f"QUBIT_COORDS(-1, -1) {readout.reference_qubit}")
    out.extend(lines[qcoords_end:first_tick_index])
    out.append(f"R {readout.reference_qubit}")
    out.append(f"H {readout.reference_qubit}")
    for qubit in readout.logical_x_qubits:
        out.append(f"CX {readout.reference_qubit} {qubit}")
    out.extend(lines[first_tick_index:terminal_measure_index])
    out.append(
        "MPP " + "*".join(
            [f"X{readout.reference_qubit}"] + [f"X{q}" for q in readout.logical_x_qubits]
        )
    )
    out.append(terminal_measurement_line)
    out.append(f"M {readout.reference_qubit}")

    found_observable = False
    for line in lines[terminal_measure_index + 1:]:
        shifted = _shift_tail_measurement_refs(line, terminal_data_count)
        if shifted.startswith("OBSERVABLE_INCLUDE("):
            if found_observable:
                raise ValueError(
                    "Expected a single logical observable in the source memory circuit tail."
                )
            shifted = shifted.replace("OBSERVABLE_INCLUDE(0)", "OBSERVABLE_INCLUDE(1)", 1)
            shifted += " rec[-1]"
            found_observable = True
        out.append(shifted)

    if not found_observable:
        raise ValueError("Could not find source logical observable line to rewrite.")

    out.append(f"OBSERVABLE_INCLUDE(0) rec[-{terminal_data_count + 2}]")
    return stim_mod.Circuit("\n".join(out)), readout


def derive_class4_targets_from_observable_flips(obs_flips: np.ndarray) -> dict[str, np.ndarray]:
    if obs_flips.ndim != 2 or obs_flips.shape[1] != 2:
        raise ValueError(
            "bell-pair logical class4 derivation requires observable_flips with shape (shots, 2), "
            f"got {obs_flips.shape}"
        )

    logical_z_flip = obs_flips[:, 0].astype(np.uint8, copy=False)
    logical_x_flip = obs_flips[:, 1].astype(np.uint8, copy=False)
    logical_class4 = (
        logical_x_flip.astype(np.uint8, copy=False)
        + (logical_z_flip.astype(np.uint8, copy=False) << 1)
    ).astype(np.uint8, copy=False)
    return {
        "logical_x_flip": logical_x_flip,
        "logical_z_flip": logical_z_flip,
        "logical_class4": logical_class4,
    }


def logical_class4_histogram(logical_class4: np.ndarray) -> dict[str, int]:
    counts = np.bincount(logical_class4.astype(np.int64, copy=False), minlength=4)
    return {
        LOGICAL_CLASS4_LABELS[index]: int(counts[index])
        for index in range(4)
    }
