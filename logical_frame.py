from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import stim  # type: ignore
except ImportError:
    stim = None

from logical_targets import supervised_logical_error_axis_for_basis


class MissingStimError(ImportError):
    """Raised when Stim-dependent logical-frame utilities are used without Stim."""


def _require_stim() -> Any:
    if stim is None:
        raise MissingStimError(
            "Stim is required but not installed in this Python environment."
        )
    return stim


def _sorted_qubit_coords(circuit: Any) -> dict[int, tuple[int, int]]:
    coord_map = circuit.get_final_qubit_coordinates()
    out: dict[int, tuple[int, int]] = {}
    for qubit, coord in coord_map.items():
        if len(coord) < 2:
            continue
        out[int(qubit)] = (int(coord[0]), int(coord[1]))
    return out


def infer_rotated_data_qubits(circuit: Any) -> list[int]:
    """
    Infer data-qubit indices from the current rotated Stim scaffold.

    In the generated rotated memory circuits used here, data qubits occupy
    odd-odd lattice coordinates.
    """
    coords = _sorted_qubit_coords(circuit)
    return sorted(
        qubit
        for qubit, (x, y) in coords.items()
        if (x % 2 == 1) and (y % 2 == 1)
    )


def infer_rotated_logical_string_qubits(circuit: Any) -> dict[str, list[int]]:
    """
    Infer canonical logical-string supports on the current rotated scaffold.

    For the current scaffold:
    - logical X: left-most column of data qubits
    - logical Z: top row of data qubits
    """
    coords = _sorted_qubit_coords(circuit)
    data_qubits = infer_rotated_data_qubits(circuit)
    if not data_qubits:
        raise ValueError("Could not infer any rotated data qubits from circuit coordinates.")

    min_x = min(coords[qubit][0] for qubit in data_qubits)
    min_y = min(coords[qubit][1] for qubit in data_qubits)

    logical_x = sorted(qubit for qubit in data_qubits if coords[qubit][0] == min_x)
    logical_z = sorted(qubit for qubit in data_qubits if coords[qubit][1] == min_y)
    return {"x": logical_x, "z": logical_z}


def _terminal_data_measurement_line(circuit: Any) -> tuple[int, str]:
    lines = str(circuit).splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if line.startswith("M ") or line.startswith("MX "):
            return index, line
    raise ValueError("Could not find terminal data measurement line in circuit text.")


def strip_terminal_data_measurement(circuit: Any) -> Any:
    stim_mod = _require_stim()
    line_index, _ = _terminal_data_measurement_line(circuit)
    prefix = "\n".join(str(circuit).splitlines()[:line_index])
    return stim_mod.Circuit(prefix)


def terminal_data_measurement_basis(circuit: Any) -> str:
    _, line = _terminal_data_measurement_line(circuit)
    if line.startswith("M "):
        return "z"
    if line.startswith("MX "):
        return "x"
    raise AssertionError(f"Unexpected terminal measurement line: {line!r}")


def terminal_data_measurement_qubits(circuit: Any) -> list[int]:
    _, line = _terminal_data_measurement_line(circuit)
    tokens = line.split()
    return [int(token) for token in tokens[1:]]


def _pauli_string_for_qubits(
    *,
    num_qubits: int,
    axis: str,
    qubits: list[int],
) -> Any:
    stim_mod = _require_stim()
    chars = ["I"] * int(num_qubits)
    pauli_char = axis.upper()
    for qubit in qubits:
        chars[int(qubit)] = pauli_char
    return stim_mod.PauliString("+" + "".join(chars))


def _mpp_target_text(axis: str, qubits: list[int]) -> str:
    pauli_char = axis.upper()
    return "*".join(f"{pauli_char}{qubit}" for qubit in qubits)


def build_single_axis_probe_circuit(
    circuit: Any,
    *,
    axis: str,
) -> Any:
    stim_mod = _require_stim()
    logical_strings = infer_rotated_logical_string_qubits(circuit)
    probe_prefix = strip_terminal_data_measurement(circuit)
    probe_suffix = stim_mod.Circuit(
        f"MPP {_mpp_target_text(axis, logical_strings[axis])}"
    )
    return probe_prefix + probe_suffix


def exact_logical_probe_expectation(
    circuit: Any,
    *,
    axis: str,
) -> int:
    stim_mod = _require_stim()
    logical_strings = infer_rotated_logical_string_qubits(circuit)
    pauli = _pauli_string_for_qubits(
        num_qubits=int(circuit.num_qubits),
        axis=axis,
        qubits=logical_strings[axis],
    )
    sim = stim_mod.TableauSimulator()
    sim.do(strip_terminal_data_measurement(circuit))
    return int(sim.peek_observable_expectation(pauli))


def sample_logical_probe_mean(
    circuit: Any,
    *,
    axis: str,
    shots: int,
    seed: int | None = None,
) -> float:
    if shots < 1:
        raise ValueError("shots must be >= 1")
    stim_mod = _require_stim()
    logical_strings = infer_rotated_logical_string_qubits(circuit)
    pauli = _pauli_string_for_qubits(
        num_qubits=int(circuit.num_qubits),
        axis=axis,
        qubits=logical_strings[axis],
    )
    prefix = strip_terminal_data_measurement(circuit)
    total = 0.0
    for shot in range(shots):
        sim_seed = None if seed is None else seed + shot
        sim = (
            stim_mod.TableauSimulator()
            if sim_seed is None
            else stim_mod.TableauSimulator(seed=sim_seed)
        )
        sim.do(prefix)
        total += float(sim.measure_observable(pauli))
    return total / float(shots)


@dataclass(frozen=True, slots=True)
class LogicalFrameStructure:
    variant: str
    source_basis: str
    terminal_data_measurement_basis: str
    directly_measured_logical_observable: str
    supervised_logical_error_axis: str
    data_qubits: list[int]
    logical_x_qubits: list[int]
    logical_z_qubits: list[int]
    measurement_line_matches_inferred_data_qubits: bool
    supports_joint_per_shot_logical_frame: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LogicalFrameAuditReport:
    structure: LogicalFrameStructure
    exact_probe_expectation: dict[str, int]
    expected_measurement_mean: dict[str, float]
    empirical_probe_mean: dict[str, float]
    deterministic_logical_observable: dict[str, bool]
    supports_true_per_shot_logical_class4: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure.to_dict(),
            "exact_probe_expectation": dict(self.exact_probe_expectation),
            "expected_measurement_mean": dict(self.expected_measurement_mean),
            "empirical_probe_mean": dict(self.empirical_probe_mean),
            "deterministic_logical_observable": dict(self.deterministic_logical_observable),
            "supports_true_per_shot_logical_class4": self.supports_true_per_shot_logical_class4,
            "notes": list(self.notes),
        }


def describe_logical_frame_structure(
    *,
    circuit: Any,
    basis: str,
    variant: str,
) -> LogicalFrameStructure:
    data_qubits = infer_rotated_data_qubits(circuit)
    logical_strings = infer_rotated_logical_string_qubits(circuit)
    measured_basis = terminal_data_measurement_basis(circuit)
    measured_qubits = terminal_data_measurement_qubits(circuit)
    supervised_error_axis = supervised_logical_error_axis_for_basis(basis)

    notes = [
        "The current rotated memory scaffold prepares a logical eigenstate in the measurement basis of the experiment.",
        f"basis={basis} directly measures the logical {measured_basis.upper()} operator on the terminal data readout.",
        f"Logical {supervised_error_axis.upper()} error supervision is indirect and inferred from flips of that measured logical observable.",
        "One basis-specific memory circuit does not provide a joint same-shot logical X/Z frame.",
    ]
    if variant == "xzzx":
        notes.append(
            "variant='xzzx' still reuses the rotated Stim scaffold, so this logical-frame structure is scaffold-level, not Willow-native XZZX."
        )

    return LogicalFrameStructure(
        variant=variant,
        source_basis=basis,
        terminal_data_measurement_basis=measured_basis,
        directly_measured_logical_observable=measured_basis,
        supervised_logical_error_axis=supervised_error_axis,
        data_qubits=data_qubits,
        logical_x_qubits=list(logical_strings["x"]),
        logical_z_qubits=list(logical_strings["z"]),
        measurement_line_matches_inferred_data_qubits=measured_qubits == data_qubits,
        supports_joint_per_shot_logical_frame=False,
        notes=notes,
    )


def audit_ideal_logical_frame_support(
    *,
    circuit: Any,
    basis: str,
    variant: str,
    sample_shots: int = 1024,
    seed: int | None = 12345,
) -> LogicalFrameAuditReport:
    structure = describe_logical_frame_structure(
        circuit=circuit,
        basis=basis,
        variant=variant,
    )

    exact_probe_expectation = {
        axis: exact_logical_probe_expectation(circuit, axis=axis)
        for axis in ("x", "z")
    }
    expected_measurement_mean = {
        axis: 0.5 * (1.0 - float(exact_probe_expectation[axis]))
        for axis in ("x", "z")
    }
    empirical_probe_mean = {
        "x": sample_logical_probe_mean(
            circuit,
            axis="x",
            shots=sample_shots,
            seed=seed,
        ),
        "z": sample_logical_probe_mean(
            circuit,
            axis="z",
            shots=sample_shots,
            seed=None if seed is None else seed + 1,
        ),
    }
    deterministic_logical_observable = {
        axis: abs(exact_probe_expectation[axis]) == 1
        for axis in ("x", "z")
    }

    notes = [
        "exact_probe_expectation is computed before the terminal destructive data measurement.",
        "A value of 0 means the logical observable is not fixed by the prepared single-basis memory state.",
        "A deterministic same-shot logical class4 target would require both logical observables to be available without collapsing one another.",
    ]
    if not all(deterministic_logical_observable.values()):
        notes.append(
            "At least one logical observable is structurally random in the current single-basis scaffold, so true per-shot logical_class4 is unsupported."
        )

    return LogicalFrameAuditReport(
        structure=structure,
        exact_probe_expectation=exact_probe_expectation,
        expected_measurement_mean=expected_measurement_mean,
        empirical_probe_mean=empirical_probe_mean,
        deterministic_logical_observable=deterministic_logical_observable,
        supports_true_per_shot_logical_class4=all(deterministic_logical_observable.values()),
        notes=notes,
    )
