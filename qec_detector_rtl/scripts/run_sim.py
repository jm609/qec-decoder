"""Run the fixed P2 bench locally or on the course server; standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "rtl/measurement_to_detection.v",
    "tb/tb_measurement_to_detection.sv",
    "scripts/run_sim.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path, width: int) -> list[str]:
    lines = path.read_text(encoding="ascii").splitlines()
    if any(len(line) != width or set(line) - {"0", "1"} for line in lines):
        raise ValueError(f"Invalid {width}-bit records: {path}")
    return lines


def validate_vectors(vectors: Path) -> dict:
    vectors = vectors.resolve()
    manifest = json.loads((vectors / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"], manifest["status"], manifest["measurement_count"],
        manifest["detector_count"], manifest["append_observables"]) != (1, "complete", 33, 24, False):
        raise ValueError("Unsupported vector manifest")
    if manifest["circuit_config"] != {"distance": 3, "rounds": 3, "basis": "z", "variant": "stim_rotated"}:
        raise ValueError("The RTL supports only the fixed d3/r3/z circuit")
    for name, checksum in manifest["files_sha256"].items():
        path = (vectors / name).resolve()
        if path.parent != vectors or digest(path) != checksum:
            raise ValueError(f"Vector checksum or path mismatch: {name}")
    if [(case["name"], case["shots"]) for case in manifest["cases"]] != [
        ("noiseless", 256), ("single_flip", 34), ("random_bits", 1024), ("si1000", 4096),
    ]:
        raise ValueError("The P1 case set changed")
    for case in manifest["cases"]:
        for key, width in [("measurements", 33), ("expected_detectors", 24)]:
            name = case[key]
            if name not in manifest["files_sha256"]:
                raise ValueError(f"Missing checksum: {name}")
            if len(read_records(vectors / name, width)) != case["shots"]:
                raise ValueError(f"Shot count mismatch: {name}")
    return manifest


def run_sim(vectors: Path, out_root: Path, simulator: str, fsdb: bool = False) -> dict:
    vectors = vectors.resolve()
    manifest = validate_vectors(vectors)
    if simulator not in {"iverilog", "vcs"} or (fsdb and simulator != "vcs"):
        raise ValueError("FSDB requires the VCS backend")
    compiler = shutil.which(simulator)
    runtime = shutil.which("vvp") if simulator == "iverilog" else None
    if compiler is None or (simulator == "iverilog" and runtime is None):
        raise FileNotFoundError(f"Simulator is not on PATH: {simulator}")
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    sim_dir = out_root / "sim"
    sim_dir.mkdir()
    report = {
        "schema_version": 1, "stage": "P2", "status": "running", "simulator": simulator,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "manifest_sha256": digest(vectors / "manifest.json"),
        "source_sha256": {name: digest(SOURCE_ROOT / name) for name in SOURCE_FILES},
        "commands": [], "cases": [], "fsdb_requested": fsdb,
        "scope": "Basic always-ready bench; P3 stress/timing and synthesis are not verified",
    }

    def execute(label: str, command: list[str]) -> subprocess.CompletedProcess:
        entry = {"label": label, "command": command}
        report["commands"].append(entry)
        try:
            result = subprocess.run(
                command, cwd=sim_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            entry["error"] = str(error)
            raise
        entry["returncode"] = result.returncode
        (sim_dir / f"{label}.log").write_text(
            "command=" + repr(command) + f"\nreturncode={result.returncode}\n"
            + result.stdout + result.stderr, encoding="utf-8",
        )
        return result

    try:
        executable = sim_dir / ("detector.vvp" if simulator == "iverilog" else "simv")
        sources = []
        for name in SOURCE_FILES[:2]:
            snapshot = sim_dir / Path(name).name
            shutil.copyfile(SOURCE_ROOT / name, snapshot)
            sources.append(snapshot.name)
        if simulator == "iverilog":
            command = [compiler, "-v", "-g2012", "-Wall", "-s", "tb_measurement_to_detection", "-o", executable.name, *sources]
        else:
            command = [compiler, "-full64", "-sverilog", "-top", "tb_measurement_to_detection", "-o", executable.name]
            if fsdb:
                command += ["-debug_access+all", "+define+DUMP_FSDB"]
            command += sources
        compiled = execute("compile", command)
        if compiled.returncode != 0:
            raise RuntimeError(f"Compiler failed ({compiled.returncode}); see compile.log")
        run_command = [runtime, str(executable)] if runtime else [str(executable)]
        for case in manifest["cases"]:
            name = case["name"]
            actual = sim_dir / f"{name}.actual.01"
            command = run_command + [
                f"+measurements={vectors / case['measurements']}",
                f"+expected={vectors / case['expected_detectors']}",
                f"+actual={actual}", f"+shots={case['shots']}",
            ]
            wave = None
            if name == "single_flip":
                wave = sim_dir / ("single_flip.fsdb" if fsdb else "single_flip.vcd")
                command.append(f"+{'fsdb' if fsdb else 'vcd'}={wave}")
            result = execute(name, command)
            if result.returncode != 0 or f"PASS shots={case['shots']} " not in result.stdout:
                raise RuntimeError(f"{name}: simulation failed; see {name}.log")
            observed = read_records(actual, 24)
            expected = read_records(vectors / case["expected_detectors"], 24)
            if observed != expected:
                raise RuntimeError(f"{name}: saved RTL outputs differ from the frozen oracle")
            if wave is not None and (not wave.is_file() or wave.stat().st_size == 0):
                raise RuntimeError("The requested waveform was not generated")
            report["cases"].append({
                "name": name, "shots": len(observed), "detector_bits": len(observed) * 24,
                "mismatches": 0, "actual_sha256": digest(actual),
            })
        case = manifest["cases"][0]
        negative = execute("expected_failure", run_command + [
            f"+measurements={vectors / case['measurements']}",
            f"+expected={vectors / case['expected_detectors']}",
            f"+actual={sim_dir / 'expected_failure.actual.01'}",
            f"+shots={case['shots']}", "+force_failure",
        ])
        if negative.returncode == 0 or "MISMATCH" not in negative.stdout + negative.stderr:
            raise RuntimeError("The bench did not reject an intentionally incorrect expected bit")
        report["negative_check_returncode"] = negative.returncode
        if simulator == "iverilog":
            for label, tool in [("compiler_version", compiler), ("runtime_version", runtime)]:
                version = execute(label, [tool, "-V"])
                if version.returncode != 0:
                    raise RuntimeError(f"{label} failed")
                report[label] = (version.stdout + version.stderr).splitlines()[0]
        else:
            report["tool_version_source"] = "VCS banner in sim/compile.log"
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (out_root / "verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--simulator", choices=("vcs", "iverilog"), default="vcs")
    parser.add_argument("--fsdb", action="store_true", help="Requires the course VCS/Verdi dumper environment")
    args = parser.parse_args()
    report = run_sim(args.vectors, args.out_root, args.simulator, args.fsdb)
    print(f"PASS {report['simulator']}: {sum(case['shots'] for case in report['cases'])} shots")
    print(args.out_root / "verification.json")


if __name__ == "__main__":
    main()
