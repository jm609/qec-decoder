from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import datetime as dt
import json

from logical_targets import logical_axis_target_name_for_basis


SCHEMA_VERSION = "dual_axis_manifest.v1"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=json_default),
        encoding="utf-8",
    )


def resolve_manifest_family_dir(manifest_path: Path, raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw.resolve()

    candidate = (manifest_path.parent / raw).resolve()
    if candidate.exists():
        return candidate

    return raw.resolve()


def expected_target_name_from_metadata(metadata: dict[str, Any]) -> str:
    targets = metadata.get("targets", {}) if isinstance(metadata, dict) else {}
    if isinstance(targets, dict):
        target_name = targets.get("logical_axis_flip_name")
        if isinstance(target_name, str) and target_name:
            return target_name
    basis = str(metadata.get("circuit", {}).get("basis"))
    return logical_axis_target_name_for_basis(basis)


def family_stage_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    return str(metadata.get("family")), str(metadata.get("stage"))


def load_family_metadata(family_dir: Path) -> dict[str, Any]:
    metadata_path = family_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in family_dir={family_dir}")
    return read_json(metadata_path)


def validate_manifest_role(manifest_path: Path, manifest: dict[str, Any], *, expected_basis: str) -> None:
    basis = str(manifest.get("basis"))
    if basis != expected_basis:
        raise ValueError(
            f"Manifest basis mismatch for {manifest_path}: expected {expected_basis!r}, got {basis!r}"
        )
    family_dirs = manifest.get("family_dirs")
    if not isinstance(family_dirs, dict) or not family_dirs:
        raise ValueError(f"Manifest {manifest_path} is missing a non-empty family_dirs mapping")


def validate_shared_experiment_shape(
    x_manifest: dict[str, Any],
    z_manifest: dict[str, Any],
    *,
    x_manifest_path: Path,
    z_manifest_path: Path,
) -> None:
    for key in ("distance", "rounds", "variant", "shots"):
        x_value = x_manifest.get(key)
        z_value = z_manifest.get(key)
        if x_value != z_value:
            raise ValueError(
                f"Manifest mismatch for key={key!r}: "
                f"{x_manifest_path} has {x_value!r}, {z_manifest_path} has {z_value!r}"
            )

    x_families = set((x_manifest.get("family_dirs") or {}).keys())
    z_families = set((z_manifest.get("family_dirs") or {}).keys())
    if x_families != z_families:
        raise ValueError(
            f"Family set mismatch between manifests. x_only={sorted(x_families - z_families)}, "
            f"z_only={sorted(z_families - x_families)}"
        )


def build_pair_entry(
    *,
    family: str,
    x_manifest_path: Path,
    z_manifest_path: Path,
    x_family_dir: Path,
    z_family_dir: Path,
    x_metadata: dict[str, Any],
    z_metadata: dict[str, Any],
) -> dict[str, Any]:
    x_family, x_stage = family_stage_from_metadata(x_metadata)
    z_family, z_stage = family_stage_from_metadata(z_metadata)
    if x_family != family or z_family != family:
        raise ValueError(
            f"Family metadata mismatch while pairing {family!r}: x={x_family!r}, z={z_family!r}"
        )
    if x_stage != z_stage:
        raise ValueError(
            f"Stage mismatch while pairing {family!r}: x={x_stage!r}, z={z_stage!r}"
        )

    x_circuit = x_metadata.get("circuit", {})
    z_circuit = z_metadata.get("circuit", {})
    if str(x_circuit.get("basis")) != "x" or str(z_circuit.get("basis")) != "z":
        raise ValueError(
            f"Basis mismatch while pairing {family!r}: x_basis={x_circuit.get('basis')!r}, "
            f"z_basis={z_circuit.get('basis')!r}"
        )

    for key in ("distance", "rounds", "variant", "num_observables"):
        if x_circuit.get(key) != z_circuit.get(key):
            raise ValueError(
                f"Circuit mismatch for family={family!r}, key={key!r}: "
                f"x={x_circuit.get(key)!r}, z={z_circuit.get(key)!r}"
            )

    x_target_name = expected_target_name_from_metadata(x_metadata)
    z_target_name = expected_target_name_from_metadata(z_metadata)
    if x_target_name != "logical_z_flip":
        raise ValueError(
            f"Expected basis-x manifest to supervise logical_z_flip, got {x_target_name!r} "
            f"for family={family!r}"
        )
    if z_target_name != "logical_x_flip":
        raise ValueError(
            f"Expected basis-z manifest to supervise logical_x_flip, got {z_target_name!r} "
            f"for family={family!r}"
        )

    return {
        "family": family,
        "stage": x_stage,
        "shared_circuit": {
            "distance": x_circuit.get("distance"),
            "rounds": x_circuit.get("rounds"),
            "variant": x_circuit.get("variant"),
            "num_observables_per_basis": x_circuit.get("num_observables"),
        },
        "logical_x_source": {
            "basis": "z",
            "family_dir": z_family_dir.as_posix(),
            "manifest_path": z_manifest_path.as_posix(),
            "metadata_json": (z_family_dir / "metadata.json").as_posix(),
            "samples_npz": (z_family_dir / "samples.npz").as_posix(),
            "target_key": "logical_axis_flip",
            "target_name": "logical_x_flip",
            "target_name_from_metadata": z_target_name,
        },
        "logical_z_source": {
            "basis": "x",
            "family_dir": x_family_dir.as_posix(),
            "manifest_path": x_manifest_path.as_posix(),
            "metadata_json": (x_family_dir / "metadata.json").as_posix(),
            "samples_npz": (x_family_dir / "samples.npz").as_posix(),
            "target_key": "logical_axis_flip",
            "target_name": "logical_z_flip",
            "target_name_from_metadata": x_target_name,
        },
    }


def build_dual_axis_manifest(
    *,
    x_manifest_path: Path,
    z_manifest_path: Path,
) -> dict[str, Any]:
    x_manifest = read_json(x_manifest_path)
    z_manifest = read_json(z_manifest_path)

    validate_manifest_role(x_manifest_path, x_manifest, expected_basis="x")
    validate_manifest_role(z_manifest_path, z_manifest, expected_basis="z")
    validate_shared_experiment_shape(
        x_manifest,
        z_manifest,
        x_manifest_path=x_manifest_path,
        z_manifest_path=z_manifest_path,
    )

    family_pairs: dict[str, Any] = {}
    for family in sorted((x_manifest.get("family_dirs") or {}).keys()):
        x_family_dir = resolve_manifest_family_dir(x_manifest_path, x_manifest["family_dirs"][family])
        z_family_dir = resolve_manifest_family_dir(z_manifest_path, z_manifest["family_dirs"][family])
        x_metadata = load_family_metadata(x_family_dir)
        z_metadata = load_family_metadata(z_family_dir)
        family_pairs[family] = build_pair_entry(
            family=family,
            x_manifest_path=x_manifest_path,
            z_manifest_path=z_manifest_path,
            x_family_dir=x_family_dir,
            z_family_dir=z_family_dir,
            x_metadata=x_metadata,
            z_metadata=z_metadata,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "x_manifest_path": x_manifest_path.as_posix(),
        "z_manifest_path": z_manifest_path.as_posix(),
        "shared_experiment": {
            "distance": x_manifest.get("distance"),
            "rounds": x_manifest.get("rounds"),
            "variant": x_manifest.get("variant"),
            "shots": x_manifest.get("shots"),
            "families": sorted(family_pairs.keys()),
        },
        "logical_axes": {
            "logical_x_flip_source_basis": "z",
            "logical_z_flip_source_basis": "x",
            "paired_target_key": "logical_axis_flip",
        },
        "supports_per_shot_logical_class4": False,
        "notes": [
            "This manifest pairs basis-z memory datasets supervising logical_x_flip with basis-x memory datasets supervising logical_z_flip.",
            "The paired datasets come from separate sampled shots and therefore do not define per-shot logical_class4 labels.",
            "Use this manifest for axis-wise training and evaluation while the project migrates toward full logical-class supervision.",
        ],
        "family_pairs": family_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair basis-x and basis-z sample_dataset manifests into a dual-axis manifest."
    )
    parser.add_argument("--x-manifest", type=Path, required=True)
    parser.add_argument("--z-manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dual_axis_manifest(
        x_manifest_path=args.x_manifest,
        z_manifest_path=args.z_manifest,
    )
    write_json(args.out_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
