from __future__ import annotations

"""
run_dual_axis_pymatching.py

Run paired axis-wise PyMatching evaluations from a dual_axis_manifest.

Current scope
-------------
- logical_x_flip is evaluated from the basis-z source datasets.
- logical_z_flip is evaluated from the basis-x source datasets.
- Saves one manifest-level PyMatching evaluation per logical axis plus a
  combined dual-axis summary.
"""

from pathlib import Path
from typing import Any
import argparse
import datetime as dt
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoders import baseline_pymatching as pym_base
from tools import run_dual_axis_experiment as dual_axis_nn


SCHEMA_VERSION = "dual_axis_pymatching.v1"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _family_metrics_rollup(
    *,
    axis_name: str,
    axis_manifest_path: Path,
    raw_result: dict[str, Any],
    reference_train_families: set[str],
) -> dict[str, Any]:
    by_family = raw_result.get("by_family", {})
    per_family: list[dict[str, Any]] = []
    holdout_fer: list[float] = []
    holdout_ler: list[float] = []
    for family, result in by_family.items():
        metrics = result.get("metrics", {})
        frame_error_rate = metrics.get("frame_error_rate")
        label_error_rate = metrics.get("label_error_rate")
        seen_in_reference_training = family in reference_train_families
        if not seen_in_reference_training:
            if frame_error_rate is not None:
                holdout_fer.append(float(frame_error_rate))
            if label_error_rate is not None:
                holdout_ler.append(float(label_error_rate))
        per_family.append(
            {
                "family": family,
                "stage": result.get("stage"),
                "seen_in_reference_training": seen_in_reference_training,
                "frame_error_rate": frame_error_rate,
                "accuracy": metrics.get("accuracy"),
                "label_error_rate": label_error_rate,
                "estimated_ler_per_cycle": metrics.get("estimated_ler_per_cycle"),
            }
        )

    return {
        "logical_axis_name": axis_name,
        "axis_manifest_path": axis_manifest_path.as_posix(),
        "families_evaluated": list(raw_result.get("summary", {}).get("families_evaluated", [])),
        "reference_train_families": sorted(reference_train_families),
        "holdout_families": [
            item["family"] for item in per_family if not item["seen_in_reference_training"]
        ],
        "holdout_mean_frame_error_rate": (
            float(sum(holdout_fer) / len(holdout_fer)) if holdout_fer else None
        ),
        "holdout_mean_label_error_rate": (
            float(sum(holdout_ler) / len(holdout_ler)) if holdout_ler else None
        ),
        "per_family": per_family,
    }


def run_dual_axis_pymatching(
    *,
    dual_axis_manifest_path: Path,
    eval_families: list[str] | None,
    reference_train_families: list[str] | None,
    out_dir: Path,
    max_shots: int | None,
    allow_circuit_fallback: bool,
) -> dict[str, Any]:
    dual_axis_manifest = dual_axis_nn._read_json(dual_axis_manifest_path)
    family_pairs = dual_axis_nn._validate_dual_axis_manifest(
        dual_axis_manifest,
        dual_axis_manifest_path,
    )
    available_families = sorted(family_pairs.keys())

    requested_eval_families = list(eval_families) if eval_families is not None else available_families
    missing_eval = [family for family in requested_eval_families if family not in family_pairs]
    if missing_eval:
        raise KeyError(
            f"Requested eval_families are missing from dual-axis manifest: {missing_eval}. "
            f"Available: {available_families}"
        )

    reference_train = set(reference_train_families or [])
    missing_reference = [family for family in reference_train if family not in family_pairs]
    if missing_reference:
        raise KeyError(
            f"Requested reference_train_families are missing from dual-axis manifest: {missing_reference}. "
            f"Available: {available_families}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    axis_manifest_dir = out_dir / "axis_manifests"
    axis_manifest_dir.mkdir(parents=True, exist_ok=True)

    axis_rollups: list[dict[str, Any]] = []
    axis_result_paths: dict[str, str] = {}
    for axis_name in ("logical_x_flip", "logical_z_flip"):
        axis_manifest_path = axis_manifest_dir / f"{axis_name}.json"
        dual_axis_nn._build_axis_manifest(
            dual_axis_manifest_path=dual_axis_manifest_path,
            dual_axis_manifest=dual_axis_manifest,
            family_pairs=family_pairs,
            axis_name=axis_name,
            out_path=axis_manifest_path,
        )
        raw_result = pym_base.evaluate_manifest(
            axis_manifest_path,
            families=requested_eval_families,
            max_shots=max_shots,
            allow_circuit_fallback=allow_circuit_fallback,
        )
        axis_out_dir = out_dir / axis_name
        axis_out_dir.mkdir(parents=True, exist_ok=True)
        axis_result_path = axis_out_dir / "pymatching_eval.json"
        dual_axis_nn._write_json(axis_result_path, raw_result)
        axis_result_paths[axis_name] = axis_result_path.as_posix()
        axis_rollups.append(
            _family_metrics_rollup(
                axis_name=axis_name,
                axis_manifest_path=axis_manifest_path,
                raw_result=raw_result,
                reference_train_families=reference_train,
            )
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "decoder": "pymatching",
        "dual_axis_manifest_path": dual_axis_manifest_path.as_posix(),
        "eval_families": requested_eval_families,
        "reference_train_families": sorted(reference_train),
        "supports_true_per_shot_logical_class4": False,
        "notes": [
            "This summary evaluates PyMatching separately on logical_x_flip and logical_z_flip axis manifests.",
            "The reference_train_families field is only for comparison against neural experiments; PyMatching itself does not train.",
            "Because the paired x/z basis datasets come from separate sampled shots, this summary is axis-wise and not a true per-shot logical_class4 baseline.",
        ],
        "axes": axis_rollups,
        "axis_manifests": {
            axis_name: (axis_manifest_dir / f"{axis_name}.json").as_posix()
            for axis_name in ("logical_x_flip", "logical_z_flip")
        },
        "axis_result_paths": axis_result_paths,
    }
    dual_axis_nn._write_json(out_dir / "dual_axis_pymatching_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired axis-wise PyMatching evaluations from a dual_axis_manifest."
    )
    parser.add_argument("--dual-axis-manifest", type=Path, required=True)
    parser.add_argument("--eval-families", nargs="+", default=None)
    parser.add_argument("--reference-train-families", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--allow-circuit-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dual_axis_pymatching(
        dual_axis_manifest_path=args.dual_axis_manifest,
        eval_families=(list(args.eval_families) if args.eval_families is not None else None),
        reference_train_families=(
            list(args.reference_train_families) if args.reference_train_families is not None else None
        ),
        out_dir=args.out_dir,
        max_shots=args.max_shots,
        allow_circuit_fallback=args.allow_circuit_fallback,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=dual_axis_nn._json_default))


if __name__ == "__main__":
    main()
