from __future__ import annotations

"""
run_dual_axis_experiment.py

Run paired axis-wise experiments from a dual_axis_manifest.

Current scope
-------------
- logical_x_flip is trained/evaluated from the basis-z source datasets.
- logical_z_flip is trained/evaluated from the basis-x source datasets.
- Each axis uses the research_noise_aware_3d decoder backend.

Important limitation
--------------------
The underlying x/z basis datasets come from separate sampled shots.
This runner therefore executes two aligned binary experiments, not a true
per-shot logical_class4 experiment.
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

from decoders import research_noise_aware_3d as research


SCHEMA_VERSION = "dual_axis_experiment.v1"
AXIS_TO_SOURCE_KEY = {
    "logical_x_flip": "logical_x_source",
    "logical_z_flip": "logical_z_source",
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=_json_default),
        encoding="utf-8",
    )


def _validate_dual_axis_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    schema_version = str(manifest.get("schema_version"))
    if schema_version != "dual_axis_manifest.v1":
        raise ValueError(
            f"Expected dual_axis_manifest.v1, got {schema_version!r} for {manifest_path}"
        )
    family_pairs = manifest.get("family_pairs")
    if not isinstance(family_pairs, dict) or not family_pairs:
        raise ValueError(f"dual-axis manifest has no family_pairs: {manifest_path}")
    return family_pairs


def _axis_entry_from_pair(pair: dict[str, Any], axis_name: str) -> dict[str, Any]:
    if axis_name not in AXIS_TO_SOURCE_KEY:
        raise KeyError(f"Unsupported logical axis: {axis_name!r}")
    source = pair.get(AXIS_TO_SOURCE_KEY[axis_name])
    if not isinstance(source, dict):
        raise ValueError(
            f"Missing {AXIS_TO_SOURCE_KEY[axis_name]!r} entry for family={pair.get('family')!r}"
        )
    return source


def _build_axis_manifest(
    *,
    dual_axis_manifest_path: Path,
    dual_axis_manifest: dict[str, Any],
    family_pairs: dict[str, Any],
    axis_name: str,
    out_path: Path,
) -> dict[str, Any]:
    family_dirs: dict[str, str] = {}
    source_basis: str | None = None
    for family, pair in family_pairs.items():
        source = _axis_entry_from_pair(pair, axis_name)
        family_dirs[family] = str(source["family_dir"])
        basis = str(source.get("basis"))
        if source_basis is None:
            source_basis = basis
        elif basis != source_basis:
            raise ValueError(
                f"Axis {axis_name!r} mixes basis sources unexpectedly: {source_basis!r} vs {basis!r}"
            )

    shared = dual_axis_manifest.get("shared_experiment", {})
    axis_manifest = {
        "schema_version": "axis_manifest.v1",
        "created_at_utc": _utc_now_iso(),
        "dual_axis_manifest_path": dual_axis_manifest_path.as_posix(),
        "logical_axis_name": axis_name,
        "source_basis": source_basis,
        "distance": shared.get("distance"),
        "rounds": shared.get("rounds"),
        "basis": source_basis,
        "variant": shared.get("variant"),
        "shots": shared.get("shots"),
        "requested_families": sorted(family_dirs.keys()),
        "family_dirs": family_dirs,
        "notes": [
            "Generated from dual_axis_manifest.v1 for axis-wise experiments.",
            f"All family_dirs in this manifest supervise {axis_name}.",
        ],
    }
    _write_json(out_path, axis_manifest)
    return axis_manifest


def _summarise_axis_experiment(
    *,
    axis_name: str,
    axis_manifest_path: Path,
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    evaluation = experiment_summary.get("evaluation", {})
    per_family = evaluation.get("per_family", [])
    holdout = [entry for entry in per_family if not bool(entry.get("seen_in_training"))]

    holdout_mean_accuracy = None
    holdout_mean_auroc = None
    if holdout:
        accuracies = [float(entry["accuracy"]) for entry in holdout if entry.get("accuracy") is not None]
        aurocs = [float(entry["auroc"]) for entry in holdout if entry.get("auroc") is not None]
        holdout_mean_accuracy = float(sum(accuracies) / len(accuracies)) if accuracies else None
        holdout_mean_auroc = float(sum(aurocs) / len(aurocs)) if aurocs else None

    return {
        "logical_axis_name": axis_name,
        "axis_manifest_path": axis_manifest_path.as_posix(),
        "train_families": list(experiment_summary.get("training", {}).get("train_families", [])),
        "eval_families": list(evaluation.get("eval_families", [])),
        "holdout_families": list(evaluation.get("holdout_families", [])),
        "holdout_mean_accuracy": holdout_mean_accuracy,
        "holdout_mean_auroc": holdout_mean_auroc,
        "experiment_summary_path": (axis_manifest_path.parent.parent / axis_name / "experiment_summary.json").as_posix(),
    }


def run_dual_axis_experiment(
    *,
    dual_axis_manifest_path: Path,
    train_families: list[str],
    eval_families: list[str] | None,
    out_dir: Path,
    fill_value: float,
    max_shots: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    num_blocks: int,
    dense_hidden_dim: int,
    device_arg: str,
) -> dict[str, Any]:
    dual_axis_manifest = _read_json(dual_axis_manifest_path)
    family_pairs = _validate_dual_axis_manifest(dual_axis_manifest, dual_axis_manifest_path)

    available_families = sorted(family_pairs.keys())
    missing_train = [family for family in train_families if family not in family_pairs]
    if missing_train:
        raise KeyError(
            f"Requested train_families are missing from dual-axis manifest: {missing_train}. "
            f"Available: {available_families}"
        )
    if eval_families is not None:
        missing_eval = [family for family in eval_families if family not in family_pairs]
        if missing_eval:
            raise KeyError(
                f"Requested eval_families are missing from dual-axis manifest: {missing_eval}. "
                f"Available: {available_families}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    axis_manifest_dir = out_dir / "axis_manifests"
    axis_manifest_dir.mkdir(parents=True, exist_ok=True)

    axis_summaries: dict[str, Any] = {}
    axis_rollups: list[dict[str, Any]] = []
    for axis_name in ("logical_x_flip", "logical_z_flip"):
        axis_manifest_path = axis_manifest_dir / f"{axis_name}.json"
        _build_axis_manifest(
            dual_axis_manifest_path=dual_axis_manifest_path,
            dual_axis_manifest=dual_axis_manifest,
            family_pairs=family_pairs,
            axis_name=axis_name,
            out_path=axis_manifest_path,
        )
        axis_out_dir = out_dir / axis_name
        axis_summary = research.run_manifest_experiment(
            manifest=axis_manifest_path,
            train_families=list(train_families),
            eval_families=(list(eval_families) if eval_families is not None else None),
            out_dir=axis_out_dir,
            fill_value=fill_value,
            max_shots=max_shots,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            dense_hidden_dim=dense_hidden_dim,
            device_arg=device_arg,
        )
        axis_summaries[axis_name] = axis_summary
        axis_rollups.append(
            _summarise_axis_experiment(
                axis_name=axis_name,
                axis_manifest_path=axis_manifest_path,
                experiment_summary=axis_summary,
            )
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "decoder_backend": "research_noise_aware_3d",
        "dual_axis_manifest_path": dual_axis_manifest_path.as_posix(),
        "supports_true_per_shot_logical_class4": False,
        "notes": [
            "This runner executes two aligned axis-wise experiments.",
            "logical_x_flip uses basis-z source datasets and logical_z_flip uses basis-x source datasets.",
            "Because the two axes come from separate sampled shots, this summary does not report true per-shot logical_class4 metrics.",
        ],
        "train_families": list(train_families),
        "eval_families": list(eval_families) if eval_families is not None else available_families,
        "axes": axis_rollups,
        "axis_experiment_dirs": {
            axis_name: (out_dir / axis_name).as_posix()
            for axis_name in ("logical_x_flip", "logical_z_flip")
        },
        "axis_manifests": {
            axis_name: (axis_manifest_dir / f"{axis_name}.json").as_posix()
            for axis_name in ("logical_x_flip", "logical_z_flip")
        },
    }
    _write_json(out_dir / "dual_axis_experiment_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired axis-wise experiments from a dual_axis_manifest."
    )
    parser.add_argument("--dual-axis-manifest", type=Path, required=True)
    parser.add_argument("--train-families", nargs="+", required=True)
    parser.add_argument("--eval-families", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fill-value", type=float, default=-0.5)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dense-hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dual_axis_experiment(
        dual_axis_manifest_path=args.dual_axis_manifest,
        train_families=list(args.train_families),
        eval_families=(list(args.eval_families) if args.eval_families is not None else None),
        out_dir=args.out_dir,
        fill_value=args.fill_value,
        max_shots=args.max_shots,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dense_hidden_dim=args.dense_hidden_dim,
        device_arg=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
