from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DECODER_DIR = REPO_ROOT / "decoders"
if str(DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_DIR))

import syndrome_edit_predecoder as sedp  # noqa: E402


DEFAULT_ANALYSIS_SPECS = (
    ("stage_a_si1000", "val"),
    ("stage_b_local", "val"),
    ("stage_c_corr", "full"),
)
DEFAULT_MARGIN_GRID = (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0)
DEFAULT_HARM_LOGIT_THRESHOLD_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, path_text = text.split("=", 1)
        return label.strip(), Path(path_text)
    path = Path(text)
    return path.parent.name, path


def _parse_analysis_spec(text: str) -> tuple[str, str]:
    if ":" not in text:
        return text, "full"
    family, split = text.split(":", 1)
    split = split.strip().lower()
    if split not in {"train", "val", "test", "full"}:
        raise ValueError(f"Unsupported split {split!r}; expected train, val, test, or full")
    return family.strip(), split


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.5)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def _row_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float32).reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)]
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)) if selected.size else None,
        "quantiles": _finite_quantiles(selected),
    }


def _component_logits_for_bundle(
    *,
    selector: torch.nn.Module,
    bundle: sedp.SelectorCandidateBundle,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray] | None:
    component_logits = getattr(selector, "component_logits", None)
    if not callable(component_logits):
        return None
    if bundle.shot_features.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    dataset = sedp.common.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(bundle.shot_features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(bundle.candidate_features, dtype=np.float32)),
    )
    loader = sedp.common._make_loader(dataset, batch_size=batch_size, shuffle=False)
    first_chunks: list[np.ndarray] = []
    second_chunks: list[np.ndarray] = []
    selector.eval()
    with torch.no_grad():
        for shot_feat, cand_feat in loader:
            first, second = component_logits(shot_feat.to(device), cand_feat.to(device))
            first_chunks.append(first.detach().cpu().numpy())
            second_chunks.append(second.detach().cpu().numpy())
    return (
        np.asarray(np.concatenate(first_chunks, axis=0), dtype=np.float32),
        np.asarray(np.concatenate(second_chunks, axis=0), dtype=np.float32),
    )


def _load_candidate_local_motif(payload: dict[str, Any]) -> sedp.LocalMotifVocabulary | None:
    raw = payload.get("candidate_local_motif_vocabulary")
    if not isinstance(raw, dict):
        return None
    return sedp.LocalMotifVocabulary(
        offset_patterns=tuple(
            tuple(tuple(int(v) for v in offset) for offset in pattern)
            for pattern in raw["offset_patterns"]
        ),
        counts=tuple(int(x) for x in raw["counts"]),
        detector_count=int(raw.get("detector_count", 0) or 0),
    )


def _load_candidate_motif(payload: dict[str, Any]) -> sedp.MotifVocabulary | None:
    raw = payload.get("candidate_motif_vocabulary")
    if not isinstance(raw, dict):
        return None
    return sedp.MotifVocabulary(
        mask_table=np.asarray(raw["mask_table"], dtype=np.uint8),
        detector_index_lists=tuple(
            tuple(int(x) for x in row) for row in raw["detector_index_lists"]
        ),
        counts=tuple(int(x) for x in raw["counts"]),
        detector_count=int(np.asarray(raw["mask_table"]).shape[1]),
    )


def _policy_specs_from_checkpoint(checkpoint: dict[str, Any]) -> list[sedp.CandidatePolicySpec]:
    inference = dict(checkpoint.get("inference", {}))
    raw_specs = list(inference.get("candidate_policy_grid") or [])
    if not raw_specs:
        payload = checkpoint.get("candidate_selector") or {}
        summary = dict(payload.get("training_summary") or {})
        raw_specs = list(summary.get("policy_specs") or [])
    return [
        sedp.CandidatePolicySpec(
            needs_edit_threshold=float(item["needs_edit_threshold"]),
            edit_threshold=float(item["edit_threshold"]),
            max_predicted_edit_weight=int(item["max_predicted_edit_weight"]),
        )
        for item in raw_specs
    ]


def _subset_for_split(
    *,
    entry: sedp.PreparedEditFamily,
    split: str,
    checkpoint: dict[str, Any],
    family: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if split == "full":
        return sedp._subset_family(entry, np.arange(entry.x.shape[0], dtype=np.int64))
    config = dict(checkpoint.get("training_config") or {})
    train_families = list(config.get("train_families") or [])
    offset = int(train_families.index(family)) if family in train_families else 0
    seed = int(config.get("seed", 0) or 0)
    bundle = sedp._build_split_bundle(
        num_shots=int(entry.x.shape[0]),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed + offset),
    )
    indices = getattr(bundle, split)
    return sedp._subset_family(entry, indices)


def _gap_summary(
    *,
    bundle: sedp.SelectorCandidateBundle,
    logits: np.ndarray,
    margin_grid: tuple[float, ...],
) -> dict[str, Any]:
    utility = np.asarray(logits, dtype=np.float32).reshape(-1)
    weights = np.asarray(bundle.candidate_edit_weight, dtype=np.int16).reshape(-1)
    scores = np.asarray(bundle.target_scores, dtype=np.float32).reshape(-1)
    best_nonzero_gap: list[float] = []
    best_positive_gap: list[float] = []
    best_negative_gap: list[float] = []
    positive_gap_counts = {str(float(margin)): 0 for margin in margin_grid}
    negative_gap_counts = {str(float(margin)): 0 for margin in margin_grid}
    nonzero_gap_counts = {str(float(margin)): 0 for margin in margin_grid}
    shots_with_positive = 0
    shots_with_negative = 0
    shots_with_nonzero = 0

    for group_slice in sedp._selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        group_weights = weights[rows]
        identity_local = np.flatnonzero(group_weights == 0)
        if identity_local.size == 0:
            continue
        identity_score = float(utility[int(rows[int(identity_local[0])])])
        nonzero_local = np.flatnonzero(group_weights > 0)
        positive_local = np.flatnonzero(np.logical_and(group_weights > 0, scores[rows] > 0.0))
        negative_local = np.flatnonzero(np.logical_and(group_weights > 0, scores[rows] < 0.0))
        if nonzero_local.size:
            shots_with_nonzero += 1
            gap = float(np.max(utility[rows[nonzero_local]]) - identity_score)
            best_nonzero_gap.append(gap)
            for margin in margin_grid:
                if gap >= float(margin):
                    nonzero_gap_counts[str(float(margin))] += 1
        if positive_local.size:
            shots_with_positive += 1
            gap = float(np.max(utility[rows[positive_local]]) - identity_score)
            best_positive_gap.append(gap)
            for margin in margin_grid:
                if gap >= float(margin):
                    positive_gap_counts[str(float(margin))] += 1
        if negative_local.size:
            shots_with_negative += 1
            gap = float(np.max(utility[rows[negative_local]]) - identity_score)
            best_negative_gap.append(gap)
            for margin in margin_grid:
                if gap >= float(margin):
                    negative_gap_counts[str(float(margin))] += 1

    return {
        "shots_with_nonzero_candidates": int(shots_with_nonzero),
        "shots_with_positive_nonzero_candidates": int(shots_with_positive),
        "shots_with_negative_nonzero_candidates": int(shots_with_negative),
        "best_nonzero_gap_quantiles": _finite_quantiles(np.asarray(best_nonzero_gap, dtype=np.float32)),
        "best_positive_gap_quantiles": _finite_quantiles(np.asarray(best_positive_gap, dtype=np.float32)),
        "best_negative_gap_quantiles": _finite_quantiles(np.asarray(best_negative_gap, dtype=np.float32)),
        "shots_with_nonzero_gap_at_least_margin": nonzero_gap_counts,
        "shots_with_positive_gap_at_least_margin": positive_gap_counts,
        "shots_with_negative_gap_at_least_margin": negative_gap_counts,
    }


def _harm_guard_selection_metric(
    *,
    bundle: sedp.SelectorCandidateBundle,
    logits: np.ndarray,
    harm_logits: np.ndarray,
    margin: float,
    max_harm_logit: float,
) -> dict[str, Any]:
    utility = np.asarray(logits, dtype=np.float32).reshape(-1)
    harm = np.asarray(harm_logits, dtype=np.float32).reshape(-1)
    if utility.shape != harm.shape:
        raise ValueError(
            "Utility and harm logit arrays must have the same shape. "
            f"Got {utility.shape} and {harm.shape}."
        )

    selected_correct = np.zeros((bundle.num_shots,), dtype=np.uint8)
    selected_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    selected_score = np.zeros((bundle.num_shots,), dtype=np.float32)
    identity_correct = np.zeros((bundle.num_shots,), dtype=np.uint8)
    selected_target_score = np.zeros((bundle.num_shots,), dtype=np.float32)

    for group_slice in sedp._selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        shot_idx = int(bundle.shot_indices[int(rows[0])])
        weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
        identity_local = np.flatnonzero(weights == 0)
        if identity_local.size == 0:
            continue
        identity_row = int(rows[int(identity_local[0])])
        identity_correct[shot_idx] = np.uint8(bundle.candidate_is_correct[identity_row])
        chosen_row = identity_row

        nonzero_local = np.flatnonzero(weights > 0)
        if nonzero_local.size:
            group_utility = utility[rows]
            group_harm = harm[rows]
            allowed = nonzero_local[group_harm[nonzero_local] <= float(max_harm_logit)]
            if allowed.size:
                best_offset = int(np.argmax(group_utility[allowed]))
                best_local = int(allowed[best_offset])
                best_pool = allowed[np.isclose(group_utility[allowed], group_utility[best_local])]
                if best_pool.size > 1:
                    best_local = int(best_pool[int(np.argmin(weights[best_pool]))])
                gap = float(group_utility[best_local] - group_utility[int(identity_local[0])])
                if gap >= float(margin):
                    chosen_row = int(rows[best_local])

        selected_correct[shot_idx] = np.uint8(bundle.candidate_is_correct[chosen_row])
        selected_weight[shot_idx] = np.int16(bundle.candidate_edit_weight[chosen_row])
        selected_score[shot_idx] = np.float32(utility[chosen_row])
        selected_target_score[shot_idx] = np.float32(bundle.target_scores[chosen_row])

    nonzero = selected_weight > 0
    improved = np.logical_and(nonzero, np.logical_and(identity_correct == 0, selected_correct > 0))
    harmed = np.logical_and(nonzero, np.logical_and(identity_correct > 0, selected_correct == 0))
    baseline_accuracy = float(np.mean(identity_correct)) if identity_correct.size else None
    selector_accuracy = float(np.mean(selected_correct)) if selected_correct.size else None
    return {
        "baseline_accuracy": baseline_accuracy,
        "selector_accuracy": selector_accuracy,
        "delta_over_identity": (
            float(selector_accuracy - baseline_accuracy)
            if selector_accuracy is not None and baseline_accuracy is not None
            else None
        ),
        "selected_nonzero": int(nonzero.sum()),
        "selected_positive_target": int(np.logical_and(nonzero, selected_target_score > 0.0).sum()),
        "selected_zero_target": int(np.logical_and(nonzero, np.isclose(selected_target_score, 0.0)).sum()),
        "selected_negative_target": int(np.logical_and(nonzero, selected_target_score < 0.0).sum()),
        "improved": int(improved.sum()),
        "harmed": int(harmed.sum()),
        "mean_selected_target_score": float(np.mean(selected_target_score)) if selected_target_score.size else None,
        "mean_selected_edit_weight": float(np.mean(selected_weight)) if selected_weight.size else None,
        "mean_selected_utility": float(np.mean(selected_score)) if selected_score.size else None,
    }


def _harm_guard_selection_grid(
    *,
    bundle: sedp.SelectorCandidateBundle,
    logits: np.ndarray,
    harm_logits: np.ndarray,
    margin_grid: tuple[float, ...],
    harm_logit_threshold_grid: tuple[float, ...],
) -> dict[str, Any]:
    return {
        str(float(margin)): {
            str(float(threshold)): _harm_guard_selection_metric(
                bundle=bundle,
                logits=logits,
                harm_logits=harm_logits,
                margin=float(margin),
                max_harm_logit=float(threshold),
            )
            for threshold in harm_logit_threshold_grid
        }
        for margin in margin_grid
    }


def _diagnose_family(
    *,
    manifest_path: Path,
    family: str,
    split: str,
    checkpoint: dict[str, Any],
    selector: torch.nn.Module,
    model: torch.nn.Module,
    policy_specs: list[sedp.CandidatePolicySpec],
    selector_payload: dict[str, Any],
    selector_training: dict[str, Any],
    batch_size: int,
    device: torch.device,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    margin_grid: tuple[float, ...],
    harm_logit_threshold_grid: tuple[float, ...],
    fill_value: float,
    max_shots: int | None,
) -> dict[str, Any]:
    _manifest, resolved = sedp._resolve_manifest_family_entries(manifest_path, [family])
    entry = sedp._prepare_edit_family(
        resolved[0][1],
        fill_value=float(fill_value),
        max_shots=max_shots,
    )
    subset = _subset_for_split(
        entry=entry,
        split=split,
        checkpoint=checkpoint,
        family=family,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    motif_vocabulary = _load_candidate_motif(selector_payload)
    local_motif_vocabulary = _load_candidate_local_motif(selector_payload)
    bundle = sedp._build_selector_candidate_bundle(
        entry=entry,
        subset=subset,
        model=model,
        batch_size=batch_size,
        device=device,
        policy_specs=policy_specs,
        selector_score_edit_penalty=float(
            selector_training.get("selector_score_edit_penalty", sedp.DEFAULT_SELECTOR_SCORE_EDIT_PENALTY)
        ),
        selector_target_mode=str(
            selector_training.get("selector_target_mode", sedp.SELECTOR_TARGET_MODE_CORRECTNESS)
        ),
        selector_harm_weight=float(
            selector_training.get("selector_harm_weight", sedp.DEFAULT_SELECTOR_HARM_WEIGHT)
        ),
        selector_miss_weight=float(
            selector_training.get("selector_miss_weight", sedp.DEFAULT_SELECTOR_MISS_WEIGHT)
        ),
        selector_policy_candidate_mode=str(
            selector_training.get(
                "selector_policy_candidate_mode",
                sedp.SELECTOR_POLICY_CANDIDATE_MODE_ALL,
            )
        ),
        selector_candidate_geometry_features=bool(
            selector_training.get(
                "selector_candidate_geometry_features",
                sedp.DEFAULT_SELECTOR_CANDIDATE_GEOMETRY_FEATURES,
            )
        ),
        selector_candidate_pattern_features=bool(
            selector_training.get(
                "selector_candidate_pattern_features",
                sedp.DEFAULT_SELECTOR_CANDIDATE_PATTERN_FEATURES,
            )
        ),
        selector_candidate_local_evidence_features=bool(
            selector_training.get(
                "selector_candidate_local_evidence_features",
                sedp.DEFAULT_SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURES,
            )
        ),
        selector_candidate_local_patch_features=bool(
            selector_training.get(
                "selector_candidate_local_patch_features",
                sedp.DEFAULT_SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURES,
            )
        ),
        motif_vocabulary=motif_vocabulary,
        local_motif_vocabulary=local_motif_vocabulary,
        local_motif_top_k=int(
            selector_training.get("selector_local_motif_top_k", sedp.DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K)
        ),
    )
    logits = sedp._selector_logits_for_bundle(
        selector=selector,
        bundle=bundle,
        batch_size=batch_size,
        device=device,
    )
    weights = np.asarray(bundle.candidate_edit_weight, dtype=np.int16).reshape(-1)
    scores = np.asarray(bundle.target_scores, dtype=np.float32).reshape(-1)
    masks = {
        "identity": weights == 0,
        "positive_nonzero": np.logical_and(weights > 0, scores > 0.0),
        "zero_nonzero": np.logical_and(weights > 0, np.isclose(scores, 0.0)),
        "negative_nonzero": np.logical_and(weights > 0, scores < 0.0),
    }
    margin_metrics: dict[str, Any] = {}
    for margin in margin_grid:
        metric = sedp._selector_selection_metric(
            bundle=bundle,
            selector_logits=logits,
            selector_emit_margin=float(margin),
        )
        selected = metric["selection"]
        margin_metrics[str(float(margin))] = {
            "selector_accuracy": metric["selector_accuracy"],
            "mean_selected_target_score": metric["mean_selected_target_score"],
            "fraction_with_any_selected_edit": metric["fraction_with_any_selected_edit"],
            "mean_selected_edit_weight": metric["mean_selected_edit_weight"],
            "selected_nonzero": int(np.sum(np.asarray(selected["selected_edit_weight"]) > 0)),
            "selected_positive_target": int(
                np.sum(
                    np.logical_and(
                        np.asarray(selected["selected_edit_weight"]) > 0,
                        np.asarray(selected["selected_target_score"]) > 0.0,
                    )
                )
            ),
            "selected_negative_target": int(
                np.sum(
                    np.logical_and(
                        np.asarray(selected["selected_edit_weight"]) > 0,
                        np.asarray(selected["selected_target_score"]) < 0.0,
                    )
                )
            ),
        }
    components = _component_logits_for_bundle(
        selector=selector,
        bundle=bundle,
        batch_size=batch_size,
        device=device,
    )
    component_summary: dict[str, Any] | None = None
    harm_guard_selection: dict[str, Any] | None = None
    if components is not None:
        first, second = components
        component_summary = {
            "first_component_name": (
                "rank_logit"
                if isinstance(selector, sedp.RiskGuardCandidateEditSelector)
                else "benefit_logit"
            ),
            "second_component_name": "harm_logit",
            "first_component_by_row_type": {
                name: _row_stats(first, mask) for name, mask in masks.items()
            },
            "harm_component_by_row_type": {
                name: _row_stats(second, mask) for name, mask in masks.items()
            },
        }
        harm_guard_selection = _harm_guard_selection_grid(
            bundle=bundle,
            logits=logits,
            harm_logits=second,
            margin_grid=margin_grid,
            harm_logit_threshold_grid=harm_logit_threshold_grid,
        )
    return {
        "family": family,
        "split": split,
        "num_shots": int(bundle.num_shots),
        "num_candidate_rows": int(bundle.shot_features.shape[0]),
        "row_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "utility_by_row_type": {
            name: _row_stats(logits, mask) for name, mask in masks.items()
        },
        "gap_summary": _gap_summary(bundle=bundle, logits=logits, margin_grid=margin_grid),
        "selection_by_margin": margin_metrics,
        "component_summary": component_summary,
        "harm_guard_selection_by_margin_threshold": harm_guard_selection,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    analysis_specs = tuple(_parse_analysis_spec(text) for text in args.analysis)
    margin_grid = tuple(float(x) for x in args.margin_grid)
    harm_logit_threshold_grid = tuple(float(x) for x in args.harm_logit_threshold_grid)
    run_results: dict[str, Any] = {}
    for label, summary_path in map(_parse_run_spec, args.run):
        summary = _read_json(summary_path)
        checkpoint_path = Path(summary["artifacts"]["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = sedp.SyndromeEditPreDecoder(**dict(checkpoint["model_kwargs"]))
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        selector_payload = dict(checkpoint["candidate_selector"])
        selector_training = dict(selector_payload.get("training_summary") or {})
        selector_model = str(
            selector_payload.get(
                "selector_model",
                selector_training.get("selector_model", sedp.SELECTOR_MODEL_SCALAR),
            )
        )
        selector = sedp._make_candidate_selector_module(
            selector_model=selector_model,
            selector_kwargs=dict(selector_payload["selector_kwargs"]),
        )
        selector.load_state_dict(selector_payload["selector_state"])
        selector = selector.to(device)
        selector.eval()
        policy_specs = _policy_specs_from_checkpoint(checkpoint)
        families: dict[str, Any] = {}
        for family, split in analysis_specs:
            families[f"{family}:{split}"] = _diagnose_family(
                manifest_path=manifest_path,
                family=family,
                split=split,
                checkpoint=checkpoint,
                selector=selector,
                model=model,
                policy_specs=policy_specs,
                selector_payload=selector_payload,
                selector_training=selector_training,
                batch_size=int(args.batch_size),
                device=device,
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
                margin_grid=margin_grid,
                harm_logit_threshold_grid=harm_logit_threshold_grid,
                fill_value=float(args.fill_value),
                max_shots=args.max_shots,
            )
        run_results[label] = {
            "summary_path": summary_path.as_posix(),
            "checkpoint_path": checkpoint_path.as_posix(),
            "selector_model": selector_model,
            "selected_inference_mode": summary.get("training", {}).get("selected_inference_mode"),
            "selector_adoption_decision": summary.get("training", {}).get("selector_adoption_decision"),
            "families": families,
        }
    return {
        "schema_version": "selector_preservation_failure_diagnostic.v1",
        "manifest_path": manifest_path.as_posix(),
        "analysis_specs": [f"{family}:{split}" for family, split in analysis_specs],
        "margin_grid": [float(x) for x in margin_grid],
        "harm_logit_threshold_grid": [float(x) for x in harm_logit_threshold_grid],
        "runs": run_results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose selector preservation failure on candidate bundles.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=experiment_summary.json")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--analysis",
        nargs="*",
        default=[f"{family}:{split}" for family, split in DEFAULT_ANALYSIS_SPECS],
    )
    parser.add_argument("--margin-grid", nargs="*", default=[str(x) for x in DEFAULT_MARGIN_GRID])
    parser.add_argument(
        "--harm-logit-threshold-grid",
        nargs="*",
        default=[str(x) for x in DEFAULT_HARM_LOGIT_THRESHOLD_GRID],
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fill-value", type=float, default=-0.5)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run(args)
    _write_json(args.out_json, payload)
    print(json.dumps({"out_json": args.out_json.as_posix(), "runs": list(payload["runs"])}, indent=2))


if __name__ == "__main__":
    main()
