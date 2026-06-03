from __future__ import annotations

"""
diagnose_predecoder_selection.py

Per-shot diagnostics for syndrome-edit pre-decoder candidate selection.
This is intentionally a research/debug tool and reuses decoder internals.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECODER_DIR = ROOT / "decoders"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_DIR))

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PyTorch is required for this diagnostic.") from exc

import syndrome_edit_predecoder as sedp


CLASS_LABELS = ["I", "X", "Z", "Y"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _parse_float_grid(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    out: list[float] = []
    for item in values:
        for part in str(item).split(","):
            text = part.strip()
            if text:
                out.append(float(text))
    return out


def _family_dir_from_manifest(manifest_path: Path, family: str) -> Path:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    family_dirs = manifest.get("family_dirs")
    if not isinstance(family_dirs, dict) or family not in family_dirs:
        raise KeyError(f"Family {family!r} not found in {manifest_path}")
    path = Path(family_dirs[family])
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_candidate_local_motif_vocabulary(payload: dict[str, Any]) -> sedp.LocalMotifVocabulary | None:
    candidate_selector = payload.get("candidate_selector")
    if not isinstance(candidate_selector, dict):
        return None
    raw = candidate_selector.get("candidate_local_motif_vocabulary")
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


def _load_candidate_motif_vocabulary(payload: dict[str, Any]) -> sedp.MotifVocabulary | None:
    candidate_selector = payload.get("candidate_selector")
    if not isinstance(candidate_selector, dict):
        return None
    raw = candidate_selector.get("candidate_motif_vocabulary")
    if not isinstance(raw, dict):
        return None
    mask_table = np.asarray(raw["mask_table"], dtype=np.uint8)
    return sedp.MotifVocabulary(
        mask_table=mask_table,
        detector_index_lists=tuple(
            tuple(int(x) for x in row) for row in raw["detector_index_lists"]
        ),
        counts=tuple(int(x) for x in raw["counts"]),
        detector_count=int(mask_table.shape[1]),
    )


def _prepare_subset(
    *,
    entry: Any,
    split: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if split == "all":
        indices = np.arange(entry.x.shape[0], dtype=np.int64)
    else:
        split_bundle = sedp._build_split_bundle(
            num_shots=int(entry.x.shape[0]),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(split_seed),
        )
        indices = getattr(split_bundle, split)
    return sedp._subset_family(entry, np.asarray(indices, dtype=np.int64))


def _status_counts(
    *,
    baseline_class: np.ndarray,
    edited_class: np.ndarray,
    target_class: np.ndarray,
    selected_weight: np.ndarray,
) -> dict[str, int]:
    baseline_correct = baseline_class == target_class
    edited_correct = edited_class == target_class
    nonzero = selected_weight > 0
    categories = {
        "selected_nonzero": nonzero,
        "improved": np.logical_and(nonzero, np.logical_and(~baseline_correct, edited_correct)),
        "harmed": np.logical_and(nonzero, np.logical_and(baseline_correct, ~edited_correct)),
        "nonzero_still_wrong": np.logical_and(nonzero, np.logical_and(~baseline_correct, ~edited_correct)),
        "nonzero_still_correct": np.logical_and(nonzero, np.logical_and(baseline_correct, edited_correct)),
        "identity": ~nonzero,
    }
    return {name: int(mask.sum()) for name, mask in categories.items()}


def _class_counter(
    *,
    target_class: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    counts = Counter(CLASS_LABELS[int(x)] for x in target_class[np.asarray(mask, dtype=bool)].tolist())
    return {label: int(counts.get(label, 0)) for label in CLASS_LABELS}


def _transition_counter(
    *,
    baseline_class: np.ndarray,
    edited_class: np.ndarray,
    target_class: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for b, e, t in zip(
        baseline_class[np.asarray(mask, dtype=bool)].tolist(),
        edited_class[np.asarray(mask, dtype=bool)].tolist(),
        target_class[np.asarray(mask, dtype=bool)].tolist(),
        strict=False,
    ):
        key = f"{CLASS_LABELS[int(b)]}->{CLASS_LABELS[int(e)]}|target={CLASS_LABELS[int(t)]}"
        counts[key] += 1
    return dict(sorted((k, int(v)) for k, v in counts.items()))


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
    }


def diagnose_checkpoint(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    family: str,
    split: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    fill_value: float,
    selector_emit_margin: float | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    family_dir = _family_dir_from_manifest(manifest_path, family)
    entry = sedp._prepare_edit_family(family_dir, fill_value=float(fill_value), max_shots=None)
    subset = _prepare_subset(
        entry=entry,
        split=split,
        split_seed=int(split_seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sedp.SyndromeEditPreDecoder(**dict(checkpoint["model_kwargs"]))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    candidate_payload = checkpoint.get("candidate_selector")
    if not isinstance(candidate_payload, dict):
        raise ValueError(f"No candidate selector payload in {checkpoint_path}")
    selector = sedp.CandidateEditSelector(**dict(candidate_payload["selector_kwargs"]))
    selector.load_state_dict(candidate_payload["selector_state"])
    selector = selector.to(device)
    selector.eval()
    training_summary = dict(candidate_payload.get("training_summary", {}))
    selector_kwargs = dict(candidate_payload["selector_kwargs"])
    inferred_patch_features = int(selector_kwargs.get("patch_feature_dim", 0) or 0) > 0
    inferred_extra_local_features = bool(inferred_patch_features)
    inference = dict(checkpoint.get("inference", {}))
    policy_specs = [
        sedp.CandidatePolicySpec(
            needs_edit_threshold=float(item["needs_edit_threshold"]),
            edit_threshold=float(item["edit_threshold"]),
            max_predicted_edit_weight=int(item["max_predicted_edit_weight"]),
        )
        for item in inference.get("candidate_policy_grid", [])
    ]

    bundle = sedp._build_selector_candidate_bundle(
        entry=entry,
        subset=subset,
        model=model,
        batch_size=int(batch_size),
        device=device,
        policy_specs=policy_specs,
        selector_score_edit_penalty=float(training_summary.get("selector_score_edit_penalty", 0.0)),
        selector_target_mode=str(training_summary.get("selector_target_mode", sedp.SELECTOR_TARGET_MODE_CORRECTNESS)),
        selector_harm_weight=float(training_summary.get("selector_harm_weight", sedp.DEFAULT_SELECTOR_HARM_WEIGHT)),
        selector_miss_weight=float(training_summary.get("selector_miss_weight", sedp.DEFAULT_SELECTOR_MISS_WEIGHT)),
        selector_policy_candidate_mode=str(
            training_summary.get("selector_policy_candidate_mode", sedp.SELECTOR_POLICY_CANDIDATE_MODE_ALL)
        ),
        selector_candidate_geometry_features=bool(training_summary.get("selector_candidate_geometry_features", False)),
        selector_candidate_pattern_features=bool(training_summary.get("selector_candidate_pattern_features", False)),
        selector_candidate_local_evidence_features=bool(
            training_summary.get(
                "selector_candidate_local_evidence_features",
                inferred_extra_local_features,
            )
        ),
        selector_candidate_local_patch_features=bool(
            training_summary.get("selector_candidate_local_patch_features", inferred_patch_features)
        ),
        motif_vocabulary=_load_candidate_motif_vocabulary(checkpoint),
        local_motif_vocabulary=_load_candidate_local_motif_vocabulary(checkpoint),
        local_motif_top_k=int(training_summary.get("selector_local_motif_top_k", sedp.DEFAULT_SELECTOR_LOCAL_MOTIF_TOP_K)),
    )
    selector_logits = sedp._selector_logits_for_bundle(
        selector=selector,
        bundle=bundle,
        batch_size=int(batch_size),
        device=device,
    )
    checkpoint_emit_margin = float(training_summary.get("selected_selector_emit_margin", 0.0) or 0.0)
    emit_margin = checkpoint_emit_margin if selector_emit_margin is None else float(selector_emit_margin)
    selection_metric = sedp._selector_selection_metric(
        bundle=bundle,
        selector_logits=selector_logits,
        selector_emit_margin=emit_margin,
        selector_nonzero_bias=float(training_summary.get("selected_selector_nonzero_bias", 0.0) or 0.0),
    )
    selected = selection_metric["selection"]
    selected_weight = np.asarray(selected["selected_edit_weight"], dtype=np.int16)
    selected_target_score = np.asarray(selected["selected_target_score"], dtype=np.float32)
    selected_mask = np.asarray(selected["selected_edit_mask"], dtype=np.uint8)

    edited_detector_events = np.asarray(subset["detector_events"], dtype=np.uint8).copy()
    edited_detector_events ^= selected_mask
    edited_obs, decode_fallback_count = sedp._safe_decode_edited_observables(
        matching=entry.matching,
        edited_detector_events=edited_detector_events,
        baseline_predicted_observables=subset["baseline_predicted_observables"],
    )
    baseline_class = sedp._logical_class4_from_observable_flips(subset["baseline_predicted_observables"])
    edited_class = sedp._logical_class4_from_observable_flips(edited_obs)
    target_class = np.asarray(subset["logical_class4"], dtype=np.uint8).reshape(-1)

    logits = np.asarray(selector_logits, dtype=np.float32).reshape(-1)
    best_nonzero_gap = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    best_nonzero_target_score = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    best_nonzero_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    oracle_available = np.zeros((bundle.num_shots,), dtype=bool)
    oracle_logit_rank = np.zeros((bundle.num_shots,), dtype=np.int16)
    oracle_nonzero_logit_rank = np.zeros((bundle.num_shots,), dtype=np.int16)
    oracle_gap_vs_identity = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    oracle_target_score = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    oracle_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    top_logit_target_score = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    top_logit_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    harm_risk = np.zeros((bundle.num_shots,), dtype=bool)
    best_harm_nonzero_gap_vs_identity = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    best_harm_nonzero_logit_rank = np.zeros((bundle.num_shots,), dtype=np.int16)
    best_harm_nonzero_target_score = np.full((bundle.num_shots,), np.nan, dtype=np.float32)
    best_harm_nonzero_weight = np.zeros((bundle.num_shots,), dtype=np.int16)
    for group_slice in sedp._selector_group_slices(bundle):
        rows = np.arange(group_slice.start or 0, group_slice.stop or 0, dtype=np.int64)
        if rows.size == 0:
            continue
        shot_idx = int(bundle.shot_indices[int(rows[0])])
        weights = np.asarray(bundle.candidate_edit_weight[rows], dtype=np.int16)
        target_scores = np.asarray(bundle.target_scores[rows], dtype=np.float32)
        identity = np.flatnonzero(weights == 0)
        nonzero = np.flatnonzero(weights > 0)
        if identity.size == 0:
            continue
        group_logits = logits[rows]
        identity_logit = float(group_logits[int(identity[0])])
        top_local = int(np.argmax(group_logits))
        top_logit_target_score[shot_idx] = np.float32(target_scores[top_local])
        top_logit_weight[shot_idx] = np.int16(weights[top_local])
        if nonzero.size == 0:
            continue
        best_local = int(nonzero[int(np.argmax(group_logits[nonzero]))])
        best_nonzero_gap[shot_idx] = float(group_logits[best_local] - identity_logit)
        best_nonzero_target_score[shot_idx] = float(bundle.target_scores[int(rows[best_local])])
        best_nonzero_weight[shot_idx] = int(bundle.candidate_edit_weight[int(rows[best_local])])
        identity_target_score = float(target_scores[int(identity[0])])
        best_nonzero_target_score_value = float(np.max(target_scores[nonzero]))
        if best_nonzero_target_score_value > identity_target_score:
            best_target_locals = nonzero[np.isclose(target_scores[nonzero], best_nonzero_target_score_value)]
            if best_target_locals.size > 1:
                best_target_weights = weights[best_target_locals]
                oracle_local = int(best_target_locals[int(np.argmin(best_target_weights))])
            else:
                oracle_local = int(best_target_locals[0])
            oracle_logit = float(group_logits[oracle_local])
            oracle_available[shot_idx] = True
            oracle_logit_rank[shot_idx] = np.int16(1 + int(np.sum(group_logits > oracle_logit)))
            oracle_nonzero_logit_rank[shot_idx] = np.int16(1 + int(np.sum(group_logits[nonzero] > oracle_logit)))
            oracle_gap_vs_identity[shot_idx] = np.float32(oracle_logit - identity_logit)
            oracle_target_score[shot_idx] = np.float32(target_scores[oracle_local])
            oracle_weight[shot_idx] = np.int16(weights[oracle_local])
        negative_nonzero = nonzero[target_scores[nonzero] < 0.0]
        if negative_nonzero.size:
            harm_risk[shot_idx] = True
            best_harm_logit = float(np.max(group_logits[negative_nonzero]))
            best_harm_locals = negative_nonzero[np.isclose(group_logits[negative_nonzero], best_harm_logit)]
            if best_harm_locals.size > 1:
                best_harm_weights = weights[best_harm_locals]
                harm_local = int(best_harm_locals[int(np.argmin(best_harm_weights))])
            else:
                harm_local = int(best_harm_locals[0])
            harm_logit = float(group_logits[harm_local])
            best_harm_nonzero_gap_vs_identity[shot_idx] = np.float32(harm_logit - identity_logit)
            best_harm_nonzero_logit_rank[shot_idx] = np.int16(1 + int(np.sum(group_logits > harm_logit)))
            best_harm_nonzero_target_score[shot_idx] = np.float32(target_scores[harm_local])
            best_harm_nonzero_weight[shot_idx] = np.int16(weights[harm_local])

    baseline_correct = baseline_class == target_class
    edited_correct = edited_class == target_class
    nonzero = selected_weight > 0
    improved = np.logical_and(nonzero, np.logical_and(~baseline_correct, edited_correct))
    harmed = np.logical_and(nonzero, np.logical_and(baseline_correct, ~edited_correct))
    still_wrong = np.logical_and(nonzero, np.logical_and(~baseline_correct, ~edited_correct))
    still_correct = np.logical_and(nonzero, np.logical_and(baseline_correct, edited_correct))
    positive_selected = np.logical_and(nonzero, selected_target_score > 0.0)
    negative_selected = np.logical_and(nonzero, selected_target_score < 0.0)
    zero_selected = np.logical_and(nonzero, np.isclose(selected_target_score, 0.0))
    oracle_selected = np.logical_and(oracle_available, selected_target_score > 0.0)
    oracle_blocked_by_margin = np.logical_and(
        oracle_available,
        np.logical_and(oracle_gap_vs_identity > 0.0, oracle_gap_vs_identity < float(emit_margin)),
    )
    oracle_below_identity = np.logical_and(oracle_available, oracle_gap_vs_identity <= 0.0)
    oracle_above_margin = np.logical_and(oracle_available, oracle_gap_vs_identity >= float(emit_margin))
    harm_above_identity = np.logical_and(harm_risk, best_harm_nonzero_gap_vs_identity > 0.0)
    harm_above_margin = np.logical_and(
        harm_risk,
        best_harm_nonzero_gap_vs_identity >= float(emit_margin),
    )

    gap_by_status = {
        "selected_nonzero": _quantiles(best_nonzero_gap[nonzero]),
        "improved": _quantiles(best_nonzero_gap[improved]),
        "harmed": _quantiles(best_nonzero_gap[harmed]),
        "still_wrong": _quantiles(best_nonzero_gap[still_wrong]),
        "still_correct": _quantiles(best_nonzero_gap[still_correct]),
        "blocked_above_zero_below_margin": _quantiles(
            best_nonzero_gap[np.logical_and(~nonzero, np.logical_and(best_nonzero_gap > 0.0, best_nonzero_gap < emit_margin))]
        ),
    }
    target_score_counts = {
        "selected_positive": int(positive_selected.sum()),
        "selected_zero": int(zero_selected.sum()),
        "selected_negative": int(negative_selected.sum()),
    }
    oracle_ranking_summary = {
        "oracle_available": int(oracle_available.sum()),
        "oracle_selected_positive_target": int(oracle_selected.sum()),
        "oracle_below_or_equal_identity_logit": int(oracle_below_identity.sum()),
        "oracle_above_identity_below_margin": int(oracle_blocked_by_margin.sum()),
        "oracle_above_margin": int(oracle_above_margin.sum()),
        "oracle_logit_rank_top1": int(np.logical_and(oracle_available, oracle_logit_rank == 1).sum()),
        "oracle_logit_rank_top3": int(
            np.logical_and(oracle_available, np.logical_and(oracle_logit_rank > 0, oracle_logit_rank <= 3)).sum()
        ),
        "oracle_nonzero_logit_rank_top1": int(
            np.logical_and(oracle_available, oracle_nonzero_logit_rank == 1).sum()
        ),
        "oracle_gap_vs_identity_quantiles": _quantiles(oracle_gap_vs_identity[oracle_available]),
        "oracle_logit_rank_quantiles": _quantiles(oracle_logit_rank[oracle_available]),
        "oracle_nonzero_logit_rank_quantiles": _quantiles(oracle_nonzero_logit_rank[oracle_available]),
        "oracle_weight_histogram": {
            str(int(weight)): int(count)
            for weight, count in zip(
                *np.unique(oracle_weight[oracle_available], return_counts=True),
                strict=False,
            )
        },
        "top_logit_target_counts_on_oracle_available": {
            "positive": int(np.logical_and(oracle_available, top_logit_target_score > 0.0).sum()),
            "zero": int(np.logical_and(oracle_available, np.isclose(top_logit_target_score, 0.0)).sum()),
            "negative": int(np.logical_and(oracle_available, top_logit_target_score < 0.0).sum()),
            "identity_weight": int(np.logical_and(oracle_available, top_logit_weight == 0).sum()),
            "nonzero_weight": int(np.logical_and(oracle_available, top_logit_weight > 0).sum()),
        },
    }
    harm_ranking_summary = {
        "harm_risk_shots": int(harm_risk.sum()),
        "harm_nonzero_above_identity": int(harm_above_identity.sum()),
        "harm_nonzero_above_margin": int(harm_above_margin.sum()),
        "harm_logit_rank_top1": int(np.logical_and(harm_risk, best_harm_nonzero_logit_rank == 1).sum()),
        "harm_logit_rank_top3": int(
            np.logical_and(
                harm_risk,
                np.logical_and(best_harm_nonzero_logit_rank > 0, best_harm_nonzero_logit_rank <= 3),
            ).sum()
        ),
        "harm_gap_vs_identity_quantiles": _quantiles(
            best_harm_nonzero_gap_vs_identity[harm_risk]
        ),
        "harm_gap_vs_identity_above_margin_quantiles": _quantiles(
            best_harm_nonzero_gap_vs_identity[harm_above_margin]
        ),
        "harm_logit_rank_quantiles": _quantiles(best_harm_nonzero_logit_rank[harm_risk]),
        "harm_weight_histogram": {
            str(int(weight)): int(count)
            for weight, count in zip(
                *np.unique(best_harm_nonzero_weight[harm_risk], return_counts=True),
                strict=False,
            )
        },
        "top_logit_target_counts_on_harm_risk": {
            "positive": int(np.logical_and(harm_risk, top_logit_target_score > 0.0).sum()),
            "zero": int(np.logical_and(harm_risk, np.isclose(top_logit_target_score, 0.0)).sum()),
            "negative": int(np.logical_and(harm_risk, top_logit_target_score < 0.0).sum()),
            "identity_weight": int(np.logical_and(harm_risk, top_logit_weight == 0).sum()),
            "nonzero_weight": int(np.logical_and(harm_risk, top_logit_weight > 0).sum()),
        },
    }

    return {
        "checkpoint": checkpoint_path.as_posix(),
        "family": family,
        "split": split,
        "split_seed": int(split_seed),
        "num_examples": int(target_class.shape[0]),
        "selected_inference_mode": str(inference.get("selected_inference_mode")),
        "checkpoint_selector_emit_margin": checkpoint_emit_margin,
        "selector_emit_margin": emit_margin,
        "candidate_local_motif_num_patterns": int(
            len(_load_candidate_local_motif_vocabulary(checkpoint).offset_patterns)
            if _load_candidate_local_motif_vocabulary(checkpoint) is not None
            else 0
        ),
        "selector_accuracy": float(np.mean(edited_correct)) if edited_correct.size else None,
        "baseline_accuracy": float(np.mean(baseline_correct)) if baseline_correct.size else None,
        "delta_over_baseline": float(np.mean(edited_correct) - np.mean(baseline_correct)),
        "decode_fallback_count": int(decode_fallback_count),
        "status_counts": _status_counts(
            baseline_class=baseline_class,
            edited_class=edited_class,
            target_class=target_class,
            selected_weight=selected_weight,
        ),
        "target_score_counts": target_score_counts,
        "oracle_ranking_summary": oracle_ranking_summary,
        "harm_ranking_summary": harm_ranking_summary,
        "selected_edit_weight_histogram": {
            str(int(weight)): int(count)
            for weight, count in zip(*np.unique(selected_weight, return_counts=True), strict=False)
        },
        "best_nonzero_gap_quantiles": _quantiles(best_nonzero_gap[np.isfinite(best_nonzero_gap)]),
        "gap_by_status": gap_by_status,
        "target_class_counts": {
            "selected_nonzero": _class_counter(target_class=target_class, mask=nonzero),
            "improved": _class_counter(target_class=target_class, mask=improved),
            "harmed": _class_counter(target_class=target_class, mask=harmed),
            "still_wrong": _class_counter(target_class=target_class, mask=still_wrong),
            "still_correct": _class_counter(target_class=target_class, mask=still_correct),
        },
        "transition_counts": {
            "improved": _transition_counter(
                baseline_class=baseline_class,
                edited_class=edited_class,
                target_class=target_class,
                mask=improved,
            ),
            "harmed": _transition_counter(
                baseline_class=baseline_class,
                edited_class=edited_class,
                target_class=target_class,
                mask=harmed,
            ),
            "still_wrong": _transition_counter(
                baseline_class=baseline_class,
                edited_class=edited_class,
                target_class=target_class,
                mask=still_wrong,
            ),
        },
        "mean_selected_target_score": float(np.mean(selected_target_score)) if selected_target_score.size else None,
        "mean_best_nonzero_target_score": float(np.nanmean(best_nonzero_target_score))
        if np.isfinite(best_nonzero_target_score).any()
        else None,
        "mean_selected_edit_weight": float(np.mean(selected_weight)) if selected_weight.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fill-value", type=float, default=-0.5)
    parser.add_argument("--selector-emit-margin", type=float, default=None)
    parser.add_argument("--selector-emit-margin-grid", nargs="*", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    margin_grid = _parse_float_grid(args.selector_emit_margin_grid)
    if margin_grid is None:
        margin_grid = [args.selector_emit_margin]
    results = []
    for checkpoint in args.checkpoint:
        for margin in margin_grid:
            results.append(
                diagnose_checkpoint(
                    checkpoint_path=checkpoint,
                    manifest_path=args.manifest,
                    family=str(args.family),
                    split=str(args.split),
                    split_seed=int(args.split_seed),
                    train_ratio=float(args.train_ratio),
                    val_ratio=float(args.val_ratio),
                    test_ratio=float(args.test_ratio),
                    batch_size=int(args.batch_size),
                    fill_value=float(args.fill_value),
                    selector_emit_margin=margin,
                )
            )
    payload = {
        "schema_version": "predecoder_selection_diagnostic.v1",
        "manifest": args.manifest.as_posix(),
        "family": str(args.family),
        "split": str(args.split),
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")
    if not bool(args.quiet):
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
