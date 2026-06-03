from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


StatusCategory = Literal["keep", "modify", "legacy"]


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    path: str
    category: StatusCategory
    summary: str
    next_action: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PROJECT_COMPONENT_STATUS: tuple[ComponentStatus, ...] = (
    ComponentStatus(
        path="config.py",
        category="keep",
        summary="Experiment configuration entry point for circuit and noise settings.",
        next_action="Keep as the canonical configuration layer for the rebuilt decoder stack.",
    ),
    ComponentStatus(
        path="circuits.py",
        category="modify",
        summary="Ideal memory-circuit builder and detector semantic export layer.",
        next_action="Extend beyond single-basis memory supervision toward true logical-class targets.",
    ),
    ComponentStatus(
        path="logical_targets.py",
        category="keep",
        summary="Shared description of which logical axis each basis-specific memory experiment can supervise.",
        next_action="Keep as the canonical supervision-mapping layer while the project bridges from axis-wise targets to true logical-class targets.",
    ),
    ComponentStatus(
        path="logical_frame.py",
        category="keep",
        summary="Scaffold-level logical-frame geometry and audit helpers for current rotated memory circuits.",
        next_action="Keep as the diagnostic layer that proves whether a circuit can support same-shot logical X/Z supervision.",
    ),
    ComponentStatus(
        path="logical_bell.py",
        category="keep",
        summary="Bell-pair logical-frame readout helpers that produce true per-shot logical_class4 labels on top of the current scaffold.",
        next_action="Keep as the first working class4 supervision path while native logical-frame circuits and class4 decoders are built out.",
    ),
    ComponentStatus(
        path="noise_si1000.py",
        category="keep",
        summary="Stage A SI1000-based noisy circuit generation.",
        next_action="Reuse as an active training and comparison noise family.",
    ),
    ComponentStatus(
        path="noise_willowcore.py",
        category="modify",
        summary="Stage B/C Willow-inspired noisy circuit generation.",
        next_action="Keep current B/C families and extend with new noise families for transfer tests.",
    ),
    ComponentStatus(
        path="sample_dataset.py",
        category="modify",
        summary="Primary dataset generator and schema owner.",
        next_action="Continue migrating targets from binary logical flip toward logical-axis and later logical-class supervision.",
    ),
    ComponentStatus(
        path="MAIN_TARGET_WORK_SCHEDULE.md",
        category="keep",
        summary="Defines the corrected main target hierarchy: 'Convolutional Neural Decoder for Surface Codes' is now an output/evaluation-format anchor rather than a required architecture, with d3/d5/d7 PyMatching baselines complete and the pre-decoder branch promoted as the most promising model family.",
        next_action="Use together with RESEARCH_PLAN_PREDECODER_MAIN.md as the broader schedule and context map.",
    ),
    ComponentStatus(
        path="RESEARCH_PLAN_PREDECODER_MAIN.md",
        category="keep",
        summary="Fixes the current research topic as transition-aware neural pre-decoding for surface-code logical-frame inference, with title candidates, novelty positioning, NVIDIA related-work boundaries, success criteria, and phased execution plan; the d3 selected gain has reproduced; the true patch-head selector now gives full d5 PyMatching-beating selected runs on seeds 1 and 3, but selected-mode adoption is still seed-sensitive.",
        next_action="Treat as the primary research plan and stabilize selector calibration / selected-mode adoption across d5 patch-head seeds.",
    ),
    ComponentStatus(
        path="MODEL_SELECTION_D3_D5_D7.md",
        category="keep",
        summary="Records the d3/d5/d7 class4 baseline table, direct neural collapse evidence, and local-edit oracle headroom that makes the PyMatching-assist pre-decoder the leading next model family.",
        next_action="Keep updated as new selected-mode pre-decoder results are produced.",
    ),
    ComponentStatus(
        path="geometry/rotated_rect.py",
        category="keep",
        summary="Rectangular lattice geometry layer for space-time syndrome volumes.",
        next_action="Keep as the default geometry interface for geometry-aware decoders.",
    ),
    ComponentStatus(
        path="decoders/baseline_pymatching.py",
        category="modify",
        summary="Classical decoding baseline built from detector error models.",
        next_action="Keep as the main classical baseline; it now reports class4 metrics on Bell-pair datasets and should later move to recovery-based evaluation.",
    ),
    ComponentStatus(
        path="decoders/baseline_rectcnn.py",
        category="keep",
        summary="Paper-style neural baseline aligned with 'Convolutional Neural Decoder for Surface Codes': rectangular syndrome lattice input, incoherent fill value, and compact CNN decoding.",
        next_action="Keep as an optional baseline/reference implementation; exact RectCNN structure is not required for the final model as long as the logical_class4 output/evaluation contract is preserved.",
    ),
    ComponentStatus(
        path="decoders/research_noise_aware_3d.py",
        category="modify",
        summary="Main research decoder with geometry and noise-context channels.",
        next_action="Treat as the active model line for multi-noise axis-wise and logical_class4 experiments until a new decoder architecture is designed.",
    ),
    ComponentStatus(
        path="decoders/factorized_logical_frame_decoder.py",
        category="modify",
        summary="First factorized logical-frame decoder with class4 main supervision, optional auxiliary axis-wise supervision, optional focal-style class4 loss, optional hierarchical non-identity auxiliary head, tempered imbalance handling, and guardrailed post-hoc temperature calibration.",
        next_action="Treat as a completed direct-neural baseline: d3 partially learns non-I classes, d5 collapses to all-I, and the d7 refresh collapses to all-X, so this is not the most promising next model family without a qualitative change.",
    ),
    ComponentStatus(
        path="decoders/multiscale_factorized_decoder.py",
        category="modify",
        summary="First multi-scale Dense3D successor to FLFD, implemented as an M3D-FLFD variant that reuses the class4 training and evaluation stack while replacing the shallow trunk with a multi-resolution encoder.",
        next_action="Keep as a completed negative-result branch: the first large d3/d5 comparisons did not beat the original FLFD and the stronger d5 run collapsed even harder, so the next architecture step should move toward an Ising-inspired pre-decoder or another nontrivial system change instead of more dense-trunk scaling alone.",
    ),
    ComponentStatus(
        path="decoders/syndrome_edit_predecoder.py",
        category="modify",
        summary="First neural syndrome-edit pre-decoder that predicts detector-bit edit masks plus a needs_edit head, then hands the edited syndrome to unchanged PyMatching; it now supports benefit/harm candidate scoring, logical-transition candidate features, selector nonzero-bias calibration, harm-margin loss, local-motif candidate selection, corrected benefit/harm-compatible router labels, transition-prior/top-k ablations, flat/group-balanced BCE candidate-compatibility heads, an auxiliary pairwise compatibility ranker, a direct main-selector pairwise benefit/harm term, motif-evidence merging for duplicate candidates, raw-policy candidate-pool disabling, geometry/placement-aware candidate features, local motif pattern/anchor candidate features, handcrafted anchor-local evidence candidate features, opt-in radius-1 local-patch candidate features, and an opt-in learned patch-head selector branch; patch-head now beats full d5 PyMatching on selected seeds 1 and 3.",
        next_action="Follow RESEARCH_PLAN_PREDECODER_MAIN.md: keep patch-head as the active representation and stabilize selector calibration / selected-mode adoption across seeds.",
    ),
    ComponentStatus(
        path="tools/build_dual_axis_manifest.py",
        category="keep",
        summary="Pairs basis-x and basis-z memory datasets into a dual-axis manifest.",
        next_action="Keep as the bridge from single-basis memory data to axis-wise supervision.",
    ),
    ComponentStatus(
        path="tools/audit_logical_frame_support.py",
        category="keep",
        summary="Audits whether the current scaffold can expose a same-shot logical X/Z frame.",
        next_action="Use before circuit redesign work to verify that true per-shot logical_class4 is or is not structurally supported.",
    ),
    ComponentStatus(
        path="tools/run_dual_axis_experiment.py",
        category="keep",
        summary="Runs aligned logical_x_flip and logical_z_flip experiments from a dual-axis manifest.",
        next_action="Keep as the current experiment runner for paired axis-wise training and evaluation.",
    ),
    ComponentStatus(
        path="tools/run_dual_axis_pymatching.py",
        category="keep",
        summary="Runs aligned axis-wise PyMatching baselines from a dual-axis manifest.",
        next_action="Keep as the classical comparison runner for paired logical-axis experiments.",
    ),
    ComponentStatus(
        path="tools/evaluate_hybrid_fallback.py",
        category="keep",
        summary="Evaluates confidence-aware FLFD plus PyMatching fallback over threshold sweeps on class4 manifests.",
        next_action="Use to compare neural-only, threshold-hybrid, and fallback-all behavior on larger class4 runs; current evidence shows threshold sweeps become meaningful only after the class4 data regime is normalized.",
    ),
    ComponentStatus(
        path="tools/evaluate_learned_hybrid_router.py",
        category="keep",
        summary="Trains a frozen-feature logistic router on FLFD and PyMatching outputs, with correctness-target and prefer-neural-target modes plus explicit metadata/noise-context features, then evaluates learned hybrid routing over class4 manifests.",
        next_action="Use as the preferred hybrid-routing experiment path after a stable FLFD checkpoint exists; the current next question is whether larger class4 runs improve holdout routing quality beyond PyMatching instead of only fixing the smoke-level fallback-all pathology.",
    ),
    ComponentStatus(
        path="tools/build_pymatching_edit_targets.py",
        category="keep",
        summary="Builds derived syndrome-edit supervision targets by searching for small local detector-bit edits that turn wrong PyMatching shots into correct ones.",
        next_action="Use as the artifact-building layer for the leading pre-decoder branch; d3/d5/d7 target manifests show strong local-edit oracle headroom, so the next step is benefit/harm calibrated selection rather than more sampling-only tuning.",
    ),
    ComponentStatus(
        path="legacy_archive/decoders/baseline_nn.py",
        category="legacy",
        summary="Flat binary logical-flip baseline without geometry-aware structure.",
        next_action="Keep only for historical comparison; do not use for new decoder development.",
    ),
    ComponentStatus(
        path="legacy_archive/decoders/baseline_tracknn.py",
        category="legacy",
        summary="Track-based binary decoder using the older track representation.",
        next_action="Keep only for comparison on past experiments; exclude from the rebuilt mainline.",
    ),
    ComponentStatus(
        path="legacy_archive/decoders/baseline_trackformer.py",
        category="legacy",
        summary="TrackFormer-style binary decoder built on the legacy track representation.",
        next_action="Keep only for reference and ablation comparisons; exclude from the rebuilt mainline.",
    ),
    ComponentStatus(
        path="legacy_archive/decoders/track_common.py",
        category="legacy",
        summary="Shared track representation helpers used by legacy track-based decoders.",
        next_action="Keep only while legacy track models remain in the repository.",
    ),
    ComponentStatus(
        path="legacy_archive/baseline_trackformer_eventcentric_v3_fast.py",
        category="legacy",
        summary="Standalone experimental trackformer variant from the previous binary-flip line.",
        next_action="Keep as a historical experiment artifact only.",
    ),
    ComponentStatus(
        path="artifacts/",
        category="keep",
        summary="Generated datasets, checkpoints, and evaluation outputs.",
        next_action="Treat as generated outputs and compare against them, but do not treat them as source modules.",
    ),
)


def iter_status_by_category(category: StatusCategory) -> list[ComponentStatus]:
    return [item for item in PROJECT_COMPONENT_STATUS if item.category == category]


def status_summary_counts() -> dict[str, int]:
    return {
        "keep": len(iter_status_by_category("keep")),
        "modify": len(iter_status_by_category("modify")),
        "legacy": len(iter_status_by_category("legacy")),
    }


def all_status_as_dicts() -> list[dict[str, str]]:
    return [item.to_dict() for item in PROJECT_COMPONENT_STATUS]
