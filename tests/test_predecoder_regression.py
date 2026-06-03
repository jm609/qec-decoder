from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders import syndrome_edit_predecoder as sedp  # noqa: E402
from tools import build_final_result_consistency_summary as final_consistency  # noqa: E402
from tools import check_main_tex_static  # noqa: E402


def _require_path(path: Path) -> None:
    if not path.exists():
        raise unittest.SkipTest(f"local regression artifact is missing: {path}")


def _selector_result(
    *,
    margin: float,
    nonzero: int,
    improved: int = 0,
    harmed: int = 0,
    profile: list[dict[str, float | int]] | None = None,
) -> dict[str, object]:
    return {
        "decision": {
            "selector_emit_margin": margin,
            "selector_margin_profile": profile or [],
        },
        "change_summary": {
            "predicted_edit_weight_histogram": {"0": 100 - nonzero, "1": nonzero},
            "num_improved_over_baseline": improved,
            "num_harmed_vs_baseline": harmed,
        },
    }


class CandidateFirstSafetyRegressionTest(unittest.TestCase):
    def test_strong_validation_delta_selects_requested_mode(self) -> None:
        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=0.90,
            global_metric=0.901,
            selector_metric=0.925,
            selector_results=[_selector_result(margin=0.0, nonzero=12, improved=9, harmed=4)],
            requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
        )

        self.assertEqual(mode, sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR)
        self.assertEqual(decision["reason"], "candidate_strong_validation_delta")

    def test_positive_delta_harm_guard_blocks_candidate(self) -> None:
        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=0.90,
            global_metric=0.90,
            selector_metric=0.906,
            selector_results=[_selector_result(margin=1.0, nonzero=12, improved=9, harmed=3)],
            requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
            positive_max_harmed=2,
        )

        self.assertEqual(mode, sedp.SELECTION_MODE_RAW_NO_EDIT)
        self.assertEqual(decision["reason"], "candidate_positive_delta_harm_guard")

    def test_tie_high_margin_with_support_selects_candidate(self) -> None:
        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=0.90,
            global_metric=0.90,
            selector_metric=0.90,
            selector_results=[_selector_result(margin=1.25, nonzero=8, improved=4, harmed=1)],
            requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
        )

        self.assertEqual(mode, sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR)
        self.assertEqual(decision["reason"], "candidate_tie_with_high_margin_evidence")

    def test_global_policy_requires_explicit_allow_flag(self) -> None:
        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=0.90,
            global_metric=0.92,
            selector_metric=None,
            selector_results=None,
            requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
            allow_global=False,
        )
        self.assertEqual(mode, sedp.SELECTION_MODE_RAW_NO_EDIT)
        self.assertEqual(decision["reason"], "default_no_edit")

        mode, decision = sedp._candidate_first_safety_adoption_mode(
            no_edit_metric=0.90,
            global_metric=0.92,
            selector_metric=None,
            selector_results=None,
            requested_mode=sedp.SELECTION_MODE_LOCAL_MOTIF_SELECTOR,
            allow_global=True,
            global_min_delta=0.01,
        )
        self.assertEqual(mode, sedp.SELECTION_MODE_GLOBAL_POLICY)
        self.assertEqual(decision["reason"], "global_delta_clears_guard")


class SelectorHelperRegressionTest(unittest.TestCase):
    def test_candidate_feature_dim_and_patch_slice_offsets(self) -> None:
        dim = sedp._selector_candidate_feature_dim(
            sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM,
            candidate_geometry_features=True,
            candidate_pattern_features=True,
            candidate_local_evidence_features=True,
            candidate_local_patch_features=True,
        )
        expected = (
            sedp.SELECTOR_CANDIDATE_BASE_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM
            + 27
        )
        self.assertEqual(dim, expected)

        patch_slice = sedp._candidate_local_patch_feature_slice(
            candidate_geometry_features=True,
            candidate_pattern_features=True,
            candidate_local_evidence_features=True,
            candidate_local_patch_features=True,
        )
        assert patch_slice is not None
        self.assertEqual(
            patch_slice.start,
            sedp.SELECTOR_CANDIDATE_BASE_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_GEOMETRY_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_PATTERN_FEATURE_DIM
            + sedp.SELECTOR_CANDIDATE_LOCAL_EVIDENCE_FEATURE_DIM,
        )
        self.assertEqual(patch_slice.stop - patch_slice.start, sedp.SELECTOR_CANDIDATE_LOCAL_PATCH_FEATURE_DIM)
        self.assertIsNone(sedp._candidate_local_patch_feature_slice(candidate_local_patch_features=False))

    def test_candidate_policy_top_k_and_needs_threshold(self) -> None:
        probs = np.asarray([0.2, 0.9, 0.4, 0.9], dtype=np.float32)
        chosen = sedp._choose_candidate_indices_for_policy(
            probs,
            needs_edit_prob=0.8,
            needs_edit_threshold=0.5,
            edit_threshold=0.5,
            max_predicted_edit_weight=2,
        )
        np.testing.assert_array_equal(chosen, np.asarray([1, 3], dtype=np.int64))

        blocked = sedp._choose_candidate_indices_for_policy(
            probs,
            needs_edit_prob=0.2,
            needs_edit_threshold=0.5,
            edit_threshold=0.5,
            max_predicted_edit_weight=2,
        )
        self.assertEqual(blocked.shape, (0,))

    def test_benefit_harm_scores_baseline_correct_and_incorrect(self) -> None:
        baseline_correct = sedp._candidate_selector_target_scores(
            candidate_is_correct=np.asarray([1, 0, 1], dtype=np.uint8),
            candidate_edit_weight=np.asarray([0, 1, 2], dtype=np.int16),
            selector_target_mode=sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM,
            selector_score_edit_penalty=0.1,
            selector_harm_weight=2.0,
            selector_miss_weight=0.25,
        )
        np.testing.assert_allclose(baseline_correct, np.asarray([0.0, -2.1, -0.2], dtype=np.float32))

        baseline_incorrect = sedp._candidate_selector_target_scores(
            candidate_is_correct=np.asarray([0, 1, 0], dtype=np.uint8),
            candidate_edit_weight=np.asarray([0, 1, 2], dtype=np.int16),
            selector_target_mode=sedp.SELECTOR_TARGET_MODE_BENEFIT_HARM,
            selector_score_edit_penalty=0.1,
            selector_harm_weight=2.0,
            selector_miss_weight=0.25,
        )
        np.testing.assert_allclose(baseline_incorrect, np.asarray([0.0, 0.9, -0.45], dtype=np.float32))

    def test_transition_feature_matrix_layout(self) -> None:
        features = sedp._candidate_transition_feature_matrix(
            baseline_predicted_observables=np.asarray([[0, 0], [1, 0]], dtype=np.uint8),
            edited_predicted_observables=np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
        )
        self.assertEqual(features.shape, (2, 27))
        np.testing.assert_array_equal(features[:, 0:2], np.asarray([[1, 0], [0, 1]], dtype=np.float32))
        np.testing.assert_array_equal(features[:, 2], np.asarray([1, 1], dtype=np.float32))
        np.testing.assert_array_equal(features[0, 3:7], np.asarray([1, 0, 0, 0], dtype=np.float32))
        np.testing.assert_array_equal(features[0, 7:11], np.asarray([0, 0, 1, 0], dtype=np.float32))
        self.assertEqual(float(features[0, 11 + 2]), 1.0)
        self.assertEqual(float(features[1, 11 + 11]), 1.0)

    def test_patch_head_forward_shape_and_invalid_slice_guard(self) -> None:
        if sedp.torch is None:
            self.skipTest("PyTorch is not installed")
        sedp.torch.manual_seed(0)
        selector = sedp.CandidateEditSelector(
            shot_feature_dim=2,
            candidate_feature_dim=5,
            hidden_dim=4,
            dropout=0.0,
            patch_feature_offset=2,
            patch_feature_dim=3,
            patch_hidden_dim=2,
        )
        out = selector(
            sedp.torch.zeros((4, 2), dtype=sedp.torch.float32),
            sedp.torch.zeros((4, 5), dtype=sedp.torch.float32),
        )
        self.assertEqual(tuple(out.shape), (4,))

        with self.assertRaises(ValueError):
            sedp.CandidateEditSelector(
                shot_feature_dim=2,
                candidate_feature_dim=5,
                hidden_dim=4,
                dropout=0.0,
                patch_feature_offset=4,
                patch_feature_dim=2,
            )


class SummaryArtifactRegressionTest(unittest.TestCase):
    def test_final_result_tables_match_consolidated_json(self) -> None:
        _require_path(ROOT / "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json")
        _require_path(ROOT / "PREDECODER_FINAL_RESULT_TABLES.md")
        summary = final_consistency.build_summary(
            ROOT / "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json",
            ROOT / "PREDECODER_FINAL_RESULT_TABLES.md",
        )
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["num_failed"], 0)

    def test_seed_expansion_ci_summary_values(self) -> None:
        path = ROOT / "artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        by_distance = {row["distance"]: row for row in data["distance_results"]}

        d3 = by_distance["d3"]
        self.assertEqual(d3["selected_delta"]["class_counts"], {"harmful": 0, "neutral": 0, "positive": 8})
        self.assertAlmostEqual(d3["selected_delta"]["mean"], 0.006591796875)
        self.assertAlmostEqual(d3["selected_delta"]["bootstrap_ci_95_low"], 0.0045166015625)
        self.assertAlmostEqual(d3["selected_delta"]["bootstrap_ci_95_high"], 0.008544921875)

        d5 = by_distance["d5"]
        self.assertEqual(d5["mode_counts"], {"local_motif_selector": 2, "raw_no_edit": 6})
        self.assertEqual(d5["candidate_harm_blocked_seeds"], [4, 6])
        self.assertAlmostEqual(d5["selected_delta"]["mean"], 0.005615234375)
        self.assertAlmostEqual(d5["selected_delta"]["bootstrap_ci_95_low"], 0.0)

    def test_paired_statistics_summary_values(self) -> None:
        path = ROOT / "artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        by_distance = {row["distance"]: row for row in data["distance_results"]}

        self.assertEqual(data["schema_version"], "predecoder_d3_d5_paired_statistics.v1")
        self.assertAlmostEqual(
            by_distance["d3"]["sign_test_excluding_zero_deltas"]["one_sided_positive_p"],
            0.00390625,
        )
        self.assertAlmostEqual(
            by_distance["d3"]["exact_sign_flip_mean_test_excluding_zero_deltas"]["two_sided_p"],
            0.0078125,
        )
        self.assertAlmostEqual(
            by_distance["d5"]["sign_test_excluding_zero_deltas"]["one_sided_positive_p"],
            0.25,
        )
        self.assertEqual(by_distance["d5"]["mode_counts"], {"local_motif_selector": 2, "raw_no_edit": 6})

    def test_d7_validation_heldout_scatter_summary_values(self) -> None:
        path = ROOT / "artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["num_rows"], 58)
        self.assertEqual(data["validation_positive_count"], 22)
        self.assertEqual(data["validation_positive_harmful_count"], 13)
        self.assertEqual(data["validation_positive_true_positive_count"], 5)
        self.assertAlmostEqual(data["validation_positive_false_positive_ratio"], 13 / 22)
        self.assertLess(data["pearson_correlation"], 0.0)

    def test_oracle_recovery_distribution_summary_values(self) -> None:
        path = ROOT / "artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        by_distance = {row["distance"]: row for row in data["distance_results"]}

        self.assertAlmostEqual(
            by_distance["d3"]["selected_recovery_fraction_distribution"]["mean"],
            0.10384615384615384,
        )
        self.assertAlmostEqual(
            by_distance["d5"]["selected_recovery_fraction_distribution"]["mean"],
            0.0625,
        )
        self.assertAlmostEqual(
            by_distance["d7"]["selected_recovery_fraction_distribution"]["mean"],
            0.0013611615245009074,
        )
        self.assertAlmostEqual(
            by_distance["d7"]["candidate_oracle_recovery_fraction_distribution"]["mean"],
            0.8684210526315778,
        )

    def test_hyperparameter_sensitivity_summary_values(self) -> None:
        path = ROOT / "artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = {row["value"]: row for row in data["rows"]}

        self.assertEqual(data["schema_version"], "predecoder_hyperparameter_sensitivity.v1")
        self.assertEqual(set(rows), {0.25, 0.5, 1.0})
        self.assertAlmostEqual(rows[0.5]["mean_selected_delta_over_no_edit"], 0.0022786458333333335)
        self.assertEqual(
            rows[0.5]["selected_delta_class_counts"],
            {"harmful": 0, "neutral": 1, "positive": 2},
        )
        self.assertAlmostEqual(rows[0.25]["seed0_selected_delta"], -0.0029296875)
        self.assertEqual(rows[1.0]["mode_counts"], {"raw_no_edit": 3})

    def test_main_tex_static_check_has_no_errors(self) -> None:
        _require_path(ROOT / "artifacts/figures/predecoder_v2")
        summary = check_main_tex_static.build_summary(
            ROOT / "main.tex",
            ROOT / "artifacts/figures/predecoder_v2",
        )

        self.assertTrue(summary["pass"])
        self.assertEqual(summary["num_failed_errors"], 0)
        self.assertEqual(summary["counts"]["predecoder_figures"], 6)
        self.assertEqual(summary["counts"]["bibitems"], 10)
        self.assertFalse(summary["stale_matches"]["old_evaluation_score"])
        self.assertFalse(summary["stale_matches"]["d5_uniformly_positive_claim"])

    def test_figure_package_summary_covers_all_manuscript_figures(self) -> None:
        path = ROOT / "artifacts/figures/predecoder_v2/predecoder_v2_figure_summary.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        figure_paths = {Path(row["outputs"]["png"]).name for row in data["figures"]}

        self.assertEqual(data["schema_version"], "predecoder_matplotlib_figures.v2")
        self.assertEqual(len(data["figures"]), 6)
        self.assertEqual(
            figure_paths,
            {
                "fig1_predecoder_pipeline_v2.png",
                "fig2_model_architecture_v2.png",
                "fig3_main_accuracy_comparison_v2.png",
                "fig4_d7_oracle_gap_false_positive_v2.png",
                "fig5_d7_validation_heldout_scatter_v2.png",
                "fig6_oracle_recovery_distribution_v2.png",
            },
        )
        for row in data["figures"]:
            self.assertTrue((ROOT / row["outputs"]["png"]).exists())
            self.assertTrue((ROOT / row["outputs"]["pdf"]).exists())

    def test_overleaf_package_manifest_contains_required_files(self) -> None:
        path = ROOT / "artifacts/overleaf_predecoder_package/overleaf_package_manifest.json"
        _require_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        copied = {Path(row).as_posix() for row in data["copied_files"]}

        self.assertTrue(data["pass"])
        self.assertEqual(data["compiler"], "XeLaTeX")
        self.assertEqual(data["main_file"], "main.tex")
        self.assertEqual(data["figure_priority"], ["pdf", "png"])
        self.assertFalse(data["missing_required"])
        self.assertTrue((ROOT / data["zip_path"]).exists())
        self.assertIn("artifacts/overleaf_predecoder_package/main.tex", copied)
        self.assertIn("artifacts/overleaf_predecoder_package/main.bib", copied)
        self.assertIn("artifacts/overleaf_predecoder_package/OVERLEAF_COMPILE_GUIDE.md", copied)
        for name in [
            "fig1_predecoder_pipeline_v2.png",
            "fig2_model_architecture_v2.png",
            "fig3_main_accuracy_comparison_v2.png",
            "fig4_d7_oracle_gap_false_positive_v2.png",
            "fig5_d7_validation_heldout_scatter_v2.png",
            "fig6_oracle_recovery_distribution_v2.png",
        ]:
            self.assertIn(f"artifacts/overleaf_predecoder_package/artifacts/figures/predecoder_v2/{name}", copied)


if __name__ == "__main__":
    unittest.main()
