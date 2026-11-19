from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from decoders import research_noise_aware_3d as volume
from decoders import syndrome_edit_predecoder as sedp
from tools import evaluate_learned_hybrid_router as router


class LabelFreeContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "stage": "C",
            "qc_stats": {
                "detector_event_fraction": 0.1,
                "logical_flip_fraction": 0.2,
                "logical_x_flip_fraction": 0.3,
                "logical_z_flip_fraction": 0.4,
            },
        }
        self.changed = copy.deepcopy(self.metadata)
        for name in self.changed["qc_stats"]:
            if name.startswith("logical_"):
                self.changed["qc_stats"][name] = 1.0
        self.layout = SimpleNamespace(
            valid_mask=np.ones((2, 2, 2), dtype=np.uint8),
            time_steps=2, height=2, width=2,
        )

    def test_volume_context_ignores_labels_and_keeps_checkpoint_width(self) -> None:
        original, names = volume._build_context_planes(self.metadata, self.layout)
        changed, changed_names = volume._build_context_planes(self.changed, self.layout)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(names, changed_names)
        self.assertEqual(original.shape, (28, 2, 2, 2))
        self.assertTrue(np.all(original[names.index("reserved_label_context")] == 0))
        # A missing label aggregate must also produce the same inference input.
        for name in list(self.changed["qc_stats"]):
            if name.startswith("logical_"):
                del self.changed["qc_stats"][name]
        missing, _ = volume._build_context_planes(self.changed, self.layout)
        np.testing.assert_array_equal(original, missing)

    def test_router_context_ignores_all_three_label_aggregates(self) -> None:
        prepared = {"x": np.ones((2, 8, 2, 2, 2), dtype=np.float32)}
        original, names = router._metadata_context_features(
            metadata=self.metadata, prepared=prepared, num_shots=2,
        )
        changed, _ = router._metadata_context_features(
            metadata=self.changed, prepared=prepared, num_shots=2,
        )
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(original.shape, (2, 30))
        reserved = [i for i, name in enumerate(names) if "reserved" in name]
        self.assertEqual(len(reserved), 3)
        self.assertTrue(np.all(original[:, reserved] == 0))

    def test_selector_action_does_not_depend_on_evaluation_targets(self) -> None:
        bundle = SimpleNamespace(
            num_shots=1, num_detectors=2,
            shot_features=np.zeros((2, 1), dtype=np.float32),
            candidate_features=np.zeros((2, 100), dtype=np.float32),
            shot_indices=np.array([0, 0]),
            candidate_edit_mask=np.array([[0, 0], [1, 0]], dtype=np.uint8),
            candidate_edit_weight=np.array([0, 1]),
            target_scores=np.array([0.0, 1.0]),
            candidate_is_correct=np.array([0, 1], dtype=np.uint8),
        )
        logits = np.array([0.0, 2.0], dtype=np.float32)
        original = sedp._select_candidate_masks_from_logits(
            bundle=bundle, selector_logits=logits, selector_emit_margin=1.0,
        )
        bundle.target_scores = np.array([100.0, -100.0])
        bundle.candidate_is_correct = 1 - bundle.candidate_is_correct
        changed = sedp._select_candidate_masks_from_logits(
            bundle=bundle, selector_logits=logits, selector_emit_margin=1.0,
        )
        np.testing.assert_array_equal(original["selected_edit_mask"], changed["selected_edit_mask"])
        np.testing.assert_array_equal(changed["selected_edit_mask"], [[1, 0]])


if __name__ == "__main__":
    unittest.main()
