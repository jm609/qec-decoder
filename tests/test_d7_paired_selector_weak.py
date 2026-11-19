from types import SimpleNamespace
import unittest

import numpy as np

from tools.evaluate_d7_paired_selector_weak import decoded_from_transition, sedp


class PairedSelectorWeakTest(unittest.TestCase):
    def test_reuses_the_two_encoded_matching_observables_in_shot_order(self):
        raw = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        shot = np.array([0, 0, 1, 1])
        decoded = np.array([[0, 1], [1, 1], [1, 0], [0, 1]], dtype=np.uint8)
        transition = sedp._candidate_transition_feature_matrix(
            baseline_predicted_observables=raw[shot], edited_predicted_observables=decoded,
        )
        geometry = np.zeros((len(shot), sedp.SELECTOR_CANDIDATE_BASE_FEATURE_DIM))
        bundle = SimpleNamespace(candidate_features=np.column_stack((geometry, transition)),
                                 shot_indices=shot, decode_fallback_count=0)
        np.testing.assert_array_equal(decoded_from_transition(bundle, raw), decoded)


if __name__ == '__main__':
    unittest.main()
