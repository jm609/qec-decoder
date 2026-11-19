from types import SimpleNamespace
import unittest

import numpy as np

from tools.probe_d7_z_transition import counts, patch_features, patch_lookup, recovery_cutoff


class ZTransitionProbeTest(unittest.TestCase):
    def test_patch_anchors_on_first_coordinate_and_padding_has_no_events(self):
        layout = SimpleNamespace(
            detector_time_index=np.array([0, 0]),
            row_index_by_detector=np.array([0, 0]),
            col_index_by_detector=np.array([1, 0]),
            detector_index_volume=np.array([[[1, 0]]]),
        )
        lookup, order = patch_lookup(layout)
        geometry, events = patch_features(np.array([[1, 0]]), np.array([[1, 1]]), lookup, order)
        np.testing.assert_array_equal(np.flatnonzero(geometry[0, :27]), [13, 14])
        np.testing.assert_array_equal(np.flatnonzero(events[0, :27]), [14])
        self.assertEqual(events[0, -1], 1)

    def test_recovery_point_admits_all_score_ties(self):
        scores = np.array([0.4, 0.8, 0.8, 0.4, 0.9])
        outcomes = np.array([1, 1, -1, -1, 0])
        cutoff = recovery_cutoff(scores, outcomes, target=1)
        self.assertEqual(counts(outcomes, scores >= cutoff), {'repairs': 1, 'harms': 1, 'both_wrong': 1})


if __name__ == '__main__':
    unittest.main()
