import unittest

import numpy as np

from tools.run_d7_paired_selector import Cases, count_routes, preserving_rows, select_rows


class PairedSelectorTest(unittest.TestCase):
    def test_rejection_uses_preserving_edit_or_raw_and_keeps_other_shots_on_base_route(self):
        raw = np.array([0, 1, 3])
        prediction = np.array([0, 2, 0, 1, 3, 1, 3, 2, 2])
        weights = np.tile([0, 1, 1], 3)
        shot = np.repeat(np.arange(3), 3)
        utility = np.tile([0.0, 1.0, 0.1], 3)
        keeping = preserving_rows(raw, prediction, shot, weights, utility)
        np.testing.assert_array_equal(keeping, [2, 5, len(shot)])

        case = Cases(raw, np.array([0, 3, 3]), prediction, weights, np.array([1, 4, 7]),
                     np.array([0, 3, 6]), np.array([0, 3, 6]), np.array([0, 1]), keeping[:2],
                     np.empty((2, 0)), np.empty((2, 0)))
        self.assertEqual(case.outcomes().tolist(), [-1, 1])
        fixed, free = select_rows(case, accept_z=np.array([False, True]))
        np.testing.assert_array_equal(fixed, [2, 4, 7])
        np.testing.assert_array_equal(free, [0, 4, 6])
        fixed_counts = count_routes([case], [fixed])
        free_counts = count_routes([case], [free])
        self.assertEqual((fixed_counts['edits'], fixed_counts['corrections'], fixed_counts['harms']), (3, 1, 1))
        self.assertEqual((free_counts['edits'], free_counts['corrections'], free_counts['harms']), (1, 1, 0))


if __name__ == '__main__':
    unittest.main()
