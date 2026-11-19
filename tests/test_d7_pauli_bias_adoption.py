import unittest

import numpy as np

from tools.train_d7_pauli_bias_adoption import apply_policy, select_threshold


class PauliBiasAdoptionTest(unittest.TestCase):
    def setUp(self):
        self.head = dict(center=[0.0], scale=[1.0],
                         beta=[[0.2, 0.8, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.data = dict(features=np.zeros((2, 1)), raw=np.array([0, 2], dtype=np.uint8),
                         class_rows=np.array([[0, 1, 2, 3], [0, -1, 2, 3]]),
                         classes=np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=np.uint8),
                         truth=np.array([1, 2], dtype=np.uint8))

    def test_inference_uses_candidate_support_and_ignores_truth(self):
        selected, rows, _ = apply_policy(self.data, self.head, 0.0)
        np.testing.assert_array_equal(selected, [1, 2])
        np.testing.assert_array_equal(rows, [1, 0])
        poisoned = {**self.data, "truth": self.data["truth"] ^ np.uint8(3)}
        np.testing.assert_array_equal(apply_policy(poisoned, self.head, 0.0)[0], selected)
        np.testing.assert_array_equal(apply_policy(self.data, self.head, 1.0)[0], self.data["raw"])

    def test_threshold_selection_keeps_raw_when_development_has_no_net_gain(self):
        harmful = {**self.data, "truth": self.data["raw"].copy()}
        threshold, grid = select_threshold([harmful], self.head)
        self.assertEqual(threshold, 1.0)
        self.assertEqual(max(row["mean_gain_pp"] for row in grid), 0)
        profitable, _ = select_threshold([self.data], self.head)
        self.assertLess(profitable, 1.0)
        self.assertEqual(apply_policy(self.data, self.head, profitable)[0][0], 1)


if __name__ == "__main__":
    unittest.main()
