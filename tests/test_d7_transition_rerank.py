import unittest

import numpy as np

from tools.run_d7_transition_rerank import compare, development_rows, select_rows


def bits(classes):
    classes = np.asarray(classes, dtype=np.uint8)
    return np.column_stack((classes >> 1, classes & 1))


class TransitionRerankTest(unittest.TestCase):
    def test_calibrated_transition_preserves_fixed_action_and_counts_recoveries(self):
        arrays = {
            'd0_mwpm_raw': bits([0, 2, 1]),
            'd0_mwpm_predicted': bits([0, 2, 0, 2, 0, 2, 1, 3, 1]),
            'truth': bits([0, 0, 3]),
            'utility': np.array([2, 3, 1, 0, 2, 1, 2, 1, 0]),
            'shot_index': np.repeat(np.arange(3), 3),
            'weights': np.tile([0, 1, 1], 3),
            'top1_rows': np.array([1, 4, 6]),
        }
        bias = np.zeros((4, 4))
        bias[0, 2], bias[2, 2], bias[1, 2] = -3, 1, 2
        calibration = {'utility_scale': 1, 'transition_bias_by_raw_class': bias.tolist()}
        _, scores, target = development_rows(arrays)
        np.testing.assert_array_equal(scores[:, 0], [0, 1, 0])
        np.testing.assert_array_equal(scores[:, 2], [1, 2, -1])
        np.testing.assert_array_equal(target, [0, 2, 2])

        rows = select_rows(arrays, calibration)
        np.testing.assert_array_equal(rows['fixed'], [2, 4, 6])
        np.testing.assert_array_equal(rows['free'], [0, 4, 7])
        metrics = compare(arrays, calibration)
        self.assertEqual(metrics['top1']['harmed'], 1)
        self.assertEqual(metrics['fixed']['avoided_harms'], 1)
        self.assertEqual(metrics['fixed']['retained_corrections'], 1)
        self.assertEqual(metrics['fixed']['new_corrections'], 0)
        self.assertEqual(metrics['free']['new_corrections'], 1)


if __name__ == '__main__':
    unittest.main()
