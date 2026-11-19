import unittest

import numpy as np

from tools.analyze_d7_logical_transitions import analyze_cell, summarize


def observable_bits(classes):
    classes = np.asarray(classes, dtype=np.uint8)
    return np.column_stack((classes >> 1, classes & 1))


class LogicalTransitionTest(unittest.TestCase):
    def test_oracle_gap_decomposes_by_edit_decision(self):
        identity = np.arange(0, 21, 3)
        top = identity + [1, 1, 0, 1, 1, 1, 0]
        utility = np.zeros(21, dtype=np.float32)
        utility[top] = 1
        arrays = {
            'truth': observable_bits([0, 1, 2, 2, 2, 2, 0]),
            'd0_mwpm_raw': observable_bits([0, 1, 0, 0, 1, 1, 0]),
            'd0_mwpm_predicted': observable_bits([
                0, 1, 3, 1, 3, 1, 0, 2, 3, 0, 1, 2, 1, 3, 0, 1, 2, 0, 0, 1, 3,
            ]),
            'd0_mwpm_failure': np.zeros(21, dtype=bool),
            'shot_index': np.repeat(np.arange(7), 3),
            'weights': np.tile([0, 1, 1], 7),
            'utility': utility,
            'top1_rows': top,
            'branch_rows': top,
            'final_rows': identity,
        }

        result = summarize([analyze_cell(arrays, 'd0_mwpm')])
        self.assertEqual(result['top1']['selected_correct'], 2)
        self.assertEqual(result['top1']['fixed_edit_oracle_correct'], 4)
        self.assertEqual(result['top1']['oracle_correct'], 6)
        for name in ('both_correct', 'improved', 'harm_alternative_edit', 'harm_requires_no_edit',
                     'missed_recoverable_edited', 'missed_recoverable_no_edit', 'no_correct_candidate'):
            with self.subTest(name=name):
                self.assertEqual(result['top1'][name], 1)
        self.assertEqual(result['final']['harm_requires_no_edit'], 0)
        self.assertEqual(result['class_scores']['missed_correct_class_utility_gap_mean'], 1)


if __name__ == '__main__':
    unittest.main()
