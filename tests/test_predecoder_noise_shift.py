import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools import run_predecoder_noise_shift as shift
from tools.run_predecoder_noise_shift import Decoder, decode_edits, summarize_predictions, top_rows


class NoiseShiftTest(unittest.TestCase):
    def test_import_does_not_read_local_protocol(self) -> None:
        with patch.object(shift.rv, 'load', side_effect=AssertionError('Import read a local artifact')):
            loaded = importlib.reload(shift)
        self.assertEqual((loaded.G0, loaded.G1), (0.05, 0.25))
        self.assertEqual(loaded.PANEL, [2, 11, 54, 57])

    def test_top1_prefers_no_edit_on_exact_ties(self) -> None:
        bundle = SimpleNamespace(
            candidate_edit_weight=np.array([0, 1, 0, 1, 2]),
            shot_indices=np.array([0, 0, 1, 1, 1]), num_shots=2,
        )
        raw, best = top_rows(bundle, np.array([0.5, 0.5, 0.0, 0.2, 0.2]))
        np.testing.assert_array_equal(raw, [0, 2])
        np.testing.assert_array_equal(best, [0, 3])

    def test_unmatchable_edit_falls_back_to_same_cell_raw_prediction(self) -> None:
        class Matching:
            def decode_batch(self, events, *, enable_correlations):
                raise ValueError('No perfect matching could be found')

            def decode(self, event, *, enable_correlations):
                if not enable_correlations:
                    raise AssertionError('The correlated decoder flag was lost')
                if event[0]:
                    raise ValueError('No perfect matching could be found')
                return [0, 1]

        decoder = Decoder(Matching(), correlated=True)
        raw = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.uint8)
        events = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.uint8)
        masks = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8)
        predicted, failed = decode_edits(decoder, events, masks, raw)
        np.testing.assert_array_equal(predicted, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(failed, [True, False, False])
        self.assertEqual(decoder.counts(), {'batch_calls': 1, 'single_calls': 2, 'rows': 4})

    def test_oracle_keeps_raw_correct_shots_and_final_can_return_raw(self) -> None:
        arrays = {
            'truth': np.array([[1, 0], [1, 1]]), 'shot_index': np.array([0, 0, 1, 1]),
            'weights': np.array([0, 1, 0, 1]), 'top1_rows': np.array([1, 3]),
            'branch_rows': np.array([0, 3]), 'final_rows': np.array([0, 2]),
            'd0_mwpm_raw': np.array([[0, 0], [1, 1]]),
            'd0_mwpm_predicted': np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            'd0_mwpm_failure': np.zeros(4, dtype=bool),
        }
        result = summarize_predictions(arrays, ['d0_mwpm'])['d0_mwpm']
        self.assertEqual(result['oracle_gain_pp'], 50.0)
        self.assertEqual(result['top1']['gain_pp'], 0.0)
        self.assertEqual(result['branch']['gain_pp'], -50.0)
        self.assertEqual(result['final']['attempted_edits'], 0)
        self.assertIsNone(result['final']['harm_per_edit'])


if __name__ == '__main__':
    unittest.main()
