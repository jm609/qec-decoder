import unittest

import numpy as np

try:
    from tools.run_d7_matching_rerank import matching_feature, rerank_rows
except FileNotFoundError as error:
    if error.filename and error.filename.endswith('protocol.json'):
        raise unittest.SkipTest('The local frozen revalidation protocol is missing') from error
    raise


class MatchingRerankTest(unittest.TestCase):
    def test_fixed_rate_keeps_original_edit_shots_and_reranks_nonzero_masks(self) -> None:
        shot = np.repeat(np.arange(2), 3)
        weights = np.tile([0, 1, 1], 2)
        logits = np.array([0.5, 0.4, 0.3, 1.0, 1.1, 0.7])
        cost_feature = np.array([0.0, 1.0, 0.0, 0.0, -1.0, -0.1])
        baseline = np.array([0, 4])
        fixed = rerank_rows(logits, cost_feature, shot, weights, baseline, 1.0, fixed_edit_rate=True)
        free = rerank_rows(logits, cost_feature, shot, weights, baseline, 1.0, fixed_edit_rate=False)
        np.testing.assert_array_equal(fixed, [0, 5])
        np.testing.assert_array_equal(free, [1, 3])
        np.testing.assert_array_equal(rerank_rows(
            logits, cost_feature, shot, weights, baseline, 0.0, fixed_edit_rate=True,
        ), baseline)

    def test_matching_cost_uses_shot_correct_identity_and_edited_syndrome(self) -> None:
        import pymatching

        matching = pymatching.Matching()
        matching.add_edge(0, 1, fault_ids={0}, weight=2)
        matching.add_boundary_edge(0, weight=3)
        matching.add_boundary_edge(1, weight=4)
        events = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        masks = np.array([[0, 0], [1, 1], [0, 0], [0, 1]], dtype=np.uint8)
        raw, decoded, feature, failure = matching_feature(matching, events, masks, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(raw[:, 0], [1, 0])
        np.testing.assert_array_equal(decoded[:, 0], [1, 0, 0, 1])
        np.testing.assert_allclose(feature, [0, 1, 0, 1/3])
        self.assertFalse(failure.any())


if __name__ == '__main__':
    unittest.main()
