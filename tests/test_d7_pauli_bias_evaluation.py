import unittest

import numpy as np
from scipy.stats import t

from tools.evaluate_d7_pauli_bias import summarize


def record(seed, improved, harmed):
    net = improved - harmed
    comparison = dict(shots=1000, raw_correct=900, selected_correct=900 + net,
                      improved=improved, harmed=harmed, gain_pp=net / 10)
    return dict(
        local_seed=seed,
        primary={**comparison, "logical_changes": improved + harmed, "decode_fallbacks": 0},
        matched_vs_uniform=comparison,
        selected_vs_uniform=comparison,
    )


class PauliBiasEvaluationTest(unittest.TestCase):
    def test_primary_interval_uses_map_variation_and_strict_positive_lower_bound(self):
        rows = [record(1, 20, 10), record(2, 10, 20), record(3, 20, 10)]
        result = summarize(rows)
        primary = result["primary"]
        gains = np.array([1.0, -1.0, 1.0])
        half = t.ppf(0.975, 2) * gains.std(ddof=1) / np.sqrt(3)
        np.testing.assert_allclose(primary["ci95_pp"], [gains.mean() - half, gains.mean() + half])
        self.assertAlmostEqual(primary["gain_pp"], 1 / 3)
        self.assertEqual((primary["improved"], primary["harmed"], primary["shots"]), (50, 40, 3000))
        self.assertFalse(primary["supports_improvement"])
        self.assertEqual(result["logical_changes"], 90)
        self.assertTrue(summarize([record(1, 20, 10), record(2, 20, 10)])["primary"]["supports_improvement"])
        self.assertFalse(summarize([record(1, 10, 10), record(2, 10, 10)])["primary"]["supports_improvement"])


if __name__ == "__main__":
    unittest.main()
