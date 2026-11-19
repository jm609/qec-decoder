import unittest

import numpy as np

from tools.evaluate_d7_paired_selector import summarize


class PairedSelectorEvaluationTest(unittest.TestCase):
    def test_summary_bootstraps_the_same_maps_for_both_routes(self):
        def model(margin_net, head_net):
            net = {'top1': 0, 'margin': margin_net, 'fixed': head_net, 'free': head_net}
            routes = {name: {'shots': 1000, 'edits': 20, 'corrections': 10 + count, 'harms': 10,
                             'changed_rows_vs_top1': 0, 'gain_pp': count / 10}
                      for name, count in net.items()}
            return {'routes': routes, 'candidate_rows': 3000, 'top1_z_shots': 20,
                    'paired_shots': 20, 'accepted_z': 10, 'fixed_free_prediction_differences': 0}

        records = [
            {'local_seed': 1, 'models': {'a': model(3, 4), 'b': model(1, 2)}},
            {'local_seed': 2, 'models': {'a': model(-1, 0), 'b': model(-3, -2)}},
        ]
        result = summarize(records, 'test')
        free = result['routes']['free']
        self.assertEqual((free['shots'], free['corrections'], free['harms']), (4000, 44, 40))
        self.assertAlmostEqual(free['gain_vs_raw_pp'], 0.1)
        np.testing.assert_allclose(free['interval_vs_raw_95_pp'], [-0.1, 0.3])
        np.testing.assert_allclose(free['interval_vs_margin_95_pp'], [0.1, 0.1])


if __name__ == '__main__':
    unittest.main()
