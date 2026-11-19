from itertools import product
import unittest

import numpy as np
import pymatching

from tools.d7_boundary_matching_costs import BoundaryCosts, checked_costs, z_gap


class BoundaryMatchingCostsTest(unittest.TestCase):
    def test_four_classes_against_exhaustive_subsets(self) -> None:
        matching = pymatching.Matching()
        matching.add_edge(0, 1, weight=1.0)
        matching.add_edge(1, 2, weight=1.5)
        matching.add_boundary_edge(0, weight=2.0)
        matching.add_boundary_edge(1, fault_ids={0, 1}, weight=4.0)
        matching.add_boundary_edge(2, fault_ids={1}, weight=2.5)
        matching.add_edge(3, 4, weight=2.0)
        matching.add_boundary_edge(3, weight=1.5)
        matching.add_boundary_edge(4, fault_ids={0}, weight=3.0)
        edges = matching.edges()
        brute = np.full((32, 4), np.inf)
        for subset in product((0, 1), repeat=len(edges)):
            syndrome, logical, cost = 0, 0, 0.0
            for enabled, (u, v, data) in zip(subset, edges, strict=True):
                if enabled:
                    syndrome ^= (1 << u) ^ (0 if v is None else 1 << v)
                    logical ^= sum(2 >> bit for bit in data['fault_ids'])
                    cost += data['weight']
            brute[syndrome, logical] = min(brute[syndrome, logical], cost)
        syndromes = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.uint8)
        solver = BoundaryCosts(matching)
        _, costs = checked_costs(matching, solver, syndromes)
        np.testing.assert_allclose(costs, brute, atol=1e-5)
        np.testing.assert_allclose(solver.solve(syndromes, independent=True), brute, atol=1e-9)

    def test_unreachable_and_unsupported_edges(self) -> None:
        matching = pymatching.Matching()
        matching.add_boundary_edge(0, fault_ids={1}, weight=1.0)
        solver = BoundaryCosts(matching)
        costs = solver.solve(np.array([[0], [1]], dtype=np.uint8))
        np.testing.assert_array_equal(z_gap(np.array([0, 1]), costs), [[0, 1], [0, 1]])
        matching.add_edge(0, 1, fault_ids={0}, weight=2.0)
        with self.assertRaises(ValueError):
            BoundaryCosts(matching)


if __name__ == '__main__':
    unittest.main()
