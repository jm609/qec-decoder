"""Class-constrained costs for ordinary MWPM with observable labels on boundary edges."""

from itertools import product

import networkx as nx
import numpy as np
import pymatching
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


class BoundaryCosts:
    """Make each boundary label a detector and enumerate its parity."""

    def __init__(self, matching: pymatching.Matching):
        if matching.boundary:
            raise ValueError('Explicit boundary nodes are unsupported')
        n = matching.num_detectors
        edges = matching.edges()
        internal = [(u, v) for u, v, _ in edges if v is not None]
        u, v = zip(*internal) if internal else ([], [])
        count, component = connected_components(
            coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False,
        )
        self.components = [np.flatnonzero(component == i) for i in range(count)]
        augmented = pymatching.Matching()
        ports: dict[tuple[int, int], int] = {}
        for u, v, edge in edges:
            label = sum(2 >> obs for obs in edge['fault_ids']) if edge['fault_ids'] <= {0, 1} else -1
            if label < 0 or (v is not None and label != 0) or edge['weight'] <= 0:
                raise ValueError('Expected positive-weight edges with two observables on the boundary only')
            if v is None:
                # The parity of each group counts how many of its physical boundary edges are selected.
                v = ports.setdefault((int(component[u]), label), n + len(ports))
            augmented.add_edge(u, v, weight=edge['weight'])
        if augmented.num_detectors != n + len(ports) or len(ports) > 8:
            raise ValueError('Expected numbered detectors and at most eight boundary groups')
        self.n = n
        self.augmented = augmented
        self.ports = ports
        # shortcut: enumerate boundary parity patterns for this graph; revisit if the number of groups grows.
        self.patterns: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
        for bits in product((0, 1), repeat=len(ports)):
            parities = [0] * count
            label = 0
            for bit, (group, value) in zip(bits, ports, strict=True):
                if bit:
                    parities[group] ^= 1
                    label ^= value
            self.patterns.setdefault(tuple(parities), []).append((label, bits))

    def solve(self, syndromes: np.ndarray, *, independent: bool = False) -> np.ndarray:
        rows = np.asarray(syndromes)
        if rows.ndim != 2 or rows.shape[1] != self.n or np.any((rows != 0) & (rows != 1)):
            raise ValueError('Expected binary syndrome rows with one column per detector')
        rows = rows.astype(np.uint8, copy=False)
        parities = np.column_stack([np.bitwise_xor.reduce(rows[:, group], axis=1)
                                    for group in self.components])
        result = np.full((len(rows), 4), np.inf)
        if independent:
            edges = self.augmented.edges()
            u, v, w = zip(*[(u, v, e['weight']) for u, v, e in edges])
            shape = (self.augmented.num_detectors,) * 2
            adjacency = coo_matrix((w, (u, v)), shape=shape).tocsr()
        for parity, patterns in self.patterns.items():
            subset = np.flatnonzero(np.all(parities == parity, axis=1))
            if not len(subset):
                continue
            for label, bits in patterns:
                events = np.column_stack((rows[subset], np.broadcast_to(
                    np.array(bits, dtype=np.uint8), (len(subset), len(bits)),
                )))
                if independent:
                    weights = [metric_closure_cost(adjacency, event) for event in events]
                else:
                    _, weights = self.augmented.decode_batch(events, return_weights=True)
                result[subset, label] = np.minimum(result[subset, label], weights)
        return result


def metric_closure_cost(adjacency, syndrome: np.ndarray) -> float:
    """Independent T-join calculation using Dijkstra and NetworkX matching."""
    active = np.flatnonzero(syndrome)
    if not active.size:
        return 0.0
    distances = dijkstra(adjacency, directed=False, indices=active)[:, active]
    complete = nx.Graph()
    complete.add_nodes_from(range(len(active)))
    for (i, j), weight in np.ndenumerate(distances):
        if i < j and np.isfinite(weight):
            complete.add_edge(i, j, weight=weight)
    matching = nx.min_weight_matching(complete)
    if len(matching) * 2 != len(active):
        raise ValueError('The augmented syndrome has no perfect matching')
    return sum(complete[u][v]['weight'] for u, v in matching)


def checked_costs(matching: pymatching.Matching, solver: BoundaryCosts,
                  syndromes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, optimum = matching.decode_batch(syndromes, return_weights=True)
    labels = 2 * prediction[:, 0] + prediction[:, 1]
    costs = solver.solve(syndromes)
    np.testing.assert_allclose(costs[np.arange(len(labels)), labels], optimum, atol=3e-4, rtol=0)
    np.testing.assert_allclose(costs.min(axis=1), optimum, atol=3e-4, rtol=0)
    return labels, costs


def z_gap(classes: np.ndarray, costs: np.ndarray) -> np.ndarray:
    indices = np.arange(len(classes))
    gap = costs[indices, classes ^ 2] - costs[indices, classes]
    unreachable = np.isposinf(gap)
    if np.any(gap < -3e-4) or np.isnan(gap).any():
        raise ValueError('Expected a nonnegative gap or an unreachable alternative')
    return np.column_stack((np.where(unreachable, 0.0, np.maximum(gap, 0.0)), unreachable))
