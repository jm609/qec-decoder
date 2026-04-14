from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from qiskit.transpiler import CouplingMap


DATA_QUBITS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DATA_POSITIONS = {
    0: (0, 2),
    1: (2, 2),
    2: (4, 2),
    3: (0, 0),
    4: (2, 0),
    5: (4, 0),
    6: (0, -2),
    7: (2, -2),
    8: (4, -2),
}

X_MEASURE_QUBITS = [9, 10, 11, 12]
X_MEASURE_POSITIONS = {
    9: (1, 1),
    10: (3, -1),
    11: (-1, -1),
    12: (5, 1),
}

Z_MEASURE_QUBITS = [13, 14, 15, 16]
Z_MEASURE_POSITIONS = {
    13: (3, 1),
    14: (1, -1),
    15: (1, 3),
    16: (3, -3),
}

POSITIONS = {
    **DATA_POSITIONS,
    **X_MEASURE_POSITIONS,
    **Z_MEASURE_POSITIONS,
}

EDGES = [
    (9, 0), (9, 1), (9, 3), (9, 4),
    (10, 4), (10, 5), (10, 7), (10, 8),
    (11, 3), (11, 6),
    (12, 2), (12, 5),
    (13, 1), (13, 2), (13, 4), (13, 5),
    (14, 3), (14, 4), (14, 6), (14, 7),
    (15, 0), (15, 1),
    (16, 7), (16, 8),
]


def build_coupling_map() -> CouplingMap:
    coupling_map = CouplingMap(EDGES)
    coupling_map.make_symmetric()
    return coupling_map


def build_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(DATA_QUBITS)
    graph.add_nodes_from(X_MEASURE_QUBITS)
    graph.add_nodes_from(Z_MEASURE_QUBITS)
    graph.add_edges_from(EDGES)
    return graph


def draw_architecture(*, save_path: Path | None = None) -> CouplingMap:
    graph = build_graph()
    coupling_map = build_coupling_map()

    plt.figure(figsize=(9, 9))
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=DATA_QUBITS,
        node_color="white",
        edgecolors="black",
        node_size=1000,
        label="Data Qubit",
    )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=X_MEASURE_QUBITS,
        node_color="gold",
        edgecolors="black",
        node_size=1000,
        label="X-Measure Qubit",
    )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=Z_MEASURE_QUBITS,
        node_color="mediumseagreen",
        edgecolors="black",
        node_size=1000,
        label="Z-Measure Qubit",
    )
    nx.draw_networkx_edges(graph, POSITIONS, width=2.5, alpha=0.6)
    nx.draw_networkx_labels(graph, POSITIONS, font_size=12, font_family="sans-serif", font_weight="bold")

    plt.title("d=3 Rotated Surface Code Architecture", fontsize=18, fontweight="bold", pad=20)
    plt.legend(scatterpoints=1, loc="upper right", fontsize=11)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return coupling_map


def main() -> None:
    coupling_map = draw_architecture()
    print(f"Total qubits in CouplingMap: {coupling_map.size()}")
    print("Coupling map representation created.")


if __name__ == "__main__":
    main()
