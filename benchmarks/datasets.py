import warnings
from typing import Tuple, Optional

import networkx as nx
import numpy as np


def generate_barbell_graph(m1: int = 10, m2: int = 5) -> Tuple[nx.Graph, np.ndarray]:
    """Return a barbell graph and role labels.

    Nodes on the path joining the two cliques are assigned label 1 while
    nodes inside the cliques receive label 0.
    """
    G = nx.barbell_graph(m1, m2)
    y = np.zeros(G.number_of_nodes(), dtype=int)
    path_nodes = list(range(m1, m1 + m2))
    y[path_nodes] = 1
    return G, y


def generate_tree_graph(r: int = 2, h: int = 3) -> Tuple[nx.Graph, np.ndarray]:
    """Return a balanced tree and role labels based on node depth."""
    G = nx.balanced_tree(r, h)
    depth = nx.shortest_path_length(G, source=0)
    labels = []
    for v in G.nodes():
        if depth[v] == 0:
            labels.append(0)  # root
        elif G.degree[v] == 1:
            labels.append(2)  # leaves
        else:
            labels.append(1)  # internal nodes
    return G, np.asarray(labels, dtype=int)


def load_wikipedia_voting_graph() -> Tuple[Optional[nx.Graph], Optional[np.ndarray]]:
    """Attempt to load the Wikipedia voting network using karateclub."""
    try:
        from karateclub.dataset import GraphReader
    except Exception as err:  # pragma: no cover
        warnings.warn(f"karateclub is required for the Wikipedia dataset: {err}")
        return None, None

    try:
        reader = GraphReader("wiki-vote")
        G = reader.get_graph()
        return G, None
    except Exception as err:  # pragma: no cover
        warnings.warn(f"Unable to load Wikipedia voting graph: {err}")
        return None, None
