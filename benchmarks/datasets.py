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


def generate_ring_of_cliques(
    n_cliques: int = 4, clique_size: int = 4
) -> Tuple[nx.Graph, np.ndarray]:
    """Return a ring of cliques graph and role labels.

    Nodes that bridge cliques receive label ``1`` while all other nodes are
    labelled ``0``.
    """

    G = nx.ring_of_cliques(n_cliques, clique_size)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    # Bridge nodes have one extra edge connecting to the next clique.
    for node, degree in G.degree():
        if degree > clique_size - 1:
            labels[node] = 1
    return G, labels


def generate_grid_graph(m: int = 5, n: int = 5) -> Tuple[nx.Graph, np.ndarray]:
    """Return an ``m`` by ``n`` grid graph and role labels.

    Labels correspond to corners (0), edge nodes (1) and inner nodes (2).
    """

    G = nx.grid_2d_graph(m, n)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for (i, j), idx in mapping.items():
        if (i in {0, m - 1}) and (j in {0, n - 1}):
            labels[idx] = 0  # corners
        elif i in {0, m - 1} or j in {0, n - 1}:
            labels[idx] = 1  # edges
        else:
            labels[idx] = 2  # interior
    return G, labels


def generate_star_graph(n_leaves: int = 10) -> Tuple[nx.Graph, np.ndarray]:
    """Return a star graph with role labels for center and leaves."""

    G = nx.star_graph(n_leaves)
    labels = np.ones(G.number_of_nodes(), dtype=int)
    labels[0] = 0  # center node
    return G, labels


def generate_house_graph() -> Tuple[nx.Graph, np.ndarray]:
    """Return a house graph and role labels.

    The graph consists of a square with a triangle on top. Nodes at the base of
    the roof (degree 3) are labelled ``1``, the roof apex is labelled ``2`` and
    the remaining nodes (degree 2 on the square base) are labelled ``0``.
    """

    G = nx.house_graph()
    degrees = dict(G.degree())
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for node, deg in degrees.items():
        if deg == 3:
            labels[node] = 1
        elif all(degrees[n] == 3 for n in G.neighbors(node)):
            labels[node] = 2
    return G, labels


def load_wikipedia_voting_graph() -> Tuple[Optional[nx.Graph], Optional[np.ndarray]]:
    """Load the Wikipedia voting network.

    Attempts to use ``karateclub`` if available. If ``karateclub`` is not
    installed or fails to retrieve the data, the function falls back to
    downloading the edge list from the SNAP repository. If all methods fail,
    ``(None, None)`` is returned.
    """

    # First try to use karateclub if available
    try:  # pragma: no cover - optional dependency
        from karateclub.dataset import GraphReader

        reader = GraphReader("wiki-vote")
        G = reader.get_graph()
        return G, None
    except Exception:
        pass

    # Fallback: attempt to download from SNAP
    url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
    try:  # pragma: no cover - network access
        import gzip
        import io
        from urllib.request import urlopen

        with urlopen(url) as resp:
            data = gzip.decompress(resp.read()).decode("utf-8")
        G = nx.parse_edgelist(io.StringIO(data), nodetype=int, create_using=nx.DiGraph())
        return G, None
    except Exception as err:  # pragma: no cover
        warnings.warn(f"Unable to load Wikipedia voting graph: {err}")
        return None, None
