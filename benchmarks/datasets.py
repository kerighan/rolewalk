import os
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
        # SNAP ids are sparse; karateclub requires consecutive 0..n-1 indices
        G = nx.convert_node_labels_to_integers(G)
        return G, None
    except Exception as err:  # pragma: no cover
        warnings.warn(f"Unable to load Wikipedia voting graph: {err}")
        return None, None


# ---------------------------------------------------------------------------
# Synthetic "shapes on a cycle" graphs, as in the GraphWave paper
# (Donnat et al., KDD 2018): motifs are attached to a base cycle and every
# node's role is its position inside its motif.
# ---------------------------------------------------------------------------

def _shape_house() -> nx.Graph:
    return nx.house_graph()


def _shape_star(n_leaves: int = 5) -> nx.Graph:
    return nx.star_graph(n_leaves)


def _shape_clique(k: int = 5) -> nx.Graph:
    return nx.complete_graph(k)


def _shape_fan(n_blades: int = 6) -> nx.Graph:
    """Hub joined to every node of a path."""
    G = nx.path_graph(range(1, n_blades + 1))
    G.add_edges_from((0, i) for i in range(1, n_blades + 1))
    return G


def _shape_diamond() -> nx.Graph:
    return nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])


def _shape_tree(r: int = 2, h: int = 2) -> nx.Graph:
    return nx.balanced_tree(r, h)


SHAPES = {
    "house": _shape_house,
    "star": _shape_star,
    "clique": _shape_clique,
    "fan": _shape_fan,
    "diamond": _shape_diamond,
    "tree": _shape_tree,
}


def _refine_colors(G: nx.Graph, colors: dict) -> dict:
    """1-WL colour refinement until the partition stops changing."""
    n_classes = len(set(colors.values()))
    while True:
        signatures = {
            v: (colors[v], tuple(sorted(colors[u] for u in G.neighbors(v))))
            for v in G
        }
        palette = {s: i for i, s in enumerate(sorted(set(signatures.values())))}
        new = {v: palette[signatures[v]] for v in G}
        if len(palette) == n_classes:
            return new
        colors, n_classes = new, len(palette)


def generate_shapes_graph(
    shapes: Tuple[str, ...] = ("house",) * 10,
    spacing: int = 3,
    seed: int = 0,
) -> Tuple[nx.Graph, np.ndarray]:
    """Attach ``shapes`` to a base cycle, one every ``spacing`` cycle nodes.

    The cycle has ``len(shapes) * spacing`` nodes and shape order is shuffled
    with ``seed``. Labels:

    - plain cycle nodes share one role;
    - a cycle node carrying a shape gets one role per shape type;
    - inside a shape, roles are the orbits of the shape with its attachment
      node (node 0) marked, obtained by colour refinement. The same position
      in two copies of a shape gets the same role.
    """
    rng = np.random.default_rng(seed)
    shapes = list(shapes)
    rng.shuffle(shapes)
    n_cycle = len(shapes) * spacing

    G = nx.cycle_graph(n_cycle)
    labels = [0] * n_cycle
    roles = {}  # (kind, shape name, local colour) -> global role id

    def role(key):
        if key not in roles:
            roles[key] = len(roles) + 1
        return roles[key]

    for k, name in enumerate(shapes):
        anchor = k * spacing
        S = SHAPES[name]()
        colors = _refine_colors(S, {v: int(v == 0) for v in S})
        offset = G.number_of_nodes()
        G.add_edges_from((u + offset, v + offset) for u, v in S.edges())
        G.add_edge(anchor, offset)
        labels[anchor] = role(("anchor", name))
        labels.extend(role(("shape", name, colors[v])) for v in sorted(S))

    return G, np.asarray(labels, dtype=int)


def generate_houses(n_shapes: int = 10, seed: int = 0):
    """GraphWave paper setting: houses on a cycle."""
    return generate_shapes_graph(("house",) * n_shapes, spacing=3, seed=seed)


def generate_varied(n_per_shape: int = 5, seed: int = 0):
    """Every motif type, ``n_per_shape`` copies each."""
    return generate_shapes_graph(
        tuple(s for s in SHAPES for _ in range(n_per_shape)), spacing=3, seed=seed
    )


AIRPORTS_URL = "https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _fetch(filename: str) -> str:
    """Download ``filename`` from the struc2vec repository once, then cache it."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):  # pragma: no cover - network access
        from urllib.request import urlretrieve

        os.makedirs(DATA_DIR, exist_ok=True)
        urlretrieve(AIRPORTS_URL + filename, path)
    return path


def load_airports(region: str) -> Tuple[nx.Graph, np.ndarray]:
    """Air-traffic network of ``region`` (brazil, europe or usa).

    The role benchmark of struc2vec (Ribeiro et al., KDD 2017), reused by most
    later role-embedding papers. Labels are quartiles of airport activity
    (passengers or flights), so they reflect structural importance rather
    than location. Nodes are relabelled ``0..n-1`` in sorted id order.
    """
    G = nx.read_edgelist(_fetch(f"{region}-airports.edgelist"), nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = {}
    with open(_fetch(f"labels-{region}-airports.txt")) as f:
        next(f)  # header
        for line in f:
            node, label = line.split()
            labels[int(node)] = int(label)
    G.add_nodes_from(labels)  # a few labelled airports have no route
    nodes = sorted(G)
    index = {v: i for i, v in enumerate(nodes)}
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))  # node order 0..n-1
    H.add_edges_from((index[u], index[v]) for u, v in G.edges())
    return H, np.asarray([labels[v] for v in nodes], dtype=int)


RONE_URL = "https://raw.githubusercontent.com/cspjiao/RONE/main/dataset/clf/"


def load_rone(name: str) -> Tuple[nx.Graph, np.ndarray]:
    """A role benchmark of the RONE survey (Jiao et al., IEEE TBD 2021).

    ``reality-call`` (phone calls), ``actor`` (co-occurrence on Wikipedia
    pages) and ``film`` are labelled by activity or importance classes;
    ``<lang>-wiki-talk`` (ht, br, cy, oc, eo, gl) label bots (1) and admins
    (2) among Wikipedia users. Edges are undirected as distributed. Unlabelled
    nodes get label -1.
    """
    paths = {}
    for ext in ("edge", "lbl"):
        path = os.path.join(DATA_DIR, "rone", f"{name}.{ext}")
        if not os.path.exists(path):  # pragma: no cover - network access
            from urllib.request import urlretrieve

            os.makedirs(os.path.dirname(path), exist_ok=True)
            urlretrieve(f"{RONE_URL}{name}.{ext}", path)
        paths[ext] = path
    edges = np.loadtxt(paths["edge"], dtype=np.int64, ndmin=2)
    labelled = np.loadtxt(paths["lbl"], dtype=np.int64, ndmin=2)
    n = int(max(edges.max(), labelled[:, 0].max())) + 1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges[edges[:, 0] != edges[:, 1]].tolist())
    y = np.full(n, -1, dtype=int)
    y[labelled[:, 0]] = labelled[:, 1]
    return G, y


# Directed motifs, attachment node 0. out_star / in_star and chain /
# reverse_chain are the same shape once directions are dropped: only a method
# that reads directions can tell their roles apart.
DIRECTED_SHAPES = {
    "out_star": lambda: nx.DiGraph([(0, i) for i in range(1, 6)]),  # broadcaster
    "in_star": lambda: nx.DiGraph([(i, 0) for i in range(1, 6)]),  # collector
    "chain": lambda: nx.DiGraph([(0, 1), (1, 2), (2, 3)]),  # cascade outwards
    "reverse_chain": lambda: nx.DiGraph([(1, 0), (2, 1), (3, 2)]),  # inwards
    "feed_forward": lambda: nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3)]),
    "cycle": lambda: nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]),
    "mutual_star": lambda: nx.DiGraph(
        [e for i in range(1, 5) for e in ((0, i), (i, 0))]),
    "bi_fan": lambda: nx.DiGraph([(0, 2), (0, 3), (1, 2), (1, 3), (0, 1)]),
}


def _refine_colors_directed(G: nx.DiGraph, colors: dict) -> dict:
    """Directed 1-WL: refine on successor and predecessor colours."""
    n_classes = len(set(colors.values()))
    while True:
        signatures = {
            v: (colors[v],
                tuple(sorted(colors[u] for u in G.successors(v))),
                tuple(sorted(colors[u] for u in G.predecessors(v))))
            for v in G
        }
        palette = {s: i for i, s in enumerate(sorted(set(signatures.values())))}
        new = {v: palette[signatures[v]] for v in G}
        if len(palette) == n_classes:
            return new
        colors, n_classes = new, len(palette)


def generate_directed_shapes(n_per_shape: int = 5, spacing: int = 3,
                             seed: int = 0) -> Tuple[nx.DiGraph, np.ndarray]:
    """Directed counterpart of :func:`generate_shapes_graph`.

    Every motif of ``DIRECTED_SHAPES`` hangs, ``n_per_shape`` times, from a
    directed base cycle through an edge cycle node -> motif node 0. Roles
    are the same as in the undirected version, with orbits computed by
    directed colour refinement.
    """
    rng = np.random.default_rng(seed)
    shapes = [s for s in DIRECTED_SHAPES for _ in range(n_per_shape)]
    rng.shuffle(shapes)
    n_cycle = len(shapes) * spacing

    G = nx.DiGraph(nx.cycle_graph(n_cycle, create_using=nx.DiGraph))
    labels = [0] * n_cycle
    roles = {}

    def role(key):
        if key not in roles:
            roles[key] = len(roles) + 1
        return roles[key]

    for k, name in enumerate(shapes):
        anchor = k * spacing
        S = DIRECTED_SHAPES[name]()
        colors = _refine_colors_directed(S, {v: int(v == 0) for v in S})
        offset = G.number_of_nodes()
        G.add_nodes_from(offset + v for v in sorted(S))
        G.add_edges_from((u + offset, v + offset) for u, v in S.edges())
        G.add_edge(anchor, offset)
        labels[anchor] = role(("anchor", name))
        labels.extend(role(("shape", name, colors[v])) for v in sorted(S))

    return G, np.asarray(labels, dtype=int)


def load_wiki_rfa() -> Tuple[nx.DiGraph, np.ndarray]:
    """Votes of Wikipedia requests for adminship (SNAP wiki-RfA), directed.

    An edge u -> v means u voted on v's request (support, neutral or
    oppose alike: only the structure is kept). Labels: 0 = voter only,
    1 = candidate never elected, 2 = elected admin. Votes without a voter
    name are dropped. Nodes are relabelled ``0..n-1`` in name order.
    """
    import gzip

    path = os.path.join(DATA_DIR, "rfa", "wiki-RfA.txt.gz")
    if not os.path.exists(path):  # pragma: no cover - network access
        from urllib.request import urlretrieve

        os.makedirs(os.path.dirname(path), exist_ok=True)
        urlretrieve("https://snap.stanford.edu/data/wiki-RfA.txt.gz", path)
    edges, elected, candidates = set(), set(), set()
    record = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                src, tgt = record.get("SRC"), record.get("TGT")
                if tgt:
                    candidates.add(tgt)
                    if record.get("RES") == "1":
                        elected.add(tgt)
                    if src and src != tgt:
                        edges.add((src, tgt))
                record = {}
            elif ":" in line:
                key, value = line.split(":", 1)
                record[key] = value
    names = sorted({u for e in edges for u in e})
    index = {u: i for i, u in enumerate(names)}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from((index[u], index[v]) for u, v in edges)
    y = np.array([2 if u in elected else 1 if u in candidates else 0
                  for u in names])
    return G, y


BREC_RANGES = {"basic": (0, 60), "regular": (60, 160), "extension": (160, 260),
               "cfi": (260, 360), "4-vertex": (360, 380), "distance-regular": (380, 400)}


def load_brec(n_relabel: int = 32, n_pairs: int = 400):
    """BREC pairs (Wang & Zhang 2023) as networkx graphs.

    Pair ``i`` occupies entries ``[i * 2 * n_relabel, (i + 1) * 2 * n_relabel)``
    of the file, alternating relabelled copies of its two graphs; entries
    after the 400 pairs are BREC's reliability copies. Returns, per pair, the
    two graphs and a relabelled copy of the first (isomorphic to it). Download
    BREC_data_all.zip from github.com/GraphPKU/BREC and unzip it into
    data/brec.
    """
    path = os.path.join(DATA_DIR, "brec", "brec_v3.npy")
    raw = np.load(path)  # fixed-width graph6 bytes: no pickle involved
    graph = lambda i: nx.from_graph6_bytes(raw[i].rstrip(b"\x00"))  # noqa: E731
    step = 2 * n_relabel
    return [(graph(i * step), graph(i * step + 1), graph(i * step + 2))
            for i in range(n_pairs)]
