"""Numerical checks of the statements of paper/theory.md.

1. Degree and triangles (Prop. 3): on a d-regular graph the return
   probability after three lazy steps is (1 + 3d + 2 t_u) / (d + 1)^3.
2. Stability (Prop. 4): adding one edge a-b moves the characteristic
   function of every node at step w by at most w ||theta|| 2 / (d + 2),
   with d the smaller endpoint degree.
3. Distance-regular graphs (Prop. 6): graphs with the same intersection
   array give every node the same embedding (Shrikhande / 4x4 rook,
   Chang / T(8), Doob / Hamming H(3, 4)).
4. RoleWalk vs 1-WL (Prop. 7): on all connected graphs up to 7 nodes,
   nodes that RoleWalk confuses but 3-round colour refinement separates,
   and the converse.

    python theory_checks.py
"""
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import identity
from sklearn.preprocessing import normalize

from rolewalk import RoleWalk

TOL = 1e-6


def lazy_walk(G):
    A = nx.to_scipy_sparse_array(G, format="csr")
    return normalize(identity(len(G)) + A, norm="l1").toarray()


def check_triangles():
    worst = 0.0
    for seed, (d, n) in enumerate(itertools.product((3, 4, 5), (12, 20, 40))):
        G = nx.random_regular_graph(d, n, seed=seed)
        H3 = np.linalg.matrix_power(lazy_walk(G), 3)
        t = np.array([nx.triangles(G, u) for u in G])
        worst = max(worst, np.abs(np.diag(H3) - (1 + 3 * d + 2 * t) / (d + 1) ** 3).max())
    print(f"1. triangles: max error of the return-probability formula {worst:.1e}")


def check_stability(n_trials=200):
    model = RoleWalk(tol=0, shells=False)
    m = model.theta.shape[1]
    norm = np.linalg.norm(model.theta)
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(n_trials):
        G = nx.powerlaw_cluster_graph(60, 2, 0.3, seed=trial)
        non_edges = list(nx.non_edges(G))
        a, b = non_edges[rng.integers(len(non_edges))]
        d = min(G.degree(a), G.degree(b))
        X = model.transform(G)
        G.add_edge(a, b)
        Y = model.transform(G)
        for w in range(model.walk_len):
            block = slice(2 * m * w, 2 * m * (w + 1))
            moved = np.linalg.norm(X[:, block] - Y[:, block], axis=1).max()
            worst = max(worst, moved / ((w + 1) * norm * 2 / (d + 2)))
    print(f"2. stability: largest move / bound = {worst:.3f} (must be <= 1)")


def chang_graph():
    """Seidel switching of T(8) with respect to a perfect matching of K8."""
    T8 = nx.line_graph(nx.complete_graph(8))
    T8 = nx.relabel_nodes(T8, {e: tuple(sorted(e)) for e in T8})
    S = {(0, 1), (2, 3), (4, 5), (6, 7)}
    G = nx.Graph()
    G.add_nodes_from(T8)
    for u, v in itertools.combinations(T8, 2):
        if (u in S) != (v in S):
            if not T8.has_edge(u, v):
                G.add_edge(u, v)
        elif T8.has_edge(u, v):
            G.add_edge(u, v)
    return T8, G


def shrikhande():
    G = nx.Graph()
    for i, j in itertools.product(range(4), repeat=2):
        for di, dj in ((0, 1), (1, 0), (1, 1)):
            G.add_edge((i, j), ((i + di) % 4, (j + dj) % 4))
    return G


def check_distance_regular():
    K4 = nx.complete_graph(4)
    rook = nx.cartesian_product(K4, K4)
    T8, chang = chang_graph()
    pairs = {
        "Shrikhande / 4x4 rook (SRG(16,6,2,2))": (shrikhande(), rook),
        "Chang / T(8) (SRG(28,12,6,4))": (chang, T8),
        "Doob D(1,1) / Hamming H(3,4) (diameter 3)":
            (nx.cartesian_product(shrikhande(), K4), nx.cartesian_product(rook, K4)),
    }
    for label, (G, H) in pairs.items():
        assert nx.is_distance_regular(G) and nx.is_distance_regular(H)
        assert nx.intersection_array(G) == nx.intersection_array(H)
        assert not nx.is_isomorphic(G, H)
        for curvature in (False, True):
            X = np.vstack([RoleWalk(tol=0, walk_len=5, curvature=curvature).transform(F)
                           for F in (G, H)])
            spread = np.abs(X - X[0]).max() / np.abs(X).max()
            print(f"3. {label}, curvature={curvature}: "
                  f"largest difference between any two nodes {spread:.1e}")


def wl_colours(G, rounds):
    hashes = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=rounds)
    return {u: h[-1] for u, h in hashes.items()}


def check_wl(rounds=3):
    graphs = [G for G in nx.graph_atlas_g()[1:] if nx.is_connected(G)]
    model = RoleWalk(tol=0)
    keys, colours, where = [], [], []
    for g, G in enumerate(graphs):
        X = model.transform(G)
        col = wl_colours(G, rounds)
        for i, u in enumerate(G):
            keys.append(X[i])
            colours.append(col[u])
            where.append((g, u))
    keys = np.array(keys, dtype=np.float64)
    keys = np.round(keys / np.abs(keys).max(), 6)
    by_rolewalk, by_wl = defaultdict(set), defaultdict(set)
    for k, c in zip(map(bytes, keys), colours):
        by_rolewalk[k].add(c)
        by_wl[c].add(k)
    wl_finer = [k for k, cs in by_rolewalk.items() if len(cs) > 1]
    rw_finer = [c for c, ks in by_wl.items() if len(ks) > 1]
    print(f"4. {len(graphs)} connected graphs <= 7 nodes, {len(keys)} nodes: "
          f"{len(set(map(bytes, keys)))} RoleWalk classes, "
          f"{len(set(colours))} classes of {rounds}-round 1-WL")
    print(f"   RoleWalk classes split by 1-WL: {len(wl_finer)}; "
          f"1-WL classes split by RoleWalk: {len(rw_finer)}")
    if wl_finer:
        k = wl_finer[0]
        idx = [i for i, kk in enumerate(map(bytes, keys)) if kk == k]
        i, j = next((i, j) for i, j in itertools.combinations(idx, 2)
                    if colours[i] != colours[j])
        for name, t in (("u", where[i]), ("v", where[j])):
            G = graphs[t[0]]
            print(f"   {name}: node {t[1]} of graph with edges {sorted(G.edges())}")


if __name__ == "__main__":
    check_triangles()
    check_stability()
    check_distance_regular()
    check_wl()
