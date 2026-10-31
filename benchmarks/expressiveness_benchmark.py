"""Which non-isomorphic graphs can each structural encoding tell apart?

Two graphs are told apart when the sorted multisets of their node embeddings
differ (relative tolerance ``TOL``). Every method here is
permutation-invariant, so isomorphic graphs always match; the script checks
this on random relabellings, which also bounds the numerical noise.

Graph sets, all generated locally:

- ``atlas``: every pair of non-isomorphic connected graphs with up to 7
  nodes that 1-WL colour refinement cannot distinguish (networkx atlas);
- ``regular``: random 3- and 4-regular graphs on 10 to 16 nodes, one per
  isomorphism class; all graphs of the same size and degree are
  1-WL-equivalent, so every pair among them is a test;
- ``csl``: circular skip link graphs (41-cycle plus chords of length
  2, 3, 4, 5, 6, 9, 11, 12, 13, 16), all 4-regular, so 1-WL-equivalent;
- classic pairs: decalin / bicyclopentyl, two triangles / hexagon, and
  Shrikhande / 4x4 rook's graph (strongly regular with equal parameters,
  beyond 3-WL);
- ``brec``: the 400 pairs of BREC (Wang & Zhang 2023) by category, if
  data/brec/brec_v3.npy is present (see ``datasets.load_brec``).

    python expressiveness_benchmark.py
"""
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from rolewalk import RoleWalk
import role_benchmark as rb
from datasets import BREC_RANGES, DATA_DIR, load_brec
import os

TOL = 1e-4
CSL_SKIPS = (2, 3, 4, 5, 6, 9, 11, 12, 13, 16)

METHODS = {
    "rolewalk": lambda G: RoleWalk(tol=0).transform(G),
    "rolewalk_curvature": lambda G: RoleWalk(tol=0, curvature=True).transform(G),
    "rolewalk_noshells": lambda G: RoleWalk(tol=0, shells=False).transform(G),
    "rolewalk_w5": lambda G: RoleWalk(tol=0, walk_len=5).transform(G),
    "rwse": lambda G: rb.rwse(nx.convert_node_labels_to_integers(G)),
    "degree": lambda G: np.array([[d] for _, d in G.degree()], dtype=float),
}


def signature(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float64)


def differ(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the multisets of rows differ, beyond ``TOL`` (relative).

    Rows are matched by optimal assignment: sorting them would be fragile,
    as rounding noise can swap two nearly equal rows.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if a.shape != b.shape:
        return True
    cost = cdist(a, b, "chebyshev")
    rows, cols = linear_sum_assignment(cost)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return cost[rows, cols].max() > TOL * scale


def wl_hash(G: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(G, iterations=len(G))


def atlas_pairs():
    """1-WL-equivalent pairs of non-isomorphic connected graphs (<= 7 nodes)."""
    groups = defaultdict(list)
    for G in nx.graph_atlas_g()[1:]:
        if nx.is_connected(G):
            groups[len(G), G.number_of_edges(), wl_hash(G)].append(G)
    return [(a, b) for graphs in groups.values()
            for a, b in itertools.combinations(graphs, 2)]


def regular_pairs(per_size=30, seed=0):
    """Pairs of non-isomorphic d-regular graphs of the same size."""
    pairs = []
    for d, n in itertools.product((3, 4), (10, 12, 14, 16)):
        graphs = []
        for k in range(per_size * 4):
            G = nx.random_regular_graph(d, n, seed=seed + k)
            if nx.is_connected(G) and not any(nx.is_isomorphic(G, H) for H in graphs):
                graphs.append(G)
            if len(graphs) == per_size:
                break
        pairs += list(itertools.combinations(graphs, 2))
    return pairs


def classic_pairs():
    decalin = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6),
                        (6, 7), (7, 8), (8, 9), (9, 5)])
    bicyclopentyl = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5),
                              (5, 6), (6, 7), (7, 8), (8, 9), (9, 5)])
    rook = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
    shrikhande = nx.Graph()
    for i, j in itertools.product(range(4), repeat=2):
        for di, dj in ((0, 1), (1, 0), (1, 1)):
            shrikhande.add_edge((i, j), ((i + di) % 4, (j + dj) % 4))
    return {
        "decalin / bicyclopentyl": (decalin, bicyclopentyl),
        "two triangles / hexagon": (nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)),
                                    nx.cycle_graph(6)),
        "Shrikhande / 4x4 rook (SRG, beyond 3-WL)": (shrikhande, rook),
    }


def main():
    rng = np.random.default_rng(0)
    pairs = atlas_pairs()
    regular = regular_pairs()
    csl = [nx.circulant_graph(41, [1, s]) for s in CSL_SKIPS]
    classic = classic_pairs()
    brec = (load_brec() if os.path.exists(os.path.join(DATA_DIR, "brec", "brec_v3.npy"))
            else None)
    print(f"{len(pairs)} 1-WL-equivalent atlas pairs, {len(regular)} regular "
          f"pairs, {len(csl)} CSL classes")

    rows = []
    for name, f in METHODS.items():
        sig = lambda G: signature(f(G))  # noqa: E731
        # sanity: relabelled copies must match
        noise = 0
        for G in csl[:3] + [p[0] for p in pairs[:20]]:
            perm = dict(zip(G, rng.permutation(len(G))))
            H = nx.relabel_nodes(G, {v: int(perm[v]) for v in G})
            H = nx.Graph(H.subgraph(sorted(H)))
            noise += differ(sig(G), sig(H))
        atlas = sum(differ(sig(a), sig(b)) for a, b in pairs)
        cache = {}
        def csig(G):  # noqa: E306
            if id(G) not in cache:
                cache[id(G)] = sig(G)
            return cache[id(G)]
        reg = sum(differ(csig(a), csig(b)) for a, b in regular)
        sigs = [sig(G) for G in csl]
        classes = len({min(j for j in range(len(sigs)) if not differ(sigs[i], sigs[j]))
                       for i in range(len(sigs))})
        row = {"method": name, "false_differences": noise,
               "atlas": f"{atlas}/{len(pairs)}",
               "regular": f"{reg}/{len(regular)}", "csl_classes": f"{classes}/{len(csl)}"}
        for label, (a, b) in classic.items():
            row[label] = "yes" if differ(sig(a), sig(b)) else "no"
        if brec:
            told, false = 0, 0
            for category, (lo, hi) in BREC_RANGES.items():
                n = sum(differ(sig(brec[i][0]), sig(brec[i][1])) for i in range(lo, hi))
                row[f"brec_{category}"] = f"{n}/{hi - lo}"
                told += n
            false = sum(differ(sig(g), sig(c)) for g, _, c in brec)
            row["brec_total"] = f"{told}/{len(brec)}"
            row["brec_false_differences"] = false
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("expressiveness.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
