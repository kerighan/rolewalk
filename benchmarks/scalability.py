"""Embedding time against graph size, up to a million nodes.

Graphs: Barabási-Albert (m=3, hubs: the worst case for walk powers) and
Watts-Strogatz (k=6, p=0.1, bounded degree). RoleWalk runs with its
defaults (``tol=1e-4``, onion layers), with and without curvature, on one
thread. Rows are embedded in blocks of ``BLOCK`` nodes, which only bounds
memory: each block gives exactly the rows of the full computation.

    OMP_NUM_THREADS=1 python scalability.py
"""
import resource
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import identity
from sklearn.preprocessing import normalize

from rolewalk import RoleWalk, compute_embedding, node_curvature

SIZES = (10_000, 30_000, 100_000, 300_000, 1_000_000)
BLOCK = 50_000
GRAPHS = {
    "barabasi_albert": lambda n: nx.barabasi_albert_graph(n, 3, seed=0),
    "watts_strogatz": lambda n: nx.watts_strogatz_graph(n, 6, 0.1, seed=0),
}


def embed(G, curvature):
    model = RoleWalk(curvature=curvature)
    start = time.perf_counter()
    n = len(G)
    A = nx.to_scipy_sparse_array(G, format="csr")
    H = normalize(identity(n) + A, norm="l1").tocsr()
    kappa = node_curvature(A) if curvature else None
    for lo in range(0, n, BLOCK):
        rows = np.arange(lo, min(lo + BLOCK, n))
        X = compute_embedding(H, model.theta, model.walk_len, model.aggregate,
                              model.tol, rows, model.shells, kappa)
    return time.perf_counter() - start, X.shape[1]


def main():
    rows = []
    for name, make in GRAPHS.items():
        for n in SIZES:
            G = make(n)
            for curvature in (False, True):
                seconds, dim = embed(G, curvature)
                row = {"graph": name, "n_nodes": n, "n_edges": G.number_of_edges(),
                       "curvature": curvature, "dim": dim, "time": round(seconds, 2),
                       "peak_rss_gb": round(resource.getrusage(
                           resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv("scalability.csv", index=False)


if __name__ == "__main__":
    main()
