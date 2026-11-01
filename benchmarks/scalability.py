"""Embedding time against graph size, up to a million nodes.

Graphs: Barabási-Albert (m=3, hubs: the worst case for walk powers) and
Watts-Strogatz (k=6, p=0.1, bounded degree). RoleWalk runs with its
defaults (``tol=1e-4``, onion layers), with and without curvature, with the
fused numba kernel, on one thread and on 8 (rows are independent). Times
include the conversion of the networkx graph.

    OMP_NUM_THREADS=1 python scalability.py
"""
import resource
import time

import networkx as nx
import numpy as np
import pandas as pd
from rolewalk import RoleWalk

SIZES = (10_000, 30_000, 100_000, 300_000, 1_000_000)
THREADS = (1, 8)
GRAPHS = {
    "barabasi_albert": lambda n: nx.barabasi_albert_graph(n, 3, seed=0),
    "watts_strogatz": lambda n: nx.watts_strogatz_graph(n, 6, 0.1, seed=0),
}


def embed(G, curvature, n_jobs):
    RoleWalk().transform(nx.path_graph(3))  # compile the kernel first
    start = time.perf_counter()
    X = RoleWalk(curvature=curvature, n_jobs=n_jobs).transform(G)
    return time.perf_counter() - start, X.shape[1]


def main():
    rows = []
    for name, make in GRAPHS.items():
        for n in SIZES:
            G = make(n)
            for curvature, n_jobs in ((False, 1), (True, 1), (False, 8)):
                seconds, dim = embed(G, curvature, n_jobs)
                row = {"graph": name, "n_nodes": n, "n_edges": G.number_of_edges(),
                       "curvature": curvature, "n_jobs": n_jobs, "dim": dim,
                       "time": round(seconds, 2),
                       "peak_rss_gb": round(resource.getrusage(
                           resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv("scalability.csv", index=False)


if __name__ == "__main__":
    main()
