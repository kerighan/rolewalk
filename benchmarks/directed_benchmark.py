"""Roles that only directions reveal: directed motifs on a directed cycle.

No public role benchmark is directed, so this one is synthetic: the motifs of
``datasets.DIRECTED_SHAPES`` (broadcaster and collector stars, cascades in
both directions, feed-forward loops, cycles, mutual stars, bi-fans) hang from
a directed cycle, then edges are randomly removed and added. Several pairs
of roles are identical once directions are dropped.

Methods that read directions get the directed graph: RoleWalk, RWSE (on the
out-walk) and in/out log-degrees. The others only handle undirected graphs
and get the symmetrised one, as does ``rolewalk_undirected``, the ablation
that isolates what directions bring. Scores are those of
``synthetic_benchmark.py``.

    python directed_benchmark.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd

import role_benchmark as rb
from datasets import generate_directed_shapes
from synthetic_benchmark import add_noise, score


def degree_in_out(G: nx.DiGraph) -> np.ndarray:
    return np.log1p([[G.in_degree(v), G.out_degree(v)] for v in range(len(G))])


DIRECTED = {
    "rolewalk": rb.rolewalk,
    "rolewalk_curvature": rb.rolewalk_curvature,
    "rolewalk_noshells": rb.rolewalk_noshells,
    "rwse": rb.rwse,
    "rwse_inout": rb.rwse_inout,
    "refex_directed": rb.refex_directed,
    "rolewalk_mixed": rb.rolewalk_mixed,
    "rolewalk_curvature_mixed": rb.rolewalk_curvature_mixed,
    "topk_layers": rb.topk_layers,
    "local_stats": rb.local_stats,
    "degree": degree_in_out,
}
UNDIRECTED = {
    "rolewalk_undirected": rb.rolewalk,
    **{m: rb.METHODS[m] for m in ("refex", "resd", "graphwave", "role2vec",
                                  "struc2vec") if m in rb.METHODS},
}
METHODS = {**DIRECTED, **UNDIRECTED}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="*", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.05, 0.1])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-per-shape", type=int, default=5)
    parser.add_argument("--output", default="directed.csv")
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        G, y = generate_directed_shapes(args.n_per_shape, seed=seed)
        for level in args.noise:
            H = add_noise(G, level, np.random.default_rng(seed))
            for method in args.methods:
                graph = H if method in DIRECTED else nx.Graph(H.to_undirected())
                start = time.perf_counter()
                X = METHODS[method](graph)
                elapsed = time.perf_counter() - start
                rows.append({"n_nodes": len(G), "seed": seed, "noise": level,
                             "method": method, "time": elapsed,
                             **score(X, y, seed)})
                print(f"seed={seed} noise={level} {method}: "
                      f"ari={rows[-1]['ari']:.3f} acc={rows[-1]['accuracy']:.3f} "
                      f"({elapsed:.2f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(df.groupby(["noise", "method"])[["ari", "accuracy", "macro_f1", "time"]]
              .mean().unstack("noise").round(3))


if __name__ == "__main__":
    main()
