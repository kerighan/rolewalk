"""How much does pruning change RoleWalk?

For tol in {0, 1e-5, 1e-4, 1e-3}: time, relative error of the embedding
against the exact one (tol = 0), per block (characteristic function,
layers), and accuracy of the usual classifier (50 splits). Stress test: the
centre of a star with many leaves, whose step-2 leaf mass is dropped although
half of it returns to the centre at step 3; and a hub in a Barabasi-Albert
graph.

    python pruning_study.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time

import networkx as nx
import numpy as np
import pandas as pd

from rolewalk import RoleWalk
from datasets import load_airports, load_rone, load_wiki_rfa
from role_benchmark import within

TOLS = (0.0, 1e-5, 1e-4, 1e-3)


def blocks(model):
    """Column indices of the CF and layer blocks."""
    m, L = model.theta.shape[1], model.walk_len
    cf, layers, start = [], [], 0
    for w in range(1, L + 1):
        cf += range(start, start + 2 * m)
        start += 2 * m
        layers += range(start, start + w + 1)
        start += w + 1
    return np.array(cf), np.array(layers)


def rel_error(X, X0):
    """Median and max over nodes of ||x - x0|| / ||x0||."""
    e = np.linalg.norm(X - X0, axis=1) / np.maximum(np.linalg.norm(X0, axis=1), 1e-12)
    return np.median(e), e.max()


def datasets():
    for r in ("brazil", "europe", "usa"):
        yield r, *load_airports(r)
    for n in ("reality-call", "actor"):
        yield n, *load_rone(n)
    G, y = load_wiki_rfa()
    yield "rfa-candidates", G, np.where(y > 0, y, -1)


def main():
    rows = []
    for name, G, y in datasets():
        exact = None
        for tol in TOLS:
            model = RoleWalk(walk_len=3, tol=tol)
            t = time.perf_counter()
            X = model.transform(G)
            dt = time.perf_counter() - t
            if exact is None:
                exact = X
            cf, layers = blocks(model)
            if G.is_directed():  # out- then in-walk, same layout
                half = X.shape[1] // 2
                cf, layers = np.r_[cf, cf + half], np.r_[layers, layers + half]
            keep = y >= 0
            rows.append({
                "dataset": name, "n_nodes": len(G), "tol": tol, "time": dt,
                "err_cf_median": rel_error(X[:, cf], exact[:, cf])[0],
                "err_cf_max": rel_error(X[:, cf], exact[:, cf])[1],
                "err_layers_median": rel_error(X[:, layers], exact[:, layers])[0],
                "err_layers_max": rel_error(X[:, layers], exact[:, layers])[1],
                "accuracy": within(X[keep], y[keep], 50).accuracy.mean()})
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in rows[-1].items()}, flush=True)

    # stress tests: hubs
    for name, G, hub in (("star 20k", nx.star_graph(20000), 0),
                         ("BA 100k hub", nx.barabasi_albert_graph(100000, 3, seed=0), None)):
        if hub is None:
            hub = max(G.degree, key=lambda x: x[1])[0]
        exact = RoleWalk(walk_len=3, tol=0).transform(G, nodes=[hub])
        for tol in TOLS[1:]:
            model = RoleWalk(walk_len=3, tol=tol)
            X = model.transform(G, nodes=[hub])
            cf, layers = blocks(model)
            rows.append({"dataset": name, "n_nodes": len(G), "tol": tol,
                         "err_cf_max": rel_error(X[:, cf], exact[:, cf])[1],
                         "err_layers_max": rel_error(X[:, layers], exact[:, layers])[1]})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("pruning.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
