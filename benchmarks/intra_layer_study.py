"""Walks that count their steps inside an onion layer.

A step of the lazy walk from u either pauses (the self-loop), or moves out,
in, or inside a layer (distance to u unchanged, the edge closes an odd
cycle: a triangle through u in layer 1). Pauses carry no information, and
labelling them like moves inside a layer confuses a triangle with a slowed
down back-and-forth.

``exact``: the mass of every destination layer, split by the number of
intra-layer steps (0..w), with one BFS per root.

``parity``: the same up to parity, with no BFS. Moves out and in change the
layer by one, intra-layer moves do not, so

    #intra = #moves - layer = w - #pauses - layer   (mod 2).

Counting pauses mod 2 is the walk H_ = D^-1 (A - I), whose entries are
sum over paths of (-1)^#pauses p: (H^w +- H_^w) / 2 split the mass by
parity, and the layers come from the supports as in RoleWalk. Output: per
step, the layer masses of walks with an odd number of intra-layer steps.

1. On all connected graphs up to 7 nodes, the node classes each adds to
   RoleWalk (tol = 0).
2. Airports: accuracy of a logistic regression on 50 splits.

    python intra_layer_study.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity

from rolewalk import RoleWalk
from datasets import load_airports
from quaternion_study import n_classes
from role_benchmark import within


def _prune(T, tol):
    if tol:
        T.data[np.abs(T.data) < tol] = 0
        T.eliminate_zeros()
    return T


def parity_layers(G, theta, walk_len=3, tol=1e-4):
    A = csr_matrix(nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr",
                                            dtype=np.float64))
    inv = diags(1 / (1 + np.asarray(A.sum(axis=1)).ravel()))
    I = identity(len(G), format="csr")
    H, H_ = csr_matrix(inv @ (I + A)), csr_matrix(inv @ (A - I))
    scale = np.linalg.norm(theta)
    T, T_ = I, I
    supports, feats = [I], []
    for w in range(walk_len):
        T, T_ = _prune(T @ H, tol), _prune(T_ @ H_, tol)
        sign = (-1) ** (w + 1)  # sum (-1)^#moves p = (-1)^w H_^w
        even, odd = (T + sign * T_) / 2, (T - sign * T_) / 2
        layers = []
        for M in (even, odd):
            within = [np.asarray(M.multiply(S).sum(axis=1)).ravel() for S in supports]
            within.append(np.asarray(M.sum(axis=1)).ravel())
            layers.append(np.diff(np.column_stack(within), prepend=0, axis=1))
        # odd intra-layer steps: odd moves to an even layer, or the converse
        ell = np.arange(w + 2)
        feats.append(scale * np.where(ell % 2 == 0, layers[1], layers[0]))
        S = T.copy()
        S.data[:] = 1
        supports.append(S)
    return np.hstack(feats)


def exact_layers(G, theta, walk_len=3):
    """Per root: mass by (step, layer, number of intra-layer steps)."""
    nodes = list(G)
    index = {v: i for i, v in enumerate(nodes)}
    deg = dict(G.degree())
    scale = np.linalg.norm(theta)
    L, K = walk_len + 1, walk_len + 1
    out = np.zeros((len(G), walk_len * L * K))
    for u in nodes:
        dist = nx.single_source_shortest_path_length(G, u, cutoff=walk_len)
        mass = {(u, 0): 1.0}
        for w in range(walk_len):
            new = {}
            for (x, k), p in mass.items():
                q = p / (deg[x] + 1)
                new[(x, k)] = new.get((x, k), 0) + q
                for y in G[x]:
                    key = (y, k + (dist[x] == dist[y]))
                    new[key] = new.get(key, 0) + q
            mass = new
            for (y, k), p in mass.items():
                out[index[u], (w * L + dist[y]) * K + k] += scale * p
    return out


def main(n_splits=50):
    graphs = [G for G in nx.graph_atlas_g()[1:] if nx.is_connected(G)]
    model = RoleWalk(tol=0)
    base = [model.transform(G) for G in graphs]
    print(f"{len(graphs)} connected graphs <= 7 nodes, {sum(map(len, graphs))} nodes")
    print(f"  rolewalk: {n_classes(np.vstack(base))} classes")
    for name, f in (("parity", lambda G: parity_layers(G, model.theta, tol=0)),
                    ("exact", lambda G: exact_layers(G, model.theta))):
        X = np.vstack([np.hstack([b, f(G)]) for b, G in zip(base, graphs)])
        print(f"  rolewalk + {name}: {n_classes(X)} classes")

    model = RoleWalk(walk_len=3)
    rows = []
    for region in ("brazil", "europe", "usa"):
        G, y = load_airports(region)
        t = time.perf_counter()
        X0 = model.transform(G)
        variants = {"rolewalk": (X0, time.perf_counter() - t)}
        for name, f in (("parity", lambda G: parity_layers(G, model.theta)),
                        ("exact", lambda G: exact_layers(G, model.theta))):
            t = time.perf_counter()
            X = f(G)
            variants[f"rolewalk+{name}"] = (np.hstack([X0, X]), time.perf_counter() - t)
        for name, (X, dt) in variants.items():
            scores = within(X, y, n_splits)
            rows.append({"dataset": region, "method": name, "dim": X.shape[1],
                         "time": dt, "accuracy": scores.accuracy.mean(),
                         "accuracy_std": scores.accuracy.std()})
    df = pd.DataFrame(rows)
    df.to_csv("intra_layer.csv", index=False)
    print(df.pivot(index="method", columns="dataset", values="accuracy")
          .assign(mean=lambda d: d.mean(axis=1)).round(3))
    print(df.groupby("method")[["dim", "time"]].mean().round(3))


if __name__ == "__main__":
    main()
