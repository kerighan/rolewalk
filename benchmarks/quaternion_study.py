"""Walks that keep the order of their moves between onion layers.

Each step of the lazy walk from a root u multiplies the walk's weight by a
unit quaternion chosen from the move: by the type of move (stay in the same
layer, out, in) or by the layer of arrival. This is a walk on the 8-sheeted
cover of the graph with voltages in the quaternion group Q8. After w steps,
the mass of each destination is split by group element, and each part gets
its own characteristic function. With commuting labels, the product along a
path only depends on its arrival layer, which gives back the per-layer
characteristic functions (``layercf``): Q8 is the first case that sees the
order of moves.

``q8_*`` keeps the 8 group elements apart; ``quat_*`` folds g and -g into
one signed quaternion component (4 parts), where reordered paths can cancel.

1. On all connected graphs up to 7 nodes, counts the node classes that each
   encoding adds to RoleWalk (exact, tol = 0).
2. Airports (struc2vec): accuracy of a logistic regression on 50 splits,
   alone and concatenated with RoleWalk.

    python quaternion_study.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from sklearn.preprocessing import normalize

from rolewalk import RoleWalk
from datasets import load_airports
from role_benchmark import within

# Q8 as (sign, unit): index g < 4 is +e_g, g >= 4 is -e_(g-4), e = (1, i, j, k)
_UNITS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _q8_index(q):
    c = int(np.flatnonzero(q)[0])
    return c if q[c] > 0 else c + 4


_ELEMENTS = [s * _UNITS[c] for s in (1, -1) for c in range(4)]
MUL = np.array([[_q8_index(_qmul(a, b)) for b in _ELEMENTS] for a in _ELEMENTS])
# right multiplication by s as an 8 x 8 permutation: (M @ RIGHT[s])[:, g s] = M[:, g]
RIGHT = np.zeros((8, 8, 8), dtype=np.float64)
for _s in range(8):
    for _g in range(8):
        RIGHT[_s, _g, MUL[_g, _s]] = 1

LABELS = {
    "step": lambda dx, dy: np.select([dy > dx, dy < dx], [1, 2], 0),  # out i, in j
    "layer": lambda dx, dy: np.minimum(dy, 3),  # arrival layer 0..3 -> 1, i, j, k
}


def _cf(values, theta):
    """sum_j (exp(i theta v_j) - 1) over the nonzero v_j, real then imaginary."""
    v = values[values != 0][:, None] * theta
    return np.concatenate([(np.cos(v) - 1).sum(0), np.sin(v).sum(0)])


def lazy_walk(G):
    A = nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr", dtype=np.float64)
    return csr_matrix(normalize(identity(len(G)) + A, norm="l1"))


def q8_embedding(G, theta, walk_len=3, labels="step", fold=False):
    H = lazy_walk(G)
    nodes = list(G)
    index = {v: i for i, v in enumerate(nodes)}
    parts = 4 if fold else 8
    out = np.zeros((len(G), walk_len * parts * 2 * theta.shape[1]))
    for u in nodes:
        dist = nx.single_source_shortest_path_length(G, u, cutoff=walk_len)
        ball = np.array([index[v] for v in dist])
        d = np.array(list(dist.values()))
        sub = H[ball][:, ball].tocoo()
        unit = LABELS[labels](d[sub.row], d[sub.col])
        steps = [csr_matrix((sub.data[unit == s], (sub.col[unit == s], sub.row[unit == s])),
                            shape=sub.shape) for s in range(4)]
        M = np.zeros((len(ball), 8))
        M[0, 0] = 1  # the root comes first in dist
        feats = []
        for w in range(walk_len):
            M = sum(S @ M @ RIGHT[s] for s, S in enumerate(steps) if S.nnz)
            cols = M[:, :4] - M[:, 4:] if fold else M
            feats += [_cf(cols[:, g], theta) for g in range(parts)]
        out[index[u]] = np.concatenate(feats)
    return out


def layer_cf(G, theta, walk_len=3):
    """Abelian control: one characteristic function per arrival layer."""
    H = lazy_walk(G)
    nodes = list(G)
    index = {v: i for i, v in enumerate(nodes)}
    out = np.zeros((len(G), walk_len * (walk_len + 1) * 2 * theta.shape[1]))
    T = identity(len(G), format="csr")
    for w in range(walk_len):
        T = T @ H
        for u in nodes:
            dist = nx.single_source_shortest_path_length(G, u, cutoff=walk_len)
            row = T[index[u]].toarray().ravel()
            layer = np.full(len(G), -1)
            layer[[index[v] for v in dist]] = list(dist.values())
            k = 2 * theta.shape[1]
            for ell in range(walk_len + 1):
                start = (w * (walk_len + 1) + ell) * k
                out[index[u], start:start + k] = _cf(np.where(layer == ell, row, 0), theta)
    return out


def encodings(theta):
    return {
        "layercf": lambda G: layer_cf(G, theta),
        "q8_step": lambda G: q8_embedding(G, theta, labels="step"),
        "quat_step": lambda G: q8_embedding(G, theta, labels="step", fold=True),
        "q8_layer": lambda G: q8_embedding(G, theta, labels="layer"),
        "quat_layer": lambda G: q8_embedding(G, theta, labels="layer", fold=True),
    }


def n_classes(X):
    X = X / max(np.abs(X).max(), 1e-12)
    return len({row.tobytes() for row in np.round(X, 6)})


def small_graphs():
    graphs = [G for G in nx.graph_atlas_g()[1:] if nx.is_connected(G)]
    model = RoleWalk(tol=0)
    enc = encodings(model.theta)
    base = [model.transform(G) for G in graphs]
    stack = lambda Xs: np.vstack(Xs)  # noqa: E731
    print(f"{len(graphs)} connected graphs <= 7 nodes, {sum(map(len, graphs))} nodes")
    print(f"  rolewalk: {n_classes(stack(base))} classes")
    for name, f in enc.items():
        extra = [np.hstack([b, f(G)]) for b, G in zip(base, graphs)]
        print(f"  rolewalk + {name}: {n_classes(stack(extra))} classes")


def airports(n_splits=50):
    model = RoleWalk(walk_len=3)
    enc = encodings(model.theta)
    rows = []
    for region in ("brazil", "europe", "usa"):
        G, y = load_airports(region)
        t = time.perf_counter()
        base = model.transform(G)
        variants = {"rolewalk": (base, time.perf_counter() - t)}
        for name, f in enc.items():
            t = time.perf_counter()
            X = f(G)
            dt = time.perf_counter() - t
            variants[name] = (X, dt)
            variants[f"rolewalk+{name}"] = (np.hstack([base, X]), dt)
        for name, (X, dt) in variants.items():
            scores = within(X, y, n_splits)
            rows.append({"dataset": region, "method": name, "dim": X.shape[1],
                         "time": dt, "accuracy": scores.accuracy.mean(),
                         "accuracy_std": scores.accuracy.std()})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("quaternion.csv", index=False)
    print(df.pivot(index="method", columns="dataset", values="accuracy")
          .assign(mean=lambda d: d.mean(axis=1)).sort_values("mean").round(3))


if __name__ == "__main__":
    small_graphs()
    airports()
