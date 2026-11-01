"""One characteristic function of the whole rooted trajectory, instead of
per-step characteristic functions plus onion layers.

Each node j reached from u has a trajectory z(j) = (1[j = u], p^1_uj,
p^2_uj, p^3_uj). A projection a gives the walk stopped at a random time,
R_a = sum_t a_t H^t, and its characteristic function samples the joint
characteristic function of the trajectories along a. Mixed-time projections
see whether large probabilities at different steps sit on the same node,
which per-step multisets forget; the root indicator plays the role of the
layers. Compared, at about the same dimension, with RoleWalk.

    python trajectory_study.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import identity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from rolewalk import RoleWalk, compute_iter
from datasets import generate_directed_shapes, load_airports
from quaternion_study import n_classes
from role_benchmark import classifier, within
from synthetic_benchmark import add_noise, score

MARGINAL = [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
MIXED = [(.5, .5, 0, 0), (.5, 0, .5, 0), (.5, 0, 0, .5), (0, .5, .5, 0), (0, 0, .5, .5)]


def powers(G, L=3, directed_part=None):
    A = nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr", dtype=np.float64)
    if directed_part == "in":
        A = A.T.tocsr()
    H = normalize(identity(len(G)) + A, norm="l1").tocsr()
    P = [identity(len(G), format="csr")]
    for _ in range(L):
        P.append((P[-1] @ H).tocsr())
    return P


def trajectory_cf(G, projections, theta):
    parts = ["out", "in"] if G.is_directed() else ["out"]
    feats = []
    for part in parts:
        P = powers(G, directed_part=part)
        for a in projections:
            R = sum(c * M for c, M in zip(a, P) if c).tocsr()
            feats.append(compute_iter(R, theta))
    return np.hstack(feats)


def variants():
    rw = RoleWalk(tol=0)
    th3 = rw.theta[:, [0, 4, 9]]
    th6 = rw.theta[:, [0, 2, 4, 6, 8, 9]]
    return {
        "rolewalk (69)": lambda G: RoleWalk(tol=0).transform(G),
        "marginal, 6 freq (36)": lambda G: trajectory_cf(G, MARGINAL, th6),
        "marginal+mixed, 3 freq (48)": lambda G: trajectory_cf(G, MARGINAL + MIXED, th3),
        "marginal+mixed, 6 freq (96)": lambda G: trajectory_cf(G, MARGINAL + MIXED, th6),
        "marginal+mixed, 10 freq (160)": lambda G: trajectory_cf(G, MARGINAL + MIXED, rw.theta),
    }


def main():
    V = variants()
    rows = []
    graphs = [G for G in nx.graph_atlas_g()[1:] if nx.is_connected(G)]
    for name, f in V.items():
        row = {"variant": name,
               "atlas_classes": n_classes(np.vstack([f(G) for G in graphs]))}
        air = {r: load_airports(r) for r in ("brazil", "europe", "usa")}
        E = {r: f(G) for r, (G, y) in air.items()}
        row["airports_acc"] = np.mean([within(E[r], air[r][1], 50).accuracy.mean() for r in air])
        row["transfer"] = np.mean([
            accuracy_score(air[t][1], classifier().fit(E[s], air[s][1]).predict(E[t]))
            for s in air for t in air if s != t])
        directed = []
        for seed in range(3):
            G, y = generate_directed_shapes(5, seed=seed)
            H = add_noise(G, 0.05, np.random.default_rng(seed))
            directed.append(score(f(H), y, seed))
        row["directed_acc"] = np.mean([d["accuracy"] for d in directed])
        row["directed_ari"] = np.mean([d["ari"] for d in directed])
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("trajectory.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
