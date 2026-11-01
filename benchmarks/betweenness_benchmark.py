"""Learned betweenness against the best approximations, on new graphs.

Exact betweenness (Brandes) runs a shortest-path search from every node.
Approximations sample: source sampling (NetworKit EstimateBetweenness, k
sources) and KADABRA (adaptive path sampling, error eps with probability
1 - delta), the state of the art. A learned estimator instead fits a
regressor once, on one 2,000-node graph of a family, and predicts the
betweenness of new, larger graphs of that family from their embeddings:
this is what transfer across graphs buys.

For each family and test size, the estimates are compared to the exact
betweenness (Spearman) with their time on one core; the learned
estimators' time is the test graph's features plus prediction (training is
a one-off, reported apart). Exact values come from NetworKit (8 threads).

    python betweenness_benchmark.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings

import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rolewalk import RoleWalk
from baselines import local_stats
from centrality_benchmark import largest_component, regressor

warnings.filterwarnings("ignore")

FAMILIES = {
    "barabasi_albert": lambda n, s: nx.barabasi_albert_graph(n, 3, seed=s),
    "powerlaw_cluster": lambda n, s: nx.powerlaw_cluster_graph(n, 3, 0.3, seed=s),
    "watts_strogatz": lambda n, s: nx.connected_watts_strogatz_graph(n, 6, 0.1, seed=s),
}
LEARNED = {
    "rolewalk": lambda G: RoleWalk(walk_len=4).transform(G),
    "rolewalk_local": lambda G: np.hstack([RoleWalk(walk_len=4).transform(G), local_stats(G)]),
    "local_stats": local_stats,
    "degree": lambda G: np.log1p(np.array([d for _, d in G.degree()], dtype=float))[:, None],
}


def to_nk(G):
    return nk.nxadapter.nx2nk(nx.convert_node_labels_to_integers(G))


def exact(G):
    nk.setNumberOfThreads(8)
    b = nk.centrality.Betweenness(to_nk(G), normalized=True)
    b.run()
    nk.setNumberOfThreads(1)
    return np.array(b.scores())


def timed(f):
    start = time.perf_counter()
    out = f()
    return out, time.perf_counter() - start


def estimate(g, k):
    e = nk.centrality.EstimateBetweenness(g, k, True, False)
    e.run()
    return np.array(e.scores())


def kadabra(g, eps):
    e = nk.centrality.KadabraBetweenness(g, eps, 0.1)
    e.run()
    return np.array(e.scores())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="*", default=sorted(FAMILIES))
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--sizes", type=int, nargs="*", default=[5000, 20000, 50000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="*", default=[10, 30, 100, 300, 1000])
    parser.add_argument("--eps", type=float, nargs="*", default=[0.05, 0.02, 0.01, 0.005])
    parser.add_argument("--output", default="betweenness.csv")
    args = parser.parse_args()
    nk.setNumberOfThreads(1)

    rows = []
    for family in args.families:
        train = largest_component(FAMILIES[family](args.n_train, 0))
        y = exact(train)
        y_fit = np.log(y + 1e-6 * y.max())
        models = {}
        for name, f in LEARNED.items():
            (X, model), t = timed(lambda: (lambda X: (X, regressor().fit(X, y_fit)))(f(train)))
            models[name] = (model, t)
        for n in args.sizes:
            for seed in range(1, 1 + (args.seeds if n < 50000 else 1)):
                G = largest_component(FAMILIES[family](n, seed))
                g = to_nk(G)
                truth = exact(G)
                base = {"family": family, "n_nodes": len(G), "seed": seed}
                for name, (model, train_time) in models.items():
                    pred, t = timed(lambda: model.predict(LEARNED[name](G)))
                    rows.append({**base, "method": name, "param": None, "time": t,
                                 "spearman": spearmanr(truth, pred)[0],
                                 "train_time": train_time})
                for k in args.k:
                    est, t = timed(lambda: estimate(g, k))
                    rows.append({**base, "method": "sampling", "param": k, "time": t,
                                 "spearman": spearmanr(truth, est)[0]})
                for eps in args.eps:
                    est, t = timed(lambda: kadabra(g, eps))
                    rows.append({**base, "method": "kadabra", "param": eps, "time": t,
                                 "spearman": spearmanr(truth, est)[0]})
                pd.DataFrame(rows).to_csv(args.output, index=False)
                print(pd.DataFrame([r for r in rows if r["family"] == family
                                    and r["n_nodes"] == len(G) and r["seed"] == seed])
                      [["method", "param", "time", "spearman"]].round(3).to_string(index=False),
                      flush=True)


if __name__ == "__main__":
    main()
