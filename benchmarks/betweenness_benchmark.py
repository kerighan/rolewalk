"""RoleWalk vs source sampling for estimating betweenness centrality.

Exact betweenness (Brandes) runs a shortest-path search from every node. The
usual shortcut runs it from ``k`` sampled sources only. RoleWalk instead
learns betweenness once, on a small graph where it is cheap, and then
predicts it from embeddings on new graphs of the same family.

For each family, a regressor is trained on a 2,000-node graph; on a separate
test graph (``--n-test`` nodes), both estimates are compared to the exact
betweenness: Spearman correlation and time. RoleWalk's time is the test
graph's embedding plus prediction; training is a one-off and reported apart.

    python betweenness_benchmark.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rolewalk import RoleWalk
from centrality_benchmark import largest_component, regressor

warnings.filterwarnings("ignore")

FAMILIES = {
    "barabasi_albert": lambda n, s: nx.barabasi_albert_graph(n, 3, seed=s),
    "powerlaw_cluster": lambda n, s: nx.powerlaw_cluster_graph(n, 3, 0.3, seed=s),
    "watts_strogatz": lambda n, s: nx.connected_watts_strogatz_graph(n, 6, 0.1, seed=s),
}


def betweenness(G: nx.Graph, k=None) -> np.ndarray:
    values = nx.betweenness_centrality(G, k=k, seed=0)
    return np.array([values[v] for v in G])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="*", default=sorted(FAMILIES),
                        choices=sorted(FAMILIES))
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--k", type=int, nargs="*", default=[10, 30, 100, 300, 1000])
    parser.add_argument("--output", default="betweenness.csv")
    args = parser.parse_args()

    rows = []
    for family in args.families:
        train = largest_component(FAMILIES[family](args.n_train, 0))
        test = largest_component(FAMILIES[family](args.n_test, 1))

        start = time.perf_counter()
        y = betweenness(train)
        model = regressor().fit(
            RoleWalk(walk_len=4).transform(train), np.log(y + 1e-6 * y.max()))
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        exact = betweenness(test)
        rows.append({"family": family, "method": "exact", "k": len(test),
                     "time": time.perf_counter() - start, "spearman": 1.0})

        start = time.perf_counter()
        pred = model.predict(RoleWalk(walk_len=4).transform(test))
        rows.append({"family": family, "method": "rolewalk", "k": None,
                     "time": time.perf_counter() - start,
                     "spearman": spearmanr(exact, pred)[0],
                     "train_time": train_time})

        for k in args.k:
            start = time.perf_counter()
            approx = betweenness(test, k)
            rows.append({"family": family, "method": "sampling", "k": k,
                         "time": time.perf_counter() - start,
                         "spearman": spearmanr(exact, approx)[0]})
        print(pd.DataFrame([r for r in rows if r["family"] == family])
              .round(3).to_string(index=False), flush=True)

    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
