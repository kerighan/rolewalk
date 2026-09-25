"""Benchmark on GraphWave-style synthetic graphs, averaged over seeds.

Motifs are attached to a base cycle (see ``datasets.generate_shapes_graph``),
random edges are removed and added, and each embedding is scored with the
metrics of the GraphWave paper (k-means homogeneity / completeness / ARI)
plus supervised accuracy, macro-F1 and retrieval mAP.

    python synthetic_benchmark.py
    python synthetic_benchmark.py --datasets varied_large --seeds 3
"""
import os

# Many small BLAS calls: multithreading oversubscribes the CPU and was ~80x
# slower on a 20-core machine. Single-threaded also keeps timings comparable.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings
from typing import Callable, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rolewalk import RoleWalk, mean_average_precision
from datasets import generate_houses, generate_varied

try:
    from karateclub.node_embedding.structural import GraphWave
except ImportError:  # pragma: no cover
    GraphWave = None  # type: ignore


DATASETS: Dict[str, Callable[[int], Tuple[nx.Graph, np.ndarray]]] = {
    "houses": lambda seed: generate_houses(n_shapes=10, seed=seed),
    "houses_large": lambda seed: generate_houses(n_shapes=100, seed=seed),
    "varied": lambda seed: generate_varied(n_per_shape=5, seed=seed),
    "varied_large": lambda seed: generate_varied(n_per_shape=50, seed=seed),
}


def rolewalk(**kwargs) -> Callable[[nx.Graph], np.ndarray]:
    params = dict(walk_len=3, embedding_dim=None)
    params.update(kwargs)
    return lambda G: RoleWalk(**params).transform(G)


def graphwave(G: nx.Graph) -> np.ndarray:
    model = GraphWave()
    model.fit(G.copy())  # fit() adds self-loops in place
    return model.get_embedding()


METHODS: Dict[str, Callable[[nx.Graph], np.ndarray]] = {
    "rolewalk": rolewalk(),
    "rolewalk_w5": rolewalk(walk_len=5),
    "rolewalk_linspace": rolewalk(theta_scheme="linspace"),  # <= 1.0 default
}
if GraphWave is not None:
    METHODS["graphwave"] = graphwave
else:  # pragma: no cover
    warnings.warn("GraphWave is unavailable; skipping it.")


def add_noise(G: nx.Graph, level: float, rng: np.random.Generator) -> nx.Graph:
    """Remove then add ``level * |E|`` random edges.

    Added edges are drawn by rejection sampling, which stays cheap on large
    sparse graphs where listing every non-edge would not.
    """
    H = G.copy()
    edges = list(H.edges())
    n_change = int(level * len(edges))
    if n_change == 0:
        return H
    for idx in rng.choice(len(edges), size=n_change, replace=False):
        H.remove_edge(*edges[idx])
    n = H.number_of_nodes()
    added = 0
    while added < n_change:
        u, v = rng.integers(n, size=2)
        if u != v and not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1
    return H


def score(X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
    n_roles = len(np.unique(y))
    pred = KMeans(n_roles, n_init=10, random_state=seed).fit_predict(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    y_hat = clf.fit(X_train, y_train).predict(X_test)

    return {
        "homogeneity": homogeneity_score(y, pred),
        "completeness": completeness_score(y, pred),
        "ari": adjusted_rand_score(y, pred),
        "accuracy": accuracy_score(y_test, y_hat),
        "macro_f1": f1_score(y_test, y_hat, average="macro"),
        "map": mean_average_precision(X, y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", default=["houses", "varied"],
                        choices=sorted(DATASETS))
    parser.add_argument("--methods", nargs="*", default=sorted(METHODS),
                        choices=sorted(METHODS))
    parser.add_argument("--noise", type=float, nargs="*",
                        default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="synthetic.csv")
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for seed in range(args.seeds):
            G, y = DATASETS[dataset](seed)
            for level in args.noise:
                # same noisy graph for every method: paired comparison
                H = add_noise(G, level, np.random.default_rng(seed))
                for method in args.methods:
                    start = time.perf_counter()
                    X = METHODS[method](H)
                    elapsed = time.perf_counter() - start
                    rows.append({
                        "dataset": dataset,
                        "n_nodes": G.number_of_nodes(),
                        "seed": seed,
                        "noise": level,
                        "method": method,
                        "time": elapsed,
                        **score(X, y, seed),
                    })
                    print(f"{dataset} seed={seed} noise={level} {method}: "
                          f"ari={rows[-1]['ari']:.3f} acc={rows[-1]['accuracy']:.3f} "
                          f"({elapsed:.2f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    metrics = ["ari", "homogeneity", "accuracy", "macro_f1", "map", "time"]
    summary = df.groupby(["dataset", "noise", "method"])[metrics].agg(["mean", "std"])
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary.round(3))


if __name__ == "__main__":
    main()
