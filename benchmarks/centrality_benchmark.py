"""Can node centralities be predicted from RoleWalk embeddings?

For each graph and centrality, a regressor is trained on half of the nodes
and scored on the other half (Spearman correlation between predicted and true
values, averaged over 2-fold cross-validation). Features are:

- ``degree``: log-degree only, the obvious baseline;
- ``rolewalk``: the RoleWalk embedding (``rolewalk_noshells``: version 1.2,
  without onion layers; ``rolewalk_curvature``: with ``curvature=True``).
  Adding the degree to it changes nothing: on unweighted graphs, its first
  walk step is a function of the degree;
- the inductive baselines of ``role_benchmark.py``: RWSE, ReFeX and
  GraphWave. struc2vec, Role2Vec and RESD learn coordinates specific to each
  graph, so they cannot transfer; neither can ReFeX, whose number of
  features depends on the graph (correlation pruning, quantile bins).

Graphs have 1,000-2,000 nodes: on the small classic real graphs (karate,
Les Misérables...) there are too few nodes to train a regressor on.

With ``--transfer``, the regressor is instead trained on one graph and tested
on another graph of the same family (different seed), which only works if the
embedding describes structure the same way across graphs; ``--test-scale k``
makes the test graphs k times larger, at the same mean degree.

    python centrality_benchmark.py
    python centrality_benchmark.py --transfer
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings
from typing import Callable, Dict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from rolewalk import RoleWalk
from datasets import generate_varied
import role_benchmark as rb
from baselines import RefexSchema, local_stats, topk_layers

warnings.filterwarnings("ignore")

# graph families of 2,000 nodes; ``k`` scales the size at fixed mean degree
GRAPHS: Dict[str, Callable[..., nx.Graph]] = {
    "barabasi_albert": lambda s, k=1: nx.barabasi_albert_graph(2000 * k, 3, seed=s),
    "powerlaw_cluster": lambda s, k=1: nx.powerlaw_cluster_graph(2000 * k, 3, 0.3, seed=s),
    "watts_strogatz": lambda s, k=1: nx.connected_watts_strogatz_graph(2000 * k, 6, 0.1, seed=s),
    "sbm": lambda s, k=1: nx.stochastic_block_model(
        [400 * k, 400 * k, 400 * k, 200 * k, 100 * k],
        (np.full((5, 5), 0.0005) + np.diag([0.02, 0.02, 0.02, 0.04, 0.08])) / k,
        seed=s),
    "random_geometric": lambda s, k=1: nx.random_geometric_graph(2000 * k, 0.04 / np.sqrt(k), seed=s),
    "shapes": lambda s, k=1: generate_varied(n_per_shape=20 * k, seed=s)[0],
}

def eigenvector(G: nx.Graph) -> dict:
    # eigsh on the symmetric adjacency converges where networkx's eigs does not
    A = nx.to_scipy_sparse_array(G, dtype=float)
    _, vec = eigsh(A, k=1, which="LA", maxiter=10_000)
    return dict(zip(G, np.abs(vec[:, 0])))


CENTRALITIES: Dict[str, Callable[[nx.Graph], dict]] = {
    "degree": lambda G: dict(G.degree()),
    "pagerank": nx.pagerank,
    "betweenness": nx.betweenness_centrality,
    "closeness": nx.closeness_centrality,
    "harmonic": nx.harmonic_centrality,
    "eigenvector": eigenvector,
    "clustering": nx.clustering,
    "core_number": nx.core_number,
}


def largest_component(G: nx.Graph) -> nx.Graph:
    G = G.subgraph(max(nx.connected_components(G), key=len))
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


FEATURES: Dict[str, Callable[[nx.Graph], np.ndarray]] = {
    "degree": lambda G: np.log1p(np.array([d for _, d in G.degree()], dtype=float))[:, None],
    "rolewalk": lambda G: RoleWalk(walk_len=4).transform(G),
    "rolewalk_noshells": lambda G: RoleWalk(walk_len=4, shells=False).transform(G),
    "rolewalk_curvature": lambda G: RoleWalk(walk_len=4, curvature=True).transform(G),
    "rwse": rb.rwse,
    "refex": rb.refex,
}
if "graphwave" in rb.METHODS:
    FEATURES["graphwave"] = rb.graphwave
    FEATURES["graphwave_centered"] = rb.graphwave_centered
FEATURES["heat_cf"] = rb.heat_cf
FEATURES["topk_layers"] = topk_layers
FEATURES["local_stats"] = local_stats
FEATURES["digraphwave"] = rb.digraphwave
FEATURES["rolewalk_local"] = lambda G: np.hstack([FEATURES["rolewalk"](G), local_stats(G)])
SELECTED = list(FEATURES)


def features(G: nx.Graph) -> Dict[str, np.ndarray]:
    return {name: FEATURES[name](G) for name in SELECTED}


def targets(G: nx.Graph, cache: str = None) -> Dict[str, np.ndarray]:
    """Exact centralities, cached on disk under ``cache`` (slow on 8k nodes)."""
    path = os.path.join("data", "centrality_targets", f"{cache}.npz") if cache else None
    if path and os.path.exists(path):
        return dict(np.load(path))
    out = {}
    for name, f in CENTRALITIES.items():
        values = f(G)
        out[name] = np.array([values[v] for v in G], dtype=float)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **out)
    return out


def regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return spearmanr(y_true, y_pred)[0]


def within_graph(name: str, G: nx.Graph, seed: int):
    feats, ys = features(G), targets(G)
    rows = []
    for target, y in ys.items():
        # heavy-tailed centralities are easier to fit on a log scale
        y_fit = np.log(y + 1e-6 * (y.max() + 1e-12))
        for feat, X in feats.items():
            preds = np.zeros_like(y)
            for train, test in KFold(2, shuffle=True, random_state=seed).split(X):
                preds[test] = regressor().fit(X[train], y_fit[train]).predict(X[test])
            rows.append({"graph": name, "seed": seed, "target": target,
                         "features": feat, "spearman": spearman(y, preds)})
    return rows


def transfer(name: str, G_train: nx.Graph, G_test: nx.Graph, seed: int, scale: int = 1):
    f_train, f_test = features(G_train), features(G_test)
    y_train = targets(G_train, f"{name}_{seed}_x1")
    y_test = targets(G_test, f"{name}_{seed + 100}_x{scale}")
    if "refex" in SELECTED:
        # ReFeX with its feature schema fitted on the training graph
        schema = RefexSchema().fit(G_train)
        f_train["refex_frozen"] = schema.transform(G_train)
        f_test["refex_frozen"] = schema.transform(G_test)
    rows = []
    for target in CENTRALITIES:
        y = y_train[target]
        y_fit = np.log(y + 1e-6 * (y.max() + 1e-12))
        for feat in f_train:
            if f_train[feat].shape[1] != f_test[feat].shape[1]:
                continue  # graph-specific features (ReFeX)
            pred = regressor().fit(f_train[feat], y_fit).predict(f_test[feat])
            rows.append({"graph": name, "seed": seed, "target": target,
                         "features": feat, "spearman": spearman(y_test[target], pred)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--graphs", nargs="*", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--test-scale", type=int, default=1,
                        help="with --transfer, test graphs are this many times larger")
    parser.add_argument("--features", nargs="*", default=list(FEATURES),
                        choices=list(FEATURES))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    SELECTED[:] = args.features

    output = args.output or ("centrality_transfer.csv" if args.transfer else "centrality.csv")
    rows, done, total, start = [], 0, len(args.graphs) * args.seeds, time.time()
    for name in args.graphs:
        make = GRAPHS[name]
        for seed in range(args.seeds):
            G = largest_component(make(seed))
            if args.transfer:
                rows += transfer(name, G, largest_component(make(seed + 100, args.test_scale)),
                                 seed, args.test_scale)
            else:
                rows += within_graph(name, G, seed)
            pd.DataFrame(rows).to_csv(output, index=False)  # partial results
            done += 1
            elapsed = time.time() - start
            print(f"[{done}/{total}] {name} seed={seed}  elapsed {elapsed / 60:.1f} min"
                  f"  ETA {elapsed / done * (total - done) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        table = df.groupby(["target", "features"]).spearman.mean().unstack("features")
        print(table.round(3))
        print(df.groupby(["graph", "features"]).spearman.mean().unstack("features").round(3))


if __name__ == "__main__":
    main()
