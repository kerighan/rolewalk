"""Role benchmarks of the literature, against the main baselines.

Classification (``clf``): the airport networks of struc2vec (Ribeiro et al.
2017), labelled by activity quartile, and the reality-call, actor and film
networks of the RONE survey (Jiao et al. 2021). Each embedding feeds a
logistic regression, scored on 50 stratified 80/20 splits (accuracy =
micro-F1, and macro-F1).

Retrieval (``topk``): the Wikipedia talk networks of RONE, where bots and
admins must be found among the nearest neighbours of other bots and admins
(precision@k on standardised embeddings, Euclidean distance).

Baselines:

- ``struc2vec``: the reference method on this benchmark, if the ``ge``
  package (github.com/shenweichen/GraphEmbedding) is installed;
- ``rwse``: random-walk structural encoding, the return probabilities
  diag(P^k) used as positional encoding in graph transformers (GPS);
- ``role2vec`` and ``graphwave`` from karateclub;
- ``refex`` and ``resd`` (see ``baselines.py``): ReFeX features, the input
  of RolX, and RESD (2021), the strongest method of the RONE survey;
- ``degree``: log-degree alone, since labels track activity.

Embeddings are cached in ``data/embeddings`` (``--no-cache`` to recompute),
and methods are skipped on graphs larger than ``MAX_NODES``.

With ``--transfer``, the classifier is trained on one network and tested on
the others: only inductive embeddings, whose coordinates mean the same thing
in every graph, can do this.

    python role_benchmark.py
    python role_benchmark.py --datasets brazil europe usa --transfer
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import tempfile
import time
import warnings
from typing import Callable, Dict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rolewalk import RoleWalk
from baselines import (RefexSchema, digraphwave, heat_cf, local_stats, refex,
                       refex_directed, resd, topk_layers)
from datasets import DATA_DIR, load_airports, load_rone, load_wiki_rfa

try:
    from karateclub.node_embedding.structural import GraphWave, Role2Vec
except ImportError:  # pragma: no cover
    GraphWave = Role2Vec = None  # type: ignore
try:
    from ge import Struc2Vec
except ImportError:  # pragma: no cover
    Struc2Vec = None  # type: ignore

REGIONS = ("brazil", "europe", "usa")
WIKI = ("ht", "br", "cy", "oc", "eo", "gl")
# name -> (loader, task)
DATASETS = {
    **{r: (lambda r=r: load_airports(r), "clf") for r in REGIONS},
    **{n: (lambda n=n: load_rone(n), "clf") for n in ("reality-call", "actor", "film")},
    **{f"{w}-wiki": (lambda w=w: load_rone(f"{w}-wiki-talk"), "topk") for w in WIKI},
    # directed: Wikipedia adminship votes; all users, then elected or not
    # among candidates only (voters never receive a vote, which is trivial)
    "wiki-rfa": (load_wiki_rfa, "clf"),
    "rfa-candidates": (lambda: (lambda G, y: (G, np.where(y > 0, y, -1)))(*load_wiki_rfa()),
                       "clf"),
}
# methods that read edge directions; the others get the undirected graph
DIRECTED = {"rolewalk", "rolewalk_noshells", "rolewalk_curvature", "rwse", "degree",
            "rwse_inout", "refex_directed", "rolewalk_mixed", "rolewalk_curvature_mixed",
            "topk_layers", "local_stats", "rolewalk_local", "digraphwave"}


def rolewalk(G: nx.Graph) -> np.ndarray:
    return RoleWalk(walk_len=3).transform(G)


def rolewalk_curvature(G: nx.Graph) -> np.ndarray:
    return RoleWalk(walk_len=3, curvature=True).transform(G)


def rolewalk_mixed(G: nx.Graph) -> np.ndarray:
    return RoleWalk(walk_len=3, mixed=True).transform(G)


def rolewalk_curvature_mixed(G: nx.Graph) -> np.ndarray:
    return RoleWalk(walk_len=3, curvature=True, mixed=True).transform(G)


def rolewalk_noshells(G: nx.Graph) -> np.ndarray:
    """RoleWalk 1.2: no onion layers."""
    return RoleWalk(walk_len=3, shells=False).transform(G)


def rwse(G: nx.Graph, k: int = 16) -> np.ndarray:
    """Return probabilities diag(P^1..P^k) of the random walk P = D^-1 A."""
    A = nx.to_scipy_sparse_array(G, nodelist=range(len(G)), format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A
    M = sp.identity(len(G), format="csr")
    features = []
    for _ in range(k):
        M = M @ P
        features.append(M.diagonal())
    return np.column_stack(features)


def rwse_inout(G: nx.Graph, k: int = 16) -> np.ndarray:
    """RWSE on the out-walk and on the in-walk (the same twice if undirected)."""
    if not G.is_directed():
        return rwse(G, k)
    return np.hstack([rwse(G, k), rwse(G.reverse(copy=True), k)])


def degree(G: nx.Graph) -> np.ndarray:
    if G.is_directed():
        return np.log1p([[G.in_degree(v), G.out_degree(v)] for v in range(len(G))])
    return np.log1p([[G.degree(v)] for v in range(len(G))])


def graphwave(G: nx.Graph) -> np.ndarray:
    model = GraphWave()
    model.fit(G.copy())  # fit() adds self-loops in place
    return model.get_embedding()


def graphwave_centered(G: nx.Graph) -> np.ndarray:
    """GraphWave without its 1 / n: n (chi - 1), the sum form of RoleWalk."""
    X = graphwave(G)
    half = X.shape[1] // 2
    return np.hstack([len(G) * (X[:, :half] - 1), len(G) * X[:, half:]])


def role2vec(G: nx.Graph) -> np.ndarray:
    model = Role2Vec(seed=0)
    model.fit(G.copy())
    return model.get_embedding()


def struc2vec(G: nx.Graph) -> np.ndarray:
    # parameters of the struc2vec paper: 10 walks of length 80 per node,
    # 128 dimensions, window 10, all three optimisations
    H = nx.relabel_nodes(G, str)
    with tempfile.TemporaryDirectory() as tmp:
        model = Struc2Vec(H, walk_length=80, num_walks=10, workers=1,
                          temp_path=tmp + "/")
        model.train(embed_size=128, window_size=10, workers=1)
        emb = model.get_embeddings()
    return np.vstack([emb[str(v)] for v in range(len(G))])


METHODS: Dict[str, Callable[[nx.Graph], np.ndarray]] = {
    "rolewalk": rolewalk,
    "rolewalk_noshells": rolewalk_noshells,
    "rolewalk_curvature": rolewalk_curvature,
    "rwse": rwse,
    "degree": degree,
    "refex": refex,
    "resd": resd,
    "heat_cf": heat_cf,
    "rwse_inout": rwse_inout,
    "refex_directed": refex_directed,
    "rolewalk_mixed": rolewalk_mixed,
    "topk_layers": topk_layers,
    "digraphwave": digraphwave,
    "local_stats": local_stats,
    "rolewalk_local": lambda G: np.hstack([rolewalk(G), local_stats(G)]),
    "rolewalk_curvature_mixed": rolewalk_curvature_mixed,
}
if GraphWave is not None:
    METHODS["graphwave"] = graphwave
    METHODS["graphwave_centered"] = graphwave_centered
    METHODS["role2vec"] = role2vec
else:  # pragma: no cover
    warnings.warn("karateclub is unavailable; skipping GraphWave and Role2Vec.")
if Struc2Vec is not None:
    METHODS["struc2vec"] = struc2vec
else:  # pragma: no cover
    warnings.warn("ge is unavailable; skipping struc2vec.")

# embeddings defined graph by graph (word2vec spaces, arbitrary axes) cannot
# be compared across networks
INDUCTIVE = {"rolewalk", "rolewalk_noshells", "rolewalk_curvature", "rwse",
             "degree", "graphwave", "graphwave_centered", "heat_cf", "topk_layers",
             "local_stats", "rolewalk_local", "digraphwave"}
# struc2vec took 6 minutes on 1,190 nodes and grows faster than linearly
MAX_NODES = {"struc2vec": 3000, "graphwave": 10000, "graphwave_centered": 10000}  # graphwave: dense n x n
K = (5, 10, 25, 50)


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def within(X: np.ndarray, y: np.ndarray, n_splits: int) -> pd.DataFrame:
    splits = StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=0)
    rows = []
    for train, test in splits.split(X, y):
        y_hat = classifier().fit(X[train], y[train]).predict(X[test])
        rows.append({"accuracy": accuracy_score(y[test], y_hat),
                     "macro_f1": f1_score(y[test], y_hat, average="macro")})
    return pd.DataFrame(rows)


def top_k(X: np.ndarray, y: np.ndarray) -> dict:
    """Precision@k of each positive class among its nearest neighbours."""
    from sklearn.neighbors import NearestNeighbors

    X = StandardScaler().fit_transform(X)
    nn_ = NearestNeighbors(n_neighbors=max(K) + 1).fit(X)
    scores = {}
    for label, name in ((1, "bots"), (2, "admins")):
        nodes = np.flatnonzero(y == label)
        if len(nodes) == 0:
            continue
        _, idx = nn_.kneighbors(X[nodes])
        # drop the query itself (first neighbour unless tied with a twin)
        hits = np.array([[j for j in row if j != v][:max(K)]
                         for v, row in zip(nodes, idx)])
        for k in K:
            scores[f"{name}@{k}"] = float((y[hits[:, :k]] == label).mean())
    return scores


def embed(name: str, method: str, G: nx.Graph, cache: bool):
    """Embedding and time of ``method`` on dataset ``name``, cached on disk."""
    path = os.path.join(DATA_DIR, "embeddings", f"{name}__{method}.npz")
    if cache and os.path.exists(path):
        saved = np.load(path)
        return saved["X"], float(saved["time"])
    if G.is_directed() and method not in DIRECTED:
        G = nx.Graph(G.to_undirected())
    start = time.perf_counter()
    X = METHODS[method](G)
    elapsed = time.perf_counter() - start
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, time=elapsed)
    return X, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--methods", nargs="*", default=sorted(METHODS),
                        choices=sorted(METHODS))
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    graphs = {name: DATASETS[name][0]() for name in args.datasets}
    embeddings, rows = {}, []
    for name, (G, y) in graphs.items():
        task = DATASETS[name][1]
        for method in args.methods:
            if len(G) > MAX_NODES.get(method, np.inf):
                print(f"{name} {method}: skipped ({len(G)} nodes)", flush=True)
                continue
            X, elapsed = embed(name, method, G, args.cache)
            embeddings[name, method] = X
            if args.transfer:
                continue
            row = {"dataset": name, "task": task, "n_nodes": len(G),
                   "method": method, "time": elapsed}
            if task == "clf":
                keep = y >= 0
                scores = within(X[keep], y[keep], args.splits)
                row.update({f"{m}_{s}": getattr(scores[m], s)()
                            for m in scores for s in ("mean", "std")})
                print(f"{name} {method}: acc={row['accuracy_mean']:.3f} "
                      f"f1={row['macro_f1_mean']:.3f} ({elapsed:.1f}s)", flush=True)
            else:
                row.update(top_k(X, y))
                print(f"{name} {method}: " + " ".join(
                    f"{k}={v:.3f}" for k, v in row.items() if "@10" in k)
                    + f" ({elapsed:.1f}s)", flush=True)
            rows.append(row)

    if args.transfer:
        names = [n for n in args.datasets if DATASETS[n][1] == "clf"]
        # ReFeX with its feature schema fitted on the source graph
        for source in names:
            G, y = graphs[source]
            schema = RefexSchema().fit(G)
            clf = classifier().fit(schema.transform(G)[y >= 0], y[y >= 0])
            for target in names:
                if target == source:
                    continue
                Gt, yt = graphs[target]
                y_hat = clf.predict(schema.transform(Gt)[yt >= 0])
                yt = yt[yt >= 0]
                rows.append({"method": "refex_frozen", "train": source, "test": target,
                             "accuracy": accuracy_score(yt, y_hat),
                             "macro_f1": f1_score(yt, y_hat, average="macro")})
        for method in args.methods:
            if method not in INDUCTIVE:
                continue
            for source in names:
                if (source, method) not in embeddings:
                    continue
                G, y = graphs[source]
                clf = classifier().fit(embeddings[source, method][y >= 0], y[y >= 0])
                for target in names:
                    if target == source or (target, method) not in embeddings:
                        continue
                    y = graphs[target][1]
                    y_hat = clf.predict(embeddings[target, method][y >= 0])
                    y = y[y >= 0]
                    rows.append({"method": method, "train": source, "test": target,
                                 "accuracy": accuracy_score(y, y_hat),
                                 "macro_f1": f1_score(y, y_hat, average="macro")})
                    print(f"{method} {source}->{target}: "
                          f"acc={rows[-1]['accuracy']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    output = args.output or ("role_transfer.csv" if args.transfer else "roles.csv")
    df.to_csv(output, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", None,
                           "display.max_columns", None):
        if args.transfer:
            print(df.pivot_table(index="method", columns=["train", "test"],
                                 values="accuracy").round(3))
            return
        clf_rows = df[df.task == "clf"]
        if len(clf_rows):
            print(clf_rows.pivot_table(index="method", columns="dataset",
                                       values="accuracy_mean").round(3))
        topk_rows = df[df.task == "topk"]
        if len(topk_rows):
            values = [c for c in ("bots@10", "admins@10") if c in topk_rows]
            print(topk_rows.pivot_table(index="method", columns="dataset",
                                        values=values).round(3))
        print(df.pivot_table(index="method", columns="dataset", values="time").round(2))


if __name__ == "__main__":
    main()
