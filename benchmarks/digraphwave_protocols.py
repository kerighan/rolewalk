"""RoleWalk on the Enron and arXiv protocols of Digraphwave (Ceylan et al.,
TKDD), with their data, splits and metrics, against their published numbers.

Data and published results come from the authors' repository
(``digraphwave-project``, passed as ``--repo``):

- node classification: Enron email types (bot / enron / external, full
  91k-node graph) and simplified employee roles (internal 23k-node
  subgraph); standardised features, balanced logistic regression,
  3-fold stratified cross-validation repeated 5 times (random_state 113),
  macro F1;
- network alignment: the internal Enron subgraph aligned to the full graph,
  and arXiv aligned to a permuted copy, after adding a share p of random
  edges to the union; top-k accuracy of the nearest neighbours (euclidean)
  of each node of the second graph among the nodes of the first, 5
  repetitions per noise level.

RoleWalk runs with its defaults (mixed probes on directed graphs), raw
coordinates; nothing is tuned.

``--experiment snap:<name>`` applies the arXiv protocol (a permuted copy,
noise edges) to a directed SNAP graph in ``data/snap/<name>.txt.gz``, with
Digraphwave (radius 2 and 3, the authors' code) and ReFeX run here
(``--methods``), since no published numbers exist.

    python digraphwave_protocols.py --repo <path> --task classification
    python digraphwave_protocols.py --repo <path> --task alignment
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from rolewalk import RoleWalk

NOISE = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
TOPK = [1, 5, 10, 25, 50]


def read_edges(path):
    with open(path) as f:
        n = int(f.readline().strip("%\n "))
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="%")
    w = df[2].to_numpy(float) if df.shape[1] > 2 else np.ones(len(df))
    return n, df[[0, 1]].to_numpy(np.int64), w


def to_csr(n, edges, weights, weighted, undirected):
    w = weights if weighted else np.ones(len(edges))
    A = sp.csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    A.sum_duplicates()
    if undirected:
        A = A.maximum(A.T) if not weighted else (A + A.T) / 2
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def _directed_blocks(X, which, walk_len=3, m=10):
    """Columns of a directed RoleWalk embedding (no curvature): per walk
    (out, then in) and step, the characteristic function (2m) then the
    layers (w + 1); then the two mixed probes (2m each)."""
    per_walk = sum(2 * m + w + 1 for w in range(1, walk_len + 1))
    cols = []
    for walk in range(2):
        start = walk * per_walk
        for w in range(1, walk_len + 1):
            if which in ("cf", "nomixed"):
                cols += range(start, start + 2 * m)
            start += 2 * m
            if which in ("layers", "nomixed"):
                cols += range(start, start + w + 1)
            start += w + 1
    if which == "cf":
        cols += range(2 * per_walk, X.shape[1])
    return X[:, cols]


_CACHE = {}


def _rolewalk(G):
    # the ablations slice one embedding of the same graph
    key = (id(G), G.number_of_edges())
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = RoleWalk().transform(G)
    return _CACHE[key]


def _methods():
    import baselines
    return {
        "rolewalk": _rolewalk,
        # ablations of the same walks (directed graphs)
        "rolewalk_nomixed": lambda G: _directed_blocks(_rolewalk(G), "nomixed"),
        "rolewalk_layers": lambda G: _directed_blocks(_rolewalk(G), "layers"),
        "rolewalk_cf": lambda G: _directed_blocks(_rolewalk(G), "cf"),
        "topk_layers": baselines.topk_layers,
        "local_stats": baselines.local_stats,
        "digraphwave_r2": lambda G: baselines.digraphwave(G, R=2),
        "digraphwave_r3": lambda G: baselines.digraphwave(G, R=3),
        "refex": lambda G: (baselines.refex_directed(G) if G.is_directed()
                            else baselines.refex(G)),
    }


_GRAPH = {}


def embed(A, undirected, method="rolewalk"):
    import networkx as nx
    if _GRAPH.get("A") is not A:
        _GRAPH.update(A=A, G=nx.from_scipy_sparse_array(
            A, create_using=nx.Graph if undirected else nx.DiGraph))
    G = _GRAPH["G"]
    start = time.perf_counter()
    X = _methods()[method](G)
    return X, time.perf_counter() - start


def read_snap(name):
    df = pd.read_csv(os.path.join("data", "snap", f"{name}.txt.gz"), sep=r"\s+",
                     comment="#", header=None)
    nodes, e = np.unique(df[[0, 1]].to_numpy(np.int64), return_inverse=True)
    e = e.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    return len(nodes), e, np.ones(len(e))


def classification(repo, weighted, undirected):
    rows = []
    base = os.path.join(repo, "data/enron/parsed_enron")
    for label_type, folder in (("role", os.path.join(base, "internal_subgraph")),
                               ("email_type", base)):
        n, edges, w = read_edges(os.path.join(folder, "enron_edges.tsv"))
        X, elapsed = embed(to_csr(n, edges, w, weighted, undirected), undirected)
        labels = pd.read_json(os.path.join(
            folder, f"simplified_{label_type}_labels.json"), typ="series")
        labels = labels[~labels.isnull()]
        X = StandardScaler().fit_transform(X)
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=113)
        for train, test in cv.split(np.empty(len(labels)), labels):
            i_train, i_test = labels.index[train], labels.index[test]
            clf = LogisticRegression(class_weight="balanced").fit(X[i_train], labels.loc[i_train])
            pred = clf.predict(X[i_test])
            rows.append({"label_type": label_type, "time": elapsed,
                         "macro_f1": metrics.f1_score(labels.loc[i_test], pred, average="macro"),
                         "accuracy": metrics.accuracy_score(labels.loc[i_test], pred)})
        print(f"{label_type}: macro F1 {np.mean([r['macro_f1'] for r in rows if r['label_type'] == label_type]):.3f}"
              f"  ({elapsed:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def noise_edges(edges, n, p, rng):
    """Random undirected node pairs not already linked (as in the authors' code)."""
    num = int(p * len(edges))
    if num == 0:
        return np.empty((0, 2), np.int64)
    known = set(map(tuple, np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1)))
    found = set()
    for _ in range(5):
        cand = np.unique(np.sort(rng.integers(0, n, size=(num, 2)), axis=1), axis=0)
        cand = cand[cand[:, 0] != cand[:, 1]]
        found |= set(map(tuple, cand)) - known
        if len(found) >= num:
            break
    found = np.array(list(found))
    return found[rng.permutation(len(found))[:num]]


def topk_accuracy(X, n1, g2_to_g1):
    """For each node of graph 2, rank of its true counterpart among the 50
    nearest nodes of graph 1."""
    nn = NearestNeighbors(n_neighbors=50).fit(X[:n1])
    _, ind = nn.kneighbors(X[n1 + np.arange(len(g2_to_g1))])
    hit = ind == g2_to_g1[:, None]
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1), np.inf)
    return {f"k@{k}": float((rank < k).mean()) for k in TOPK}


def alignment(repo, experiment, weighted, undirected, reps, methods=("rolewalk",), noise=NOISE):
    rng = np.random.default_rng(0)
    permuted = experiment == "arxiv" or experiment.startswith("snap:")
    if experiment == "arxiv":
        n1, e1, w1 = read_edges(os.path.join(repo, "data/arxiv/arxiv_edges.tsv"))
        weighted, undirected = False, True
    elif permuted:
        n1, e1, w1 = read_snap(experiment[5:])
        weighted = False
    else:
        base = os.path.join(repo, "data/enron/parsed_enron")
        suffix = "_undirected" if undirected else ""
        n1, e1, w1 = read_edges(os.path.join(base, f"enron_edges{suffix}.tsv"))
        n2, e2, w2 = read_edges(os.path.join(base, "internal_subgraph", "enron_edges.tsv"))
        g2_to_g1 = pd.read_csv(os.path.join(base, "internal_subgraph", "node_index_alignment.csv"),
                               index_col=0, header=None, comment="%")[1].to_numpy()
    rows, start, total = [], time.time(), len(noise) * reps * len(methods)
    for p in noise:
        for rep in range(reps):
            if permuted:
                perm = rng.permutation(n1)          # g2 node i is g1 node perm[i]
                inv = np.empty_like(perm)
                inv[perm] = np.arange(n1)
                n2, e2, w2, g2_to_g1 = n1, inv[e1], w1, perm
            edges = np.vstack([e1, e2 + n1])
            w = np.concatenate([w1, w2])
            extra = noise_edges(edges, n1 + n2, p, rng)
            if len(extra):
                # noise weights drawn from the empirical weight distribution
                edges = np.vstack([edges, extra])
                w = np.concatenate([w, rng.choice(w, size=len(extra))])
            A = to_csr(n1 + n2, edges, w, weighted, undirected)
            for method in methods:
                X, elapsed = embed(A, undirected, method)
                rows.append({"experiment": experiment, "method": method, "noise_p": p, "rep": rep,
                             "time": elapsed, **topk_accuracy(X, n1, g2_to_g1)})
                done = len(rows)
                el = time.time() - start
                print(f"[{done}/{total}] {experiment} {method} p={p} rep={rep}: "
                      f"k@1 {rows[-1]['k@1']:.3f}  elapsed {el / 60:.1f} min  "
                      f"ETA {el / done * (total - done) / 60:.1f} min", flush=True)
            if p == 0 and not permuted:
                break  # deterministic without noise
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--task", choices=["classification", "alignment"], required=True)
    parser.add_argument("--experiment", default="enron_all_internal",
                        help="enron_all_internal, arxiv or snap:<name>")
    parser.add_argument("--methods", nargs="*", default=["rolewalk"])
    parser.add_argument("--suffix", default="", help="appended to the output name")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--noise", type=float, nargs="*", default=NOISE)
    args = parser.parse_args()
    tag = f"{'weighted' if args.weighted else 'unweighted'}_{'undirected' if args.undirected else 'directed'}"
    os.makedirs("digraphwave_protocols", exist_ok=True)
    if args.task == "classification":
        out = classification(args.repo, args.weighted, args.undirected)
        out.to_csv(f"digraphwave_protocols/enron_classification_{tag}.csv", index=False)
        print(out.groupby("label_type")[["macro_f1", "accuracy"]].agg(["mean", "std"]).round(3))
    else:
        if args.experiment == "arxiv":
            tag = "unweighted_undirected"
        out = alignment(args.repo, args.experiment, args.weighted, args.undirected, args.reps,
                        args.methods, args.noise)
        name = args.experiment.replace(":", "_")
        out.to_csv(f"digraphwave_protocols/alignment_{name}_{tag}{args.suffix}.csv", index=False)
        print(out.groupby(["method", "noise_p"])["k@1"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
