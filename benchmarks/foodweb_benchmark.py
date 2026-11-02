"""Trophic roles in 20 real food webs: directed, weighted by carbon flows.

Graphs: the 20 carbon-flow food webs of igraphdata (Ulanowicz et al.; the
Pajek food-web collection), edge i -> j when carbon goes from i to j (prey
to predator), weighted by the flow. Boundary nodes (Input, Output,
Respiration) and self-loops are removed. Labels: the trophic guild of each
compartment (8 classes), given by an LLM from the compartment names alone
(``foodweb_labels.py``), so they do not depend on the graphs.

Protocol: leave one ecosystem out. Webs of the same ecosystem (wet and dry
seasons, the four Chesapeake webs, the two Crystal River webs) form a group;
a classifier is fitted on the nodes of all other groups and predicts the
guilds of the held-out group. Only inductive features can do this. ReFeX's
schema is fitted on the training webs and frozen.

    python foodweb_benchmark.py
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
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from rolewalk import RoleWalk
import role_benchmark as rb
from baselines import RefexSchema, local_stats, topk_layers

warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "foodwebs")
GROUPS = {"baydry": "florida_bay", "baywet": "florida_bay",
          "cypdry": "cypress", "cypwet": "cypress",
          "gramdry": "graminoids", "gramwet": "graminoids",
          "mangdry": "mangrove", "mangwet": "mangrove",
          "ChesLower": "chesapeake", "ChesMiddle": "chesapeake",
          "ChesUpper": "chesapeake", "Chesapeake": "chesapeake",
          "CrystalC": "crystal_river", "CrystalD": "crystal_river"}


def unweighted(G):
    H = G.copy()
    nx.set_edge_attributes(H, 1.0, "weight")
    return H


def strength(G):
    """log in/out degree and log in/out flow: activity, the trivial baseline."""
    A = nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr")
    B = A.copy()
    B.data[:] = 1
    cols = [B.sum(axis=1), B.sum(axis=0), A.sum(axis=1), A.sum(axis=0)]
    return np.log1p(np.column_stack([np.asarray(c).ravel() for c in cols]))


METHODS = {
    "rolewalk": lambda G: RoleWalk().transform(G),
    "rolewalk_curvature": lambda G: RoleWalk(curvature=True).transform(G),
    "rolewalk_unweighted": lambda G: RoleWalk().transform(unweighted(G)),
    "rolewalk_nomixed": lambda G: RoleWalk(mixed=False).transform(G),
    "rolewalk_undirected": lambda G: RoleWalk().transform(G.to_undirected()),
    "topk_layers": topk_layers,
    "local_stats": local_stats,
    "rwse_inout": rb.rwse_inout,
    "digraphwave": rb.digraphwave,
    "strength": strength,
}


def load():
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    labels = pd.read_csv(os.path.join(DATA, "labels.csv"))
    webs = {}
    for web, g in nodes.groupby("web", sort=False):
        inner = g[g.eco <= 2].node.tolist()
        edges = pd.read_csv(os.path.join(DATA, f"{web}_edges.csv"))
        edges = edges[edges.source.isin(inner) & edges.target.isin(inner)
                      & (edges.source != edges.target) & (edges.weight > 0)]
        G = nx.DiGraph()
        G.add_nodes_from(range(len(inner)))
        index = {v: i for i, v in enumerate(inner)}
        G.add_weighted_edges_from((index[s], index[t], w) for s, t, w in edges.values)
        lab = labels[labels.web == web].set_index("node").reindex(inner)
        webs[web] = (G, lab.guild.values)
    return webs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="*", default=list(METHODS) + ["refex_frozen"])
    parser.add_argument("--output", default="foodwebs.csv")
    args = parser.parse_args()

    webs = load()
    group = {w: GROUPS.get(w, w) for w in webs}
    held = sorted(set(group.values()))
    print(f"{len(webs)} webs, {len(held)} ecosystems, "
          f"{sum(pd.notna(y).sum() for _, y in webs.values())} labelled nodes", flush=True)

    feats, start = {}, time.time()
    methods = [m for m in args.methods if m in METHODS]
    for i, m in enumerate(methods):
        t = time.time()
        feats[m] = {w: METHODS[m](G) for w, (G, _) in webs.items()}
        elapsed = time.time() - start
        print(f"[{i + 1}/{len(methods)}] {m} {time.time() - t:.1f} s  "
              f"ETA {elapsed / (i + 1) * (len(methods) - i - 1):.0f} s", flush=True)

    rows = []
    for g in held:
        train = [w for w in webs if group[w] != g]
        test = [w for w in webs if group[w] == g]
        if "refex_frozen" in args.methods:
            union = nx.disjoint_union_all([webs[w][0].to_undirected() for w in train])
            schema = RefexSchema().fit(union)
            feats.setdefault("refex_frozen", {}).update(
                {w: schema.transform(webs[w][0].to_undirected()) for w in test})
            X_train = schema.transform(union)
            y_train = np.concatenate([webs[w][1] for w in train])
            frozen = (X_train, y_train)
        for m in feats:
            if m == "refex_frozen":
                X_tr, y_tr = frozen
            else:
                X_tr = np.vstack([feats[m][w] for w in train])
                y_tr = np.concatenate([webs[w][1] for w in train])
            X_te = np.vstack([feats[m][w] for w in test])
            y_te = np.concatenate([webs[w][1] for w in test])
            keep_tr, keep_te = pd.notna(y_tr), pd.notna(y_te)
            clf = rb.classifier().fit(X_tr[keep_tr], y_tr[keep_tr])
            y_hat = clf.predict(X_te[keep_te])
            rows.append({"ecosystem": g, "method": m, "n_test": keep_te.sum(),
                         "accuracy": accuracy_score(y_te[keep_te], y_hat),
                         "balanced_accuracy": balanced_accuracy_score(y_te[keep_te], y_hat),
                         "macro_f1": f1_score(y_te[keep_te], y_hat, average="macro")})
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    with pd.option_context("display.width", 200, "display.max_rows", None):
        for metric in ("accuracy", "balanced_accuracy"):
            table = df.pivot(index="ecosystem", columns="method", values=metric)
            pooled = df.assign(c=df[metric] * df.n_test).groupby("method").c.sum() \
                / df.groupby("method").n_test.sum()
            table.loc["mean"] = table.mean()
            table.loc["node-weighted"] = pooled
            print(f"\n{metric}\n{table.round(3).to_string()}")
        print("\nRoleWalk minus other, paired over held-out ecosystems (balanced accuracy)")
        wide = df.pivot(index="ecosystem", columns="method", values="balanced_accuracy")
        for m in wide.columns.drop("rolewalk"):
            d = wide["rolewalk"] - wide[m]
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {m:22s} {d.mean():+.3f}  [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}]"
                  f"  wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
