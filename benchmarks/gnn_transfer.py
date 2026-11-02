"""A trained GNN on the centrality transfer of ``centrality_benchmark.py``.

Same 60 pairs of graphs (6 families x 10 seeds; trained on a 2,000-node
graph, tested on a graph of the same family four times larger) and the same
cached targets. A GIN (4 layers, 64 hidden units, sum aggregation) maps node
inputs to the 7 log-centralities at once (standardised on the training
graph), full batch, Adam (lr 1e-3, 500 epochs), early stopping on 20% of the
training graph's nodes (mean Spearman). Inputs, which a GIN propagates:

- ``gin_degree``: log-degree and a constant;
- ``gin_rwse``: log-degree, a constant and 16-step RWSE;
- ``gin_rolewalk``: log-degree, a constant and RoleWalk (4 steps, exact),
  i.e. RoleWalk as a structural encoding.

    python gnn_transfer.py                 # all inputs, 10 seeds
"""
import argparse
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from torch import nn

import centrality_benchmark as cb
from rolewalk import RoleWalk

TARGETS = ["betweenness", "closeness", "harmonic", "pagerank",
           "clustering", "core_number", "eigenvector"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rwse(G, steps=16):
    A = nx.to_scipy_sparse_array(G, nodelist=range(len(G)), format="csr", dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A
    out, T = [], sp.identity(len(G), format="csr")
    for _ in range(steps):
        T = T @ P
        out.append(T.diagonal())
    return np.column_stack(out)


def inputs(G, kind):
    deg = np.array([d for _, d in G.degree()], dtype=float)
    base = [np.log1p(deg)[:, None], np.ones((len(G), 1))]
    if kind == "gin_rwse":
        base.append(rwse(G))
    elif kind == "gin_rolewalk":
        base.append(RoleWalk(walk_len=4, tol=0).transform(G))
    return np.hstack(base).astype(np.float32)


class GIN(nn.Module):
    def __init__(self, d_in, hidden=64, layers=4, d_out=len(TARGETS)):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                nn.Linear(hidden, hidden), nn.ReLU())
                                  for _ in range(layers))
        self.eps = nn.Parameter(torch.zeros(layers))
        self.out = nn.Linear(hidden * (layers + 1), d_out)

    def forward(self, x, A):
        h = self.inp(x)
        hs = [h]
        for eps, mlp in zip(self.eps, self.mlps):
            h = mlp((1 + eps) * h + torch.sparse.mm(A, h))
            hs.append(h)
        return self.out(torch.cat(hs, dim=1))  # jumping knowledge


def torch_adj(G):
    A = nx.to_scipy_sparse_array(G, nodelist=range(len(G)), format="coo", dtype=np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data), A.shape).coalesce().to(DEVICE)


def log_targets(y, names):
    return np.column_stack([np.log(y[t] + 1e-6 * (y[t].max() + 1e-12)) for t in names])


def mean_spearman(Y, P):
    vals = [spearmanr(Y[:, k], P[:, k])[0] for k in range(Y.shape[1]) if np.std(Y[:, k]) > 0]
    return np.nanmean(vals) if vals else -1


def fit_predict(G_train, y_train, G_test, kind, seed):
    torch.manual_seed(seed)
    X_tr, X_te = inputs(G_train, kind), inputs(G_test, kind)
    # standardise inputs and targets on the training graph
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Y = log_targets(y_train, TARGETS)
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    x_tr = torch.tensor((X_tr - mu) / sd, device=DEVICE)
    x_te = torch.tensor((X_te - mu) / sd, device=DEVICE)
    y_t = torch.tensor((Y - ym) / ys, dtype=torch.float32, device=DEVICE)
    A_tr, A_te = torch_adj(G_train), torch_adj(G_test)
    rng = np.random.default_rng(seed)
    val = rng.random(len(G_train)) < 0.2
    tr_mask = torch.tensor(~val, device=DEVICE)
    model = GIN(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, wait = -np.inf, None, 0
    for epoch in range(500):
        model.train()
        opt.zero_grad()
        loss = ((model(x_tr, A_tr) - y_t)[tr_mask] ** 2).mean()
        loss.backward()
        opt.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                pred = model(x_tr, A_tr).cpu().numpy()
            score = mean_spearman(Y[val], pred[val])
            if score > best:
                best, wait = score, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 5:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_te, A_te).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--inputs", nargs="*", default=["gin_degree", "gin_rwse", "gin_rolewalk"])
    parser.add_argument("--output", default="transfer_gnn.csv")
    args = parser.parse_args()
    rows, start = [], time.time()
    total, done = len(cb.GRAPHS) * args.seeds, 0
    for name, make in cb.GRAPHS.items():
        for seed in range(args.seeds):
            G_train = cb.largest_component(make(seed))
            G_test = cb.largest_component(make(seed + 100, 4))
            y_train = cb.targets(G_train, f"{name}_{seed}_x1")
            y_test = cb.targets(G_test, f"{name}_{seed + 100}_x4")
            for kind in args.inputs:
                pred = fit_predict(G_train, y_train, G_test, kind, seed)
                for k, t in enumerate(TARGETS):
                    rows.append({"graph": name, "seed": seed, "target": t, "features": kind,
                                 "spearman": cb.spearman(y_test[t], pred[:, k])})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            done += 1
            el = time.time() - start
            print(f"[{done}/{total}] {name} seed={seed}  elapsed {el / 60:.1f} min  "
                  f"ETA {el / done * (total - done) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["target", "features"]).spearman.mean().unstack().round(3))
    print(df.groupby("features").spearman.mean().round(3))


if __name__ == "__main__":
    main()
