"""RoleWalk as a structural encoding for graph neural networks, on ZINC.

ZINC (12k molecules, the subset of Dwivedi et al.) is the standard test of
positional and structural encodings: regress constrained solubility, scored
by test MAE (lower is better). A GINE network (edge-aware GIN, 4 layers,
under the 500k-parameter budget of the benchmark) receives each encoding,
batch-normalised and projected, added to the atom embeddings:

- ``none``: no encoding;
- ``rwse``: random-walk return probabilities diag(P^k), k = 1..16, the
  structural encoding of GPS (Rampasek et al. 2022);
- ``rolewalk``: RoleWalk embedding of each molecule;
- ``rolewalk+rwse``: both;
- ``curvature``: RoleWalk with ``curvature=True``, alone or with RWSE.

Every encoding gets the same model, budget and seeds; the best validation
epoch gives the test MAE. Needs torch_geometric.

    python pe_benchmark.py
    python pe_benchmark.py --epochs 100 --seeds 1 --encodings none rolewalk
"""
import argparse
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_scipy_sparse_matrix

from rolewalk import RoleWalk
from datasets import DATA_DIR

ENCODINGS = ("none", "rwse", "rolewalk", "rolewalk+rwse", "curvature",
             "curvature+rwse")


def rwse(A: sp.csr_matrix, k: int = 16) -> np.ndarray:
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A
    M = sp.identity(A.shape[0], format="csr")
    out = []
    for _ in range(k):
        M = M @ P
        out.append(M.diagonal())
    return np.column_stack(out)


def add_encodings(dataset, model: RoleWalk, curved: RoleWalk):
    """Attach ``rwse``, ``rolewalk`` and ``curvature`` node features."""
    graphs = []
    for data in dataset:
        n = data.num_nodes
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
        A.data[:] = 1
        G = nx.from_scipy_sparse_array(A)
        data.rwse = torch.from_numpy(rwse(A)).float()
        data.rolewalk = torch.from_numpy(model.transform(G)).float()
        data.curvature = torch.from_numpy(curved.transform(G)).float()
        graphs.append(data)
    return graphs


class GINE(nn.Module):
    def __init__(self, encoding: str, pe_dims: dict, hidden: int = 128,
                 layers: int = 4):
        super().__init__()
        self.parts = [] if encoding == "none" else encoding.split("+")
        self.atom = nn.Embedding(28, hidden)
        self.bond = nn.Embedding(4, hidden)
        self.pe = nn.ModuleDict({
            p: nn.Sequential(nn.BatchNorm1d(pe_dims[p]), nn.Linear(pe_dims[p], hidden))
            for p in self.parts})
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                                   nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=hidden)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, data):
        h = self.atom(data.x.squeeze(-1))
        for p in self.parts:
            h = h + self.pe[p](data[p])
        e = self.bond(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, data.edge_index, e)))  # residual
        return self.head(global_add_pool(h, data.batch)).squeeze(-1)


def run(encoding, splits, pe_dims, epochs, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = {name: DataLoader(graphs, batch_size=128, shuffle=name == "train")
               for name, graphs in splits.items()}
    model = GINE(encoding, pe_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-5)

    def evaluate(loader):
        model.eval()
        error = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                error += (model(batch) - batch.y).abs().sum().item()
        return error / len(loader.dataset)

    best = (np.inf, np.inf)
    for _ in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = (model(batch) - batch.y).abs().mean()
            loss.backward()
            optimizer.step()
        val = evaluate(loaders["val"])
        scheduler.step(val)
        if val < best[0]:
            best = (val, evaluate(loaders["test"]))
    n_params = sum(p.numel() for p in model.parameters())
    return {"val_mae": best[0], "test_mae": best[1], "n_params": n_params}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encodings", nargs="*", default=list(ENCODINGS),
                        choices=ENCODINGS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="pe_zinc.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(DATA_DIR, "ZINC")
    start = time.perf_counter()
    cache = os.path.join(root, "encodings.pt")
    if os.path.exists(cache):
        splits = torch.load(cache, weights_only=False)
    else:
        model, curved = RoleWalk(), RoleWalk(curvature=True)
        splits = {name: add_encodings(ZINC(root, subset=True, split=name),
                                      model, curved)
                  for name in ("train", "val", "test")}
        torch.save(splits, cache)
    pe_dims = {p: splits["train"][0][p].shape[1]
               for p in ("rwse", "rolewalk", "curvature")}
    print(f"encodings computed in {time.perf_counter() - start:.0f}s: {pe_dims}",
          flush=True)

    rows = []
    for seed in range(args.seeds):
        for encoding in args.encodings:
            start = time.perf_counter()
            result = run(encoding, splits, pe_dims, args.epochs, seed, device)
            rows.append({"encoding": encoding, "seed": seed,
                         "time": time.perf_counter() - start, **result})
            print(f"seed={seed} {encoding}: test MAE {result['test_mae']:.4f} "
                  f"(val {result['val_mae']:.4f}, {result['n_params']} params, "
                  f"{rows[-1]['time']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)

    df = pd.DataFrame(rows)
    print(df.groupby("encoding")[["test_mae", "val_mae"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
