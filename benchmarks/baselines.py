"""Re-implementations of role-embedding baselines without a usable package.

Both follow the code of the RONE survey repository (github.com/cspjiao/RONE),
which the RESD authors maintain, with its default parameters:

- ``refex``: ReFeX features (Henderson et al., KDD 2011), the input of RolX
  and RESD: egonet statistics, recursive sums and means over neighbours with
  correlation pruning, then quartile one-hot encoding;
- ``resd``: RESD (Zhang et al., Knowledge-Based Systems 2021), a variational
  autoencoder of the ReFeX features whose embedding must also predict the
  log-degree.
"""
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _base_features(G: nx.Graph) -> np.ndarray:
    n = len(G)
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    deg = np.asarray(A.sum(axis=1)).ravel()
    triangles = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2
    within = deg + triangles  # edges inside the egonet
    total = deg + A @ deg  # degrees summed over the egonet
    with np.errstate(divide="ignore", invalid="ignore"):
        clustering = np.where(deg > 1, 2 * triangles / (deg * (deg - 1)), 0)
        ratio = np.where(total > 0, within / total, 0)
    return A, np.column_stack([within, total, ratio, 1 - ratio, deg, clustering])


def refex(G: nx.Graph, iterations: int = 3, pruning: float = 0.5,
          bins: int = 4) -> np.ndarray:
    A, features = _base_features(G)
    deg = np.asarray(A.sum(axis=1)).ravel()
    rounds = [features]
    for _ in range(iterations):
        old = rounds[-1]
        total = A @ old
        mean = total / np.maximum(deg, 1)[:, None]
        new = np.column_stack([c for pair in zip(total.T, mean.T) for c in pair])
        # drop new features correlated with any feature of the previous round
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(old.T, new.T)[:old.shape[1], old.shape[1]:]
        keep = ~(np.abs(np.nan_to_num(corr)) > pruning).any(axis=0)
        if not keep.any():
            break
        rounds.append(new[:, keep])
    X = np.hstack(rounds)
    X = X / (X.max() - X.min())
    onehot = []
    for column in X.T:
        try:
            q = pd.qcut(column, bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        onehot.append(pd.get_dummies(q).values)
    return np.hstack(onehot).astype(np.float32)


def resd(G: nx.Graph, dims=(128, 64), epochs: int = 100, batch_size: int = 32,
         lr: float = 1e-3, weight_decay: float = 0.01, gamma: float = 0.3,
         seed: int = 0) -> np.ndarray:
    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(refex(G))
    deg = torch.tensor([G.degree(v) for v in range(len(G))], dtype=torch.float32)
    log_deg = torch.log(deg + 1)[:, None]
    struct = [X.shape[1], *dims]

    encoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(struct, struct[1:]))
    mu_layer = nn.Linear(struct[-1], struct[-1])
    sigma_layer = nn.Linear(struct[-1], struct[-1])
    rev = struct[::-1]
    decoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(rev, rev[1:]))
    mlp = nn.ModuleList([nn.Linear(struct[-1], struct[-1]), nn.Linear(struct[-1], 1)])
    for layer in [*encoder, *decoder, *mlp]:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.uniform_(layer.bias)
    vae = nn.ModuleList([encoder, mu_layer, sigma_layer, decoder])

    def encode(x):
        for layer in encoder:
            x = torch.tanh(layer(x))
        mu, sigma = mu_layer(x), sigma_layer(x).exp()
        return mu + sigma * torch.randn_like(sigma)

    def decode(z):
        for i, layer in enumerate(decoder):
            z = layer(z)
            if i < len(decoder) - 1:
                z = torch.tanh(z)
        return z

    def guide(z):
        for layer in mlp:
            z = torch.relu(layer(z))
        return z

    optimizer = torch.optim.Adam([
        {"params": vae.parameters(), "weight_decay": 0},
        {"params": mlp.parameters()},
    ], lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(G)),
                                    max(1, -(-len(G) // batch_size))):
            optimizer.zero_grad()
            z = encode(X[batch])
            loss = (F.mse_loss(decode(z), X[batch])
                    + gamma * F.l1_loss(guide(z), log_deg[batch]))
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        # the RESD code returns a sample of the posterior, not its mean
        return encode(X).numpy()


def heat_cf(G: nx.Graph, times=(1.0, 2.0, 3.0), theta=None,
            block: int = 512) -> np.ndarray:
    """GraphWave with fixed scales: the characteristic function of RoleWalk
    (centred, summed over nodes, same frequencies) applied to the rows of the
    continuous-time random walk exp(s (P - I)) instead of lazy-walk powers.

    No 1 / n and no scale read from the spectrum, so coordinates do not
    depend on the size or spectrum of the graph; but the heat kernel has full
    support, so the embedding is not local. ``s`` is the expected number of
    jumps, matching the walk steps of RoleWalk.
    """
    from scipy.sparse.linalg import expm_multiply

    if theta is None:
        from rolewalk import RoleWalk
        theta = RoleWalk().theta
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = len(G)
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A + sp.diags((deg == 0).astype(float))
    QT = (P - sp.identity(n)).T.tocsr()
    out = np.zeros((n, len(times) * 2 * len(theta)), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        E = sp.identity(n, format="csc")[:, start:stop].toarray()
        feats = []
        for s in times:
            V = expm_multiply(s * QT, E)  # column u = row u of exp(s (P - I))
            V[V < 0] = 0  # round-off
            angles = V[:, :, None] * theta
            feats += [(np.cos(angles) - 1).sum(0), np.sin(angles).sum(0)]
        out[start:stop] = np.hstack(feats)
    return out


def refex_directed(G: nx.DiGraph, iterations: int = 3, pruning: float = 0.5,
                   bins: int = 4) -> np.ndarray:
    """ReFeX with directions: in-, out- and mutual degrees next to the
    egonet features of the symmetrised graph, then recursive sums and means
    over out-neighbours and over in-neighbours (same pruning and binning)."""
    n = len(G)
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    mutual = np.asarray(A.multiply(A.T).sum(axis=1)).ravel()
    _, undirected = _base_features(nx.Graph(G.to_undirected()))
    features = np.column_stack([in_deg, out_deg, mutual, undirected])
    AT = A.T.tocsr()
    rounds = [features]
    for _ in range(iterations):
        old = rounds[-1]
        new = []
        for M, deg in ((A, out_deg), (AT, in_deg)):
            total = M @ old
            mean = total / np.maximum(deg, 1)[:, None]
            new += [c for pair in zip(total.T, mean.T) for c in pair]
        new = np.column_stack(new)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(old.T, new.T)[:old.shape[1], old.shape[1]:]
        keep = ~(np.abs(np.nan_to_num(corr)) > pruning).any(axis=0)
        if not keep.any():
            break
        rounds.append(new[:, keep])
    X = np.hstack(rounds)
    X = X / (X.max() - X.min())
    onehot = []
    for column in X.T:
        try:
            q = pd.qcut(column, bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        onehot.append(pd.get_dummies(q).values)
    return np.hstack(onehot).astype(np.float32)
