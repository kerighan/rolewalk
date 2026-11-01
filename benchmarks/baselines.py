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


def _walk_layers(G: nx.Graph, walk_len: int = 3):
    """RoleWalk's layer block (scaled), out- then in-walk if directed."""
    from rolewalk import RoleWalk
    model = RoleWalk(walk_len=walk_len)
    X = model.transform(G)
    m = model.theta.shape[1]
    cols, start = [], 0
    for w in range(1, walk_len + 1):
        start += 2 * m
        cols += range(start, start + w + 1)
        start += w + 1
    cols = np.array(cols)
    if G.is_directed():
        cols = np.r_[cols, cols + X.shape[1] // 2]
    return X[:, cols]


def topk_layers(G: nx.Graph, k: int = 20, walk_len: int = 3, tol: float = 1e-4) -> np.ndarray:
    """The k largest transition probabilities at each step (zero-padded)
    instead of the characteristic function, with the same layers: the same
    number of features as RoleWalk (3 x 20 + 9)."""
    from sklearn.preprocessing import normalize
    n = len(G)
    A = nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr", dtype=np.float64)
    walks = [A, A.T.tocsr()] if G.is_directed() else [A]
    feats = []
    for M in walks:
        H = normalize(sp.identity(n) + M, norm="l1").tocsr()
        T = H
        for w in range(walk_len):
            if w:
                T = (T @ H).tocsr()
                T.data[T.data < tol] = 0
                T.eliminate_zeros()
            top = np.zeros((n, k))
            for i in range(n):
                row = T.data[T.indptr[i]:T.indptr[i + 1]]
                if len(row) > k:
                    row = np.partition(row, -k)[-k:]
                row = np.sort(row)[::-1]
                top[i, :len(row)] = row
            feats.append(top)
    return np.hstack(feats + [_walk_layers(G, walk_len)])


def local_stats(G: nx.Graph) -> np.ndarray:
    """Compact local statistics, with fixed meaning across graphs: degree,
    moments of neighbour degrees, triangles, egonet boundary, clustering and
    return probabilities of the lazy walk at steps 2 and 3 (for each
    direction if directed)."""
    from sklearn.preprocessing import normalize
    n = len(G)
    A = nx.to_scipy_sparse_array(G, nodelist=list(G), format="csr", dtype=np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    feats = []
    for M in ([A, A.T.tocsr()] if G.is_directed() else [A]):
        deg = np.asarray(M.sum(axis=1)).ravel()
        logdeg = np.log1p(deg)
        nbr = M @ logdeg / np.maximum(deg, 1)
        nbr2 = M @ logdeg ** 2 / np.maximum(deg, 1)
        nbr_max = np.zeros(n)
        for i in range(n):
            js = M.indices[M.indptr[i]:M.indptr[i + 1]]
            if len(js):
                nbr_max[i] = logdeg[js].max()
        H = normalize(sp.identity(n) + M, norm="l1").tocsr()
        H2 = (H @ H).tocsr()
        ret2 = np.asarray(H.multiply(H.T).sum(axis=1)).ravel()
        ret3 = np.asarray(H2.multiply(H.T).sum(axis=1)).ravel()
        feats += [logdeg, nbr, np.sqrt(np.maximum(nbr2 - nbr ** 2, 0)), nbr_max, ret2, ret3]
    U, base = _base_features(nx.Graph(G.to_undirected()) if G.is_directed() else G)
    within, total, ratio, _, deg, clustering = base.T
    triangles = within - deg
    boundary = total - deg - 2 * within + deg  # degrees in the egonet minus inner edges twice
    feats += [np.log1p(triangles), np.log1p(np.maximum(boundary, 0)), clustering, ratio]
    return np.column_stack(feats)


class RefexSchema:
    """ReFeX whose recursive feature selection, scaling and quantile bins
    are fitted on one graph and frozen, so that another graph gets the same
    features with the same meaning (needed to transfer a model)."""

    def __init__(self, iterations: int = 3, pruning: float = 0.5, bins: int = 4):
        self.iterations, self.pruning, self.bins = iterations, pruning, bins

    def _rounds(self, G, keeps=None):
        A, features = _base_features(G)
        deg = np.asarray(A.sum(axis=1)).ravel()
        rounds, chosen = [features], []
        for r in range(self.iterations):
            old = rounds[-1]
            total = A @ old
            mean = total / np.maximum(deg, 1)[:, None]
            new = np.column_stack([c for pair in zip(total.T, mean.T) for c in pair])
            if keeps is None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.corrcoef(old.T, new.T)[:old.shape[1], old.shape[1]:]
                keep = ~(np.abs(np.nan_to_num(corr)) > self.pruning).any(axis=0)
                if not keep.any():
                    break
                chosen.append(keep)
            else:
                if r >= len(keeps):
                    break
                keep = keeps[r]
            rounds.append(new[:, keep])
        return np.hstack(rounds), chosen

    def fit(self, G):
        X, self.keeps_ = self._rounds(G)
        self.lo_, self.hi_ = X.min(), X.max()
        X = (X - self.lo_) / (self.hi_ - self.lo_)
        self.edges_ = [np.unique(np.quantile(c, np.linspace(0, 1, self.bins + 1)[1:-1]))
                       for c in X.T]
        return self

    def transform(self, G):
        X, _ = self._rounds(G, self.keeps_)
        X = (X - self.lo_) / (self.hi_ - self.lo_)
        out = []
        for c, e in zip(X.T, self.edges_):
            b = np.digitize(c, e)
            out.append(np.eye(len(e) + 1)[b])
        return np.hstack(out).astype(np.float32)
