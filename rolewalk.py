#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from scipy.sparse import identity
from sklearn.preprocessing import normalize


def compute_iter(X, T_indptr, T_data, theta, n, w, dim):
    for i in range(n):
        a, b = T_indptr[i:i+2]
        probabilities = np.expand_dims(T_data[a:b], -1)
        phi = np.mean(np.exp(1j * probabilities * theta), axis=0)
        X[i, w*dim:(w+1)*dim] = np.concatenate([phi.real, phi.imag])


def compute_embedding(X, H, n, theta, walk_len=3, offset=0):
    dim = 2 * theta.shape[1]
    T = H.copy()
    for w in range(walk_len):
        T @= H
        compute_iter(X, T.indptr, T.data, theta, n, w + offset*walk_len, dim)


class RoleWalk:
    def __init__(
        self, walk_len=3, n_samples=10, bounds=(1e-3, 100),
        embedding_dim=2, random_state=0
    ):
        self.walk_len = walk_len
        self.n_samples = n_samples
        self.bounds = bounds
        self.embedding_dim = embedding_dim

        # timesteps
        theta = np.linspace(bounds[0], bounds[1], n_samples)
        theta = theta[None, :].astype(np.float32)
        self.theta = theta

        self.random_state = random_state

    def fit_transform(self, G):
        n = len(G.nodes)
        A = nx.to_scipy_sparse_matrix(G)
        # extract raw embedding from sampling the characteristic function
        if nx.is_directed(G):
            dim = 4 * self.n_samples * self.walk_len
            X = np.zeros((n, dim), dtype=np.float32)
            H = normalize(identity(n) + A, norm="l1")
            H_T = normalize(identity(n) + A.T, norm="l1")
            del A
            compute_embedding(X, H, n, self.theta, self.walk_len, offset=0)
            compute_embedding(X, H_T, n, self.theta, self.walk_len, offset=1)
        else:
            dim = 2 * self.n_samples * self.walk_len
            X = np.zeros((n, dim), dtype=np.float32)
            H = normalize(A, norm="l1")
            compute_embedding(X, H, n, self.theta, self.walk_len)

        # random projection
        if self.embedding_dim is not None:
            r = np.random.RandomState(self.random_state)
            U = r.random(
                size=(X.shape[1], self.embedding_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(U)
            X = X @ Q
        return X

    def fit_predict(
        self, G,
        min_n_roles=2,
        max_n_roles=10,
        method="kmeans",
        metric="silhouette"
    ):
        if not isinstance(G, np.ndarray):
            X = self.fit_transform(G)
        else:
            X = G

        if metric == "silhouette":
            from sklearn.metrics import silhouette_score as get_score
        elif metric == "calinski_harabasz":
            from sklearn.metrics import calinski_harabasz_score as get_score

        if method == "kmeans":
            from sklearn.cluster import KMeans as Clusterer
            X += np.random.normal(size=X.shape, scale=1e-5)  # avoid duplicates
        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering as Clusterer
        elif method == "spectral":
            from sklearn.cluster import SpectralClustering as Clusterer

        best_score = -float("inf")
        for i in range(min_n_roles, max_n_roles+1):
            labels = Clusterer(n_clusters=i).fit_predict(X)
            score = get_score(X, labels)
            if score > best_score:
                best_score = score
                best_y = labels[:]
        return best_y
