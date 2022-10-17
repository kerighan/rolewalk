#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.sparse import identity


def fast_compute(X, T_indptr, T_data, theta, n, w, dim):
    for i in range(n):
        a, b = T_indptr[i:i+2]
        delta = b - a
        probabilities = np.expand_dims(T_data[a:b], -1)
        phi = np.mean(np.exp(1j * probabilities * theta), axis=0)
        X[i, w*dim:(w+1)*dim] = np.concatenate([phi.real, phi.imag])


def compute_embedding(X, H, n, theta, walk_len=3, offset=0):
    dim = 2 * theta.shape[1]
    T = H.copy()
    for w in range(walk_len):
        T @= H
        fast_compute(X, T.indptr, T.data, theta, n, w + offset, dim)


def rolewalk2(
    G,
    walk_len=3,
    n_samples=10,
    bounds=(1e-3, 100),
    embedding_dim=32,
    random_state=0
):
    A = nx.adjacency_matrix(G)
    n = len(G.nodes)
    theta = np.linspace(bounds[0], bounds[1], n_samples)[None, :].astype(np.float32)

    # extract raw embedding from sampling the characteristic function
    if nx.is_directed(G):
        X = np.zeros((n, 4 * n_samples * walk_len), dtype=np.float32)
        H = normalize(identity(n) + A, norm="l1")
        H_T = normalize(identity(n) + A.T, norm="l1")
        compute_embedding(X, H, n, theta, walk_len, offset=0)
        compute_embedding(X, H_T, n, theta, walk_len, offset=walk_len)
    else:
        X = np.zeros((n, 2 * n_samples * walk_len), dtype=np.float32)
        H = normalize(A, norm="l1")
        compute_embedding(X, H, n, theta, walk_len)

    # random projection
    if embedding_dim is not None:
        r = np.random.RandomState(random_state)
        U = r.random(size=(X.shape[1], embedding_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(U)
        X = X @ Q
    return X
