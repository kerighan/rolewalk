#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.sparse import identity
from numba import njit, prange


@njit(fastmath=True)
def _compute_entropy(row, delta):
    entropy = -(row * np.log(row)).sum()
    if delta > 1:
        return entropy / np.log(delta)
    return entropy


@njit(fastmath=True)
def compute_directed_embedding(
    n, n_samples, indptr, indptr_T, data, data_T,
    t_min=1, t_max=100, use_entropy=True,
):
    X = np.zeros((n, n_samples * 4 + 2 * use_entropy), dtype=np.float32)
    timesteps = np.linspace(t_min, t_max, n_samples)
    for i in prange(n):
        vec = []
        # normalized adjacency matrix
        a, b = indptr[i:i+2]
        delta = b - a
        row = data[a:b]

        # transposed normalized adjacency matrix
        a, b = indptr_T[i:i+2]
        delta_T = b - a
        row_T = data_T[a:b]

        if use_entropy:
            vec.append(_compute_entropy(row, delta))
            vec.append(_compute_entropy(row_T, delta_T))

        for t in timesteps:
            phi = np.mean(np.exp(1j * row * t))
            vec.append(phi.real)
            vec.append(phi.imag)
            # add transposed components
            phi = np.mean(np.exp(1j * row_T * t))
            vec.append(phi.real)
            vec.append(phi.imag)
        X[i] = vec
    return X


@njit(fastmath=True)
def compute_undirected_embedding(n, n_samples, indptr, data, t_min=1, t_max=100, use_entropy=True):
    X = np.zeros((n, n_samples * 2 + use_entropy), dtype=np.float32)
    timesteps = np.linspace(t_min, t_max, n_samples)
    for i in prange(n):
        vec = []
        a, b = indptr[i:i+2]
        row = data[a:b]

        if use_entropy:
            vec.append(_compute_entropy(row, b - a))
        for t in timesteps:
            phi = np.mean(np.exp(1j * row * t))
            vec.append(phi.real)
            vec.append(phi.imag)
        X[i] = vec
    return X


@njit(fastmath=True)
def _fill_in_entropy(X, i, j, row, n):
    entropy = -(row * np.log(row)).sum()
    if n > 1:
        entropy /= np.log(n)
    X[i, j] = entropy


def entropy_embedding(A, n, walk_len):
    H = normalize(.1*identity(n) + A, norm="l1")
    H_T = normalize(.1*identity(n) + A.T, norm="l1")

    H_j = H
    H_T_j = H_T

    X = np.zeros((n, 2 * walk_len), dtype=np.float32)
    for j in range(walk_len - 1):
        H_j *= H
        H_T_j *= H_T
        indptr, data = H_j.indptr, H_j.data
        indptr_T, data_T = H_T_j.indptr, H_T_j.data
        for i in prange(n):
            # out edges entropy
            a, b = indptr[i:i+2]
            row = data[a:b]
            _fill_in_entropy(X, i, 2*j, row, b-a)

            # in edges entropy
            a, b = indptr_T[i:i+2]
            row = data_T[a:b]
            _fill_in_entropy(X, i, 2*j+1, row, b-a)
    return X


def rolewalk(
    G,
    walk_len=4,
    n_samples=50,
    t_min=1,
    t_max=100,
    method="characteristic",
    use_entropy=True
):
    if method == "characteristic":
        n = len(G.nodes)
        A = nx.adjacency_matrix(G)
        if nx.is_directed(G):
            H = normalize(identity(n) + A, norm="l1")**walk_len
            H_T = normalize(identity(n) + A.T, norm="l1")**walk_len
            return compute_directed_embedding(
                n, n_samples, H.indptr, H_T.indptr, H.data, H_T.data,
                t_min=t_min, t_max=t_max, use_entropy=use_entropy)
        else:
            H = normalize(identity(n) + A, norm="l1")**walk_len
            return compute_undirected_embedding(
                n, n_samples, H.indptr, H.data,
                t_min=t_min, t_max=t_max, use_entropy=use_entropy)
    elif method == "entropy":
        n = len(G.nodes)
        A = nx.adjacency_matrix(G)
        return entropy_embedding(A, n, walk_len)

