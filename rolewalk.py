#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.sparse import identity
from numba import njit, prange


@njit(fastmath=True)
def compute_directed_embedding(n, dim, indptr, indptr_T, data, data_T, t_min=1, t_max=100):
    X = np.zeros((n, dim * 4), dtype=np.float32)
    timesteps = np.linspace(t_min, t_max, dim)
    for i in prange(n):
        vec = []
        # normalized adjacency matrix
        a, b = indptr[i:i+2]
        row = data[a:b]
        # transposed normalized adjacency matrix
        a, b = indptr_T[i:i+2]
        row_T = data_T[a:b]
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
def compute_undirected_embedding(n, dim, indptr, data, t_min=1, t_max=100):
    X = np.zeros((n, dim * 2), dtype=np.float32)
    timesteps = np.linspace(t_min, t_max, dim)
    for i in prange(n):
        vec = []
        a, b = indptr[i:i+2]
        row = data[a:b]
        for t in timesteps:
            phi = np.mean(np.exp(1j * row * t))
            vec.append(phi.real)
            vec.append(phi.imag)
        X[i] = vec
    return X


def rolewalk(G, walk_len=4, dim=50, t_min=1, t_max=100):
    n = len(G.nodes)
    A = nx.adjacency_matrix(G)
    if nx.is_directed(G):
        H = normalize(identity(n) + A, norm="l1")**walk_len
        H_T = normalize(identity(n) + A.T, norm="l1")**walk_len
        return compute_directed_embedding(
            n, dim, H.indptr, H_T.indptr, H.data, H_T.data,
            t_min=t_min, t_max=t_max)
    else:
        H = normalize(identity(n) + A, norm="l1")**walk_len
        return compute_undirected_embedding(
            n, dim, H.indptr, H.data,
            t_min=t_min, t_max=t_max)
