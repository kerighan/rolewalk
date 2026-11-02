#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def compute_iter(T, theta, aggregate="sum", chunk_size=2**20):
    """Characteristic function of each row of the transition matrix ``T``.

    Returns an array of shape (n, 2 * n_theta): real then imaginary parts.
    With ``aggregate="sum"``, row i gives ``sum_j (exp(i theta p_ij) - 1)``
    over the row's nonzeros, i.e. the characteristic function over all n
    nodes (zeros included, as in GraphWave) rescaled by n. With "mean", it
    is the average of ``exp(i theta p_ij)`` over the nonzeros only, which
    loses how many nodes a walk reaches.

    Rows are processed in blocks of about ``chunk_size`` nonzeros so that
    memory stays bounded on large graphs.
    """
    n = T.shape[0]
    indptr = T.indptr
    out = np.zeros((n, 2 * theta.shape[1]), dtype=np.float32)
    start = 0
    while start < n:
        stop = np.searchsorted(indptr, indptr[start] + chunk_size, "right") - 1
        stop = min(max(stop, start + 1), n)
        a, b = indptr[start], indptr[stop]
        angles = T.data[a:b, None].astype(np.float32) * theta
        if aggregate == "sum":
            # cos x - 1, without cancellation for small angles
            cos = -2 * np.sin(angles / 2) ** 2
        else:
            cos = np.cos(angles)
        # sums each row's block of nonzeros
        rows = csr_matrix(
            (np.ones(b - a, dtype=np.float32), np.arange(b - a),
             indptr[start:stop + 1] - a),
            shape=(stop - start, b - a))
        phi = rows @ np.hstack([cos, np.sin(angles)])
        if aggregate == "mean":
            phi /= np.maximum(np.diff(indptr[start:stop + 1]), 1)[:, None]
        out[start:stop] = phi
        start = stop
    return out


def _prune(T, tol, max_missing=None):
    """Drop entries below ``tol``; with ``max_missing``, only in rows whose
    missing mass (from earlier pruning, plus these entries) stays below it."""
    small = T.data < tol
    if max_missing is not None:
        n_rows = T.shape[0]
        row = np.repeat(np.arange(n_rows), np.diff(T.indptr))
        dropped = np.bincount(row, T.data * small, minlength=n_rows)
        missing = 1 - np.bincount(row, T.data, minlength=n_rows) + dropped
        small &= (missing <= max_missing)[row]
    T.data[small] = 0
    T.eliminate_zeros()
    return T


def compute_embedding(H, theta, walk_len=3, aggregate="sum", tol=0.0,
                      rows=None, shells=False, curvature=None,
                      max_missing=None, return_missing=False):
    """Stack ``compute_iter`` over the first ``walk_len`` powers of ``H``.

    With ``rows``, only those rows of the powers are computed, which only
    explores the walks' neighbourhoods.

    Probabilities below ``tol`` are dropped after each power, which keeps
    the powers sparse around hubs. Their mass is exactly what a row is
    missing from 1, and since ``exp(i theta p) - 1 ~ i theta p`` for small
    p, the "sum" features add it back to first order. Dropped mass can
    gather again at the next step (around a hub, many small probabilities
    lead back to it): with ``max_missing``, a row only drops entries while
    its missing mass stays below that budget, and keeps them otherwise.

    With ``shells``, each step also gives the probability mass of every
    "onion layer": nodes at distance 0, 1, ..., w of the start after w steps.
    Mass that stays in inner layers reveals walks that loop back, which the
    characteristic function alone cannot place. A node lies within k steps
    when it is in the support of the k-th power, so the layers come from the
    powers already computed: a node's layer is the first step at which it
    appears, read before pruning. Without pruning this is its distance.
    With pruning, every retained entry still gets its exact distance up to
    three steps (the layer masses lose what pruning drops); beyond, a node
    first reached through a pruned entry is labelled later than its
    distance. Layer masses are multiplied by ``||theta||``:
    for a small probability, the characteristic function is about
    ``i theta p``, so this puts a unit of mass on the same scale as in the
    characteristic function. Unscaled, they weigh ~0.02% of the variance and
    leave distances (hence k-means roles) unchanged
    (see benchmarks/kernel_grid.py).

    With ``curvature``, an array of node curvatures (see
    :func:`node_curvature`), each step also gives the mean and standard
    deviation of the curvature where the walk lands, on the same scale.

    With ``return_missing``, also return the mass each row misses at each
    step (lost to pruning).
    """
    scale = np.float32(np.linalg.norm(theta))
    features, missing = [], []
    T = H if rows is None else H[rows]
    if shells:
        supports = [identity(H.shape[0], format="csr")
                    if rows is None else identity(H.shape[0], format="csr")[rows]]
    for w in range(walk_len):
        if w > 0:
            # the densest power, so never compute one past the last step
            T = T @ H
        if shells and w < walk_len - 1:
            # support before pruning: nodes within w + 1 steps
            S = T.copy()
            S.data[:] = 1
            if shells and len(supports) > 1:
                # nodes reached by any earlier step too: the supports stay
                # nested once pruning drops entries
                S = S + supports[-1]
                S.data[:] = 1
        if w > 0 and tol:
            T = _prune(T, tol, max_missing)
        phi = compute_iter(T, theta, aggregate)
        pruned = 1 - np.asarray(T.sum(axis=1)).ravel()
        missing.append(np.maximum(pruned, 0))
        if tol and aggregate == "sum":
            phi[:, theta.shape[1]:] += pruned[:, None] * theta
        features.append(phi)
        if shells:
            within = [np.asarray(T.multiply(S).sum(axis=1)).ravel()
                      for S in supports]
            within.append(np.asarray(T.sum(axis=1)).ravel())
            layers = np.diff(np.column_stack(within), prepend=0, axis=1)
            features.append(scale * layers.astype(np.float32))
            if w < walk_len - 1:
                supports.append(S)
        if curvature is not None:
            mean = T @ curvature
            std = np.sqrt(np.maximum(T @ curvature ** 2 - mean ** 2, 0))
            features.append(scale * np.column_stack([mean, std]).astype(np.float32))
    X = np.hstack(features)
    return (X, np.column_stack(missing)) if return_missing else X


try:
    import numba
except ImportError:  # pragma: no cover
    numba = None


def _walk_rows_kernel(indptr, indices, data, rows, walk_len, theta, tol,
                      max_missing, shells, kappa, n_threads):
    """Fused walk: for each root, propagate a sparse row step by step and
    summarise it on the fly (characteristic function, layers, curvature),
    without materialising the powers. Same features as compute_embedding
    with aggregate="sum" (layers: step at which each node is first reached,
    before pruning); each thread holds O(n) workspace."""
    n = len(indptr) - 1
    m = len(theta)
    use_kappa = len(kappa) > 0
    per_step = 2 * m + 2 * use_kappa
    width = walk_len * per_step + (walk_len * (walk_len + 3) // 2 if shells else 0)
    out = np.zeros((len(rows), width), dtype=np.float32)
    missing = np.zeros((len(rows), walk_len))
    scale = np.sqrt(np.sum(theta ** 2))
    # below this probability every angle is under 0.1: Taylor series from
    # power sums instead of 2 m sines (truncation below 3e-12 of the term;
    # float32 rounding dominates)
    small_p = 0.1 / np.max(theta)
    chunks = np.linspace(0, len(rows), n_threads + 1).astype(np.int64)
    for c in numba.prange(n_threads):
        acc = np.zeros(n)
        dist = np.full(n, -1, dtype=np.int64)
        touched = np.empty(n, dtype=np.int64)
        seen = np.empty(n, dtype=np.int64)
        cur_idx = np.empty(n, dtype=np.int64)
        cur_val = np.empty(n)
        layers = np.zeros(walk_len + 1)
        M = np.zeros(9)
        for r in range(chunks[c], chunks[c + 1]):
            u = rows[r]
            n_seen = 1
            seen[0] = u
            dist[u] = 0
            n_cur = 1
            cur_idx[0] = u
            cur_val[0] = 1.0
            col = 0
            for w in range(walk_len):
                # next row = current row @ H, in a dense accumulator
                n_touch = 0
                for a in range(n_cur):
                    j = cur_idx[a]
                    p = cur_val[a]
                    for e in range(indptr[j], indptr[j + 1]):
                        k = indices[e]
                        if acc[k] == 0.0:
                            touched[n_touch] = k
                            n_touch += 1
                        acc[k] += p * data[e]
                total = 0.0
                small = 0.0
                for t in range(n_touch):
                    k = touched[t]
                    if dist[k] < 0:  # first reached now, before pruning
                        dist[k] = w + 1
                        seen[n_seen] = k
                        n_seen += 1
                    total += acc[k]
                    if acc[k] < tol:
                        small += acc[k]
                prune = w > 0 and tol > 0.0
                if prune and max_missing >= 0.0:
                    prune = (1.0 - total) + small <= max_missing
                # keep, summarise, and reset the accumulator
                n_cur = 0
                kept = 0.0
                mean = 0.0
                second = 0.0
                for d in range(walk_len + 1):
                    layers[d] = 0.0
                for d in range(9):
                    M[d] = 0.0
                for t in range(n_touch):
                    k = touched[t]
                    p = acc[k]
                    acc[k] = 0.0
                    if prune and p < tol:
                        continue
                    cur_idx[n_cur] = k
                    cur_val[n_cur] = p
                    n_cur += 1
                    kept += p
                    if p < small_p:
                        q = p
                        for d in range(1, 9):
                            M[d] += q
                            q *= p
                    else:
                        for i in range(m):
                            x = np.float32(theta[i] * p)
                            h = np.sin(np.float32(0.5) * x)
                            out[r, col + i] += np.float32(-2.0) * h * h
                            out[r, col + m + i] += np.sin(x)
                    if shells:
                        layers[dist[k]] += p
                    if use_kappa:
                        mean += p * kappa[k]
                        second += p * kappa[k] * kappa[k]
                for i in range(m):
                    t2 = theta[i] * theta[i]
                    out[r, col + i] += t2 * (-M[2] / 2 + t2 * (M[4] / 24 + t2 * (
                        -M[6] / 720 + t2 * M[8] / 40320)))
                    out[r, col + m + i] += theta[i] * (M[1] + t2 * (-M[3] / 6 + t2 * (
                        M[5] / 120 - t2 * M[7] / 5040)))
                missing[r, w] = max(1.0 - kept, 0.0)
                if tol > 0.0:  # missing mass, to first order
                    for i in range(m):
                        out[r, col + m + i] += (1.0 - kept) * theta[i]
                col += 2 * m
                if shells:
                    for d in range(w + 2):
                        out[r, col + d] = scale * layers[d]
                    col += w + 2
                if use_kappa:
                    out[r, col] = scale * mean
                    out[r, col + 1] = scale * np.sqrt(max(second - mean * mean, 0.0))
                    col += 2
            for t in range(n_seen):
                dist[seen[t]] = -1
    return out, missing


if numba is not None:
    _walk_rows_kernel = numba.njit(parallel=True, cache=True, fastmath=True)(_walk_rows_kernel)


def compute_embedding_fused(H, theta, walk_len=3, tol=0.0, rows=None,
                            shells=False, curvature=None, max_missing=None,
                            n_jobs=1, return_missing=False):
    """compute_embedding (aggregate="sum") with the fused Numba kernel:
    a few times faster and with O(n) memory per thread. Same features up
    to float32 rounding and the Taylor truncation (relative error below
    3e-12 for the default theta). With ``return_missing``, also return the
    mass each row misses at each step."""
    if numba is None:
        raise ImportError("compute_embedding_fused needs numba")
    H = H.tocsr()
    rows = np.arange(H.shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
    n_threads = max(1, min(n_jobs, len(rows)))
    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
    kappa = np.zeros(0) if curvature is None else np.asarray(curvature, dtype=np.float64)
    X, missing = _walk_rows_kernel(
        H.indptr.astype(np.int64), H.indices.astype(np.int64),
        H.data.astype(np.float64), rows.astype(np.int64), walk_len,
        np.asarray(theta, dtype=np.float64).ravel(), float(tol),
        -1.0 if max_missing is None else float(max_missing), bool(shells),
        kappa, n_threads)
    return (X, missing) if return_missing else X


def _edge_triangles_kernel(indptr, indices, rank):
    """Common neighbours of every edge (CSR positions of a symmetric
    adjacency), by enumerating each triangle once from its lowest-ranked
    node along edges oriented by rank (degree, then index). Only forward
    neighbours are scanned, at most sqrt(2 |E|) per node: O(|E|^1.5)."""
    n = len(indptr) - 1
    # forward adjacency: neighbours of higher rank, with their CSR position
    fwd_ptr = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        k = 0
        for e in range(indptr[v], indptr[v + 1]):
            if rank[indices[e]] > rank[v]:
                k += 1
        fwd_ptr[v + 1] = fwd_ptr[v] + k
    fwd = np.empty(fwd_ptr[n], dtype=np.int64)
    fwd_pos = np.empty(fwd_ptr[n], dtype=np.int64)
    for v in range(n):
        k = fwd_ptr[v]
        for e in range(indptr[v], indptr[v + 1]):
            if rank[indices[e]] > rank[v]:
                fwd[k] = indices[e]
                fwd_pos[k] = e
                k += 1
    count = np.zeros(len(indices), dtype=np.int64)
    mark = np.full(n, -1, dtype=np.int64)  # mark[z] = position of edge v-z
    for v in range(n):
        for a in range(fwd_ptr[v], fwd_ptr[v + 1]):
            mark[fwd[a]] = fwd_pos[a]
        for a in range(fwd_ptr[v], fwd_ptr[v + 1]):
            w = fwd[a]
            for b in range(fwd_ptr[w], fwd_ptr[w + 1]):
                z = fwd[b]
                if mark[z] >= 0:
                    count[fwd_pos[a]] += 1  # v-w
                    count[fwd_pos[b]] += 1  # w-z
                    count[mark[z]] += 1  # v-z
        for a in range(fwd_ptr[v], fwd_ptr[v + 1]):
            mark[fwd[a]] = -1
    return count


def _edge_common_kernel(indptr, indices, xs):
    """Common neighbours of the edges x-y of the given nodes x (sorted
    CSR of a symmetric adjacency), in the order of their CSR positions: the
    shorter neighbour list is scanned and searched in the longer one,
    O(min(d_x, d_y) log d) per edge, O(|E|^1.5 log d) at most."""
    out_ptr = np.zeros(len(xs) + 1, dtype=np.int64)
    for i in range(len(xs)):
        out_ptr[i + 1] = out_ptr[i] + indptr[xs[i] + 1] - indptr[xs[i]]
    count = np.zeros(out_ptr[-1], dtype=np.int64)
    for i in range(len(xs)):
        x = xs[i]
        for e in range(indptr[x], indptr[x + 1]):
            y = indices[e]
            a, b = x, y
            if indptr[a + 1] - indptr[a] > indptr[b + 1] - indptr[b]:
                a, b = y, x
            lo, hi = indptr[b], indptr[b + 1]
            c = 0
            for f in range(indptr[a], indptr[a + 1]):
                pos = lo + np.searchsorted(indices[lo:hi], indices[f])
                if pos < hi and indices[pos] == indices[f]:
                    c += 1
            count[out_ptr[i] + e - indptr[x]] = c
    return count


if numba is not None:
    _edge_triangles_kernel = numba.njit(cache=True)(_edge_triangles_kernel)
    _edge_common_kernel = numba.njit(cache=True)(_edge_common_kernel)


def lazy_walk(A):
    """Transition matrix of the walk that stays put with the weight of an
    average edge of the node: ``(diag(s / d) + A)`` normalised by rows, with
    ``s`` the weighted and ``d`` the unweighted out-degree (1 without
    edges). On unweighted graphs this is ``D~^-1 (I + A)``; on weighted ones
    it stays put with probability ``1 / (d + 1)`` too, and does not depend on
    the unit of the weights (nor on rescaling the edges of a node)."""
    A = csr_matrix(A, dtype=np.float64)
    A.eliminate_zeros()
    count = np.diff(A.indptr)
    strength = np.asarray(A.sum(axis=1)).ravel()
    stay = np.where(count > 0, strength / np.maximum(count, 1), 1.0)
    return normalize(diags(stay) + A, norm="l1").tocsr()


def node_curvature(A, nodes=None):
    """Mean curvature of the edges around each node.

    The curvature of an edge x-y is the overlap ``sum_z min(m_x(z), m_y(z))``
    of the one-step lazy walk distributions from x and from y: high in
    cliques (spherical, the walks stay together), low in trees (hyperbolic,
    they split). It is a probabilistic proxy of Ollivier-Ricci curvature,
    ``1 - W1(m_x, m_y)``, which it bounds from above since every unit of
    mass that does not overlap moves at least one step. Being made of
    probabilities, it keeps embeddings comparable across graphs.

    Directions are ignored. With ``nodes`` (indices), only those nodes get a
    curvature; the others are 0.

    The two closed neighbourhoods are uniform and share the
    ``c(x, y) + 2`` nodes x, y and their common neighbours, so the overlap
    is ``(c(x, y) + 2) / (max(d_x, d_y) + 1)``. With numba, common
    neighbours come from enumerating triangles along edges oriented by
    degree, O(|E|^1.5), or for a subset of nodes from intersecting the
    neighbour lists of their edges, O(min(d_x, d_y) log d) per edge.
    Without numba they come from one sparse product, whose cost is the
    number of 2-paths from the nodes (quadratic on a star).
    """
    n = A.shape[0]
    U = ((A + A.T) > 0).astype(np.float64).tocsr()
    U.setdiag(0)
    U.eliminate_zeros()
    deg = np.asarray(U.sum(axis=1)).ravel()
    if numba is not None and nodes is None:
        U.sort_indices()
        order = np.lexsort((np.arange(n), deg))
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        count = _edge_triangles_kernel(U.indptr.astype(np.int64),
                                       U.indices.astype(np.int64), rank)
        # each triangle counted one edge direction: add the other one
        C = csr_matrix((count.astype(np.float64), U.indices, U.indptr), shape=U.shape)
        C = (C + C.T).tocsr()
        C.sort_indices()
        common = np.asarray(C.multiply(U).tocsr()[U.nonzero()]).ravel()
        E = U.tocoo()  # same order as U.nonzero()
        overlap = (common + 2) / (np.maximum(deg[E.row], deg[E.col]) + 1)
        return np.bincount(E.row, overlap, minlength=n) / np.maximum(deg, 1)
    if numba is not None:
        U.sort_indices()
        xs = np.asarray(nodes, dtype=np.int64)
        count = _edge_common_kernel(U.indptr.astype(np.int64),
                                    U.indices.astype(np.int64), xs)
        x = np.repeat(xs, deg[xs].astype(np.int64))
        y = np.concatenate([U.indices[U.indptr[v]:U.indptr[v + 1]] for v in xs]) \
            if len(xs) else np.zeros(0, dtype=np.int64)
        overlap = (count + 2) / (np.maximum(deg[x], deg[y]) + 1)
        return np.bincount(x, overlap, minlength=n) / np.maximum(deg, 1)
    rows = U if nodes is None else U[nodes]
    common = (rows @ U).multiply(rows).tocoo()  # c(x, y) on the edges x-y
    E = rows.tocoo()  # edges with c = 0 are absent from ``common``
    x = E.row if nodes is None else np.asarray(nodes)[E.row]
    overlap = 2 / (np.maximum(deg[x], deg[E.col]) + 1)
    cx = common.row if nodes is None else np.asarray(nodes)[common.row]
    extra = common.data / (np.maximum(deg[cx], deg[common.col]) + 1)
    kappa = np.bincount(x, overlap, minlength=n) + np.bincount(cx, extra, minlength=n)
    return kappa / np.maximum(deg, 1)


AGGREGATES = ("sum", "mean")

THETA_SCHEMES = {
    # log-spaced: multi-scale kernel, robust to the choice of the upper bound
    "geomspace": ((1, 100), np.geomspace),
    # uniform (RoleWalk <= 1.0 behaviour): periodic kernel, degrades under
    # edge noise when the upper bound is too large
    "linspace": ((1e-3, 100), np.linspace),
}


class RoleWalk:
    """Structural role embedding from random-walk transition probabilities.

    Each node is described, for each walk step, by the empirical
    characteristic function of its transition probabilities, sampled at
    ``n_samples`` values of theta, and by the mass it keeps at each
    distance. With aggregate="sum", without curvature, pruning or
    projection, squared distances between embeddings are the MMD between
    the nodes' walk footprints (finite measures over (block, probability,
    layer), a block being a step and direction or a mixed probe) under the
    kernel ``[same block] (sum_theta [cos theta (a - b) - cos theta a
    - cos theta b + 1] + ||theta||^2 a b [same layer])``, with no layer
    term for mixed probes (see the paper), so ``theta_scheme`` and
    ``bounds`` set the kernel's shape and bandwidth.

    Parameters
    ----------
    theta_scheme : {"geomspace", "linspace"}
        How theta is spread over ``bounds``.
    bounds : tuple of float, optional
        Smallest and largest theta. Defaults to (1, 100) for "geomspace" and
        (1e-3, 100) for "linspace".
    aggregate : {"sum", "mean"}
        How the characteristic function combines a node's transition
        probabilities. "sum" (default since 1.2.0) keeps the number of nodes
        a walk reaches and is more robust to edge noise; "mean" is the
        RoleWalk <= 1.1 behaviour.
    tol : float
        Transition probabilities below ``tol`` are dropped after each walk
        step. 1e-4 is 2-3x faster on graphs with hubs; on the role
        benchmarks it changes accuracy by at most 0.005, but hubs' features
        can move (see ``max_missing``). 0 computes the walks exactly.
    shells : bool
        Also describe, at each step, how the walk's probability mass splits
        over the "onion layers" around the node (distance 0, 1, ..., step).
        Loops and triangles keep mass in inner layers. Adds
        ``walk_len * (walk_len + 3) / 2`` features for about 15% more time
        (default since 1.3.0; models saved before load with ``False``).
    curvature : bool
        Also describe the local geometry the walk explores: mean and spread,
        at each step, of the curvature of the nodes it lands on (see
        :func:`node_curvature`), from tree-like (hyperbolic) to clique-like
        (spherical). Adds ``2 * walk_len`` features.
    mixed : bool
        Directed graphs only (default since 1.4.0): also describe the walks
        that take one step
        forward then one back (``H_out H_in``: nodes sharing targets with the
        start) and the converse (shared sources), by their characteristic
        functions. Pure out- and in-walks miss these: a source pointing to
        two sinks and a source of a directed K(2, 2) have the same out- and
        in-walks. Adds ``4 * n_samples`` features.
    max_missing : float, optional
        Budget of probability mass a row may lose to pruning. Without it,
        a hub's step-2 row can lose most of its mass, which returns to the
        hub at step 3. Keeping it costs more time around hubs.
    n_jobs : int
        Threads for the fused walk (with numba). Rows are independent, so
        time scales down almost linearly.
    block_work : float
        Rows are embedded in blocks of about this many neighbour visits for
        the first two steps, which bounds memory on large graphs. Results do
        not depend on it.
    embedding_dim : int, optional
        If set, randomly project the embedding to this many dimensions. Low
        values lose role information: keep None except for plotting.
    role_names : list of str, optional
        Names of the roles found by :meth:`fit`, indexed by label. Only
        stored, and saved along with the model.

    Attributes
    ----------
    missing_ : ndarray of shape (n_nodes, n_blocks)
        After :meth:`transform`, the probability mass each embedded node
        lost to pruning at each step of each walk (out-walk then in-walk
        steps, then the two mixed probes). 0 everywhere with ``tol=0``.
    certificate_ : ndarray of shape (n_nodes,) or None
        Bound on the distance between each embedded node's features and its
        exact features (``tol=0``), in exact arithmetic (see the paper,
        certified pruning): ``||theta|| sqrt(5 sum r^2 + 4 sum r_mixed^2)``
        (4 instead of 5 without layers). None with curvature or more than
        three steps, which it does not cover.

    Notes
    -----
    Embeddings only depend on the parameters above and on each node's
    neighbourhood, so they are comparable across graphs. A model fitted on
    one graph can therefore label nodes of another graph, or of the same
    graph later on, with :meth:`predict`; :meth:`save` and :meth:`load`
    keep it between sessions.
    """

    def __init__(
        self, walk_len=3, n_samples=10, bounds=None,
        embedding_dim=None, random_state=0, theta_scheme="geomspace",
        aggregate="sum", tol=1e-4, shells=True, curvature=False,
        role_names=None, max_missing=None, block_work=1e7, mixed=True,
        n_jobs=1
    ):
        if theta_scheme not in THETA_SCHEMES:
            raise ValueError(
                f"Unknown theta_scheme {theta_scheme!r}, expected one of "
                f"{sorted(THETA_SCHEMES)}")
        if aggregate not in AGGREGATES:
            raise ValueError(
                f"Unknown aggregate {aggregate!r}, expected one of "
                f"{list(AGGREGATES)}")
        if not tol >= 0:
            raise ValueError("tol must be non-negative")
        if max_missing is not None and not max_missing >= 0:
            raise ValueError("max_missing must be non-negative")
        default_bounds, spacing = THETA_SCHEMES[theta_scheme]
        if bounds is None:
            bounds = default_bounds

        self.walk_len = walk_len
        self.n_samples = n_samples
        self.bounds = bounds
        self.theta_scheme = theta_scheme
        self.aggregate = aggregate
        self.tol = tol
        self.shells = shells
        self.curvature = curvature
        self.max_missing = max_missing
        self.mixed = mixed
        self.n_jobs = n_jobs
        self.block_work = block_work
        self.embedding_dim = embedding_dim

        # timesteps
        theta = spacing(bounds[0], bounds[1], n_samples)
        theta = theta[None, :].astype(np.float32)
        self.theta = theta

        self.random_state = random_state
        self.role_names = role_names

    def transform(self, G, nodes=None):
        """Embed the nodes of ``G``, in ``G.nodes`` order.

        With ``nodes``, only embed those nodes, in that order. Their
        features only depend on their ``walk_len``-step neighbourhoods
        (2-step with mixed probes, ``walk_len`` + 2 with curvature), and only
        those walks are computed; the graph is still converted and the
        transition matrices built in full.
        """
        n = len(G.nodes)
        A = nx.to_scipy_sparse_array(G, format="csr")
        if A.nnz and (not np.isfinite(A.data).all() or A.data.min() < 0):
            raise ValueError("edge weights must be finite and non-negative")
        rows = None
        if nodes is not None:
            index = {v: i for i, v in enumerate(G.nodes)}
            rows = np.array([index[v] for v in nodes], dtype=np.int64)
        # extract raw embedding from sampling the characteristic function
        H = lazy_walk(A)
        kappa = None
        if self.curvature:
            near = None
            if rows is not None:
                # the walks only land within walk_len steps of the rows
                U = ((A + A.T) > 0).tocsr()
                reach = np.zeros(n, dtype=bool)
                reach[rows] = True
                for _ in range(self.walk_len):
                    reach |= np.asarray(U.T @ reach).ravel() > 0
                near = np.flatnonzero(reach)
            kappa = node_curvature(A, near)
        walks = [H]
        if nx.is_directed(G):
            walks.append(lazy_walk(A.T))
        parts = [self._embed_blocks(W, rows, kappa) for W in walks]
        mixed = []
        if self.mixed and len(walks) == 2:
            H_out, H_in = walks
            mixed = [self._mixed_blocks(H_out, H_in, rows),
                     self._mixed_blocks(H_in, H_out, rows)]
        X = np.hstack([x for x, _ in parts + mixed])
        pure = np.hstack([r for _, r in parts])
        self.missing_ = np.hstack([pure] + [r for _, r in mixed])
        self.certificate_ = None
        if not self.curvature and self.walk_len <= 3 and self.aggregate == "sum":
            # proposition "certified pruning": 2 ||theta|| r per CF block,
            # ||theta|| r per layer block
            c = 5 if self.shells else 4
            sq = c * (pure ** 2).sum(axis=1)
            if mixed:
                sq = sq + 4 * sum((r ** 2).sum(axis=1) for _, r in mixed)
            self.certificate_ = np.linalg.norm(self.theta) * np.sqrt(sq)

        # random projection
        if self.embedding_dim is not None:
            r = np.random.RandomState(self.random_state)
            U = r.random(
                size=(X.shape[1], self.embedding_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(U)
            X = X @ Q
        return X

    def _row_blocks(self, H, rows, second=None):
        """Blocks of rows with about ``block_work`` neighbour visits for the
        first two steps (``H`` then ``second``, default ``H``) each."""
        if rows is None:
            rows = np.arange(H.shape[0])
        if second is None:
            second = H
        B = H.copy()
        B.data[:] = 1
        work = (B @ np.diff(second.indptr).astype(np.float64))[rows]
        cuts = np.searchsorted(np.cumsum(work),
                               np.arange(self.block_work, work.sum(), self.block_work))
        return np.split(rows, np.unique(cuts[(cuts > 0) & (cuts < len(rows))]))

    def _mixed_blocks(self, first, second, rows):
        """Characteristic function of the rows of ``first @ second``, by
        blocks whose product has at most about ``block_work`` entries before
        pruning."""
        out, missing = [], []
        for block in self._row_blocks(first, rows, second):
            T = (first[block] @ second).tocsr()
            if self.tol:
                T = _prune(T, self.tol, self.max_missing)
            phi = compute_iter(T, self.theta, self.aggregate)
            pruned = 1 - np.asarray(T.sum(axis=1)).ravel()
            if self.tol and self.aggregate == "sum":
                phi[:, self.theta.shape[1]:] += pruned[:, None] * self.theta
            out.append(phi)
            missing.append(np.maximum(pruned, 0))
        return np.vstack(out), np.concatenate(missing)[:, None]

    def _embed_blocks(self, H, rows, kappa):
        """compute_embedding over blocks of rows of similar expansion work,
        or the fused kernel when numba is available."""
        if numba is not None and self.aggregate == "sum":
            return compute_embedding_fused(
                H, self.theta, self.walk_len, self.tol, rows, self.shells,
                kappa, self.max_missing, self.n_jobs, return_missing=True)
        blocks = [compute_embedding(H, self.theta, self.walk_len, self.aggregate,
                                    self.tol, block, self.shells, kappa,
                                    self.max_missing, return_missing=True)
                  for block in self._row_blocks(H, rows)]
        return np.vstack([x for x, _ in blocks]), np.vstack([r for _, r in blocks])

    def fit(
        self, G, n_roles=None, max_n_roles=40, tolerance=0.05,
        sample_size=1000, n_init=10
    ):
        """Find roles; see :meth:`fit_predict`. Returns the model."""
        self.fit_predict(
            G, n_roles, max_n_roles, tolerance, sample_size, n_init)
        return self

    def fit_predict(
        self, G, n_roles=None, max_n_roles=40, tolerance=0.05,
        sample_size=1000, n_init=10
    ):
        """Cluster nodes into roles.

        Parameters
        ----------
        G : networkx graph or ndarray
            A graph, or embeddings already computed by :meth:`transform`.
        n_roles : int, optional
            Number of roles. If None, it is chosen from the data, between 1
            and ``max_n_roles``.
        max_n_roles : int
            Largest number of roles considered when ``n_roles`` is None.
        tolerance : float
            When choosing the number of roles, the smallest one whose
            silhouette is within ``tolerance`` of the best is kept. The best
            silhouette alone tends to split roles that edge noise blurred.
        sample_size : int
            Number of nodes on which the number of roles is chosen.
        n_init : int
            Number of k-means restarts when ``n_roles`` is given. When roles
            blend into each other, k-means has several near-equivalent
            solutions and which one comes out depends on node order; more
            restarts (e.g. 200 on graphs of a few thousand nodes) make the
            result stable.

        Returns
        -------
        labels : ndarray of shape (n_nodes,)

        Notes
        -----
        With ``n_roles`` given, this is k-means. Otherwise a Ward hierarchy
        is built on a sample of nodes, every cut is scored by silhouette, and
        the chosen cut's centroids seed a k-means on all nodes. Sets
        ``cluster_centers_``, ``n_roles_`` and, in the latter case,
        ``silhouette_scores_`` (a dict from number of roles to silhouette). On the synthetic benchmarks,
        this beat GMM (BIC), HDBSCAN, spectral clustering (eigengap) and
        k-means + silhouette, and even k-means told the true number of roles.
        """
        from sklearn.cluster import KMeans

        X = G if isinstance(G, np.ndarray) else self.transform(G)
        if n_roles is None:
            centers = self._choose_roles(
                X, max_n_roles, tolerance, sample_size)
            km = KMeans(len(centers), init=centers, n_init=1)
        else:
            km = KMeans(
                n_roles, n_init=n_init, random_state=self.random_state)
        labels = km.fit_predict(X)
        self.cluster_centers_ = km.cluster_centers_
        self.n_roles_ = km.n_clusters
        return labels

    def predict(self, G, nodes=None):
        """Label nodes with the closest role found by :meth:`fit`.

        Parameters
        ----------
        G : networkx graph or ndarray
            A graph, possibly not the one the model was fitted on, or
            embeddings computed by :meth:`transform`.
        nodes : iterable, optional
            Only label these nodes of ``G``, in that order.
        """
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("RoleWalk has no roles yet: call fit first")
        if isinstance(G, np.ndarray):
            X = G
        else:
            X = self.transform(G, nodes)
        return pairwise_distances(X, self.cluster_centers_).argmin(axis=1)

    def save(self, path):
        """Save parameters and roles to ``path`` (a .npz file)."""
        params = {
            "walk_len": self.walk_len, "n_samples": self.n_samples,
            "bounds": [float(b) for b in self.bounds],
            "embedding_dim": self.embedding_dim,
            "random_state": self.random_state,
            "theta_scheme": self.theta_scheme, "aggregate": self.aggregate,
            "tol": self.tol, "shells": self.shells,
            "curvature": self.curvature, "max_missing": self.max_missing,
            "mixed": self.mixed,
            "role_names": self.role_names,
        }
        arrays = {"theta": self.theta}
        if hasattr(self, "cluster_centers_"):
            arrays["cluster_centers"] = self.cluster_centers_
        np.savez(path, params=json.dumps(params), **arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by :meth:`save`."""
        # no pickle: loading a file cannot run code
        with np.load(path, allow_pickle=False) as data:
            params = json.loads(str(data["params"]))
            params.setdefault("shells", False)  # saved before 1.3.0
            params.setdefault("curvature", False)
            params.setdefault("mixed", False)  # saved before 1.4.0
            model = cls(**params)
            model.theta = data["theta"]
            if "cluster_centers" in data:
                model.cluster_centers_ = data["cluster_centers"]
                model.n_roles_ = len(model.cluster_centers_)
        return model

    def _choose_roles(self, X, max_n_roles, tolerance, sample_size):
        """Return the centroids of the chosen Ward cut of a node sample."""
        from scipy.cluster.hierarchy import fcluster, linkage
        from sklearn.metrics import silhouette_score

        r = np.random.RandomState(self.random_state)
        if len(X) > sample_size:
            X = X[r.choice(len(X), sample_size, replace=False)]
        X = X.astype(np.float64)
        self.silhouette_scores_ = {}
        if len(X) < 3:
            return X.mean(axis=0, keepdims=True)

        tree = linkage(X, "ward")
        # merges of points equal up to float32 rounding do not split roles
        n_splits = np.sum(tree[:, 2] > 1e-5 * np.abs(X).max())
        dists = pairwise_distances(X)
        cuts = {}
        for k in range(2, min(max_n_roles, len(X) - 1, n_splits + 1) + 1):
            labels = fcluster(tree, k, "maxclust") - 1
            # identical points cannot be split, so a cut may have fewer
            # clusters than asked for
            k = int(labels.max()) + 1
            if k < 2 or k in cuts:
                continue
            cuts[k] = labels
            self.silhouette_scores_[k] = silhouette_score(
                dists, labels, metric="precomputed")
        if not cuts:  # all nodes look alike
            return X.mean(axis=0, keepdims=True)

        best = max(self.silhouette_scores_.values())
        k = min(k for k, score in self.silhouette_scores_.items()
                if score >= best - tolerance)
        return np.stack([X[cuts[k] == c].mean(axis=0) for c in range(k)])


def pairwise_role_distances(X, metric="euclidean"):
    """Compute pairwise distances between node role embeddings.

    Parameters
    ----------
    X : ndarray of shape (n_nodes, n_features)
        Embeddings of graph nodes, typically produced by :meth:`RoleWalk.transform`.
    metric : str, optional
        Distance metric passed to :func:`sklearn.metrics.pairwise_distances`.

    Returns
    -------
    distances : ndarray of shape (n_nodes, n_nodes)
        Pairwise distance matrix.
    ranking : ndarray of shape (n_nodes, n_nodes - 1)
        For each node, indices of other nodes sorted by increasing distance.
    """

    dists = pairwise_distances(X, metric=metric)
    # ignore self-distances when ranking
    np.fill_diagonal(dists, np.inf)
    # self sorts last (infinite distance), drop it from the ranking
    order = np.argsort(dists, axis=1)[:, :-1]
    return dists, order


def mean_average_precision(X, labels, metric="euclidean"):
    """Return mean average precision for retrieving matching roles.

    Parameters
    ----------
    X : ndarray of shape (n_nodes, n_features)
        Node embeddings.
    labels : ndarray of shape (n_nodes,)
        Ground truth role labels for each node.
    metric : str, optional
        Distance metric used to rank nodes by structural similarity.

    Notes
    -----
    Nodes with no other instances of their label contribute zero to the mean.

    """

    _, ranking = pairwise_role_distances(X, metric=metric)
    aps = []
    for i, order in enumerate(ranking):
        rel = labels[order] == labels[i]
        n_rel = rel.sum()
        if n_rel == 0:
            aps.append(0.0)
            continue
        precision = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((precision[rel]).mean()))
    return float(np.mean(aps))


def _node_stats(G):
    """Per-node structural statistics, computed with sparse algebra."""
    A = nx.to_scipy_sparse_array(G, format="csr", weight=None)
    A.setdiag(0)
    A.eliminate_zeros()
    stats = {}
    if nx.is_directed(G):
        stats["in_degree"] = np.asarray(A.sum(axis=0)).ravel()
        stats["out_degree"] = np.asarray(A.sum(axis=1)).ravel()
        A = ((A + A.T) > 0).astype(np.float64)
    else:
        A = (A > 0).astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    stats["degree"] = deg
    with np.errstate(divide="ignore", invalid="ignore"):
        stats["neighbor_degree"] = np.nan_to_num((A @ deg) / deg)
        triangles = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2
        stats["clustering"] = np.nan_to_num(2 * triangles / (deg * (deg - 1)))
    return stats, A


def role_summary(G, labels, X=None, n_examples=3, n_links=3):
    """Describe each role with structural statistics.

    Parameters
    ----------
    G : networkx graph
        The graph whose nodes were labelled, in ``G.nodes`` order.
    labels : array-like of shape (n_nodes,)
        Role of each node, e.g. from :meth:`RoleWalk.fit_predict`.
    X : ndarray of shape (n_nodes, n_features), optional
        Node embeddings. If given, examples are the nodes closest to their
        role's centroid; otherwise the highest-degree nodes.
    n_examples : int
        Number of example nodes per role.
    n_links : int
        Number of most-linked roles listed per role.

    Returns
    -------
    summary : pandas.DataFrame
        One row per role, largest first:

        - ``n_nodes``, ``share``: size of the role;
        - ``degree`` (``in_degree``, ``out_degree`` if directed),
          ``neighbor_degree`` and ``clustering``: medians over the role's
          nodes. Degree and clustering ignore direction and weights;
        - ``profile``: the statistics that set the role apart, i.e. whose
          median falls in the top or bottom quarter of all nodes (e.g.
          "high degree (top 3%)");
        - ``links_to``: where the role's edges go, as shares of its edge
          endpoints; ``self`` is the share staying inside the role;
        - ``examples``: representative nodes.
    """
    import pandas as pd

    labels = np.asarray(labels)
    nodes = np.array(list(G.nodes), dtype=object)
    stats, A = _node_stats(G)
    roles, sizes = np.unique(labels, return_counts=True)
    onehot = csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)),
                                np.searchsorted(roles, labels))),
        shape=(len(labels), len(roles)))
    # edge endpoints from each role (rows) to each role (columns)
    mixing = (onehot.T @ A @ onehot).toarray()

    rows = []
    for i, role in enumerate(roles):
        mask = labels == role
        row = {"role": role, "n_nodes": int(sizes[i]),
               "share": sizes[i] / len(labels)}
        profile = []
        for name, values in stats.items():
            row[name] = np.median(values[mask])
            # mid-rank, so that ties (e.g. many zero clusterings) count half
            rank = (np.mean(values < row[name])
                    + np.mean(values <= row[name])) / 2
            name = name.replace("_", " ")
            if rank >= 3 / 4:
                profile.append((rank, f"high {name} (top {1 - rank:.0%})"))
            elif rank <= 1 / 4:
                profile.append((1 - rank, f"low {name} (bottom {rank:.0%})"))
        row["profile"] = ", ".join(p for _, p in sorted(profile)[::-1])

        out = mixing[i] / max(mixing[i].sum(), 1)
        links = [
            f"{'self' if j == i else roles[j]} ({out[j]:.0%})"
            for j in np.argsort(-out)[:n_links] if out[j] > 0]
        row["links_to"] = ", ".join(links)

        idx = np.flatnonzero(mask)
        if X is not None:
            dist = np.linalg.norm(X[idx] - X[idx].mean(axis=0), axis=1)
            idx = idx[np.argsort(dist, kind="stable")]
        else:
            idx = idx[np.argsort(-stats["degree"][idx], kind="stable")]
        row["examples"] = list(nodes[idx[:n_examples]])
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("role")
    return summary.sort_values("n_nodes", ascending=False)
