"""Run Digraphwave (Ceylan et al. 2022) on an adjacency saved by
baselines.digraphwave, in its own environment (torch + torch_sparse).

    python dgw_runner.py adjacency.npz out.npy R k_emb
"""
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import digraphwave.digraphwave as digw


def main():
    path, out, R, k_emb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    # Digraphwave's convention: adj[target, source], CSC
    adj = sp.load_npz(path).T.tocsc().astype(np.float64)
    param = digw.DigraphwaveHyperparameters.create(
        num_nodes=adj.shape[0], num_edges=adj.nnz, R=R, k_emb=k_emb,
        arctan_log_transform=False, n_flag=False, t_flag=True, a_flag=True)
    start = time.perf_counter()
    emb = digw.digraphwave(adj=adj, param=param, device_ids=None)
    np.save(out, emb)
    with open(out + ".time", "w") as f:  # compute time, without loading torch
        f.write(str(time.perf_counter() - start))


if __name__ == "__main__":
    main()
