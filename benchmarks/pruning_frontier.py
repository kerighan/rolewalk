"""Certified pruning as a deployment setting: accuracy, time, memory.

On hub-heavy graphs (Barabasi-Albert, a directed scale-free graph) and on
wiki-RfA (directed, real), each configuration runs in its own process (one
core) and reports time, peak memory, the error of every node's embedding
against the exact one (tol = 0): median, 99th percentile and maximum of the
relative error, and the per-node certificates of ``RoleWalk.certificate_``:
their maximum, and the largest ratio of observed error to certificate
(at most 1 in exact arithmetic).

    python pruning_frontier.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp
import resource
import time

import networkx as nx
import numpy as np
import pandas as pd

CONFIGS = [(0.0, None), (1e-4, None), (1e-3, None),
           (1e-4, 0.01), (1e-4, 0.05), (1e-3, 0.01), (1e-3, 0.05)]


def graphs():
    from datasets import load_wiki_rfa
    yield "barabasi_albert_50k", nx.barabasi_albert_graph(50_000, 3, seed=0)
    D = nx.DiGraph(nx.scale_free_graph(50_000, seed=0))
    D.remove_edges_from(nx.selfloop_edges(D))
    yield "scale_free_directed_50k", D
    yield "wiki_rfa", load_wiki_rfa()[0]


def run(args):
    G, tol, budget, queue = args
    from rolewalk import RoleWalk
    RoleWalk().transform(nx.path_graph(3))  # compile first
    start = time.perf_counter()
    model = RoleWalk(tol=tol, max_missing=budget)
    X = model.transform(G)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # GB
    queue.put((X, model.certificate_, model.missing_.max(), elapsed, peak))


def in_process(G, tol, budget):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    p = ctx.Process(target=run, args=((G, tol, budget, queue),))
    p.start()
    out = queue.get()
    p.join()
    return out


def main():
    rows, start = [], time.time()
    total = 3 * len(CONFIGS)
    done = 0
    for name, G in graphs():
        G = nx.convert_node_labels_to_integers(G)
        X0 = None
        for tol, budget in CONFIGS:
            X, cert, worst_missing, elapsed, peak = in_process(G, tol, budget)
            if X0 is None:
                X0 = X
            err = np.linalg.norm(X - X0, axis=1)
            rel = err / np.maximum(np.linalg.norm(X0, axis=1), 1e-12)
            rows.append({
                "graph": name, "n_nodes": len(G), "tol": tol, "max_missing": budget,
                "time": elapsed, "peak_gb": peak,
                "rel_err_median": np.median(rel), "rel_err_p99": np.quantile(rel, 0.99),
                "rel_err_max": rel.max(), "abs_err_max": err.max(),
                "certificate_max": cert.max(), "certificate_p99": np.quantile(cert, 0.99),
                "err_over_certificate_max": np.max(err / np.maximum(cert, 1e-12)
                                                    * (cert > 1e-9)),
                "missing_max": worst_missing})
            pd.DataFrame(rows).to_csv("pruning_frontier.csv", index=False)
            done += 1
            el = time.time() - start
            print(f"[{done}/{total}] {name} tol={tol} budget={budget}: {elapsed:.1f} s, "
                  f"{peak:.2f} GB, max rel err {rel.max():.3f}  "
                  f"elapsed {el / 60:.1f} min  ETA {el / done * (total - done) / 60:.1f} min",
                  flush=True)
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
