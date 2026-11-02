"""A million nodes with a missing-mass budget: the certified setting at scale.

Barabasi-Albert (m = 3) and Watts-Strogatz (k = 6, p = 0.1) graphs of
100k and 1M nodes, one core, tol 1e-4 with budgets of 5% and 1% (and the
threshold alone, for reference): time, peak memory, and the distribution of
the per-node certificates and missing masses returned by the library.

    python certified_scale.py
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

GRAPHS = {
    "barabasi_albert": lambda n: nx.barabasi_albert_graph(n, 3, seed=0),
    "watts_strogatz": lambda n: nx.watts_strogatz_graph(n, 6, 0.1, seed=0),
}
SIZES = (100_000, 1_000_000)
BUDGETS = (None, 0.05, 0.01)


def run(name, n, budget, queue):
    from rolewalk import RoleWalk
    RoleWalk().transform(nx.path_graph(3))  # compile first
    G = GRAPHS[name](n)
    start = time.perf_counter()
    model = RoleWalk(tol=1e-4, max_missing=budget)
    model.transform(G)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    cert, miss = model.certificate_, model.missing_.max(axis=1)
    queue.put({"graph": name, "n_nodes": n, "max_missing": budget, "time": elapsed,
               "peak_gb": peak, "certificate_median": np.median(cert),
               "certificate_p99": np.quantile(cert, 0.99), "certificate_max": cert.max(),
               "missing_p99": np.quantile(miss, 0.99), "missing_max": miss.max(),
               "share_rows_missing_over_1pct": (miss > 0.01).mean()})


def main():
    rows, start, total = [], time.time(), len(GRAPHS) * len(SIZES) * len(BUDGETS)
    ctx = mp.get_context("fork")
    for name in GRAPHS:
        for n in SIZES:
            for budget in BUDGETS:
                queue = ctx.Queue()
                p = ctx.Process(target=run, args=(name, n, budget, queue))
                p.start()
                p.join(timeout=3600)
                if p.is_alive():
                    p.terminate()
                    row = {"graph": name, "n_nodes": n, "max_missing": budget,
                           "time": np.nan, "status": "timeout 3600 s"}
                elif queue.empty():
                    row = {"graph": name, "n_nodes": n, "max_missing": budget,
                           "time": np.nan, "status": f"failed (exit {p.exitcode})"}
                else:
                    row = {**queue.get(), "status": "ok"}
                rows.append(row)
                pd.DataFrame(rows).to_csv("certified_scale.csv", index=False)
                el = time.time() - start
                print(f"[{len(rows)}/{total}] {name} n={n} budget={budget}: "
                      f"{row.get('time', float('nan')):.1f} s, {row['status']}  "
                      f"elapsed {el / 60:.1f} min", flush=True)
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
