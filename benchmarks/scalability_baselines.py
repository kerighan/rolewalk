"""Embedding time of the baselines on Barabasi-Albert graphs (m = 3) of
10k, 100k and 1M nodes, one core, with a time limit per run.

Each run is a separate process killed after ``--limit`` seconds and capped
at ``--memory`` GB of address space; once a method times out or runs out of
memory, larger sizes are skipped. struc2vec and GraphWave are
left out: they already stop below 10^4 nodes (dense n x n, or quadratic
structural distances). Appends to scalability_baselines.csv.

    python scalability_baselines.py
"""
import argparse
import csv
import multiprocessing as mp
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

SIZES = (10_000, 100_000, 1_000_000)
METHODS = ("rolewalk", "refex", "rwse", "heat_cf", "resd", "role2vec")


def run(method, n, queue, memory_gb):
    import resource

    cap = int(memory_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import networkx as nx
    import role_benchmark as rb

    G = nx.barabasi_albert_graph(n, 3, seed=0)
    start = time.perf_counter()
    rb.METHODS[method](G)
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    queue.put((elapsed, rss))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=float, default=3600)
    parser.add_argument("--memory", type=float, default=16)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--output", default="scalability_baselines.csv")
    args = parser.parse_args()

    new = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["method", "n_nodes", "time", "peak_rss_gb", "status"])
        for method in args.methods:
            for n in SIZES:
                queue = mp.Queue()
                p = mp.Process(target=run, args=(method, n, queue, args.memory))
                p.start()
                p.join(args.limit)
                if p.is_alive():
                    p.kill()
                    p.join()
                    row = [method, n, "", "", f"timeout {args.limit:.0f}s"]
                elif p.exitcode != 0 or queue.empty():
                    row = [method, n, "", "", f"failed (exit {p.exitcode}; "
                           f"{args.memory:.0f} GB cap)"]
                else:
                    elapsed, rss = queue.get()
                    row = [method, n, round(elapsed, 2), round(rss, 2), "ok"]
                writer.writerow(row)
                f.flush()
                print(row, flush=True)
                if row[-1] != "ok":
                    break


if __name__ == "__main__":
    main()
