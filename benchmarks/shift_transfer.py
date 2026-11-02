"""Centrality transfer under a change of generator parameters.

Same protocol as ``centrality_benchmark.py --transfer`` (regressor fitted on
one 2,000-node graph, Spearman on another), but the test graph comes from
the same family with one parameter changed, fixed in advance:

- Barabasi-Albert: m = 3 -> 5 (mean degree 6 -> 10);
- powerlaw-cluster: triangle probability 0.3 -> 0.6 (clustering);
- Watts-Strogatz: rewiring 0.1 -> 0.3;
- SBM: within-block probabilities doubled (stronger communities);
- random geometric: radius 0.04 -> 0.06 (density x2.25).

10 independent pairs of graphs per shift. Analyse with
``python transfer_analysis.py "transfer_shift/*.csv"``.

    python shift_transfer.py --shift barabasi_albert
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd

import centrality_benchmark as cb


def sbm(s, strength):
    sizes = [400, 400, 400, 200, 100]
    probs = np.full((5, 5), 0.0005) + strength * np.diag([0.02, 0.02, 0.02, 0.04, 0.08])
    return nx.stochastic_block_model(sizes, probs, seed=s)


SHIFTS = {
    "barabasi_albert": (lambda s: nx.barabasi_albert_graph(2000, 3, seed=s),
                        lambda s: nx.barabasi_albert_graph(2000, 5, seed=s)),
    "powerlaw_cluster": (lambda s: nx.powerlaw_cluster_graph(2000, 3, 0.3, seed=s),
                         lambda s: nx.powerlaw_cluster_graph(2000, 3, 0.6, seed=s)),
    "watts_strogatz": (lambda s: nx.connected_watts_strogatz_graph(2000, 6, 0.1, seed=s),
                       lambda s: nx.connected_watts_strogatz_graph(2000, 6, 0.3, seed=s)),
    "sbm": (lambda s: sbm(s, 1.0), lambda s: sbm(s, 2.0)),
    "random_geometric": (lambda s: nx.random_geometric_graph(2000, 0.04, seed=s),
                         lambda s: nx.random_geometric_graph(2000, 0.06, seed=s)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shift", choices=list(SHIFTS), required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--features", nargs="*",
                        default=["degree", "rolewalk", "rolewalk_curvature", "rwse",
                                 "graphwave_centered", "heat_cf", "topk_layers",
                                 "local_stats", "refex"])
    parser.add_argument("--output-dir", default="transfer_shift")
    args = parser.parse_args()
    cb.SELECTED[:] = args.features
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"{args.shift}.csv")

    train_graph, test_graph = SHIFTS[args.shift]
    rows, start = [], time.time()
    for seed in range(args.seeds):
        G_train = cb.largest_component(train_graph(seed))
        G_test = cb.largest_component(test_graph(seed + 100))
        # cache names distinct from the size experiment
        rows += cb.transfer(f"shift_{args.shift}", G_train, G_test, seed, scale="shift")
        pd.DataFrame(rows).assign(graph=args.shift).to_csv(output, index=False)
        elapsed = time.time() - start
        print(f"[{seed + 1}/{args.seeds}] {args.shift}  elapsed {elapsed / 60:.1f} min  "
              f"ETA {elapsed / (seed + 1) * (args.seeds - seed - 1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
