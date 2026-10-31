"""struc2vec is stochastic: average it over several runs on the airports.

Its walks are not seeded, so a single run can be lucky or unlucky (Europe
moved from 0.56 to 0.60 between two runs). Each run is scored like
``role_benchmark.py`` (50 stratified 80/20 splits); ``struc2vec_runs.csv``
keeps every run, the summary gives mean and std across runs.

    python struc2vec_runs.py --runs 5
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse

import pandas as pd

import role_benchmark as rb
from datasets import load_airports


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="struc2vec_runs.csv")
    args = parser.parse_args()

    rows = []
    for region in rb.REGIONS:
        G, y = load_airports(region)
        for run in range(args.runs):
            X = rb.struc2vec(G)
            rows.append({"dataset": region, "run": run,
                         "accuracy": rb.within(X, y, 50).accuracy.mean()})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("dataset").accuracy.agg(["mean", "std", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
