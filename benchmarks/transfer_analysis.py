"""Centrality transfer to 4x larger graphs, per family and with paired
graph-level intervals.

Reads the per-family outputs of

    python centrality_benchmark.py --transfer --test-scale 4 --seeds 10 \
        --graphs <family> --output transfer_seeds/x4_<family>.csv

Each (family, seed) is an independent pair of graphs (training graph seed s,
test graph seed s + 100, four times larger). For every competitor, the
difference with RoleWalk is taken per pair (mean over targets, or per
target), and summarised by its mean and a t-interval over pairs: pairs are
independent, so no correction for overlap is needed. Pairs with a constant
target (k-core on Barabasi-Albert) are left out.

    python transfer_analysis.py                                  # size, 4x
    python transfer_analysis.py "transfer_shift/*.csv" shift_paired.csv
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

TARGETS = ["betweenness", "closeness", "harmonic", "pagerank",
           "clustering", "core_number", "eigenvector"]
REFERENCE = "rolewalk"
OTHERS = ["rolewalk_curvature", "topk_layers", "local_stats", "refex_frozen",
          "heat_cf", "rwse", "graphwave_centered", "degree"]


def interval(d):
    d = np.asarray(d)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = 2 * stats.t.sf(abs(d.mean()) / (d.std(ddof=1) / np.sqrt(len(d))), len(d) - 1)
    return d.mean(), d.mean() - half, d.mean() + half, p


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "transfer_seeds/x4_*.csv"
    output = sys.argv[2] if len(sys.argv) > 2 else "transfer_paired.csv"
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))])
    df = df[df.target.isin(TARGETS)]
    # drop (pair, target) cells whose target is constant: NaN for every method
    wide = df.pivot_table(index=["graph", "seed", "target"], columns="features",
                          values="spearman", dropna=False)
    wide = wide[wide[REFERENCE].notna()]
    print(f"{wide.index.droplevel('target').nunique()} graph pairs, {len(wide)} cells\n")

    methods = [REFERENCE] + [m for m in OTHERS if m in wide]
    per_family = wide[methods].groupby("graph").mean().round(3)
    per_family.loc["mean"] = wide[methods].mean().round(3)
    with pd.option_context("display.width", 200):
        print("Mean Spearman per family (all targets)")
        print(per_family.to_string(), "\n")
        print("Mean Spearman per target")
        print(wide[methods].groupby("target").mean().loc[TARGETS].round(3).to_string(), "\n")

    rows = []
    pair_mean = wide[methods].groupby(["graph", "seed"]).mean()
    for other in methods[1:]:
        d = (pair_mean[REFERENCE] - pair_mean[other]).dropna()
        mean, low, high, p = interval(d)
        wins = (d > 0).mean()
        fam = (d.groupby("graph").mean() > 0).sum()
        rows.append({"other": other, "target": "mean", "diff": mean, "low": low,
                     "high": high, "p": p, "pairs won": wins,
                     "families won": f"{fam}/{d.index.get_level_values('graph').nunique()}"})
        for target in TARGETS:
            cell = wide.xs(target, level="target")
            d = (cell[REFERENCE] - cell[other]).dropna()
            if len(d) < 3:
                continue
            mean, low, high, p = interval(d)
            fam = (d.groupby("graph").mean() > 0).sum()
            rows.append({"other": other, "target": target, "diff": mean, "low": low,
                         "high": high, "p": p, "pairs won": (d > 0).mean(),
                         "families won": f"{fam}/{d.index.get_level_values('graph').nunique()}"})
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print("RoleWalk minus other: paired over graph pairs (95% t-interval)")
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
