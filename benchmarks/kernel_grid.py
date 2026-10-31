"""Grid search of RoleWalk's kernel for unsupervised role discovery.

In raw embeddings, the largest theta dominates distances (about half of
the variance at theta = 100), so k-means and nearest neighbours mostly see
one frequency. This script searches how to weight the frequencies, along
with the theta range, the walk length and the onion layers, so that
unsupervised roles (k-means ARI) are best on synthetic and real networks
together:

- ``power`` p: characteristic-function features are divided by theta^p.
  p = 0 is RoleWalk as is; p = 1 expresses every frequency in units of
  probability mass, since Im phi ~ theta * mass for small probabilities;
- ``upper``: largest theta (log-spaced from 1, ``n_samples`` = 10);
- ``walk_len``;
- ``shells``: onion layers, scaled like the first-order term of the
  characteristic function, ``||theta^(1-p)||`` per unit of mass.

The score averages two means with equal weight: synthetic motifs (varied,
houses, directed, 5% edge noise, 3 seeds) and real networks (airports,
reality-call, actor). Supervised accuracy (standardised, so insensitive to
the weighting) and airport transfer are reported as checks.

    python kernel_grid.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

from rolewalk import RoleWalk
from datasets import (generate_directed_shapes, generate_houses,
                      generate_varied, load_airports, load_rone)
from role_benchmark import classifier, within
from synthetic_benchmark import add_noise

GRID = {
    "power": [0.0, 0.5, 1.0],
    "upper": [10, 30, 100, 300],
    "walk_len": [3, 4],
    "shells": [False, True],
}


def embed(G, power, upper, walk_len, shells):
    rw = RoleWalk(walk_len=walk_len, bounds=(1, upper), shells=shells)
    X = rw.transform(G)
    theta = rw.theta.ravel()
    cf = np.tile(theta ** -power, 2)
    mass = np.linalg.norm(theta ** (1 - power))
    weights = []
    for w in range(1, walk_len + 1):
        weights += [cf, np.full(w + 1 if shells else 0, mass)]
    weights = np.concatenate(weights)
    if G.is_directed():
        weights = np.concatenate([weights, weights])
    return X * weights


def ari(X, y, seed=0):
    keep = y >= 0
    pred = KMeans(len(np.unique(y[keep])), n_init=10,
                  random_state=seed).fit_predict(X[keep])
    return adjusted_rand_score(y[keep], pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="kernel_grid.csv")
    args = parser.parse_args()

    synthetic = {}
    for seed in range(args.seeds):
        for kind, make in (("varied", lambda s: generate_varied(5, seed=s)),
                           ("houses", lambda s: generate_houses(10, seed=s)),
                           ("directed", lambda s: generate_directed_shapes(5, seed=s))):
            G, y = make(seed)
            synthetic[kind, seed] = (add_noise(G, 0.05, np.random.default_rng(seed)), y)
    real = {r: load_airports(r) for r in ("brazil", "europe", "usa")}
    real.update({n: load_rone(n) for n in ("reality-call", "actor")})
    airports = ("brazil", "europe", "usa")

    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        row = dict(params)
        for kind in ("varied", "houses", "directed"):
            row[kind] = np.mean([ari(embed(G, **params), y, s)
                                 for (k, s), (G, y) in synthetic.items() if k == kind])
        E = {name: embed(G, **params) for name, (G, y) in real.items()}
        for name, (G, y) in real.items():
            row[name] = ari(E[name], y)
        row["synthetic"] = np.mean([row[k] for k in ("varied", "houses", "directed")])
        row["real"] = np.mean([row[n] for n in real])
        row["score"] = (row["synthetic"] + row["real"]) / 2
        # checks: supervised airports and transfer between airports
        row["airports_acc"] = np.mean([within(E[r], real[r][1], 10).accuracy.mean()
                                       for r in airports])
        row["transfer"] = np.mean([
            accuracy_score(real[t][1], classifier().fit(E[s], real[s][1]).predict(E[t]))
            for s in airports for t in airports if s != t])
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(args.output, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.head(15).round(3).to_string(index=False))
        default = df[(df.power == 0) & (df.upper == 100) & (df.walk_len == 3)]
        print("\ncurrent defaults:")
        print(default.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
