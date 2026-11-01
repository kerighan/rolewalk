"""Log-spaced or uniform frequencies, with the current defaults.

Same protocol and scores as kernel_grid.py (k-means ARI on noisy synthetic
motifs and real networks, supervised airport accuracy, airport transfer),
for several ways of spreading m = 10 frequencies.

    python theta_scheme.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from rolewalk import RoleWalk
from datasets import (generate_directed_shapes, generate_houses,
                      generate_varied, load_airports, load_rone)
from kernel_grid import ari
from role_benchmark import classifier, within
from synthetic_benchmark import add_noise

SCHEMES = {
    "geomspace [1, 100] (default)": ("geomspace", (1, 100)),
    "linspace [1, 100]": ("linspace", (1, 100)),
    "linspace [0.001, 100]": ("linspace", (1e-3, 100)),
    "linspace [1, 30]": ("linspace", (1, 30)),
    "geomspace [0.3, 300]": ("geomspace", (0.3, 300)),
}
AIRPORTS = ("brazil", "europe", "usa")


def main(seeds=3):
    synthetic = {}
    for seed in range(seeds):
        for kind, make in (("varied", lambda s: generate_varied(5, seed=s)),
                           ("houses", lambda s: generate_houses(10, seed=s)),
                           ("directed", lambda s: generate_directed_shapes(5, seed=s))):
            G, y = make(seed)
            synthetic[kind, seed] = (add_noise(G, 0.05, np.random.default_rng(seed)), y)
    real = {r: load_airports(r) for r in AIRPORTS}
    real.update({n: load_rone(n) for n in ("reality-call", "actor")})

    rows = []
    for name, (scheme, bounds) in SCHEMES.items():
        model = RoleWalk(walk_len=3, theta_scheme=scheme, bounds=bounds)
        row = {"scheme": name}
        for kind in ("varied", "houses", "directed"):
            row[kind] = np.mean([ari(model.transform(G), y, s)
                                 for (k, s), (G, y) in synthetic.items() if k == kind])
        E = {n: model.transform(G) for n, (G, y) in real.items()}
        for n, (G, y) in real.items():
            row[n] = ari(E[n], y)
        row["synthetic"] = np.mean([row[k] for k in ("varied", "houses", "directed")])
        row["real"] = np.mean([row[n] for n in real])
        row["score"] = (row["synthetic"] + row["real"]) / 2
        row["airports_acc"] = np.mean([within(E[r], real[r][1], 50).accuracy.mean()
                                       for r in AIRPORTS])
        row["transfer"] = np.mean([
            accuracy_score(real[t][1], classifier().fit(E[s], real[s][1]).predict(E[t]))
            for s in AIRPORTS for t in AIRPORTS if s != t])
        rows.append(row)
        print(name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("theta_scheme.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
