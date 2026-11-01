"""Directed motifs: linear versus gradient-boosting classifier.

ReFeX's quantile one-hot features give a linear classifier piecewise
rules; this checks whether its lead on noisy directed motifs comes from the
features or from that. Same graphs as directed_benchmark.py (3 seeds), 10
stratified 50/50 splits per graph, both classifiers on the same splits.

    python directed_gb.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import role_benchmark as rb
from datasets import generate_directed_shapes
from directed_benchmark import degree_in_out
from synthetic_benchmark import add_noise

METHODS = {
    "rolewalk_curvature": rb.rolewalk_curvature,
    "rolewalk": rb.rolewalk,
    "refex_directed": rb.refex_directed,
    "local_stats": rb.local_stats,
    "topk_layers": rb.topk_layers,
    "degree": degree_in_out,
}
CLASSIFIERS = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}


def main():
    rows = []
    for seed in range(3):
        G, y = generate_directed_shapes(5, seed=seed)
        for noise in (0.0, 0.05, 0.1):
            H = add_noise(G, noise, np.random.default_rng(seed))
            splits = list(StratifiedShuffleSplit(10, test_size=0.5, random_state=seed).split(H, y))
            for method, f in METHODS.items():
                X = f(H)
                for name, make in CLASSIFIERS.items():
                    acc = [accuracy_score(y[te], make().fit(X[tr], y[tr]).predict(X[te]))
                           for tr, te in splits]
                    rows.append({"seed": seed, "noise": noise, "method": method,
                                 "classifier": name, "accuracy": np.mean(acc)})
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("directed_gb.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.groupby(["classifier", "method", "noise"]).accuracy.mean()
              .unstack("noise").round(3))


if __name__ == "__main__":
    main()
