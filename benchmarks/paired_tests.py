"""Paired differences in accuracy on the role benchmarks, with intervals.

Every method is scored on the same 50 stratified 80/20 splits; for each
split we take the difference with RoleWalk (+ curvature). Repeated random
splits share training data, so the plain t-interval is too narrow: we use
the corrected resampled t-test of Nadeau and Bengio (2003), with variance
(1/k + n_test/n_train) s^2. Embeddings come from the cache of
role_benchmark.py.

    python paired_tests.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy import stats

from role_benchmark import DATASETS, embed, within

REFERENCES = ("rolewalk", "rolewalk_curvature")
OTHERS = ("graphwave", "graphwave_centered", "heat_cf", "resd", "refex",
          "refex_directed", "role2vec", "rwse", "topk_layers", "local_stats",
          "rolewalk_mixed", "rolewalk_curvature_mixed", "degree")
NAMES = ("brazil", "europe", "usa", "reality-call", "actor", "film",
         "rfa-candidates", "wiki-rfa")
SPLITS = 50


def main():
    rows = []
    for name in NAMES:
        G, y = DATASETS[name][0]()
        keep = y >= 0
        scores = {}
        for method in REFERENCES + OTHERS:
            path = os.path.join("data", "embeddings", f"{name}__{method}.npz")
            if not os.path.exists(path):
                continue
            X, _ = embed(name, method, G, cache=True)
            scores[method] = within(X[keep], y[keep], SPLITS).accuracy.values
        for ref in REFERENCES:
            for other in scores:
                if other == ref:
                    continue
                d = scores[ref] - scores[other]
                var = (1 / SPLITS + 0.2 / 0.8) * d.var(ddof=1)
                t = d.mean() / np.sqrt(var)
                half = stats.t.ppf(0.975, SPLITS - 1) * np.sqrt(var)
                rows.append({"dataset": name, "reference": ref, "other": other,
                             "diff": d.mean(), "low": d.mean() - half,
                             "high": d.mean() + half,
                             "p": 2 * stats.t.sf(abs(t), SPLITS - 1)})
        print(name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("paired_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
