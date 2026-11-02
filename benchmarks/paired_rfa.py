"""Paired differences on wiki-RfA (tasks A and B) against RoleWalk + curv.
(with mixed probes, the default), accuracy and macro-F1, Nadeau-Bengio
corrected intervals over the 50 splits (see paired_tests.py).

    python paired_rfa.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy import stats

from role_benchmark import DATASETS, embed, within

REFERENCE = "rolewalk_curvature_mixed"
OTHERS = ("rolewalk_mixed", "local_stats", "topk_layers", "refex_directed", "refex",
          "digraphwave", "resd", "role2vec", "rwse_inout", "degree")
SPLITS = 50


def main():
    rows = []
    for name in ("rfa-candidates", "wiki-rfa"):
        G, y = DATASETS[name][0]()
        keep = y >= 0
        scores = {}
        for method in (REFERENCE,) + OTHERS:
            X, _ = embed(name, method, G, cache=True)
            scores[method] = within(X[keep], y[keep], SPLITS)
            print(name, method, flush=True)
        for other in OTHERS:
            for metric in ("accuracy", "macro_f1"):
                d = scores[REFERENCE][metric].values - scores[other][metric].values
                var = (1 / SPLITS + 0.2 / 0.8) * d.var(ddof=1)
                half = stats.t.ppf(0.975, SPLITS - 1) * np.sqrt(var)
                rows.append({"dataset": name, "other": other, "metric": metric,
                             "diff": d.mean(), "low": d.mean() - half, "high": d.mean() + half,
                             "p": 2 * stats.t.sf(abs(d.mean()) / np.sqrt(var), SPLITS - 1)})
    df = pd.DataFrame(rows)
    df.to_csv("paired_rfa.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
